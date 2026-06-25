//===- TesseractOps.td - Tesseract dialect ops -------------*- tablegen -*-===//
//
// Defines the tesseract dialect op set. The list mirrors the C++-level
// graph::Op kinds emitted by src/graph/GraphScope.cpp; both must stay in
// lockstep (see ADR-0004). Op signatures are deliberately ranked-tensor
// specific so that later lowering passes can pattern-match shapes.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERACT_IR_OPS
#define TESSERACT_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "TesseractDialect.td"

class Tesseract_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tesseract_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

def Tesseract_ConstantOp : Tesseract_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Materialize a dense constant tensor.";
  let description = [{
    Creates a constant tensor with the values encoded in `value`. The result
    type must match the attribute shape.
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Binary elementwise
//
// For M1 we keep the `SameOperandsAndResultType` contract: call sites on the
// Tesseract side either pre-broadcast through `ops::broadcast_to` or emit a
// `tesseract.broadcast_to` immediately before the binary op. This keeps op
// verifiers simple at the cost of an extra explicit op; a ranked-broadcast
// trait is tracked as part of M2.
//===----------------------------------------------------------------------===//

class Tesseract_BinaryElementwiseOp<string mnemonic> :
    Tesseract_Op<mnemonic,
                 [Pure, SameOperandsAndResultType, ElementwiseMappable]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Tesseract_AddOp : Tesseract_BinaryElementwiseOp<"add"> {
  let summary = "Elementwise addition.";
}
def Tesseract_SubOp : Tesseract_BinaryElementwiseOp<"sub"> {
  let summary = "Elementwise subtraction.";
}
def Tesseract_MulOp : Tesseract_BinaryElementwiseOp<"mul"> {
  let summary = "Elementwise multiplication.";
}
def Tesseract_DivOp : Tesseract_BinaryElementwiseOp<"div"> {
  let summary = "Elementwise true-division (floating-point dtypes only).";
}

//===----------------------------------------------------------------------===//
// Unary elementwise (same shape / dtype in and out)
//===----------------------------------------------------------------------===//

class Tesseract_UnaryElementwiseOp<string mnemonic> :
    Tesseract_Op<mnemonic,
                 [Pure, SameOperandsAndResultType, ElementwiseMappable]> {
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def Tesseract_NegOp     : Tesseract_UnaryElementwiseOp<"neg">     { let summary = "Elementwise negation."; }
def Tesseract_ReluOp    : Tesseract_UnaryElementwiseOp<"relu">    { let summary = "Elementwise max(x, 0)."; }
def Tesseract_SigmoidOp : Tesseract_UnaryElementwiseOp<"sigmoid"> { let summary = "Elementwise logistic sigmoid."; }
def Tesseract_TanhOp    : Tesseract_UnaryElementwiseOp<"tanh">    { let summary = "Elementwise hyperbolic tangent."; }
def Tesseract_ExpOp     : Tesseract_UnaryElementwiseOp<"exp">     { let summary = "Elementwise natural exponent."; }
def Tesseract_LogOp     : Tesseract_UnaryElementwiseOp<"log">     { let summary = "Elementwise natural logarithm."; }
def Tesseract_SqrtOp    : Tesseract_UnaryElementwiseOp<"sqrt">    { let summary = "Elementwise square root."; }

//===----------------------------------------------------------------------===//
// MatMul (rank-2 only at M1; batched matmul tracked by backlog B-004)
//===----------------------------------------------------------------------===//

def Tesseract_MatMulOp : Tesseract_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication of two rank-2 tensors.";
  let description = [{
    Computes `result = lhs * rhs` where `lhs : MxK`, `rhs : KxN`, and
    `result : MxN`. Only rank-2 inputs are accepted at M1.
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Quantized matmul (Wave 20 / B-037)
//
// Carries the quantization format as IR attributes so the graph compiler can
// reason about FP8 / FP4 / INT4 / INT8 weights at the IR level (vs. only at
// runtime as in B-021). `lhs : [M,K]` floating activations multiply a
// quantized weight in `qweight` (integer/byte storage) with dequantization
// metadata in `scale`; `scheme` selects the format and `group_size` is the
// channel-group width for grouped schemes. Lowering (dequant → matmul, or a
// fused dequant-matmul kernel) is a follow-up; this op pins the contract.
//===----------------------------------------------------------------------===//

def Tesseract_DequantMatMulOp : Tesseract_Op<"dequant_matmul", [Pure]> {
  let summary = "Matmul against a quantized weight, dequantized on the fly.";
  let description = [{
    Computes `result = lhs * dequant(qweight, scale, scheme)` where
    `lhs : [M, K]` is floating, `qweight` holds the packed quantized weight,
    and `scale` carries the dequantization scale(s). `scheme` is one of
    `int8`, `int4_group`, `fp8_e4m3`, `fp8_e5m2`, `fp4`. `group_size` is the
    per-channel group width for grouped schemes (`int4_group`, `fp4`) and
    must be -1 for per-tensor schemes (`int8`, `fp8_e4m3`, `fp8_e5m2`).
    `result : [M, N]` is floating.
  }];

  let arguments = (ins AnyRankedTensor:$lhs,
                       AnyRankedTensor:$qweight,
                       AnyRankedTensor:$scale,
                       StrAttr:$scheme,
                       DefaultValuedAttr<SI64Attr, "-1">:$group_size);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat =
      "$lhs `,` $qweight `,` $scale attr-dict `:` `(` type($lhs) `,`"
      " type($qweight) `,` type($scale) `)` `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reductions
//
// `dim` == -1 means "reduce all axes" (result is a 0-D tensor). `keepdim`
// controls whether the reduced axis is kept with size 1. Semantics mirror
// `tesseract::ops::sum / mean / max`.
//===----------------------------------------------------------------------===//

class Tesseract_ReductionOp<string mnemonic> : Tesseract_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyRankedTensor:$input,
                       DefaultValuedAttr<SI64Attr, "-1">:$dim,
                       DefaultValuedAttr<BoolAttr, "false">:$keepdim);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` `(` type($input) `)` `->` type($result)";
}

def Tesseract_SumOp  : Tesseract_ReductionOp<"sum">  { let summary = "Sum reduction."; }
def Tesseract_MeanOp : Tesseract_ReductionOp<"mean"> { let summary = "Mean reduction."; }
def Tesseract_MaxOp  : Tesseract_ReductionOp<"max">  { let summary = "Maximum reduction."; }

//===----------------------------------------------------------------------===//
// Softmax family
//===----------------------------------------------------------------------===//

class Tesseract_SoftmaxLikeOp<string mnemonic> :
    Tesseract_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyRankedTensor:$input,
                       DefaultValuedAttr<SI64Attr, "-1">:$dim);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Tesseract_SoftmaxOp    : Tesseract_SoftmaxLikeOp<"softmax"> {
  let summary = "Numerically stable softmax along `dim`.";
}
def Tesseract_LogSoftmaxOp : Tesseract_SoftmaxLikeOp<"log_softmax"> {
  let summary = "Numerically stable log-softmax along `dim`.";
}

//===----------------------------------------------------------------------===//
// Rotary position embedding (RoPE)
//
// Mirrors `tesseract::ops::rotary_embedding`. Rotates each adjacent pair
// `(x[..., 2j], x[..., 2j+1])` of the last dim by the per-position angle
// encoded in the `cos` / `sin` tables (`[S_table, D]`, only the first `S`
// rows are read). The result shares `input`'s shape / dtype. Kept as a
// single dialect node so the emitter stays a 1:1 mirror of the graph op;
// `--convert-tesseract-to-linalg` decomposes it into strided slices + an
// elementwise `x·cos + rotate_half(x)·sin` generic.
//===----------------------------------------------------------------------===//

def Tesseract_RotaryEmbeddingOp : Tesseract_Op<"rotary_embedding", [Pure]> {
  let summary = "Rotary position embedding over interleaved last-dim pairs.";
  let description = [{
    `input : [..., S, D]` (D even), `cos` / `sin : [S_table, D]` with
    `S_table >= S`. Produces a tensor of the same type as `input`.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$cos,
                       AnyRankedTensor:$sin);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat =
      "$input `,` $cos `,` $sin attr-dict `:` `(` type($input) `,`"
      " type($cos) `,` type($sin) `)` `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Cross-entropy with logits (composite op; kept as a single dialect op at
// M1 so lowering can emit a fused reference kernel. Split into the
// canonical log_softmax + gather + negative-log-likelihood triple once the
// indexing ops land (backlog B-003).
//===----------------------------------------------------------------------===//

def Tesseract_CrossEntropyWithLogitsOp :
    Tesseract_Op<"cross_entropy_with_logits", [Pure]> {
  let summary = "Mean softmax cross-entropy over class targets.";
  let description = [{
    `logits : [N, C]` (floating), `targets : [N]` (Int64). Produces a 0-D
    scalar `loss` of the same dtype as `logits`.
  }];
  let arguments = (ins AnyRankedTensor:$logits, AnyRankedTensor:$targets);
  let results = (outs AnyRankedTensor:$loss);
  let assemblyFormat =
      "$logits `,` $targets attr-dict `:` `(` type($logits) `,` type($targets) `)` `->` type($loss)";
}

//===----------------------------------------------------------------------===//
// Fused backward ops
//
// The autograd pass (`--tesseract-backward`) needs to generate gradients
// for ops that would otherwise require primitives we haven't lowered yet
// (notably a comparison-based mask for ReLU, and softmax + scatter for
// cross-entropy). Rather than leave those forward ops ungradable, we
// expose fused backward ops in the dialect itself. They close the AD
// contract for M1I ("graph captures and composes cleanly with
// --tesseract-backward") while letting the decomposition land later as a
// pattern rewrite — the forward ops stay single-node, the backward ops
// stay single-node, and any further splitting is a lowering detail.
//===----------------------------------------------------------------------===//

def Tesseract_ReluBackwardOp :
    Tesseract_Op<"relu_backward",
                 [Pure, SameOperandsAndResultType, ElementwiseMappable]> {
  let summary = "Gradient of relu: dx = dout * (x > 0).";
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$grad_output);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $grad_output attr-dict `:` type($result)";
}

def Tesseract_CrossEntropyWithLogitsBackwardOp :
    Tesseract_Op<"cross_entropy_with_logits_backward", [Pure]> {
  let summary = "Gradient of cross_entropy_with_logits w.r.t. logits.";
  let description = [{
    Produces `d_logits = (softmax(logits) - one_hot(targets)) * grad / N`.
    Shape matches `logits : [N, C]`. `grad` is a 0-D scalar of the same
    floating dtype as `logits`. `targets` is `[N]` Int64 (no gradient).
  }];
  let arguments = (ins AnyRankedTensor:$logits,
                       AnyRankedTensor:$targets,
                       AnyRankedTensor:$grad);
  let results = (outs AnyRankedTensor:$d_logits);
  let assemblyFormat =
      "$logits `,` $targets `,` $grad attr-dict `:` `(` type($logits) `,`"
      " type($targets) `,` type($grad) `)` `->` type($d_logits)";
}

//===----------------------------------------------------------------------===//
// Shape manipulation
//
// All of these take an explicit target shape / axis list as an attribute;
// none of them do any compute. They are marked `Pure` so that the canonical
// dead-code-elimination pass can drop unused views without extra work.
//===----------------------------------------------------------------------===//

def Tesseract_ViewOp : Tesseract_Op<"view", [Pure]> {
  let summary = "Change the shape of a contiguous tensor (share storage).";
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` `(` type($input) `)` `->` type($result)";
}

def Tesseract_ReshapeOp : Tesseract_Op<"reshape", [Pure]> {
  let summary = "Change the shape, copying if necessary.";
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` `(` type($input) `)` `->` type($result)";
}

def Tesseract_PermuteOp : Tesseract_Op<"permute", [Pure]> {
  let summary = "Permute the dimensions according to `axes`.";
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$axes);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` `(` type($input) `)` `->` type($result)";
}

def Tesseract_TransposeOp : Tesseract_Op<"transpose", [Pure]> {
  let summary = "Swap two dimensions.";
  let arguments = (ins AnyRankedTensor:$input,
                       SI64Attr:$dim_a,
                       SI64Attr:$dim_b);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` `(` type($input) `)` `->` type($result)";
}

def Tesseract_ContiguousOp :
    Tesseract_Op<"contiguous", [Pure, SameOperandsAndResultType]> {
  let summary = "Force a contiguous row-major copy.";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Tesseract_CloneOp : Tesseract_Op<"clone", [Pure, SameOperandsAndResultType]> {
  let summary = "Deep copy into fresh storage.";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Tesseract_BroadcastToOp : Tesseract_Op<"broadcast_to", [Pure]> {
  let summary = "Broadcast to an explicitly given target shape.";
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` `(` type($input) `)` `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Paged KV cache + dynamic shape (M4 Track C2 / B-045)
//
// These pin the IR-level *representation* of an LLM decode runtime as
// first-class dialect nodes (idea.md §4.1). Execution/lowering is future
// work — the goal here is a contract the later NVVM/runtime stages target.
//
// Dynamic shapes: the per-request sequence length is statically unknown, so
// the token axis is expressed with the standard MLIR dynamic dim `?`
// (`ShapedType::kDynamic`). The verifiers below deliberately skip any check
// against a dynamic dim, so `tensor<?x8x64xf32>` queries verify cleanly —
// that is the "dynamic shape as an IR concept" down payment.
//
// The KV pool is modeled as an explicit rank-4 block buffer
// `[num_blocks, block_size, num_kv_heads, head_dim]` (matching the runtime's
// `nn::PagedKVPool`), and block tables / seq-lens as integer tensors, so no
// custom MLIR type is needed at this stage.
//===----------------------------------------------------------------------===//

def Tesseract_PagedKVAllocOp : Tesseract_Op<"paged_kv_alloc", [Pure]> {
  let summary = "Allocate a paged KV block pool.";
  let description = [{
    Materializes a paged KV pool with `num_blocks` blocks of `block_size`
    tokens, each storing `num_kv_heads * head_dim` per token. The result is the
    rank-4 block buffer `[num_blocks, block_size, num_kv_heads, head_dim]`.
  }];
  let arguments = (ins SI64Attr:$num_blocks, SI64Attr:$block_size,
                       SI64Attr:$num_kv_heads, SI64Attr:$head_dim);
  let results = (outs AnyRankedTensor:$pool);
  let assemblyFormat = "attr-dict `:` type($pool)";
  let hasVerifier = 1;
}

def Tesseract_PagedKVAppendOp : Tesseract_Op<"paged_kv_append", [Pure]> {
  let summary = "Scatter new K/V vectors into paged slots.";
  let description = [{
    Writes the `kv` vectors for newly generated tokens into the pool at the
    physical positions given by `slot_mapping` (Int, one slot per token).
    Returns the updated pool (same type), preserving SSA value semantics.
  }];
  let arguments = (ins AnyRankedTensor:$pool, AnyRankedTensor:$kv,
                       AnyRankedTensor:$slot_mapping);
  let results = (outs AnyRankedTensor:$updated);
  let assemblyFormat =
      "$pool `,` $kv `,` $slot_mapping attr-dict `:` `(` type($pool) `,`"
      " type($kv) `,` type($slot_mapping) `)` `->` type($updated)";
  let hasVerifier = 1;
}

def Tesseract_PagedAttentionOp : Tesseract_Op<"paged_attention", [Pure]> {
  let summary = "Attention over a paged KV pool with dynamic seq lengths.";
  let description = [{
    Decode-time attention. `query : [num_tokens, num_heads, head_dim]` (the
    token axis may be dynamic). `k_pool` / `v_pool` are paged block buffers.
    `block_table : [num_seqs, max_blocks]` (Int) maps each sequence's logical
    blocks to physical pool blocks; `seq_lens : [num_seqs]` (Int) gives each
    sequence's current length. `scale` is the softmax temperature; `causal`
    selects masked decode. Result has the query's shape.
  }];
  let arguments = (ins AnyRankedTensor:$query, AnyRankedTensor:$k_pool,
                       AnyRankedTensor:$v_pool, AnyRankedTensor:$block_table,
                       AnyRankedTensor:$seq_lens,
                       F64Attr:$scale,
                       DefaultValuedAttr<BoolAttr, "true">:$causal);
  let results = (outs AnyRankedTensor:$out);
  let assemblyFormat =
      "$query `,` $k_pool `,` $v_pool `,` $block_table `,` $seq_lens attr-dict"
      " `:` `(` type($query) `,` type($k_pool) `,` type($v_pool) `,`"
      " type($block_table) `,` type($seq_lens) `)` `->` type($out)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Module-boundary ops (structural)
//
// `tesseract.graph` is the top-level region wrapping a function body; it is
// roughly analogous to `builtin.module` but scoped to Tesseract-owned
// functions so that passes can match on the exact region boundary.
//
// `tesseract.function` declares the signature (arguments + results) and
// hosts the body block. `tesseract.param` tags a block argument as a
// trainable parameter — this is what reverse-mode AD walks to find leaf
// tensors. `tesseract.return` is the terminator.
//===----------------------------------------------------------------------===//

def Tesseract_GraphOp : Tesseract_Op<"graph",
    [IsolatedFromAbove, NoTerminator, SymbolTable]> {
  let summary = "Container region holding one or more tesseract.function ops.";
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
}

def Tesseract_FunctionOp : Tesseract_Op<"function",
    [IsolatedFromAbove, Symbol,
     HasParent<"tesseract::ir::GraphOp">]> {
  let summary = "Function inside a tesseract.graph.";
  let description = [{
    A Tesseract function is a self-contained computation graph. Its
    arguments are the graph's inputs + params; its single block ends with a
    `tesseract.return` returning the graph outputs. We intentionally avoid
    the upstream `FunctionOpInterface` at M1 — that interface pulls in a
    sizable surface that we do not yet exercise. The minimal symbol +
    function_type pair here is enough for the MLIR textual printer and for
    the passes we ship in M1.β/γ.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name `:` $function_type attr-dict-with-keyword $body";
}

def Tesseract_ReturnOp : Tesseract_Op<"return",
    [Pure, Terminator, ReturnLike,
     HasParent<"tesseract::ir::FunctionOp">]> {
  let summary = "Terminator returning values from a tesseract.function.";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Tesseract_ParamOp : Tesseract_Op<"param", [Pure, SameOperandsAndResultType]> {
  let summary = "Tag a value as a trainable parameter of the enclosing graph.";
  let description = [{
    Wraps a tensor SSA value and annotates it with a symbolic name. Used by
    the reverse-mode AD pass to identify leaves that accumulate gradients.
  }];
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$name);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif  // TESSERACT_IR_OPS
