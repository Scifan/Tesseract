_**Agent (claude-opus-4-8)**_

> ## ✅ M4 性能闭环 — 全部已解决 (closeout 2026-06-24)
>
> 下面这轮反刍提出的每一条"地基/correctness-only/gated/无数字/honest-loss"
> 欠账,在 M4 性能闭环的 9 个 phase 里**已逐条做实并量化**(13/14 取胜,唯一
> 诚实 loss 是整模型单流 serving 延迟 vs vLLM,见下表末行)。权威总表见
> [`docs/design/external-benchmark.md`](design/external-benchmark.md) 的
> 14 行 scoreboard;每条明细在 [`bench/external/results/`](../bench/external/results/)。
>
> | 原欠账 | 现状 | 证据 |
> |--------|------|------|
> | A1 MoE 是 dense、无 GPU、无效率指标 | **融合 GPU MoE**:device 路由+grouped GEMM,胜 dense 3.5–16×、胜 PyTorch eager 1.39–2.92× | `moe_gpu.md` |
> | A2 Mamba 顺序扫描、GPU parity gated、无指标 | **chunkwise 并行 scan + 融合 conv1d kernel + GPU parity**;O(1) decode 胜 attention 2.08–4.29× | `mamba_gpu.md` |
> | B1 Python 无开销度量 | 真实尺寸算子开销 **<1–3%**(≈0%) | `python_overhead.md` |
> | B2 训练只过拟合、无 PyTorch 对照 | 与 PyTorch Adam **同配置 loss 曲线 parity** | `train_parity.md` |
> | B3 TP 单进程模拟、无多卡/无 scaling/无 backward parity | **真实 NCCL 多进程 TP=2/3**:forward+backward parity、吞吐 2.98× 胜 PyTorch 2.38×、显存 1/N | `nccl_tp.md` / `tp_multigpu.md` |
> | C1 IR attention/rope gated、无性能对照 | backward 全 op(softmax/sum 任意 dim)+ 数值 parity;**JIT 胜 eager 5.5–8.5×**;并新增 **GPU 端 gpu.module→PTX→cubin 执行 + eager-CUDA parity** | `ir_backward.md` / `ir_gpu_jit.md` |
> | B-046 外部对标 0 个数字 | **14 行外部 scoreboard**(llama.cpp / PyTorch / FlashDecoding / FP8 / NCCL / vLLM)。**全部取胜或追平**。原第 14 行的 serving 诚实 loss 已在 2026-06-25 翻盘:同精度 FP16 下 Tesseract(FP16+整模型 CUDA-graph+GQA-native fused attention)**decode 321.1 vs vLLM 305.8 tok/s(+5%)、TPOT 3.115 vs 3.275 ms、端到端 ~400 vs 420 ms 全胜**;唯一仍落后的 TTFT/prefill 经两轮优化已逼平到 7% 以内——B-024+ WMMA tensor-core flash(7.28→6.59 ms)+ B-024c stride-aware/BSHD attention 布局消除 per-layer `contiguous`/转置拷贝(`strided_copy` 25.8%→15.9%,6.59→**5.86 ms**,vs vLLM 5.47,gap 1.20×→**1.07×**);残余差距是两边共享的 cuBLAS GEMM 下限 + Tesseract 未融合的 norm/RoPE/residual launch(后续项 B-024e:BSHD-native RoPE + epilogue 融合可彻底反超) | `external-benchmark.md` / `vllm_serving.md` |
> | honest-loss #1 FlashDecoding attention 2× 慢 | **已反超**:split-K+multi-warp,胜 PyTorch SDPA 1.05–1.58× | `flashdecode.md` |
> | honest-loss #2 CPU decode 比 llama.cpp 慢 73× | **已反超**:AVX-512-VNNI W8A8,胜 Q8_0 1.62×、胜其最快 Q4_0 1.08× | `cpu_decode_vnni.md` |
> | GEMM 同精度只能 tie | 线级用 **FP8 E4M3** 取胜 PyTorch FP16 1.36–2.19× | `fp8_gemm.md` |
> | 横切:ctest skip 被当 fail / OOM | 隔离方法论 + `SKIP_RETURN_CODE` 修复;两套 build ctest 绿 | `phase1_fixes.md` |
>
> 下方为原始反刍记录,保留作历史。

我对 M4 做了一轮基于源码证据的核查(不是复述文档),下面是诚实的反刍式结论。先给总评,再逐条展开。

## 总评

M4 的十个待办都"跑通且有测试",但**完成度的定义被悄悄放宽了**:相当一部分交付是**地基/脚手架/correctness-only**,而不是"可度量的能力"。最尖锐的两点:

1. **几乎没有客观定量指标。** M2 有 ~14 条带硬性 bar 的性能基准(tok/s、GB/s、speedup);M4 只新增了 1 条 `bench_llama_decode_cpu`,且是 informational、无 bar。架构差异化(MoE/Mamba)的"优势"一个数字都没量出来。
2. **几处"名实不符"**,下面逐条点名。

并且一个硬伤:`build-cpu` 现在仍有 **3 个 ctest FAIL**(`layer_norm`/`batch_norm` CPU↔CUDA,见下),M4 exit bar 写的"全程 CPU ctest 绿"**实际未达成**。

---

## 一、逐项质量与缺漏(附证据)

**A1 MoE —— correctness-only,丢了 MoE 存在的全部意义。**
`src/nn/MoEFeedForward.cpp:132-137` 的 forward 对**每个 token 跑了全部 N 个专家**,再用 0/1 gate 加权求和:

```131:137:/home/data/qfshi/framework/src/nn/MoEFeedForward.cpp
  Tensor out;  // [..., d_model], lazily initialized to the first weighted expert
  for (int64_t e = 0; e < num_experts_; ++e) {
    Tensor y_e = experts_[static_cast<std::size_t>(e)]->forward(x);  // [..., D]
    Tensor g_e = gates.narrow(last, e, 1);                            // [..., 1]
    Tensor weighted = ops::mul(y_e, g_e);                            // [..., D]
    out = out.defined() ? ops::add(out, weighted) : weighted;
  }
```

MoE 的全部价值是**稀疏激活**(top-2/8 只算 2 个专家 → 1/4 算力)。当前实现是 dense,FLOPs 反而是等价稠密模型的 N 倍。再加上 top-k mask 走 host 端 D→H→D(`build_topk_mask`,line 48-65,CUDA 上是同步点)。结论:**MoE 只能展示 parity,无法展示任何效率优势**——而效率优势正是它唯一的存在理由。CUDA 路径也从未在 GPU 上实测过。

**A2 SSM/Mamba —— "chunkwise 并行前缀扫描"是计划里的承诺,实际是顺序扫描。**
`src/ops/cpu/SelectiveScan.cpp:104-120` 和 CUDA kernel `src/cuda/SelectiveScan.cu:58-73` 都是**逐时间步串行**(CUDA 仅在 B·D 上并行,t 轴串行)。这对 decode(L=1)没问题,但长序列 prefill 拿不到 associative-scan 的并行延迟优势——而这正是 SSM 对 attention 的卖点。另外 CUDA 有 `N<=32` 硬上限(`.cu:103`),且 GPU parity 仍是 gated、未实跑。`conv1d` 用 op 组合实现,正确但非高效。

**A3 DiT —— 仅设计,符合预期。** 这条没问题。

**B1 Python —— 只有 smoke,深度不足。** 9 个 pytest 都是"能调通"级别:没有 `LlamaModel` 真实权重加载、没有 autograd-through-`generate`、绑定面是手挑子集、**没有任何 Python↔C++ 调用开销度量**。"采用价值"没有被任何数字支撑。

**B2 单卡训练 —— 只证明了"接线对",没证明"能训练"。** `examples/llama_train.cpp` 只在**固定 batch 上过拟合**(4.26→0.0065)。没有真实数据集、没有验证集 loss、没有与 PyTorch 同配置的 loss 曲线对照。它证明 forward/backward/optimizer 连通,不证明训练质量。

**B3 张量并行 —— 单进程模拟,且未验证训练。** `SimCommBackend` 把所有 rank 放一个进程(all-reduce=求和)。**没有真实多卡、没有任何扩展性数字(吞吐/显存随 rank)、也没有 backward parity 测试**——TP 层能否正确反传完全未验证(虽然 `all_reduce`/`all_gather` 用了 autograd-aware 的 add/cat,但 `from_dense` 的 `copy_into` 走 memcpy,这条梯度链没测)。

**C1 IR —— 不是"单个 Llama block 过 IR",是 FFN+norm 子集。** 我把 `rms_norm`/`swiglu` 的**原语链**做到了过 CPU JIT 且 eager parity,但 `attention` 和 `rope` 是 gated tail。计划/ADR 的 exit bar 是"单个 Llama block 经 IR CPU JIT",**未达成**(缺 attention/rope)。而且**没有任何性能对照**(JIT 的 RMSNorm 比 eager 快多少?未测,也没加进 `bench_graph_vs_eager`)。

**C2 IR —— 仅表示,符合预期,但 verifier 偏弱。** 三个 op + round-trip + 负例都在,但没有 shape 一致性的交叉校验(如 block_table/seq_lens 维度关系),且无执行(这是设计内的)。

**B-046 外部对标 —— 这是最名不副实的一条。** 计划目标是"对标外部框架";实际产物是**一个脚手架 + 我们自己的数字**,`docs/design/external-benchmark.md` 里 llama.cpp 那一行至今是 `TODO`。**真正的外部对标数字:0 个。** "无外部 bench"这个缺口,严格说只是从"无脚手架"变成"有脚手架但仍无数字"。

**横切硬伤:** `ctest` 仍有 3 个 FAIL(`#221/#224/#315`),根因是这几个测试用 Catch2 `SKIP`(退出码 4)但其 `catch_discover_tests` 没配 `SKIP_RETURN_CODE`,ctest 把 skip 当 fail。这是既有问题(非 M4 引入),但它让"CPU ctest 全绿"这条 exit bar 失真。

---

## 二、如何客观度量(这是你问的核心,也是 M4 最大的欠账)

对照 M1/M2 的做法,**每一项都应该有定量指标,目前几乎全缺**。逐条给出"该测什么 + 怎么测":

| 工作 | 应有的客观指标 | 当前 | 障碍/前置 |
|---|---|---|---|
| A1 MoE | 稀疏算力节省比(top2/8 应 ≈1/4 dense 的 FLOPs/延迟);router 开销占比 | **测不出**(实现是 dense) | 必须先做真正的 token dispatch/combine(gather/scatter),否则无指标可言 |
| A2 Mamba | decode tok/s 与序列长 L 的关系(应 O(1)/步,attention 是 O(L)/步);显存随 L 曲线;长上下文吞吐 vs Llama | 无 | 加 `bench_mamba_vs_llama_decode`(纯 CPU 即可对比 O(L) vs O(L²)) |
| A3 DiT | (设计阶段,无) | — | M5 |
| B1 Python | `import tesseract` 后训练/推理吞吐 vs 原生 C++ 的 overhead% | 无 | 一个 pytest-benchmark 即可 |
| B2 训练 | 真实小数据集上的收敛曲线 + 与 PyTorch 同配置 loss 对照 | 仅过拟合 | 需小数据集 fixture |
| B3 TP | 真实 2 卡吞吐/显存 scaling(gated);单机至少 backward parity + shard 开销 | 无 | 多卡 gated;backward parity 可立刻补 |
| C1 IR | JIT'd RMSNorm/SwiGLU vs eager 加速比(纳入 `bench_graph_vs_eager`) | 无 | 低成本可补 |
| B-046 | llama.cpp 的 `tg`/`pp` t/s 实测填表 | **0 个外部数字** | 需 build llama.cpp + 一个 GGUF |

一句话:**M4 证明了"能做",但没有像 M1/M2 那样证明"做得多好"。** 这是它和前两个里程碑在"工程严谨度"上的真实落差。

---