# LLM training parity vs PyTorch (issue.md B2 / B-042)

Same config (vocab=64, hidden=64, 2 layers, 4 heads, ffn=128, seq=16, batch=2),
same optimizer (Adam, lr=3e-3), same task (memorize a fixed batch via next-token
cross-entropy), 100 steps. Tesseract: `examples/llama_train.cpp`. PyTorch:
`bench/external/torch_llama_train.py` (RMSNorm + RoPE causal MHA + SwiGLU,
mirroring Tesseract's LlamaModel). Init + token RNG differ across frameworks, so
this is *behavioral* parity, not bit-exact.

| step | Tesseract loss | PyTorch loss |
|------|---------------:|-------------:|
| 1    | 4.263 | 4.316 |
| 10   | 0.889 | 0.793 |
| 20   | 0.129 | 0.122 |
| 30   | 0.0407 | 0.0425 |
| 40   | 0.0199 | 0.0213 |
| 50   | 0.0126 | 0.0137 |
| 60   | 0.00939 | 0.0102 |
| 70   | 0.00763 | 0.00828 |
| 80   | 0.00650 | 0.00703 |
| 90   | 0.00568 | 0.00612 |
| 100  | 0.00504 | 0.00541 |

The two loss curves track within ~10% at every milestone and converge to the
same ~5e-3 floor. Tesseract's autograd + Adam reproduce PyTorch's training
dynamics on an identical config — the LLM training stack (RMSNorm → RoPE-MHA →
SwiGLU → lm_head, full forward+backward+optimizer) is correct and on par.
