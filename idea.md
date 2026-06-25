# 用 C++ 构建下一代深度学习框架：格局、机会与架构蓝图

> 本文档基于 2025–2026 年深度学习框架生态的实际现状（vLLM V1、SGLang、TensorRT-LLM、PyTorch 2.x/Dynamo/Inductor、JAX/Pallas、MAX/Mojo、tinygrad、MLX、Burn/Candle、FlashAttention-4/CuTe DSL、Blackwell/CUDA 13 等）进行核验与分析，避免"PyTorch 诞生于 CNN 时代"这类陈旧叙事，聚焦真正尚未被解决的工程与架构问题。

---

## 一、为什么"再写一个框架"仍然是一个合理命题

先澄清一个常被滥用的前提：**PyTorch/TensorFlow 并非停滞不前**。PyTorch 2.x 已经通过 TorchDynamo + TorchInductor + Triton 将编译路径重建为一套现代 MLIR-style 基础设施，`torch.compile` 在主流训练/推理工作负载上可提供 1.5–2× 的实测加速；PyTorch 2.6/2.7 已开始支持 Python 3.13 的 free-threading（PEP 703，无 GIL）构建；ExecuTorch 在端侧建立 AOT `.pte` + 轻量 C++ 运行时方案；JAX 通过 Pallas 把 TPU/GPU kernel 编写下沉到 Python DSL；MAX（Modular/Mojo）开源了 Graph API 并把包体积压到 <700MB；tinygrad 用约 1.9 万行代码挑战 PyTorch 在 openpilot 等生产场景落地。

真正有效的新框架机会**不在于"做一个更好的 PyTorch"**，而在以下三条已被市场验证但没有统一解的窄门：

1. **训练—推理—部署**在一个 IR、一个运行时里完成，不再经历 `torch.compile → export → ONNX → TensorRT/ExecuTorch → 运行时对不齐`的传统流水线；
2. **异构硬件下的统一 IR + 可插拔后端**，尤其面对 Blackwell 的 TMEM/FP4、AMD CDNA3/ROCm、Trainium/TPU、Apple Silicon 统一内存、国产 NPU 的爆发；
3. **非 Transformer 架构**（Mamba/Mamba-3、RWKV、线性注意力、MoE、DiT）的一等公民支持——目前所有主流框架对这些模型都是"能跑"而非"原生优化"。

C++ 仍是这一命题的最优语言选择，但理由与十年前不同：

- **绝大多数现代后端 DSL（Triton、CuTe DSL/FlashAttention-4、Pallas-Mosaic、TVM-TensorIR）都以 C++/MLIR 为宿主**，新框架如果不想沦为这些 DSL 的调用者，就必须深入 MLIR/LLVM 基础设施；
- **端侧与服务端的部署边界正在消失**，ExecuTorch、llama.cpp/ggml、MLX 都证明了纯原生运行时的价值；C++ 是唯一能在服务器、手机、MCU、WebAssembly、车机同时覆盖的语言；
- **Python 3.13t free-threading 改变了判断**：GIL 不再是"不可逾越"的，但从 Python 层面获得稳定多线程要等到 3.14+ 主流化，而且 Python 的依赖树、启动开销、内存占用问题依然存在；
- 竞争者变多了：Rust 生态（**Burn** 训练+多后端、**Candle** 推理+最小化）、Mojo（Modular 的闭源底座 + 开源 Graph API）、Julia/Flux 都在抢这条赛道，C++ 并不独占。

---

## 二、七大痛点的现状与"已解决到什么程度"

原文档列出的七个痛点在方向上是对的，但量化与"没解决"的表述多数已经过时。基于 2025–2026 年的实际情况重新评估：

### 1. Python GIL 与调度开销

- **事实**：Python 3.13（2024-10 发布）引入 PEP 703 的 free-threading 实验支持；PyTorch 2.6 在 Linux 上支持 `cp313t`，2.7 计划覆盖全平台。
- **残留问题**：GIL 本身可被关闭，但 CPython 单线程性能在 3.13t 上有明显回退；C 扩展（含绝大多数算子库）还没普遍做线程安全迁移；Python 启动与 import 开销在 serverless 场景仍是致命的（冷启动数百毫秒到秒级）。
- **结论修正**：把"GIL 不可逾越"改成"**Python 运行时在微秒级延迟、极小内存占用、秒级冷启动场景结构性不合适**"。小 batch 推理时 Python 调度开销通常占 20–50%（具体数字取决于模型大小，原文档的"70%–90%"只在玩具模型或极小 batch 下成立，作为通论会被读者反驳）。

### 2. 显存效率与 KV Cache

- **量化事实**：**PagedAttention 论文（vLLM）测得传统 KV Cache 实现浪费 60–80% 的显存**，PagedAttention 将浪费降到接近零，吞吐提升 2–4×；SGLang 的 RadixAttention 进一步在前缀缓存与多轮对话上做到 3–5× 的结构化生成加速。
- **尚未解决**：长上下文（>1M tokens）下的 KV 压缩/量化/卸载仍是开放问题；CPU↔GPU↔SSD 三级缓存调度缺乏统一抽象；训练态的梯度与 activation checkpoint 管理仍然碎片化严重。
- **结论**：**推理态的 KV 管理已基本被 vLLM/SGLang 解决**，真正的空白是"训练—推理统一的张量/激活/KV 生命周期管理器"。

### 3. 算子融合与编译

- **事实**：`torch.compile`（Dynamo + Inductor + Triton）已在绝大多数模型上可用；Inductor 后端在 GPU 上生成 Triton，在 CPU 上生成 C++/OpenMP，性能接近手写 kernel。FlashAttention-4 已切换到 CuTe DSL 支持 Hopper（SM90）与 Blackwell（SM100/SM120）。
- **残留问题**：**graph break** 仍是 `torch.compile` 的结构性痛点（用户代码中任何 Dynamo 无法捕获的 Python 构造都会中断图，2025 年的 GraphMend 论文通过源码改写把延迟降低 75%，说明问题仍然严重）；跨控制流融合、融合大算子（attention + MoE + rope + quantize 融合）仍需要手写。
- **结论**：**不是"融合能力差"，而是"动态 Python 语义 + 编译器"先天矛盾**。纯 C++ 前端从根本上消除 graph break。

### 4. 训练—推理的割裂

- **事实**：PyTorch 提供 `torch.export` → ExecuTorch（端侧）、TensorRT-LLM（服务器）、ONNX（通用）三条独立导出路径，行为与算子集不完全一致；vLLM / SGLang / TensorRT-LLM 的推理运行时与训练运行时是不同代码库；权重格式（safetensors / gguf / .pte / TensorRT engine）彼此不兼容。
- **结论**：**这仍然是最大的未解决系统问题**，也是新框架最明确的价值主张。

### 5. 分布式训练

- **事实**：PyTorch FSDP2、DeepSpeed ZeRO-3、Megatron-LM、NVIDIA NeMo、TorchTitan 已覆盖 3D/4D/5D 并行（DP × TP × PP × EP × CP）；NCCL 2.20+、NVLink 5（Blackwell NVL72 达 7.2 TB/s all-reduce）硬件层面进步巨大。
- **残留问题**：不同并行策略的组合需要手工调优（策略空间爆炸）、容错与弹性伸缩支持弱、异构集群（不同代 GPU 混用）几乎不支持。
- **结论**：存在优化空间，但"从零做一个更好的分布式"不是一个创业级差异化点，除非绑定具体硬件（如国产 NPU 集群）。

### 6. 框架复杂度

- **量化事实**：**PyTorch 仓库 ~470 万行代码**（约 192 万 Python、116 万 C++、25 万 C、7.2 万 CUDA），而不是原文档的"1000 万行"。tinygrad 约 1.9 万行（不含测试），ggml 核心数千行。完整编译 PyTorch 仍需 30 分钟–2 小时（不是"几小时"的量级错误，但在大型仓库中确实偏慢）。
- **结论**：数字要改准，核心论点仍成立——**存量代码是巨大的维护负担，也是新入者的结构性机会**。

### 7. 新硬件支持

- **事实**：Blackwell B200/GB200 于 2024 Q4 出货，CUDA 13.0 提供完整支持（tile-based programming、32B 对齐向量类型、TMEM），FlashAttention-4 与 CUTLASS 4.2 同步就绪——**头部硬件的适配窗口已经压缩到 6–12 周**，而非"半年以上"。
- **真正滞后的是长尾**：AMD MI300X 在 vLLM/SGLang 的 day-one 支持仍有差距；Intel Gaudi 3、AWS Trainium2、Google TPU v5p/v6、以及国产芯片（华为昇腾 910B、摩尔线程、沐曦、壁仞、寒武纪）的端到端体验远未打平 NVIDIA。
- **结论**：新硬件适配的真正瓶颈是**缺乏统一的下降路径（IR → tile → kernel）**，这正是 MLIR/Triton/Pallas/TensorIR 在解决的问题。新框架若不绑定 MLIR 生态，重复造轮子代价极高。

---

## 三、现有框架与运行时的"2026 现状地图"

原文档的单一表格把训练框架、推理引擎、编译器、端侧库放在同一坐标系，容易误导。以下按"是什么"重新分组：

### 3.1 通用训练/推理框架

| 框架           | 核心定位                | 长板                                       | 短板                                            |
| -------------- | ----------------------- | ------------------------------------------ | ----------------------------------------------- |
| PyTorch 2.x    | 事实标准                | 生态、`torch.compile`、FSDP2、ExecuTorch | 470 万行代码包袱；Python 前端；graph break 顽疾 |
| JAX + Flax/NNX | 函数式 + 编译优先       | XLA 融合、TPU 原生、Pallas kernel DSL      | 生态窄、调试痛、纯函数式反直觉                  |
| TensorFlow     | 遗留主导地位            | SavedModel、TFLite                         | Keras 之外基本停止创新                          |
| tinygrad       | 极简主义                | ~1.9 万行、多后端、openpilot 生产          | 仍在 alpha、算子覆盖不足                        |
| MLX (Apple)    | Apple Silicon 一等公民  | 统一内存、lazy graph、C++/Swift/Python     | 仅限 Apple 平台                                 |
| Burn (Rust)    | 类 PyTorch 的 Rust 框架 | 多后端（CUDA/Metal/WebGPU/LibTorch）       | 生态小、训练性能待验证                          |

### 3.2 LLM 推理专用引擎

| 引擎                     | 核心创新                                                     | 适用场景                                                    |
| ------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------- |
| **vLLM V1**        | PagedAttention + continuous batching + disaggregated prefill | 高吞吐通用推理，硬件覆盖最广（NV/AMD/Intel/Trainium/TPU）   |
| **SGLang**         | RadixAttention + 结构化生成                                  | 结构化输出、多轮对话、工具调用，结构化任务上 3–5× 于 vLLM |
| **TensorRT-LLM**   | TensorRT kernel 优化                                         | NVIDIA 独占，单卡单 query 最低延迟                          |
| **llama.cpp/ggml** | 纯 C/C++、无依赖、GGUF 格式                                  | 端侧、CPU、量化推理                                         |
| **MAX (Modular)**  | Mojo + MAX Graph IR                                          | 500+ 模型、AMD/NV/Apple 统一、700MB 包体                    |
| **Candle (Rust)**  | 极简、HF 原生集成                                            | serverless、边缘 LLM                                        |

### 3.3 编译器与 IR 基础设施

| 项目                             | 层级                                                 |
| -------------------------------- | ---------------------------------------------------- |
| **MLIR**                   | 通用中间表示基础设施（LLVM 子项目）                  |
| **StableHLO**              | 框架—编译器的可移植算子集（~100 ops，5 年后向兼容） |
| **OpenXLA / IREE**         | 可执行的 MLIR 编译器后端                             |
| **Triton**                 | Python DSL，tile 级并行，PyTorch Inductor 默认后端   |
| **CuTe DSL (CUTLASS 4.x)** | NVIDIA 官方 kernel DSL，FlashAttention-4 载体        |
| **Pallas**                 | JAX 的 kernel DSL（Mosaic-GPU + TPU + Triton）       |
| **TVM / TensorIR**         | 学术源头的编译器，国产芯片适配主力                   |

### 3.4 分布式训练

| 项目               | 定位                         |
| ------------------ | ---------------------------- |
| FSDP2 / TorchTitan | PyTorch 原生 3D 并行         |
| DeepSpeed          | ZeRO、Offload、MoE 专家并行  |
| Megatron-LM        | Tensor/Pipeline 并行参考实现 |
| NeMo / NeMo Run    | NVIDIA 全栈训练编排          |

**关键洞察：你要构建的"C++ 下一代框架"必须明确回答——你要在上面这张图里占据哪个格子，以及你要吞并哪几个格子。**

---

## 四、真正未被解决的问题（新框架的窄门）

### 4.1 训练—推理—部署的"一份 IR、一个运行时"

所有现存框架都在做 `训练框架 → 导出 → 推理引擎 → 端侧运行时` 的多级流水线。每一次转换都丢信息、引入不一致、制造 bug。

**新框架应追求**：

- 一份 **Graph IR**（可以复用 StableHLO 或自定义超集），训练与推理是同一张图的不同执行模式；
- 一套 **Tensor/Memory 运行时**，自动微分、KV Cache、activation checkpoint、权重共享都是一等公民的 IR 节点；
- 一个 **统一权重格式**（兼容 safetensors，扩展元数据用于量化、LoRA、动态形状）；
- 编译期决定静态部分、运行期处理动态部分，不做事后导出。

这是 **MAX 正在尝试但还没兑现**、**PyTorch 因存量无法做**、**vLLM/SGLang 因只做推理无法做**的事。

### 4.2 非 Transformer 架构的一等公民支持

- **Mamba / Mamba-3 / RWKV** 等 SSM/线性注意力模型在长上下文上有 5× 吞吐优势，但在 PyTorch/vLLM 里是"外挂算子"，没有原生的 selective scan、chunkwise 并行、state 管理；
- **DiT / Diffusion Transformer**（Sora、Stable Diffusion 3、FLUX）需要的是 timestep-aware 调度 + VAE/UNet 流水线，与 LLM 的 autoregressive pattern 完全不同；
- **MoE** 的专家并行、all-to-all 通信、负载均衡都没有标准抽象。

新框架若能把"**序列模型的三大范式（Transformer / SSM / Diffusion）**"做成正交的 IR 模块并让用户自由组合，会是一个真正的差异化点。

### 4.3 Blackwell 时代的 FP4/FP6/FP8 与 TMEM 编程模型

- Blackwell 引入专用的 **Tensor Memory（TMEM）**、warp specialization、FP4/FP6 新精度（40 PFLOPS FP4，5× Hopper）；
- 现有框架对这些新特性都是"等 cuBLAS/cuDNN/FlashAttention 升级"，没有能让用户直接表达 warp-specialized pipeline 的抽象；
- 若你构建的 C++ 框架把 CuTe DSL / Mojo-style kernel 作为一等公民，并提供 IR 到这些原语的编译路径，你就能吃到硬件迭代的红利。

### 4.4 服务端推理的前沿组合

2026 年部署一个高性能 LLM 服务所需的技术栈：

- PagedAttention / 分页 KV
- Continuous batching
- Prefix caching（RadixAttention）
- Disaggregated prefill/decode（vLLM V1 已实验）
- Speculative decoding（**EAGLE-3 可达 3–6.5×，Medusa、MTP**）
- FP8/FP4 权重 + KV 量化
- CUDA Graphs + CPU-side request router
- Structured generation（JSON schema、grammar）

**没有一个开源框架把上面全部做成一等公民**。vLLM 最全但仍有 gap，SGLang 在结构化生成最强，TensorRT-LLM 延迟最低。这是一个可以端到端重写的机会。

### 4.5 端侧统一运行时

- ExecuTorch、llama.cpp、MLX、MediaPipe、ONNX Runtime Mobile、TFLite、NCNN 七分天下；
- 每个运行时的算子集、量化格式、硬件后端都不兼容；
- 如果你从 C++ 出发，主打"一份模型 / 一个 `.so` 二进制 / 覆盖 x86/ARM/RISC-V/NPU/WebGPU/WASM"，在端侧 AI 爆发期（XR、机器人、具身智能、车机）有极强的竞争力。

---

## 五、定位建议：先选窄门，再图扩张

不建议一开始就做通用框架。根据上面的空白，有四条可行路径，按"差异化锐度 / 市场规模 / 技术难度" 三维排序：

| # | 方向                                            | 卖点                            | 主要对手                           | 为何现在有机会                       |
| - | ----------------------------------------------- | ------------------------------- | ---------------------------------- | ------------------------------------ |
| A | **训练—推理一体化的大模型框架**          | 一份 IR，训练即部署，天然分布式 | PyTorch + vLLM 组合                | 存量框架没法无痛统一；MAX 在做但闭源 |
| B | **非 Transformer / 混合架构一等公民框架** | Mamba/SSM/MoE/DiT 原生优化      | PyTorch + 第三方 CUDA kernel       | SSM/DiT 是 2026–2028 的主流增量     |
| C | **端侧统一 AI 运行时**                    | 一套 C++ 运行时跨 7 个平台      | ExecuTorch / llama.cpp / MLX       | 端侧爆发且碎片化严重                 |
| D | **国产/非 NVIDIA 硬件优先的训练框架**     | MLIR-first，后端可插拔          | MindSpore / OneFlow / PaddlePaddle | 硬件自主化驱动，政策窗口             |

**最推荐的初始切入点**：**方向 A**（训练—推理一体化）且**先做 LLM + SSM 推理**，早期把训练当作"同 IR 的反向执行模式"预留接口，在用户验证后再逐步打开训练通路。这样：

- 第一年可达到 vLLM 可比性能（可公平对齐的 benchmark 是 Llama-3/Qwen/DeepSeek）；
- 天然支持 Mamba/RWKV；
- C++ 运行时+插件架构天然向端侧迁移；
- 训练能力是 18 个月后的大杀器（一旦训练打通，立刻具备对 PyTorch 的替代叙事）。

---

## 六、C++ 新框架的核心架构蓝图

对原文档那张"六层架构图"进行重构，加入编译器、IR、调度器、硬件抽象四个维度，这也是现代框架（PyTorch 2.x / MAX / XLA）的事实结构：

```
┌──────────────────────────────────────────────────────────────┐
│  前端  Frontend (C++ core API + optional Python bindings)    │
│  - 即时模式 (eager) / 图模式 (graph) / 脚本 DSL              │
├──────────────────────────────────────────────────────────────┤
│  高层库  Layers, Optimizers, Loss, Distributed Strategies    │
├──────────────────────────────────────────────────────────────┤
│  自动微分  Autograd (tape + graph-mode hybrid)               │
├──────────────────────────────────────────────────────────────┤
│  图 IR / 中间表示  Graph IR (MLIR-based, StableHLO compatible)│
│  - 训练/推理统一   - 显式内存/KV Cache 节点                   │
│  - 动态 shape 一等公民  - 量化/稀疏/MoE 作为 IR 属性           │
├──────────────────────────────────────────────────────────────┤
│  编译器 pass  Lowering → Tile IR → Kernel IR                 │
│  - 融合 / 分块 / 管线化 / 自动并行 (3D/4D parallel)          │
│  - 针对后端的选择性 lowering（Triton / CuTe / Metal / ROCm）  │
├──────────────────────────────────────────────────────────────┤
│  运行时  Runtime: Scheduler + Memory Planner + Executor      │
│  - Continuous batching / PagedKV / CUDA Graph / Spec decode  │
│  - 分布式通信（NCCL / RCCL / oneCCL / 自研）                  │
├──────────────────────────────────────────────────────────────┤
│  硬件抽象  HAL: GPU / CPU / NPU / Metal / WebGPU / WASM      │
└──────────────────────────────────────────────────────────────┘
```

### 6.1 关键实现决策

1. **IR 选型**：**直接在 MLIR 之上构建自定义 dialect**，下降路径对齐 StableHLO / Linalg / Tile 方言。不要自己从零设计 IR，也不要用 ONNX（语义太弱）。
2. **Kernel 生成**：上层不手写 CUDA。融合策略由编译器决定，后端生成交给 **Triton（GPU）+ LLVM/OpenMP（CPU）+ Metal Performance Shaders（Apple）+ 可插拔 NPU 后端**。手写只限于 FlashAttention 级 hot path。
3. **内存规划**：把 KV Cache、activation、权重、梯度都建模成 **Buffer with lifetime annotation**，由 memory planner 在编译期求解 register/SRAM/HBM/DRAM 多级分配。
4. **自动微分**：采用 **tape + graph 混合**。eager 模式走 tape（调试体验），graph 模式走反向 IR 重写（性能+可部署）。训练与推理共享前向 IR。
5. **分布式**：把并行策略（DP/TP/PP/EP/CP）作为 IR 的**变换 pass**，而不是用户手写（参考 JAX 的 `shard_map`、GSPMD）。
6. **API 设计**：C++ 作为一等前端（模板 + CRTP + 值语义），pybind11/nanobind 作为可选 Python 绑定；考虑对 Rust / Swift / C# FFI 友好（头文件保持 ABI 稳定）。
7. **包体与启动**：目标静态链接单二进制 < 50MB（CPU-only）/ < 500MB（含 CUDA runtime，对标 MAX 的 <700MB）；冷启动 < 100ms。

### 6.2 分阶段里程碑（建议 24 个月）

| 阶段                                   | 周期         | 目标                                                          |
| -------------------------------------- | ------------ | ------------------------------------------------------------- |
| **M0：张量与自动微分**           | 第 1–3 月   | CPU 张量、eager autograd、MNIST 可训练                        |
| **M1：MLIR IR + Graph 模式**     | 第 4–7 月   | 自定义 dialect、lowering 到 LLVM/CPU、ResNet-50 能跑          |
| **M2：CUDA 后端 + Triton 集成**  | 第 8–11 月  | GPU eager + graph、Llama-7B 推理与 PyTorch 可比               |
| **M3：LLM 推理一等公民**         | 第 12–15 月 | PagedKV、continuous batching、speculative decoding，对标 vLLM |
| **M4：SSM / MoE / DiT 原生支持** | 第 16–19 月 | Mamba-3、Mixtral、DiT 首发原生优化                            |
| **M5：分布式训练 + 端侧运行时**  | 第 20–24 月 | FSDP 级训练 + ExecuTorch 级端侧部署                           |

> **M4 落地修订（2026-06-22，见 [ADR-0006](docs/adr/0006-m4-parallel-abc-scope.md) +
> [docs/m4-plan.md](docs/m4-plan.md)）：** 实际执行中，M4 把本表的"SSM/MoE/DiT 原生"
> （方向 A）与原 M5 的"分布式训练 + Python frontend"（方向 B）以及补齐"一份 IR、训练
> 推理一体"未兑现部分（方向 C）**三线并行推进**。其中 MoE + SSM/Mamba 为主线并在本里程碑
> 落地；**DiT 运行时与 at-scale 分布式顺延到 M5**（本里程碑仅出设计 + 单机原型）；LLM 的
> GPU codegen 因本机 LLVM 缺 NVPTX 后端仍为 gated tail。

每个阶段都必须有**可复现的基准测试**（vs PyTorch/vLLM/SGLang/llama.cpp 的对齐数据），拒绝 hand-wave 的"更快"。

---

## 七、进阶方向（M5 之后）

1. **FP4/FP6 + MXFP8** 的端到端训练—推理支持（Blackwell 专属优势）
2. **JIT 自动调优**（参考 Ansor、AutoTVM，把 schedule 搜索做进框架）
3. **异构集群调度**（不同代 GPU 混跑 + 弹性容错）
4. **WebGPU / WASM 后端**（浏览器端 LLM，对标 transformers.js + ExecuTorch Web）
5. **形式化验证的数值精度保证**（医疗/金融场景关键差异化）
6. **Agent-native 执行模型**（把工具调用、结构化生成、KV 复用做成 IR 原语）

---

## 八、避坑清单（基于他人失败经验）

1. **不要自造 IR 后又自造编译器**：MLIR/LLVM 已经吃掉了这层基础设施，与其并行不如共生。
2. **不要轻视 Python 生态**：即便 C++ 是核心，也必须提供零摩擦的 Python 绑定与 HuggingFace 权重加载路径。用户迁移成本高于一切性能数字。
3. **不要在算子层重复造轮子**：cuBLAS/cuDNN/cuSPARSE/CUTLASS/FlashAttention/NCCL/RCCL 都应作为后端而非重写对象。
4. **不要绑死一种硬件**：哪怕起步只支持 NVIDIA，也要在 HAL 层预留 AMD/Apple/NPU 接口，否则后期改造代价无限大。
5. **基准测试即产品**：从 Day 1 起就建立可对外复现的 perf 仓库，每个 commit 都跑。社区信任建立在数字上，不在 README 上。
6. **文档、示例、模型动物园是一等工作**：tinygrad 花了 5 年还停在 alpha，最大原因是缺生态。
7. **提防"全部重写"诱惑**：保持和 PyTorch eager、HuggingFace、safetensors、ONNX 的互操作性，是用户愿意尝试的前提。

---

## 九、优秀参考项目（按学习价值排序）

### 学习系统架构

- **PyTorch（`aten/`、`torch/_dynamo/`、`torch/_inductor/`）**——现代化 C++ 张量系统 + Python 编译器前端的事实范本
- **JAX + OpenXLA**——函数式编译型框架的最干净实现
- **MAX / Mojo 开源部分**（MAX Graph API）——2025 年新生代框架的设计语言

### 学习 LLM 推理

- **vLLM**——PagedAttention + 调度器 + V1 engine，LLM 推理事实标准
- **SGLang**——RadixAttention + 结构化生成
- **llama.cpp / ggml**——极致轻量的 C/C++ 推理实现
- **TensorRT-LLM**——NVIDIA 官方的 kernel 级优化范例

### 学习轻量框架与 Rust 竞争者

- **tinygrad**——约 1.9 万行代码的极简主义（不是"几千行"，别弄错数字）
- **Candle**——Rust 生态的 HuggingFace 原生推理库
- **Burn**——Rust 多后端训练框架
- **MLX**——Apple Silicon 与统一内存架构的原生实践

### 学习编译器与 kernel DSL

- **MLIR / LLVM**——必修
- **Triton**——Python DSL 的 tile 级并行，PyTorch Inductor 默认后端
- **CUTLASS 4.x / CuTe DSL**（含 FlashAttention-4）——NVIDIA 官方 kernel DSL
- **Pallas**（JAX）——跨 TPU/GPU 的 kernel DSL 设计
- **TVM / TensorIR**——学术源头 + 国产芯片适配

### 学习端侧部署

- **ExecuTorch**——PyTorch 官方端侧方案，AOTInductor + `.pte`
- **ONNX Runtime + ORT Mobile**——跨平台事实标准
- **MediaPipe / TFLite / NCNN**——工业端侧实践

---

## 十、总结：一句话战略

> **不要做一个更好的 PyTorch；做一个 PyTorch + vLLM 组合从架构上做不到的东西——那就是"一份 IR、一个 C++ 运行时、训练和推理无缝、非 Transformer 架构原生、端到端从服务器到端侧"。**

接下来可以细化的两个方向（任选其一或都要）：

- **A. 微内核 + 插件化架构的 C++ 代码设计**：dialect 定义、pass 注册机制、后端 ABI、热插拔内存分配器；
- **B. 训练—推理一体化的实现方案**：统一 IR 的训练模式与推理模式差异如何在同一张图上表达、权重格式、KV Cache 生命周期、自动微分 pass 的具体工程化。

如需进一步展开任何章节，或直接落地到具体代码骨架（MLIR dialect、张量类模板、调度器），请明确告知优先方向。
