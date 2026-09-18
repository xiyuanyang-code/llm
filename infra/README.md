# Introduction

这里从零阅读五个开源 LLM 基础设施项目的源码,并整理成中文专题。每个专题下是一系列按代码阅读顺序组织的文章。

LLM 训练有三个关键的变量: **算力/算法/数据**。而算力和算法的问题，说到底是 infra 的问题。

与其在课堂上听着枯燥无味的 LLM 训练课堂，不如深入到行业最一线的训练代码中。

因此，本网站将会作为个人学习 LLM Training Algorithms 的学习记录网站，网站将会**逐行深入**学习开源的 LLM Training Infra, 包括:

- GPU 并行计算底层引擎: `Megatron-LM`
- LLM 核心训练算法引擎: `slime`, `ms-swift`
- LLM 核心推理引擎: `vllm`, `sglang`
- Agentic 时代的沙箱训练和评测: `harbor`

除少量辅助性文字由大语言模型辅助生成外，其他文字和代码均由本人亲自书写。

## 专题

### slime

slime 是 THUDM 出品的、为 RL scaling 设计的 LLM post-training 框架,连接 Megatron 与 SGLang,支撑了 GLM 系列模型的训练闭环。

在 slime 专题中，我们会了解到若干核心的 LLM 训练算法:

- 具体的训练算法是如何构建的: SFT, GRPO, PPO, OPD 等经典的适用于 LLM 的模型训练算法
    - 几乎所有重要的 LLM 核心算法都在 slime 中得到了支持，slime 也将会是我们重点讲解的核心训练引擎!
- 如何将 GPU 底层并行 (Megatron) 和 大模型推理引擎(sglang) 融合进入训练引擎中
- 如何利用 ray 实现多机 GPU 的核心调度

详细的章节目录

- [Slime Training](./slime/slime-training.md)

More on the way...