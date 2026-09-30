## Decoding and Prefiling

在 Advanced Attention 章节过后，我们将视角放在全局的 LLM forward 计算过程中：

- Prefill 阶段：模型处理输入 Prompt，计算 QK 矩阵，存储 KV-Cache
	- 在这个阶段对于 Full-Attention 来说，需要完成多个 $o(D^2)$ 的大矩阵乘法来计算 Attention Score
	- 因此是计算密集型的操作
- Decode 阶段：模型开始实现自回归解码

为了简化计算，我们考虑最简单的 Full-Attention 的情况，且不考虑多头注意力：
- 之前我们计算过，单次 Attention 涉及:  $\text{FLOPs}_{prefill} \approx \underbrace{6N d^2}_{\text{QKV 投影}} + \underbrace{4N^2 d}_{\text{Attention 矩阵运算}}$
- Memory Bound $\text{Bytes}_{prefill} = \underbrace{3d^2 \times \text{bytes}}_{\text{权重 (固定)}} + \underbrace{c_1 N d \times \text{bytes}}_{\text{线性激活值/KV 读写}} + \underbrace{c_2 N^2 \times \text{bytes}}_{\text{Attention 矩阵读写}}$

因此，计算强度定义为：
$$I_{prefill} = \frac{6N d^2 + 4N^2 d}{3d^2 + c_1 N d + c_2 N^2}$$
我们假设 N 很大，此时计算强度为 $4d/c_2$ 是一个相对较大的数值，是计算密集型

在 Decoding 计算，每一次只需要实现一次向量-矩阵乘法，因此 FLOPS 是 $\text{FLOPs}_{decode} = 6d^2 + 4Nd$, 是线性的（对于生成一个 token 来说），但是每次计算需要把模型权重 && KV-Cache 实现完整的搬运, 因此 Memory Bound = $(3d^2 + 2Nd) \times \text{bytes}$。
因此，在 Decode 阶段，计算强度定义为 
$$I_{decode} = \frac{6d^2 + 4Nd}{(3d^2 + 2Nd) \times \text{bytes}} = \frac{2}{\text{bytes}}$$
因此，我们得到了一个关键结论：

- Prefiling 阶段是 Computation-bounded 的计算，因此关键的优化点在于如何优化 $O(N^2)$ 的矩阵乘法计算，例如引入 sparse attention，linear attention 来优化矩阵乘法的计算复杂度
- Decoding 阶段的计算强度和 $N$ 无关，甚至在极端情况下（例如我们上面计算的最简单计算情况），模型的计算强度是一个很低的常数。



## Speculative Decoding

## Multi-Token Prediction

DeepSeek: https://arxiv.org/pdf/2412.19437


