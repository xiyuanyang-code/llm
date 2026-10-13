在 Advanced Attention 章节过后，我们将视角放在全局的 LLM forward 计算过程中：

- Prefill 阶段：模型处理输入 Prompt，计算 QK 矩阵，存储 KV-Cache
	- 在这个阶段对于 Full-Attention 来说，需要完成多个 $o(D^2)$ 的大矩阵乘法来计算 Attention Score
	- 因此是计算密集型的操作
- Decode 阶段：模型开始实现自回归解码

在本章节，我们将重点放在两个部分上：

- FLOPs 分析: 在 Transformer 的每一个步骤上，模型花费的计算复杂度是多少? 哪一个地方是最为耗时的步骤?
- Decoding Methods: 在 Attention 章节我们详细介绍了若干 efficient attention 的方法，主要是在 prefiling 阶段（当然也有在 decoding 阶段）进行模型优化，在这里，我们详细介绍两个比较主流的 efficient decoding 算法, speculative decoding 和 mutli-token-predictions.

## LLM Training FLOPs

### Forward Pass FLOPs

我们首先从最简单的 Attention & LLM 出发，计算单次 Attention 计算中，不同组件的计算复杂度耗时。
#### Basic: Counting Foundations

一般来说，一次浮点运算（加减乘除）被定义为 1 FLOP，我们来看常见的两种数学计算:

- 向量的点积计算 Dot Products for $u,v \in \mathcal{V}$:
	- 向量的维度为 $[d,1]$, 则需要进行 $d$ 次乘法和 $d-1$ 次加法，得到一个标量值
	- FLOPS: $2d - 1 \approx 2d$
- 矩阵乘法计算 Matrix Multiplications (GEMM):
	- 不考虑矩阵乘法在计算复杂度上的优化，只考虑最简单的 GEMM 朴素矩阵乘法
	- 对于 $A([p,q] \times B([q,s]) = C([p,s]))$,. 计算复杂度是 $ps (2q-1) \approx 2pqs$
#### Multi-Head Attention Compute

不考虑 Batch-size, 我们来看 Attention 计算部分，发生了什么矩阵计算:

- Input: $X \in [L , d]$
- Q, K, V Projection:
	- $Q = X W_Q \in [L, d]$, $W_Q \in [d, d]$
	- $K = X W_K \in [L, d]$, $W_K \in [d, d]$
	- $V = X W_V \in [L, d]$, $W_V \in [d, d]$
	- Compute 计算: $3 \times Ld(2d-1) = 6Ld^2 - 3Ld$
	- 对于 MHA 来说，每一个头投影的矩阵是 $Q_i = X W_{Q, i} \in [L, d_h]$
		- 最终，每一个 head 的 output  $[L, d_h]$  拼接成 $[L, d]$ 进入 FFN Layer
		- Compute 计算: $3 \times h \times (L d_h (2d-1)) = 6Ld^2 - 3Ld$
- Atten Scores Compute
	- 对于每一个头，模型会输出对应的  Attention Score 矩阵 $[L, L]$, 具体是 $Q K^T$ 这两个矩阵相乘，Compute: $h \times L^2(2d_h - 1) = 2L^2 d - L^2h$
- Scaling: 除以 $\sqrt{d_k}$ 的操作:
	- Compute: $h \times L^2 \times 1 = h L^2$
- Stable Softmax:
	- 因为 Softmax 存在 e 指数幂的运算，数值稳定性在数值较大处往往会溢出，因此实际工程上往往采用 Stable Softmax 的运算技巧，找到最大的数，然后分子分母同时除以 $e^{a_{\max}}$（数学上等价），保证数值稳定性
	- Compute: 对于一次 softmax 运算，需要有 $L$ 次减法, $L$ 次除法, $L-1$ 次加法: $3L-1$
		- 注意，这里没有只考虑的四则运算，如果只考虑时间复杂度，这里的 FLOPS 应该是 $k L$
		- Softmax 是针对每一个 query 对它全部可见的 key 去做的，在不考虑掩码的情况下，应该是 $h \times L \times (3L-1)$
- Weighted Values & Output Projections
	- $O_i = A_i \times V_i \in [L, d_h]$
	- Compute: $h \times L d_h (2L-1) = 2L^2 d - Ld$
	- 最终 $O \in [L,d]$, 过一个 $W_o \in [d, d]$ Compute: $L d (2d -1) = 2Ld^2 - Ld$

| Component          | Exact FLOPs                | Leading FLOPs |
| ------------------ | -------------------------- | ------------- |
| Q, K, V Projection | $6Ld^2-3Ld$                | $6Ld^2$       |
| Attention Scores   | $2L^2d-hL^2$               | $2L^2d$       |
| Scaling            | $hL^2$                     | $hL^2$        |
| Stable Softmax     | $3hL^2-hL$                 | $3hL^2$       |
| Weighted Values    | $2L^2d-Ld$                 | $2L^2d$       |
| Output Projection  | $2Ld^2-Ld$                 | $2Ld^2$       |
| Total MHA          | $8Ld^2+4L^2d+3hL^2-5Ld-hL$ | $8Ld^2+4L^2d$ |
在实际 Casual Attention 的实践中，相当大一部分的计算可以被减半(full-size 的矩阵变成一个下三角矩阵):
- 计算 attention 矩阵 -> 下三角矩阵
- scaling: 下三角矩阵
- softmax: 下三角矩阵
- values: 下三角 矩阵

替换只需要 $L^2$ -> $\frac{L^2 + L}{2}$

#### FFN, Normalizations, Output Costs

标准 FFN 可以写为 ($[L, d]$ -> $[L, d]$)
$$
\operatorname{FFN}(X)=\operatorname{ReLU}(XW_1)W_2
$$

对于升维矩阵: Compute $Lrd(2d - 1)$ (假设升维维度是 d -> rd)
对于降维矩阵: Compute $Ld(2rd - 1)$
不同的激活函数对应的 compute 不一样（比如 ReLU 0 FLOPs，SiLU 和 GELU 等引入较多的算术运算）

对于 一个标准的 Pre-RMSNorm 阶段，norm 会发生在:
$$
H^{(l)} = X^{(l)} + \text{MHA}(\text{RMSNorm}(X^{(l)}))
$$
$$
X^{(l+1)} = H^{(l)} + \text{FFN}(\text{RMSNorm}(H^{(l)}))
$$

每一次 Norm 都是对 $[L, d]$ 做的，对最后一个维度做归一化。
对于一个单个 $d$ 维的向量，计算 RMSNorm 需要 $4d+2$ 次计算(先计算 mean square，再计算归一化后的值)
因此 计算量: $L(4d+2) \times (2n+1)$ (在所有 blocks 的最后，还要加一个 RMSNorm)

> LayerNorm 的计算会更加复杂

> tokenize 的过程是一个 lookup 的查表过程，因此我们认为是一个 memory-bound 的过程，FLOPs 几乎为 0

#### Rope

与传统 Position Embedding 不同，RoPE 不直接与输入 \(X\) 相加，而是在 Q、K Projection 之后，对 Q 和 K 的特征进行旋转。

对于每一对维度，rope 需要做：
- 4 次乘法
- 2 次加减法

因此，对于 $h$ 个 head, $d_h / 2$ 个 pair，每个 pair 做 6 个 FLOPs, total compute:

$$
h \times (d_h / 2) \times 2 \times 6 \times L = 6Ld 
$$
#### What is leading?

- Four Projections: $8 n L d^2$
- FFN: $4 r n L d^2$
- Score / Value Products (Attention Score 矩阵): $4 n L^2 d$
- Vocab Size: $2 L d V$
### Architectural Variants

#### Gated FFNs

在 Gated FFNs 中，有三个矩阵计算和一个 SiLU 激活函数:

- 三个矩阵计算，假设中间维度是 $r d$, 则 leadingflops 就是 $6 r L d^2$
- SiLU 需要 4 次算术运算和一次指数运算
#### GQA






## Speculative Decoding

## Multi-Token Prediction

DeepSeek: https://arxiv.org/pdf/2412.19437


