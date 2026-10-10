在上一讲，我们从最基本的 RL 理论出发，一路推导到了最经典的 RL 理论算法: PPO

但是，在大语言模型的 RL 训练中，不可以直接照搬对应的 RL 算法，而需要进行对应场景的适配。在这一讲，我们将会以 Math 场景出发，介绍经典的 RLHF Pipeline，包含 PPO, GRPO 等训练算法。

![[grpo.png]]

## PPO for LLMs

在 RLHF-PPO 中，会出现四个模型:

- Policy Model: 被训练的核心模型，输入仍然是 $\pi_{\theta}(a \mid s)$, 在这里 $s$ 就是给定的 prompt (例如一个数学题), 输出就是对应的 response, 得到一个 reward score.
- Value Model: 在 PPO 中训练的 Value Network
- Reward Model: ==冻结==，针对模型输出 response 进行打分，类似于之前 rl 中的 $r$
- Reference Model: ==冻结==，主要避免过度微调，设计的 KL 散度 reward

最原始的 PPO 优化目标就是尽可能提升 reward-model 的分数:

$$
\max_\theta J(\theta) = \mathbb E_{o\sim\pi_\theta(\cdot\mid q)} [r_{\mathrm{RM}}(q,o)]
$$
> 注意, 这里是单轮 RL，没有多步的操作

我们需要额外加一个 KL 惩罚:

$$
J_{\mathrm{RLHF}}(\theta) = \mathbb E_{q,o\sim\pi_\theta} \left[ r_{\mathrm{RM}}(q,o) \right] - \beta \mathbb E_q\left[ D_{\mathrm{KL}} \left( \pi_\theta(\cdot\mid q) \Vert \pi_{\mathrm{ref}}(\cdot\mid q) \right) \right] 
$$

>! 在我们之前的推导中，我们的数学期望内部是 $R(\tau)$，要求这一项对 $\theta$ 求梯度的时候，梯度为 0

在加入 KL 散度之后，这个式子还成立吗?

将期望展开：

$$
J(\theta) = \sum_a\pi_\theta(a\mid s) \left[ r(a)-\beta\log\frac{\pi_\theta(a\mid s)} {\pi_{\mathrm{ref}}(a\mid s)} \right]
$$

现在对 $\theta$ 求梯度。

$$
\begin{aligned} \nabla_\theta J ={}& \underbrace{ \sum_a\nabla_\theta\pi_\theta(a\mid s) \left[ r(a)-\beta\log\frac{\pi_\theta(a\mid s)} {\pi_{\mathrm{ref}}(a\mid s)} \right] }_{\text{第一项：Policy Probability 的梯度}}\\[8pt] &- \underbrace{ \beta\sum_a\pi_\theta(a\mid s) \nabla_\theta\log\frac{\pi_\theta(a\mid s)} {\pi_{\mathrm{ref}}(a\mid s)} }_{\text{第二项：KL Reward 自身的梯度}} \end{aligned}
$$

但是由于 Reference Model 固定不变，且 $\sum_a\nabla_\theta\pi_\theta(a\mid s) = 0$ (Score Function 零均值性质)，因此第二项的梯度为 0。第一项的梯度，可以看成 reward 被替换成了带有 KL 散度的 reward。可以进行 reward 计算。

### Token Level or Sequence Level?


