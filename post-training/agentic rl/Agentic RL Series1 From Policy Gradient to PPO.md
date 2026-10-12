> 本文将会快速从 RL 的基础理论讲起，介绍 RL Basis, Policy Gradients, Actor-Critic Methods 等 RL 中的基本概念，在这个基础上介绍最经典的一种强化学习算法 PPO

## RL Basis

对于一个统一的马尔可夫过程：$\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma)$, 给定参数化策略 $\pi_\theta(a\mid s)$，我们给出的最终目标是:

$$
 \max_\theta J(\theta) = \mathbb E_{\tau\sim\pi_\theta} \left[\sum_{t=0}^{T}\gamma^t r(s_t,a_t)\right]
$$
即从参数化策略中采样一个策略，我们希望这个策略的最终 reward 可以被最大化。

为了优化这个目标，我们引入如下函数和指标:
- $G_t=\sum_{k=0}^{T-t}\gamma^k r_{t+k}$: 从 $t$ 时刻开始可以得到的全部 reward
- $V^\pi(s) =\mathbb E_\pi[G_t\mid s_t=s]$: 从 $t$ 时刻开始，在当前状态 $s$ 下采样运行策略 $\pi$，得到的 reward 的数学期望
- $Q^\pi(s,a) =\mathbb E_\pi[G_t\mid s_t=s,a_t=a]$: 从 $t$ 时刻开始，在当前状态 $s$ 下运行 action $a$, 采样运行策略 $\pi$，得到的 reward 的数学期望
- $A^\pi(s,a) =Q^\pi(s,a)-V^\pi(s)$: Advantage
	- 因为对当前状态 $s$, 根据策略 $\pi$ 采样出来的 $Q^\pi(s,a)$ 的数学期望就是 $V^\pi(s)$.

$$
 V^\pi(s)=\mathbb E_{a\sim\pi(\cdot\mid s)}[Q^\pi(s,a)] 
 $$
 在 RL 算法中，存在两种优化方法:

- 如果我们能够建模出来 $Q^\pi(s,a)$, 我们就可以根据这个函数对策略 $\pi$ 进行优化
	- 直接设置 $\pi(a|s) = 1$, 对于最大的 Q 对应的 action
	- 这可以保证 $\pi$ 的优化是递增的，即更新后的旧策略一定优于新策略
	- 这种方式需要依赖对 $Q$ Function 的很好的拟合 -> ==**Value Based RL**==!
- 直接通过 reward 的梯度信号优化 policy (policy gradient)
	- e.g. REINFORCE, PPO, GRPO

## Policy Based RL (Policy Gradient)

考虑一条轨迹: $\tau=(s_0,a_0,s_1,a_1,s_2,a_2,s_3, ...)$
RL 优化的目标就是 sample 这样一条轨迹之后，这条轨迹的最终 reward 会被最大化。

首先，给出数学上的 Log-Derivative Trick（本质就是 log 导数的 trick，在这里省略证明）：
$$
 \nabla_\theta p_\theta(\tau) = p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)
$$
对于一个离散的轨迹空间，期望的求和形式是:
$$
J(\theta)= \sum_\tau p_\theta(\tau)R(\tau)
$$
### 基本的概率梯度形式

对两边求 $\theta$ 的梯度:
$$
\nabla_\theta J(\theta) = \nabla_\theta \sum_\tau p_\theta(\tau)R(\tau) = \sum_\tau R(\tau)\nabla_\theta p_\theta(\tau)
$$

> 我们默认 $R(\tau)$ 的给出往往是环境的反馈等等，没有额外依赖 $\theta$.

因此，代入公式:

$$
\nabla_\theta J(\theta) = \sum_\tau R(\tau)\nabla_\theta p_\theta(\tau) = \sum_\tau R(\tau) p_\theta(\tau)\nabla_\theta\log p_\theta(\tau) = \mathbb E_{\tau\sim p_\theta} \left[ R(\tau) \nabla_\theta\log p_\theta(\tau) \right] 
$$

因此，我们需要求解的梯度，可以被写成一个**针对旧策略采样**的数学期望的形式。我们可以使用 MCMC 等方法进行 policy gradient 的估计。

### 轨迹概率分解

接下来，我们继续攻关，尝试对 $\nabla_\theta\log p_\theta(\tau)$ 进行轨迹的概率分解：

考虑一个只有两个 Action 的轨迹, $\tau=(s_0,a_0,s_1,a_1,s_2)$，完整的轨迹概率是:

$$
 \begin{aligned} p_\theta(\tau) ={}& \rho_0(s_0)\\ &\times\pi_\theta(a_0\mid s_0)\\ &\times P(s_1\mid s_0,a_0)\\ &\times\pi_\theta(a_1\mid s_1)\\ &\times P(s_2\mid s_1,a_1) \end{aligned}
 $$

我们取对数，可以计算:
$$
\begin{aligned} \log p_\theta(\tau) ={}& \log\rho_0(s_0)\\ &+\log\pi_\theta(a_0\mid s_0)\\ &+\log P(s_1\mid s_0,a_0)\\ &+\log\pi_\theta(a_1\mid s_1)\\ &+\log P(s_2\mid s_1,a_1) \end{aligned}
$$

很显然，在这样子的累和项中，只有包含 $\pi_\theta (a \mid s)$ 的项，会产生梯度 (剩下的项只和环境有关，不会产生对模型参数的梯度)
$$
\nabla_\theta\log p_\theta(\tau) = \nabla_\theta\log\pi_\theta(a_0\mid s_0) + \nabla_\theta\log\pi_\theta(a_1\mid s_1) 
$$
将这个式子进行推广:
$$
\nabla_\theta\log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta\log\pi_\theta(a_t\mid s_t)
$$
因此，代入公式，我们得到:
$$
\nabla_\theta J(\theta) = \mathbb E_\tau \left[ R(\tau) \sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t) \right]
$$
### From Reward to Reward-to-Go

现在的梯度估计有个问题，轨迹级别的 reward 可能方差很大，可能导致梯度估计方差很高，而且不同 Action 的学习信号不够精细。

首先，我们的第一个优化是把 reward 换成 reward-to-go:

我们最终会优化成为的式子是:

$$
\nabla_\theta J(\theta) = \mathbb E_\tau \left[ \sum_{t=0}^{T-1} \gamma^t G_t \nabla_\theta\log\pi_\theta(a_t\mid s_t) \right] 
$$

对比一下这两个式子:

- Current: $\nabla_\theta J(\theta) = \mathbb E_\tau \left[ R(\tau) \sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t) \right]$
- Target: $\nabla_\theta J(\theta) = \mathbb E_\tau \left[ \sum_{t=0}^{T-1} \gamma^t G_t \nabla_\theta\log\pi_\theta(a_t\mid s_t) \right]$

两者最核心的区别是，对于一条采样轨迹，每一步 Action 的梯度，在 current 中乘了 $R(\tau)$ (整条轨迹的 reward), 在 target 中乘了 reward-to-go: $\gamma^t G_t$.

> Reward to go: $G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k$
> Reward: $R(\tau)=\sum_{k=0}^{T-1}\gamma^k r_k$

因此，我们可以换个角度思考，我们只需要证明两者存在差异的部分是 0，我们展开原始式子，把 $R(\tau)$ 乘进去, 定义 $g_t=\nabla_\theta\log\pi_\theta(a_t\mid s_t)$:

$$
\nabla_\theta J(\theta) = \mathbb E_\tau \left[ \sum_{t=0}^{T-1} R(\tau) g_t \right] = \mathbb E_\tau\left[ \sum_{t=0}^{T-1} \left(\sum_{k=0}^{T-1}\gamma^k r_k\right)g_t \right]
$$

我们将 reward 拆分成过去历史的 reward 和现在的 reward:

$$
\sum_{k=0}^{T-1}\gamma^k r_k = \underbrace{\sum_{k=0}^{t-1}\gamma^k r_k}_{\text{过去的 Reward}} + \underbrace{\sum_{k=t}^{T-1}\gamma^k r_k}_{\text{当前及未来的 Reward}}
$$
我们考虑第一项 需要证明的是:
$$
\mathbb E_\tau\left[ \sum_t g_t \sum_{k<t}\gamma^k r_k \right] = 0
$$
给定当前状态 \(s_t\)，对当前 Action 的所有可能取值求期望：

$$
\begin{aligned} \mathbb E_{a_t\sim\pi_\theta}[g_t\mid s_t] &=\sum_a\pi_\theta(a\mid s_t) \nabla_\theta\log\pi_\theta(a\mid s_t)\\ &=\sum_a\pi_\theta(a\mid s_t) \frac{\nabla_\theta\pi_\theta(a\mid s_t)} {\pi_\theta(a\mid s_t)}\\ &=\sum_a\nabla_\theta\pi_\theta(a\mid s_t)\\ &=\nabla_\theta\sum_a\pi_\theta(a\mid s_t)\\ &=\nabla_\theta 1\\ &=0 \end{aligned}
$$
例如，我们想证明 $\mathbb E_\tau[r_0g_1]=0$, 在 Step 1 选择 Action 之前，$r_0$ 已经确定。用 $H_1$ 表示 Step 1 选择 Action 前的完整历史信息 ($s_0, a_0, r_1, s_1$)。利用条件期望:

$$
\begin{aligned} \mathbb E_\tau[r_0g_1] &=\mathbb E_{H_1}\left[ r_0\mathbb E[g_1\mid H_1] \right]\\ &=\mathbb E_{H_1}[r_0\times0]\\ &=0 \end{aligned}
$$

> 这里对 $\tau$ 轨迹的采样可以看作是两个步骤的采样，首先采样 $s_0, a_0, r_1, s_1$, 然后再这个条件下进一步采样 $a_1$。数学上可以证明这样的采样顺序变化，最终的期望求解是等价的

同理，我们可以使用相同的方式证明: $\mathbb E[r_0g_2]=0$.

$$
\begin{aligned} \mathbb E_\tau[r_0g_2] &=\mathbb E_{H_2}\left[ r_0\mathbb E[g_2\mid H_2] \right]\\ &=\mathbb E_{H_2}[r_0\times0]\\ &=0 \end{aligned}
$$

进一步归纳:
$$
\begin{aligned} \mathbb E_\tau[r_kg_t](k<t) &=\mathbb E_{H_t}\left[ r_k\mathbb E[g_t\mid H_t] \right]\\ &=\mathbb E_{H_t}[r_k\times0]\\ &=0 \end{aligned}
$$
$$
\mathbb E_\tau\left[ g_t\sum_{k=0}^{t-1}\gamma^k r_k \right]=0
$$

原命题证明成立。

因此，我们进一步优化了 Policy Gradient 的形式:

$$
\nabla_\theta J(\theta) = \mathbb E_\tau \left[ \sum_{t=0}^{T-1} \gamma^t G_t \nabla_\theta\log\pi_\theta(a_t\mid s_t) \right] 
$$
### From Reward-to-Go to Advantages

我们现在使用 $G_t$ 来决定一个 action 的梯度应该被抑制还是鼓励。然而，绝对的 reward 的度量并不准确。因次，我们引入 baseline 和 advantage: $\hat{A_t}=G_t-b(s_t)$
同时，我们可以证明数学上的等价性：

$$
 \begin{aligned} \nabla_\theta J(\theta) =\mathbb E_\tau\Bigg[ \sum_{t=0}^{T-1} \gamma^t \underbrace{\left(G_t-b(s_t)+b(s_t)\right)}_{=G_t} \nabla_\theta\log\pi_\theta(a_t\mid s_t) \Bigg] \end{aligned} 
$$

考虑后一项 $\mathbb E_\tau\left[ \sum_{t=0}^{T-1} \gamma^t b(s_t) \nabla_\theta\log\pi_\theta(a_t\mid s_t) \right]$:
类似的方法，我们首先可以证明单个时间步下: $\mathbb E_\tau\left[ b(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t) \right]=0$:

$$
\begin{aligned} &\mathbb E_\tau[ b(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t) ]\\ &= \mathbb E_{H_t}\left[ b(s_t) \mathbb E_{a_t\sim\pi_\theta} [ \nabla_\theta\log\pi_\theta(a_t\mid s_t) ] \right]\\ &= \mathbb E_{H_t}[b(s_t)\times0]\\ &=0 \end{aligned} 
$$

因此，我们证明了后一项是 0，这个代表这梯度的等价性，与此同时，引入 gradient 可以降低整个梯度估计器 $(G_t-b(s_t))\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ 的方差，使得梯度估计可以更加稳定。

> Actor-Critic 方法本质只是更好的估计 $b(s_t)$, 一个只和 state 有关的量: $b(s_t)=V_\phi(s_t)$.

我们可以挑选一个合适的度量来替换我们的 $b(s_t)$, 我们这里选择: $A^\pi(s_t,a_t)= Q^\pi(s_t,a_t)-V^\pi(s_t)$.

回顾条件期望的性质和 Q 的定义:

$$
\mathbb E_{\tau}[ Q^\pi(s_t,a_t) \nabla_\theta\log\pi_\theta(a_t\mid s_t) ]  = \mathbb E_{\tau}[ \mathbb E_\pi[G_t\mid s_t,a_t] \nabla_\theta\log\pi_\theta(a_t\mid s_t) ] = \mathbb E_{\tau}[ G_t \nabla_\theta\log\pi_\theta(a_t\mid s_t) ]
$$

因此，最终，我们定义了 $b(s_t)$ 的具体形式，写成了:

$$
\nabla_\theta J(\theta) = \mathbb E_\tau\left[ \sum_{t=0}^{T-1} \gamma^t \nabla_\theta\log\pi_\theta(a_t\mid s_t) A^{\pi_\theta}(s_t,a_t) \right]
$$
$$
 \nabla_\theta J(\theta) \propto \mathbb E_{s,a\sim\pi_\theta} \left[ \nabla_\theta\log\pi_\theta(a\mid s) A^{\pi_\theta}(s,a) \right]
 $$
### Summary

总结一下，Policy Gradient 究竟在做什么? 关键在于求 $J(\theta)$ 的梯度：
- 为什么要求? $J(\theta)$ 是 RL Loss 的优化目标，需要进行梯度下降
- 为什么要这么求?

- Step1: 定义形式
	- Form: $J(\theta)=\mathbb E_\tau[R(\tau)]$
- Step2: 梯度展开 && 求期望
	- Form: $\nabla_\theta J=\sum_\tau R(\tau)\nabla_\theta p_\theta(\tau)$
	- 存在问题: 离散采样不可微，无法求梯度
- Step3: Log-Derivative Trick
	- Form: $\nabla_\theta J=\mathbb E_\tau[R(\tau)\nabla_\theta\log p_\theta(\tau)]$
	- 存在问题: 梯度项 $p_{\theta}(\tau)$ 由环境和策略共同决定
- Step4: 轨迹概率分解
	- Form: $\nabla_\theta J(\theta) = \mathbb E_\tau \left[ R(\tau) \sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t) \right]$
	- **我们得到了第一个可求解的梯度形式**! 但是我们还可以做更多优化!
- Step5: Reward -> Reward to Go
	- Form: $\nabla_\theta J(\theta) = \mathbb E_\tau \left[ \sum_{t=0}^{T-1} \gamma^t G_t \nabla_\theta\log\pi_\theta(a_t\mid s_t) \right]$
	- 使得整体方差更小，更稳定
- Step6: Reward to Go -> Advantages
	- Form: $\nabla_\theta J(\theta) = \mathbb E_\tau\left[ \sum_{t=0}^{T-1} \gamma^t \nabla_\theta\log\pi_\theta(a_t\mid s_t) A^{\pi_\theta}(s_t,a_t) \right]$

$$
\nabla_\theta J\propto\mathbb E_{s,a}[\nabla_\theta\log\pi_\theta(a\mid s)\,A^\pi(s,a)]
$$

## REINFORCE

我们从理论角度出发，推导了 Policy Gradient 的数学形式:

$$
\nabla_\theta J(\theta) = \mathbb E_\tau\left[ \sum_{t=0}^{T-1} \gamma^t \nabla_\theta\log\pi_\theta(a_t\mid s_t) A^{\pi_\theta}(s_t,a_t) \right]
$$

这里面唯一的未知量就是 $A^{\pi_{\theta}}$, 在上文推导中，我们默认这个是一个已知量，在实际算法求解中，需要被估计计算。

REINFORCE 算法的梯度估计方式非常简单，通过蒙特卡洛采样的方式，直接放弃求解 Advantages，而是统计 $G_t$ ($G_t$ 可以通过蒙特卡罗采样的方式进行)

因此，REINFORCE 方法只是把外层的 $\mathbb{E}$ 换成了一个 MCMC，具体公式就是:

$$
 \hat g = \frac1N\sum_{i=1}^{N} \sum_{t=0}^{T_i-1} \gamma^t G_{i,t} \nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t}) 
$$

接下来，根据梯度进行梯度上升（因为这里不是 loss，而是一个最大化目标）

$$
\theta\leftarrow\theta+\eta\hat g
$$

## Actor-Critic

回到 REINFORCE 没有解决的那个问题: $\nabla_\theta J(\theta) \propto \mathbb E_{s,a\sim\pi_\theta} \left[ \nabla_\theta\log\pi_\theta(a\mid s) A^{\pi_\theta}(s,a) \right]$ 中 $\hat{A}$ 的估计是一个痛点。Actor-Critic 架构提出可以更新一个 V 网络作为可学习的 Value Function:

$$
V_\phi(s)\approx\mathbb E[G_t\mid s_t=s] = V^{\pi}(s)
$$
> 结合 Value Function 的具体定义，Value Function 的作用就是评估当前状态的“价值”，类似于一个 teacher 的作用，提供打分监督。

在有一个近似的 Value Network 后（利用 rollout 的数据进行训练），就可以使用经典的近似算法来估计 $A_t$ 了，比如:

- MCMC: $\hat{A}_t = G_t - V_\phi(s_t)$
- TD Error: $\hat A_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$

> 具体的更新方式，在此处不做展开。

## PPO

PPO 的本质是对 Actor-Critic 网络的改进，因此其本身仍然适配在 Actor-Critic 的网络结构下，但是引入了对应的优化。

在每一次梯度更新是，算法依赖的是策略 $\pi_{\theta}$ 对应的数据采样，但是每一次做完梯度更新后，旧策略的数据会被丢弃，无法继续使用。Actor-Critic 是一种 On Policy 的算法。

在实际应用中，数据收集的利用效率低下，因此可以加入重要性采样的手段，提升数据利用效率，同时避免 Off Policy 带来的问题。

### 重要性采样

假设有一个函数 $f(a)$， 计算期望：

$$
\mathbb E_{a\sim\pi_\theta}[f(a)] = \sum_a\pi_\theta(a\mid s)f(a) =\sum_a \pi_{\theta_{\mathrm{old}}}(a\mid s) \frac{\pi_\theta(a\mid s)} {\pi_{\theta_{\mathrm{old}}}(a\mid s)} f(a)
$$
因此:
$$
\mathbb E_{a\sim\pi_\theta}[f(a)] = \mathbb E_{a\sim\pi_{\theta_{\mathrm{old}}}} \left[ \frac{\pi_\theta(a\mid s)} {\pi_{\theta_{\mathrm{old}}}(a\mid s)} f(a) \right]
$$
这个添加的比例就是重要性采样，也就是说，可以通过添加一个比例系数，来保证期望无偏性。

因此，在 PPO 中，我们定义 probability ratio 代表重要性采样:

$$
\rho_t(\theta)= \frac{ \pi_\theta(a_t\mid s_t) }{ \pi_{\theta_{\mathrm{old}}}(a_t\mid s_t) } 
$$
定义代理目标函数，在旧策略的数据采样下进行:

$$
L^{\mathrm{PG}}(\theta) = \mathbb E_{s,a\sim\pi_{\mathrm{old}}} \left[ \rho_t(\theta)\hat A_t \right]
$$

$$
\nabla_\theta L^{\mathrm{PG}} = \mathbb E_{\mathrm{old}} \left[ \nabla_\theta\rho_t(\theta)\hat A_t \right] = \mathbb E_{\mathrm{old}} \left[ \rho_t(\theta) \nabla_\theta\log\pi_\theta(a_t\mid s_t) \hat A_t \right]
$$
当新旧策略相同时，代理目标函数的梯度就是真正 $J(\theta)$ 的梯度! 但是当两者偏移较远时，此时 $\rho_t(\theta)$ 会导致两者存在偏差。

### Clipping

在上面的框架下，我们没有做对新旧 Policy 差异的约束，如果模型在旧策略下拿到了很高的 $A$, 此时优化器会提高该 action 的激励（哪怕和现在的 policy 已经偏离）

因此，PPO 引入了一个 Clip 机制:

$$
L^{\mathrm{CLIP}}(\theta) = \mathbb E_t \left[ \min\left( \rho_t\hat A_t, \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t \right) \right]
$$

相比于之前的 $L^{\mathrm{PG}}(\theta) = \mathbb E_{s,a\sim\pi_{\mathrm{old}}} \left[ \rho_t(\theta)\hat A_t \right]$, 其主要的改进是后一项：
- $\operatorname{clip}(x,l,u)$ 代表的是将 $x$ 的值限制约束在 $[1-\epsilon,1+\epsilon]$ 中间
- 即，如果 IS 的值超过了 $1 + \epsilon$ 或者低于 $1 - \epsilon$，代表此时针对这个动作新旧策略的概率分布差异太大，需要被 clip 掉
	- 例如 $\rho_t$ 过大，则被 clip 到最大值，此时的结果是 $1+\epsilon$ 
	- 例如 $\rho_t$ 过小，被 clip 到最小值，但是此时 $\hat{A_t}$ 的预测值应该是一个负数 (做出了一个坏操作，对应概率被下调)
		- 此时取 min 仍然是被 clip 的那一项

此外，PPO 作为经典的 Actor-Critic 架构，会使用 GAE 来估计 Advantage:

定义 TD_error:

$$
\delta_t= r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$
对应的估计是多阶 TD-Error 的加权结果。
$$
\hat A_t^{\mathrm{GAE}} = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l\delta_{t+l}
$$

### Off-Policy or On-Policy?

我们具体梳理在工程实现中，PPO 的实现方式是什么:

- Iteration $X$:
	- **==Rollout==**: 使用 Policy $\theta_0$ 进行 rollout (运行 rollout batch sizes 次)
	- **==PPO Update==**: for Epoch $s$:
		- 对于每一个 mini-batch 聚合轨迹数据，计算梯度，进行梯度下降
		- $\theta_0$ -> $\theta_{0,1}$ -> $\theta_{0,2}$ -> ...
		- 注意，哪怕后续发生了多次偏移，在单个 iteration 中，所有的数据都使用 $\theta_0$ 的数据进行更新，通过 IS 进行采样修复
	- 最终，得到新的 Policy $\theta_1$
- Iteration $X+1$: ...
	- 使用 $\theta_1$ 进行轨迹数据 rollout（不再使用上一个 iteration 中的数据）

假设我们规定超参数:
- Rollout Batch Size = 1024 Responses
- Training Mini-batch Size = 256 Responses
- PPO Epoch = 4

则在单个 epoch 中 minibatches 的数量是: 1024/256 = 4, 因此在一个 iterations 内，需要做 16 次 gradient descent ($\texttt{epoch\_num} \times \texttt{mini\_batch\_size}$). 也就是说，可能在一个 iteration 中会有 16 个 policy checkpoint! 但是这些都使用相同的数据计算，只不过被赋予了不同的重要性采样。

因此，PPO 本质上是一种 On-Policy 的算法，但是为了提升数据复用利用效率，允许一部分数据的 Off-Policy 复用。

