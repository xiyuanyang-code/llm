在上一讲，我们从最基本的 RL 理论出发，一路推导到了最经典的 RL 理论算法: PPO

在 PPO 基础上，一系列经典的强化学习算法被应用到了 LLM 后训练中，我们会从 timeline 出发，了解GRPO 时代和前 GRPO 时代中，那些经典的 LLM RL 算法。

![[grpo.png]]

## Timeline

我们有必要梳理一下在 LLM 中经典 RL 算法被提出的时间线:

- 2017 年，PPO 算法首次被 OpenAI 提出 (https://arxiv.org/pdf/1707.06347)
- 2022 年，RLHF-PPO 被提出，强化学习算法首次被应用到大语言模型的训练中 (https://arxiv.org/pdf/2203.02155)
- 2022 年，业界出现了一系列针对 RLHF-PPO 的算法改进：
	- 2023 年，Remax 被提出 (https://arxiv.org/pdf/2310.10505)，删除了 Value-Model
	- 2023 年，DPO 被提出(https://arxiv.org/pdf/2305.18290), 删除了 Reward-Model 和 Value-Model，直接在偏好数据对上进行训练
- 2024 年，DeepSeek-Math 提出 GRPO 算法 (https://arxiv.org/pdf/2402.03300), 通过强化学习在可验证领域大规模提升模型表现
	- 直到 2024 年，Tulu3 Tech Report (https://arxiv.org/pdf/2411.15124) 才定义了 RLVR: 将环境反馈直接作为 reward，而不依赖 reward model（==不过其实很早就开始了==）
	- 进入了 RLVR (Math, Coding) 的时代
- After 2024: 一个崭新的时代（在第三讲详细说明）


## PPO for LLMs

### Vanilla PPO

在 LLM 的场景下，我们首先来看如何把 vanilla-ppo 算法迁移到 LLM 建模中去:

LLM 本身的参数作为 Policy $\theta$, 我们只考虑单次 LLM 交互的场景，且**暂时不考虑 Next-Token-Prediction** 的过程，given prompt $x$, 输出 response $y$, 一个基于环境的打分器给出 reward = $R(x, y)$.

因此，目标函数可以写成一个更简单的形式:

$$
J(\theta)=\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)]
$$

在原始的 Policy Gradient 中，形式是:

$$
\nabla_\theta J(\theta) = \mathbb E_\tau\left[ \sum_{t=0}^{T-1} \gamma^t \nabla_\theta\log\pi_\theta(a_t\mid s_t) A^{\pi_\theta}(s_t,a_t) \right]
$$

但是在这里，对轨迹 $\tau$ 采样就等价于对输出 response $y$ 采样，且不存在 $\sum_{0}^{T-1}$(因为只考虑单轮), 此时可以简化 Policy Gradient 的形式是:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)} \left[ \nabla_\theta\log\pi_\theta(y\mid x) A^{\pi_\theta}(x,y) \right]
$$

其中: $A^{\pi_\theta}(x,y) =R(x,y)- \mathbb E_{y'\sim\pi_\theta(\cdot\mid x)}[R(x,y')]$， 后者在 Vanilla-PPO 中通过一个 Value Network 来进行拟合。

### RLVR-PPO/RLHF-PPO

在 RLHF-PPO 中提出了一个核心改进，即引入了 Reference Model 和 Reward Model:

- Policy Model: 被训练的核心模型，输入仍然是 $\pi_{\theta}(a \mid s)$, 在这里 $s$ 就是给定的 prompt (例如一个数学题), 输出就是对应的 response, 得到一个 reward score.
- Value Model: 在 PPO 中训练的 Value Network
- Reward Model: ==冻结==，针对模型输出 response 进行打分，类似于之前 rl 中的 $r$
	- 在 RLHF 场景下，奖励信号是根据 reward-model（使用偏好数据对进行训练）给出的
	- 在 RLVR 场景下，奖励信号直接根据环境给出
- ==Reference Model==: ==冻结==，主要避免过度微调，设计的 KL 散度 reward

我们需要额外加一个 KL 惩罚:

$$
J_{\mathrm{RLHF}}(\theta) = \mathbb E_{q,o\sim\pi_\theta} \left[ r_{\mathrm{RM}}(q,o) \right] - \beta \mathbb E_q\left[ D_{\mathrm{KL}} \left( \pi_\theta(\cdot\mid q) \Vert \pi_{\mathrm{ref}}(\cdot\mid q) \right) \right]
$$

>在我们之前的推导中，我们的数学期望内部是 $R(\tau)$，要求这一项对 $\theta$ 求梯度的时候，梯度为 0, 但是在引入包含 $\theta$ 的 KL 散度项之后，Policy Gradient 还成立吗?

将期望展开：

$$
J(\theta) = \sum_y\pi_\theta(y\mid x) \left[ r(x,y)-\beta\log\frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} \right]
$$

现在对 $\theta$ 求梯度。根据梯度计算的运算法则，左导右不导，右导左不导:

$$
\begin{aligned} \nabla_\theta J ={}& \underbrace{ \sum_y\nabla_\theta\pi_\theta(y\mid x) \left[ r(x,y)-\beta\log\frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} \right] }_{\text{第一项：Policy Probability 的梯度}}\\[8pt] &- \underbrace{ \beta\sum_y\pi_\theta(y\mid x) \nabla_\theta\log\frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} }_{\text{第二项：KL Reward 自身的梯度}} \end{aligned}
$$
下面，我们证明第二项是 0，我们基于梯度化简第二项:

$$
\sum_y\pi_\theta(y\mid x) \nabla_\theta\log\frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} = \sum_y\pi_\theta(y\mid x) \nabla_\theta\log{\pi_\theta(y\mid x)} = \sum_y \nabla_\theta\pi_\theta(y\mid x)
$$

我们惊喜的发现，我们可以使用第一讲推导的零均值性质:
$$
\sum_y \nabla_\theta\pi_\theta(y\mid x) = \nabla_\theta\sum_y \pi_\theta(y\mid x) = \nabla_\theta 1 = 0
$$
因此，我们可以推导，哪怕在加上 KL 散度后，我们的 reward 可以被写为:

$$
\nabla_\theta J =  \sum_y\nabla_\theta\pi_\theta(y\mid x) \left[ r(x,y)-\beta\log\frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} \right] = \sum_y \pi_\theta(y\mid x)\nabla_\theta\log\pi_\theta(y\mid x) \left[ r(x,y)-\beta\log\frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)} \right]
$$

此时，我们可以认为 $r'(x,y) = r(x,y)-\beta\log\frac{\pi_\theta(y\mid x)} {\pi_{\mathrm{ref}}(y\mid x)}$，这个项已经被转移到了梯度之外，后续剩下的 PPO 变形都与这个无关了，因此，加上 KL 散度的 reward 项仍然成立。

> 注意，但是这个不是一个普适的性质，因为我们必须得保证第二项  (KL Reward 自身期望的梯度为 0 )

### Token Level or Sequence Level?

在上文无论是 Vanilla-PPO 还是 RLHF-PPO，都是在 sequence-level 上做的，也就是说，我们假想模型 $\pi_{\theta}$ 给定输入 $x$, 直接输出 $y$，忽略了 Next-Token-Prediction 的方式。在 PPO 中，LLM 的 Auto-Regressive（自回归）特性实际上会带来一个很重要的结果：

> 整个 Response 的 KL Divergence，可以分解成每一个 Token 的 Conditional KL Divergence。

具体来说，Next-token-prediction 可以通过概率的方式建模出来:

$$
\log\pi_\theta(y\mid x) =\sum_{t=1}^{|y|} \log\pi_\theta(y_t\mid x,y_{<t})
$$

具体到 KL 散度，可以被分解为:

$$
 \begin{aligned} \log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} &=\log\frac{\prod_{t=1}^T\pi_\theta(y_t\mid s_t)} {\prod_{t=1}^T\pi_{\mathrm{ref}}(y_t\mid s_t)} \\ &=\sum_{t=1}^T\log\frac{\pi_\theta(y_t\mid s_t)} {\pi_{\mathrm{ref}}(y_t\mid s_t)} \end{aligned}
$$
我们可以定义每个 token 采样的 log-ratio: $k_t=\log\pi_\theta(y_t\mid s_t) -\log\pi_{\mathrm{ref}}(y_t\mid s_t)$:

$$
\boxed{ D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}}) = \mathbb E_{y\sim\pi_\theta} \left[\sum_{t=1}^T k_t\right] }
$$
因为模型只会在最后的 sequence-level 给出一个 reward，因此我们需要手动构造 token-level 的 reward，在 RLHF-PPO 中，token-level 的 reward 是这样给出的:

$$
\tilde r_t= \begin{cases} -\beta k_t,&t<T\\[4pt] R(x,y)-\beta k_t,&t=T \end{cases}
$$

也就是说：
- 中间 Token：只获得 KL Penalty。
- 最后一个 Token：获得 KL Penalty 和最终任务 Reward。

> 具体代码的实现细节可以查看: https://github.com/huggingface/blog/blob/main/the_n_implementation_details_of_rlhf_with_ppo.md

这样的 reward 定义可以保证，累和后的 reward 等于 $R(x,y)$（不考虑 discount factor）. 因此，一次 PPO forward 的过程可以被表征成: $(s_0 = x, a_0 = y_1, r_0, s_1 = (x, y_1), a_2 = y_2, \dots)$ 的一个 POMDP 过程，后续按照 PPO 的方式可以正常进行强化学习。

> 如果不是最后一个 token，如何获得奖励? 回忆 PPO 中，我们的 $A(s,a)$ 是从 reward-to-go 推导出来了，因此，理论上每一个 token 的奖励回传的时候，都包含了最后一个 token 的 result-driven 的奖励。（实际使用 GAE 进行 reward estimate）

因此，最终在 token-level 上，RLHF-PPO 的 objective 是:

$$
L^{\mathrm{CLIP}}(\theta)= \mathbb E_{y\sim\pi_{\mathrm{old}}} \left[ \frac{1}{T}\sum_{t=1}^T \min\left( \rho_t(\theta)\hat A_t,\, \operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t \right) \right]
$$

## Direct Preference Optimization (DPO)

### TL;DR

在 22 年提出的 RLHF-PPO 中，模型首先根据偏好数据对学习出一个 Reward-Model 进行打分，然后进行 PPO Update。但是，reward model 给出的监督信号往往是不准确的，同时会存在 reward-hacking 等问题。
因此，DPO 希望 LLM 可以直接在偏好数据集上学习，而不再需要“绕个路”，依赖 reward model 提供的奖励信号。

> 在 RLVR 上，这个问题其实并不存在，因为环境会给出最准确的奖励信号，但是在其他 hard-to-verify 的问题 && 指令偏好对齐的场景下，reward 信号的问题显的极为重要。

因此，DPO 和 PPO 在训练目标和训练方式上存在很大的差异:

- DPO 算法是离线 off-policy 的，也就是训练中不需要引入任何的 rollout 过程
- DPO 算法需要的数据是静态的离线偏好数据对 (这也是 RLHF 的一种格式之一。)

DPO 的亮点就在于它可以绕过 reward model，直接在静态数据上产生梯度，进行训练。
### Removing the RL from RLHF

- Recall we want to maximize the following objective in RLHF
$$\mathbb{E}_{\hat{y} \sim p_{\theta}^{RL}(\hat{y}|x)}[RM_{\phi}(x, \hat{y}) - \beta \log \left(\frac{p_{\theta}^{RL}(\hat{y}|x)}{p^{PT}(\hat{y}|x)}\right)]$$
There is a closed form solution to this:
$$p^{*}(\hat{y}|x) = \frac{1}{Z(x)} p^{PT}(\hat{y}|x) \exp(\frac{1}{\beta} RM(x, \hat{y}))$$
> $Z(x)$ is a normalization constant.

- Rearrange this via a log transformation
$$RM(x, \hat{y}) = \beta \left(\log p^{*}(\hat{y}|x) - \log p^{PT}(\hat{y}|x)\right) + \beta \log Z(x) = \beta \log \frac{p^{*}(\hat{y}|x)}{p^{PT}(\hat{y}|x)} + \beta \log Z(x)$$
- This holds true for any arbitrary LMs, thus

$$RM_{\theta}(x, \hat{y}) = \beta \log \frac{p_{\theta}^{RL}(\hat{y}|x)}{p^{PT}(\hat{y}|x)} + \beta \log Z(x)$$

What does that tells us? The ideal RM model can be computed if we get the optimized policy $p_\theta^{\text{RL}}$.
### Algorithm

Hence, we can derive the reward model directly:
$$RM_\phi(x, \hat{y}) \propto \log \frac{p_{\theta}^{RL}(\hat{y}|x)}{p^{PT}(\hat{y}|x)}$$
> We just ignore $Z(x)$ for it is constants.

Defining **Direct Preference Optimization**:
Previous: Highly relied on trained reward model (bottleneck)
$$L_{RL}(\theta) = -\mathbb{E}_{\hat{y} \sim p_{\theta}^{RL}(\hat{y}|x)}[RM_{\phi}(x, \hat{y}) - \beta \log \left(\frac{p_{\theta}^{RL}(\hat{y}|x)}{p^{PT}(\hat{y}|x)}\right)]$$
Now: Using DPO into a loss function, learning directly from $p^{PT}$ to $p_{\theta}^{RL}$
$$L_{DPO}(\theta) = -\mathbb{E}_{(x,y_W,y_l)\sim D} \log\sigma\left(\beta \log\frac{p_{\theta}^{RL}(y_w|x)}{p^{PT}(y_w|x)} - \beta \log\frac{p_{\theta}^{RL}(y_l|x)}{p^{PT}(y_l|x)}\right)$$
$$L_{DPO}(\theta) = -\mathbb{E}_{(x,y_W,y_l)\sim D} \log\sigma\left(\beta \log\frac{p_{\theta}^{RL}(y_w|x) p^{PT}(y_l|x)}{p_{\theta}^{RL}(y_l|x) p^{PT}(y_w|x)}\right)$$
> Using the data directly (learn directly!)
## GRPO for LLMs

### Advantages Estimations

在 RLHF 时代，后训练 RL 更关注的是偏好对齐，DPO 给出了在偏好数据集上进行训练的最优解。在 RLHF 时代过后，就是以 GRPO 为主导的 **RLVR 时代**，模型可以在**可验证的环境**上进行 RL 训练，提升特定领域的能力。

在 PPO 中，一共会涉及四个模型，管线异常复杂，工程复杂度高。因此，在 2023 年，RL 的研究重点是: 能否针对这个 Pipeline 进行简化?

- PPO 如何估计 Advantages? 利用 GAE，根据 Value 网络进行:
	- Define TD-Error: $\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$
	- $$ \hat A_t^{\mathrm{PPO}} = \sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}$$
	- Advantage 中包含 sequence-level 的奖励加权，但是每个 token 给出的 reward 只是 KL 散度
- DPO: 直接在静态偏好数据集上进行简化，省略 reward-model 的存在
- **Remax && GRPO**: **PPO 使用 Value Model 估计 Advantage；ReMax 和 GRPO 则尝试使用模型自身生成的其他 Responses 来构造 Baseline，从而不再需要训练 Value Model。**

受到篇幅限制，我们在这里只重点展开 GRPO 给出的算法改进：

- Group Sampling:
	- 对于同一个 Prompt \(x\)，从 Old Policy 采样 $G$ 条 Responses
	- $y_1,\ldots,y_G \overset{\mathrm{iid}}{\sim} \pi_{\theta_{\mathrm{old}}}(\cdot\mid x)$
- GRPO 的关键改进在于: 不再依赖 Value-Model 进行 Value estimate，而是依赖统计量进行估计。

Define $R_i=R(x,y_i)$, GRPO 计算 Group Mean：
$$
\mu_R=\frac1G\sum_{i=1}^G R_i
$$
再计算 Group Standard Deviation：

$$
\sigma_R = \sqrt{\frac1G\sum_{i=1}^G(R_i-\mu_R)^2}
$$
最后得到：
$$
\hat A_i^{\mathrm{GRPO}} = \frac{R_i-\mu_R}{\sigma_R}
$$
在 token-level 的时候，一个 sequence 上生成的所有 token 都享有这个 advantages（因为 GRPO 不再利用 GAE 进行 Advantages 估计，因此需要保证每一个 token 都能拿到 sequence-level 的 advantages）

在原始 GRPO 的 Outcome Supervision 中，同一个 Response 的全部 Token 共享同一个 Advantage: (i 是 rollout 标记, t 是对应 token-index)

$$
\hat A_{i,t}=\hat A_i^{\mathrm{GRPO}}
$$
在原始的 Policy Gradient 中，形式是:

$$
\nabla_\theta J(\theta) = \mathbb E_\tau\left[ \sum_{t=0}^{T-1} \gamma^t \nabla_\theta\log\pi_\theta(y_t\mid s_t) A^{\pi_\theta}(s_t,y_t) \right]
$$

因此，在 GRPO 中，原始的 Policy Gradient 可以写成:

$$\nabla_\theta J \approx \frac1G\sum_{i=1}^G \hat A_i \sum_{t=1}^{T_i} \nabla_\theta\log\pi_\theta(y_{i,t}\mid s_{i,t})
$$

> 这里使用近似号，是因为 GRPO 的组内标准化 Advantage 并非严格无偏的真实 Advantage Estimator。

### Vanilla-GRPO

同样，我们继续加上 PPO-Clip 和 KL 散度约束，形成最终的 GRPO 原始版形式:

$$
\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})} {\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}
$$

$$
\begin{aligned} J_{\mathrm{GRPO}}^{\mathrm{clip}}(\theta) ={}& \operatorname*{\mathbb E}_{\substack{ q\sim P(Q)\\ \{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q) }} \Bigg[ \frac1G\sum_{i=1}^G\frac1{T_i}\sum_{t=1}^{T_i}\\ &\min\Big( \rho_{i,t}(\theta)\hat A_i,\, \operatorname{clip}\left( \rho_{i,t}(\theta),1-\epsilon,1+\epsilon \right)\hat A_i \Big) \Bigg] \end{aligned}
$$
原始 GRPO 将 Reference KL 作为单独的惩罚项，理论上的 KL Divergence 定义为：

$$
D_{\mathrm{KL}}\left( \pi_\theta(\cdot\mid s) \Vert \pi_{\mathrm{ref}}(\cdot\mid s) \right) = \sum_{v\in\mathcal V} \pi_\theta(v\mid s) \log\frac{\pi_\theta(v\mid s)} {\pi_{\mathrm{ref}}(v\mid s)}
$$

其中 $\mathcal V$ 是 Vocabulary，$s=(q,o_{i,<t})$ 表示当前 Token 的上下文。

但原始 GRPO 并没有直接对整个 Vocabulary 计算 KL，而是使用一个基于采样 Token 的 KL Estimator：
$$
\begin{aligned} \widehat D_{\mathrm{KL},i,t}(\theta) ={}& \frac{\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})} {\pi_\theta(o_{i,t}\mid q,o_{i,<t})}\\ &-\log \frac{\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})} {\pi_\theta(o_{i,t}\mid q,o_{i,<t})} -1 \end{aligned}
$$
这个形式有两个重要性质：
- 对任意采样 Token，估计值非负。
- 如果 Token 从当前 Policy $\pi_\theta$ 中采样，那么这个估计量的期望等于对应状态的 Forward KL，即 $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}}$)。

定义 KL Regularization：

$$
J_{\mathrm{GRPO}}^{\mathrm{KL}}(\theta) = \operatorname*{\mathbb E}_{\substack{ q\sim P(Q)\\ \{o_i\}_{i=1}^G\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q) }} \left[ \frac1G\sum_{i=1}^G \frac1{T_i}\sum_{t=1}^{T_i} \widehat D_{\mathrm{KL},i,t}(\theta) \right]
$$

$$
J_{\mathrm{GRPO}}(\theta) = J_{\mathrm{GRPO}}^{\mathrm{clip}}(\theta) -\beta J_{\mathrm{GRPO}}^{\mathrm{KL}}(\theta)
$$

> GRPO 中的 KL 设计展开又是一个可以讲很久的 topic，我们暂时在此处忽略。具体可以参考 http://joschu.net/blog/kl-approx.html 这里的 KL Approximations

## Summary

在 DeepSeek-Math 的论文中，将所有的 RL 和训练算法统一起来，形成了一个实在精彩的段落，在本章节进行细节诠释 (https://arxiv.org/pdf/2402.03300, 5.2: Insights of Reinforcement Learning)

In this section, we provide a unified paradigm to analyze different training methods, such as **SFT, RFT, DPO, PPO, GRPO**, and further conduct experiments to explore the factors of the unified paradigm.
Generally, the gradient with respect to the parameter $\theta$ of a training method can be written as:

$$
\nabla_{\theta}\mathcal{J}_{\textcolor{red}{\mathcal{A}}}(\theta) = \mathbb{E}[\underbrace{(q,o) \sim \textcolor{red}{\mathcal{D}}}_{Data \ Source}]\left( \frac{1}{|o|} \sum_{t=1}^{|o|} \underbrace{GC_{{\mathcal{A}}}(q, o, t, \textcolor{red}{\pi_{{rf}}})}_{Gradient \ Coefficient} \nabla_{\theta}\log \pi_{\theta}(o_t | q, o_{<t})\right).
$$

- Data Source $\mathcal{D}$ 代表采样的数据源
- Reward Function $\pi_{{rf}}$： 给定的 reward
- Algorithm $\mathcal{A}$: processes the training data and the reward signal to the gradient coefficient $GC$ that determines the magnitude of the penalty or reinforcement for the data.

如何理解 Gradient Coefficient? 在上文，核心的梯度通过 Policy Gradient 被传导到 $\nabla_{\theta}\log \pi_{\theta}(o_t | q, o_{<t})$ 中。算法的核心就在于，对特定 token 的梯度进行强化，对特定 token 的梯度进行弱化或者乘法。

- $GC>0$：沿着增加该 Token Log Probability 的方向更新，即 Reinforcement。
- $GC<0$：沿着降低该 Token Log Probability 的方向更新，即 Penalty。
- $|GC|$ 越大：该 Token 的梯度贡献被赋予越大的权重。
- $GC=0$：该 Token 对这个梯度表达式不产生贡献。

![[grpo_2.png]]
### SFT

SFT 的目标是最大化正确答案的 Log Likelihood：

$$
J_{\mathrm{SFT}}(\theta) = \mathbb E_{(q,o)\sim\mathcal D_{\mathrm{SFT}}} \left[ \frac1{|o|}\sum_t\log\pi_\theta(o_t\mid s_t) \right]
$$
直接求导：

$$
\nabla_\theta J_{\mathrm{SFT}} = \mathbb E\left[ \frac1{|o|}\sum_t \underbrace{1}_{GC_{\mathrm{SFT}}} \nabla_\theta\log\pi_\theta(o_t\mid s_t) \right ]
$$
因此，在 SFT 中，所有数据集中的 token 都被赋予了相同的梯度。

### RFT && Online RFT

RFT(Rejection Sampling Fine-tuning) 的数据池是 SFT 采样出来的数据。先从模型采样 Responses，然后通过 Verifier 判断结果：
$$\mathbb I(o)= \begin{cases} 1,&\text{答案正确}\\ 0,&\text{答案错误} \end{cases}$$

于是：
$$
GC_{\mathrm{RFT}}=\mathbb I(o)
$$
对于 Online RFT，GC 没有变化，只是数据是从实时的 Policy $\theta$ 采样得到，而不是 SFT 的静态采样数据。
### DPO

DPO 在静态偏好数据集上进行训练，GC 的具体表现形式如下:

$$
GC_{DPO}(q,o,t) = \sigma\left(\beta\log \frac{\pi_{\theta}(o^-_t | q, o^-_{<t})}{\pi_{\text{ref}}(o^-_t | q, o^-_{<t})} - \beta\log \frac{\pi_{\theta}(o^+_t | q, o^+_{<t})}{\pi_{\text{ref}}(o^+_t | q, o^+_{<t})}\right)
$$
### PPO

对 PPO 来说，就是 GAE 估计的 Advantage，对于大于 0 的 advantage 的 token 梯度强化，对于小于 0 的 advantage 的 token 梯度施加惩罚。
$$
GC_{\mathrm{PPO}}(q,o,t)=\hat A_t^{\mathrm{GAE}}
$$

其中：
$$
\hat A_t^{\mathrm{GAE}} = \sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}
$$
> 这里的 PPO 是指最初始的 PPO 形式，没有加上 KL 散度项 (不是 RLHF-PPO)
### GRPO

原始 GRPO 的目标包含：

$$
J_{\mathrm{GRPO}} = J_{\mathrm{clip}}-\beta J_{\mathrm{KL}}
$$

其中单 Token 的 KL Estimator 是：

$$
k_t= \frac{\pi_{\mathrm{ref}}(o_t\mid s_t)} {\pi_\theta(o_t\mid s_t)} -\log\frac{\pi_{\mathrm{ref}}(o_t\mid s_t)} {\pi_\theta(o_t\mid s_t)}-1
$$
令 $r_t=\frac{\pi_{\mathrm{ref}}(o_t\mid s_t)} {\pi_\theta(o_t\mid s_t)}$, 由于 Reference Model 不参与训练：
$$
\nabla_\theta r_t = -r_t\nabla_\theta\log\pi_\theta(o_t\mid s_t)
$$
因此：
$$
\begin{aligned} \nabla_\theta(-\beta k_t) &=-\beta\nabla_\theta(r_t-\log r_t-1)\\ &=\beta(r_t-1)\nabla_\theta\log\pi_\theta(o_t\mid s_t) \end{aligned}
$$
因此，GRPO 的这一部分可以被写为 **Grouped Advantages** 和 **KL 梯度计算**的部分。从几何意义上，也非常明显：
- 在组间具备优势，得到正向的 GC
- 如果 ref-model 产生这个 token 的概率更大，此时说明当前 Policy 相比于 Reference Model，低估了这个 Token 的概率。（拉回来）

> GRPO 两项的 GC 是两个独立的梯度，前者希望给得到正确  response 的 value 更高的价值，后者希望给偏离于 ref-model 的 token 一些很大的惩罚。

$$
GC_{GRPO}(q, o, t, \pi_{\theta_{rm}}) = \hat{A}_{i,t} + \beta \left(\frac{\pi_{ref}(o_{i,t}|o_{i,<t})}{\pi_{\theta}(o_{i,t}|o_{i,<t})} - 1\right)
$$

### OPD

On Policy Distillation 是一个比较新颖的 RL 方式，通过引入教师模型，给予 teacher-level 的 token 信号监督，可以实现更稠密的监督信号。OPD 的数据来自 Student 自己生成的 Rollout，但监督信号来自 Teacher 对 Student 所访问的 Prefix 给出的概率分布。

固定一个 Prefix $s$，Reverse-KL OPD 希望最小化：
$$
D_{\mathrm{KL}}(p_\theta\Vert p_T) = \sum_{v\in\mathcal V} p_\theta(v\mid s) \log\frac{p_\theta(v\mid s)}{p_T(v\mid s)}
$$

如果写成最大化目标：$J_{\mathrm{OPD}}(\theta;s) = -D_{\mathrm{KL}}(p_\theta\Vert p_T)$

和 RL 一样，如法炮制展开求梯度：

$$
J_{\mathrm{OPD}} = -\sum_vp_\theta(v\mid s) \log\frac{p_\theta(v\mid s)}{p_T(v\mid s)}
$$
$$
\begin{aligned} \nabla_\theta J_{\mathrm{OPD}} ={}&-\sum_v\nabla_\theta p_\theta(v\mid s) \left( \log\frac{p_\theta(v\mid s)}{p_T(v\mid s)}+1 \right) \end{aligned}
$$

我们再次利用零均值性质: $\sum_v\nabla_\theta p_\theta(v\mid s)=0$ 和 log 对数技巧:

$$
\begin{aligned} \nabla_\theta J_{\mathrm{OPD}} &= -\sum_v\nabla_\theta p_\theta(v\mid s) \log\frac{p_\theta(v\mid s)}{p_T(v\mid s)}\\ &= \sum_vp_\theta(v\mid s) \log\frac{p_T(v\mid s)}{p_\theta(v\mid s)} \nabla_\theta\log p_\theta(v\mid s) \end{aligned}
$$
$$
\nabla_\theta J_{\mathrm{OPD}}(\theta;s) = \mathbb E_{v\sim p_\theta(\cdot\mid s)} \left[ \underbrace{ \log\frac{p_T(v\mid s)}{p_\theta(v\mid s)} }_{GC_{\mathrm{OPD}}} \nabla_\theta\log p_\theta(v\mid s) \right]
$$

因此，对于常见的 Reverse-KL On-Policy Distillation，在固定 Token Prefix 上使用当前 Student Policy 采样时，可以得到：
$$
GC_{\mathrm{OPD}}(q,o,t) = \log\frac{ \pi_{\mathrm{teacher}}(o_t\mid q,o_{<t}) }{ \pi_\theta(o_t\mid q,o_{<t})}
$$

## On the Insights of Distributions: SFT, RL, OPD

### KL Divergence

我们关注的问题就是从算法原理角度，能否统一所有的后训练算法? 下面的这篇推文从 KL 散度的视角出发，给出了详细的解释。
相关链接:
- https://zhuanlan.zhihu.com/p/2070328173166866733
- https://nrehiew.github.io/blog/sft_rl_opd/

首先，我们重新给出 KL 散度的数学定义，KL 散度的作用是衡量两个概率分布之间的差异情况，但优化方向不同，会导致截然不同的行为：Forward KL 倾向于 Mode Covering（覆盖多个模式），Reverse KL 倾向于 Mode Seeking（集中于某个模式）。

假设有两个概率分布：
- $P(x)$：目标分布（Target Distribution），希望模型学习的分布。
- $Q_\theta(x)$：模型分布（Model Distribution），可以通过参数 $\theta$ 优化。

KL Divergence（相对熵）定义为：
$$
 D_{\mathrm{KL}}(P\|Q)=\mathbb E_{x\sim P}\left[\log\frac{P(x)}{Q(x)}\right]
$$
对于离散分布：
$$
D_{\mathrm{KL}}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

对于连续分布：
$$
D_{\mathrm{KL}}(P\|Q)=\int P(x)\log\frac{P(x)}{Q(x)}dx
$$

KL 分布有两个重要的数学性质:
- $D_{\mathrm{KL}}(P\|Q)\geq 0$
- $D_{\mathrm{KL}}(P\|Q)=0\iff P=Q$

但是，KL 散度不是对称的，如果交换 Q 和 P 的位置，会产生不同的优化目标: $D_{\mathrm{KL}}(P\|Q)\neq D_{\mathrm{KL}}(Q\|P)$

### Forward KL & Reverse KL

Forward KL:
$$
D_{\mathrm{KL}}(P\|Q_{\theta})=\mathbb E_{x\sim P}\left[\log\frac{P(x)}{Q_{\theta}(x)}\right] = E_{x\sim P}\left[\log P(x)\right] - E_{x\sim P}\left[\log Q_{\theta}(x)\right]
$$

因为前一项是针对 P（目标分布）的一个固定值，和 $\theta$ 无关，因此此时 KL 散度等价于后一项：对于在 P 分布中出现的 x，需要最大化他们在当前参数分布中出现的概率。**Forward KL 要求模型尽可能提高目标分布中真实样本的概率。**

> 这本质上就是最大似然估计! 我们需要找到最优的参数 $\theta$ 来最大化概率。

Reverse KL对当前参数分布中进行采样，检验该分布在目标分布中出现的比例。
$$
 D_{\mathrm{KL}}(Q_\theta\|P) =\mathbb E_{x\sim Q_\theta}\left[\log\frac{Q_\theta(x)}{P(x)}\right]
$$

展开：
$$
D_{\mathrm{KL}}(Q_\theta\|P) =\mathbb E_{Q_\theta}[\log Q_\theta(x)] -\mathbb E_{Q_\theta}[\log P(x)]
$$

注意到 $H(Q_\theta)=-\mathbb E_{Q_\theta}[\log Q_\theta(x)]$

因此：$D_{\mathrm{KL}}(Q_\theta\|P) =-H(Q_\theta)-\mathbb E_{Q_\theta}[\log P(x)]$

所以最小化 Reverse KL 等价于最大化：

$$
\max_\theta\left( \mathbb E_{x\sim Q_\theta}[\log P(x)] +H(Q_\theta) \right)
$$

这里的目标函数有两项组成：
- 第一项约束在当前模型采样的区域，P(x) 必须需要有很高的概率
- 第二项约束了自身分布的熵，熵过小说明分布坍缩了

### Mode Seeking & Mode Covering

我们考虑极端的情况，假设在前向 KL 中，存在 $P(x) > 0, Q(x) \to 0$ 的情况，此时因为 Q 在分母上，因此 KL 散度会变的无穷大。这说明**前向 KL 会严重惩罚那些在当前分布中缺乏，但是在目标分布中存在** 的分布，展现为 ==Mode Covering== (覆盖目标分布，哪怕方差很大)

举一个具体的例子, 考虑一个目标分布有两个模式，我们在前向 KL 作为优化目标的情况下，尝试使用单峰去拟合双峰：

$$
P(x)=\frac12\mathcal N(-4,1)+\frac12\mathcal N(4,1)
$$

$$
Q_\theta(x)=\mathcal N(\mu,\sigma^2)
$$
由于：
$$
\log Q_\theta(x) =-\frac12\log(2\pi\sigma^2) -\frac{(x-\mu)^2}{2\sigma^2}
$$

所以最小化 Forward KL 等价于最小化：

$$
\mathcal L(\mu,\sigma^2) =\frac12\log(2\pi\sigma^2) +\frac{\mathbb E_P[(x-\mu)^2]}{2\sigma^2}
$$
对两个参数分别求导得到 0，可以得到：

- $\mu^*=\mathbb E_P[x] = 0$
- $(\sigma^2)^*=\mathrm{Var}_P(x) = 17$

![[kl_1.png]]

对于 reverse kl，情况会不一样，因为此时数据是在 $P(x)$ 上进行采样的，我们来看 Reverse KL 的优化目标:
$$
\max_\theta\left( \mathbb E_{x\sim Q_\theta}[\log P(x)] +H(Q_\theta) \right)
$$
- 首先，Q 本身的分布不可以过于尖锐（熵的约束）
- 其次，**只需要在 Q 分布的区域，目标分布的出现概率较大即可**

因此，目标参数不会选择拟合 P 分布中所有的 Mode，而是会选择性找到一个比较温和的局部分布，使其具备一定拟合性即可，因此我们把这种定义为 ==Mode Seeking==.

### SFT: Off-Policy, Mode-Covering, Forward-KL

定义 Forward KL：

$$
D_{\mathrm{KL}}(p_{\mathrm{data}}\|q_\theta) =\mathbb E_{x\sim p_{\mathrm{data}}} \left[\log\frac{p_{\mathrm{data}}(x)}{q_\theta(x)}\right]
$$

因此, KL 散度可以被写成:

$$
\begin{aligned} D_{\mathrm{KL}}(p_{\mathrm{data}}\|q_\theta) &=\mathbb E_{p_{\mathrm{data}}}[\log p_{\mathrm{data}}(x)]\\ &\quad-\mathbb E_{p_{\mathrm{data}}}[\log q_\theta(x)]\\ &=-H(p_{\mathrm{data}})+H(p_{\mathrm{data}},q_\theta) \end{aligned}
$$
- 第一项是静态数据的分布熵，是一个固定值
- 第二项是模型参数和数据分布的 cross-entropy，这也是我们需要最小化的目标

而我们在上文证明了，前向 KL 散度本质上就是 MLE（最大似然估计），而 SFT 的目标本质也只是一种最大似然估计!

$\max_\theta \prod_{i=1}^{N}q_\theta(y_i|x_i)$ -> $\max_\theta\sum_{i=1}^{N}\log q_\theta(y_i|x_i)$ -> $\mathcal L_{\mathrm{SFT}} =-\frac1N\sum_{i=1}^{N}\log q_\theta(y_i|x_i)$

因此，SFT 的优化目标本质上就是 forward KL 散度，算法需要强制拟合数据集内部出现的所有文本模式，是一种优化强度很高的拟合方式。因此，对 SFT 来说，梯度会被均匀的施加在所有的 token 梯度上，导致模型产生灾难性遗忘等严重问题。

### RL: On-Policy, Mode-Seeking, Reverse-KL

对于 RL 来说，是一种 On Policy 的算法，同时依靠策略梯度提升序列期望奖励。

我们考虑 RLHF-PPO 的优化目标:

$$
\max_\theta J(\theta) = \mathbb E_{y\sim q_\theta}[R(x,y)] -\beta D_{\mathrm{KL}}(q_\theta\|q_{\mathrm{ref}})
$$

展开 Reverse KL:

$$
D_{\mathrm{KL}}(q_\theta\|q_{\mathrm{ref}}) = \mathbb E_{y\sim q_\theta} \left[\log\frac{q_\theta(y|x)}{q_{\mathrm{ref}}(y|x)}\right]
$$
代入并合并期望:

$$
\begin{aligned} J(\theta) &=\mathbb E_{q_\theta}[R(x,y)]\\ &\quad-\beta\mathbb E_{q_\theta} \left[\log\frac{q_\theta(y|x)}{q_{\mathrm{ref}}(y|x)}\right] \end{aligned}

$$
$$
 J(\theta) = \mathbb E_{q_\theta}\left[ R(x,y) -\beta\log\frac{q_\theta(y|x)}{q_{\mathrm{ref}}(y|x)} \right]
$$
我们构造一个新的概率分布, $Z(x)$ 是一个归一化常数 $Z(x) = \sum_y q_{\mathrm{ref}}(y|x) \exp\left(\frac{R(x,y)}{\beta}\right)$, 这一项可以被理解为: 在 **Reference Model 的原始分布基础上**，根据 Reward 重新加权后得到的理想目标分布。

$$
p_{\mathrm{reward}}(y|x) = \frac{1}{Z(x)} q_{\mathrm{ref}}(y|x) \exp\left(\frac{R(x,y)}{\beta}\right)
$$
根据定义：

$$
\log p_{\mathrm{reward}}(y|x) = \log q_{\mathrm{ref}}(y|x) +\frac{R(x,y)}{\beta} -\log Z(x)
$$
$$
D_{\mathrm{KL}}(q_\theta\|p_{\mathrm{reward}}) = \mathbb E_{q_\theta} \left[\log\frac{q_\theta(y|x)}{p_{\mathrm{reward}}(y|x)}\right]
$$
$$
\begin{aligned} &D_{\mathrm{KL}}(q_\theta\|p_{\mathrm{reward}})\\ &=\mathbb E_{q_\theta}\left[ \log q_\theta(y|x) -\log q_{\mathrm{ref}}(y|x) -\frac{R(x,y)}{\beta} +\log Z(x) \right] \end{aligned}
$$
整理可以得到:
$$
\begin{aligned} D_{\mathrm{KL}}(q_\theta\|p_{\mathrm{reward}}) &= D_{\mathrm{KL}}(q_\theta\|q_{\mathrm{ref}})-\frac1\beta\mathbb E_{q_\theta}[R(x,y)] +\log Z(x) \\ &= -\frac1\beta [\mathbb E_{q_\theta}[R(x,y) - \beta D_{\mathrm{KL}}(q_\theta\|q_{\mathrm{ref}})]+ \log Z(x)\end{aligned}
$$
由于 $\beta$ 和 $Z(x)$ 对 $\theta$ 的梯度都是 0，因此我们将原始优化目标的梯度写成了一个等价的 Reverse KL 的反向梯度的等价形式。RL 会将自身的分布温和地拟合到 reward 加权后的的 ref-model 分布中。后向 KL 属于 Mode Seeking 范式，所以 RL 的更新会更加温和，是在本身自身高概率（或者具备一定概率）的输出区域进行微调。

### OPD: On-Policy, Reverse-KL

固定一个 Prefix $s$，Reverse-KL OPD 希望最小化：
$$
D_{\mathrm{KL}}(p_\theta\Vert p_T) = \sum_{v\in\mathcal V} p_\theta(v\mid s) \log\frac{p_\theta(v\mid s)}{p_T(v\mid s)}
$$
如果写成最大化目标：$J_{\mathrm{OPD}}(\theta;s) = -D_{\mathrm{KL}}(p_\theta\Vert p_T)$ 天然就是一个后向 KL 的过程!

> 因为 OPD 的信号监督细化到了 token 级别，因此，一般的 OPD 会要求模型必须使用相同的 tokenizer 和词表进行训练，保证最后输出层的语义 logits 对齐。目前学术界也在研究一些前沿的 OPD 技巧，让跨次表的 token mapping 成为可能。

OPD 相比于纯 RL 具备更完善的奖励信号，因此现在业界的通用范式往往是: SFT 出一个专家模型，通过 OPD/MOPD 等方式对学生模型进行训练。一方面 OPD 也可以缓解 SFT 的灾难性遗忘的问题。

最优策略集合存在无数可行分布，RL仅在模型当前采样邻域内寻优，自动选择**与原始模型KL距离最小**的最优分布，全程不会强行拉扯至遥远外部分布。OPD继承这套On-Policy约束，因此哪怕教师存在能力退化，学生也不会完全复刻教师的缺陷分布。