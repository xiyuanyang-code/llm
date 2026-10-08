## DeepSeek V4.1 Tech Report

核心的后训练 Pipeline：SFT + RL + OPD

> nstead, our efforts are concentrated almost entirely on what the model is trained on rather than how it is optimized.

关键的 scale 变量是合成更多高质量的可执行、可验证的环境 (instance)，以及在训练量和训俩资源上的横向 scale。
### Environment

> Based on the interfaces observed in the returned data, we construct a large set of mocked tools that reproduce the interfaces and behaviors of real-world tools and system.

Agent 环境构建: 基于真实的用户数据和用户交互轨迹，构建出 Mock API，类似于 Trajectory2Instance 的过程

 >In parallel, we collect negative feedback and model failure cases submitted by internal employees at scale.
 
 失败的轨迹，也被用来执行作为训练环境

> Then, multiple distinct agents attempt the task, and an independent quality-inspection agent reviews the environment together with the solving agents’ trajectories, checking for environment issues, factual errors, mismatches between evaluation points and task descriptions, and hackability risks.

一种针对 training instance 的 RSI 说明是有必要的
### RL

> We scale RL runs in two dimensions, i.e., training compute and the number of scaffolds.

现在流行的 agentic rl 是 black-boxed agentic rl! 两个重要的 scaling rl 的维度: (1) training compute (2) the number of scaffolds (agent harness)

- 对于第一个维度，就是体现为 rl steps 和高质量 instance 的扩展
- 对于第二个维度，就是体现为 rl harness 的重要性
![[deepseek-v41-rl.png]]

> To extend effective RL compute beyond a single training run, we use model merging to reinitialize successive RL runs. 

在 RL 的起点处，做 Model Merge（==针对不同 scaffold 单训 然后做 model-merge==），比单独训练更加高效

#### DSec: RL Environment

DeepSeek 自建的 sandbox platform

思考：一个适合大规模 rollout 的 sandbox platform 需要支持什么样的 feature:

- 大规模的资源吞吐和并发稳定性
- 支持沙箱文件通信 (e.g. file snapshot, 核心日志文件导出)
- ==安全性==:权限管理 (agent && root), 防止逃逸和 hacking

Dsec 实现的机制: sharding(分片) and scheduling with relaxed consistency(弱化的一致性约束). 本质上来说就是一直去中心化的调度系统。

> **大规模横向扩展计算。**  
> DSec 通过两种互补机制实现扩展：分片（sharding）以及基于弱一致性的调度（scheduling with relaxed consistency）。为了支持大量机器，我们将计算节点划分为多个分片，也就是所谓的 scale units（扩展单元）。这种分片方式还可以通过隔离不同实验的工作负载来减小故障的波及范围（blast radius），从而避免某个内存占用极高的任务耗尽与其他无关任务共享的资源。DSec 并没有采用 Kubernetes 这类现成的编排系统，而是使用了一个自定义的 placement engine（任务放置/部署引擎） 来调度海量 sandbox。其核心思想是：牺牲强全局一致性，以换取更好的可扩展性。这一设计基于一个关键观察：对于 agentic sandbox 的任务放置而言，只要每个计算节点都能够独立执行本地安全约束，那么系统实际上只需要满足最终一致性（eventual consistency），而不需要所有调度器在任何时刻都保持完全一致。具体来说，placement engine 会部署成多个彼此独立的副本，而这些副本之间不会进行同步协调。每个副本都会根据最近采集到的资源使用情况，预测哪些节点还有可用资源，并据此做出“足够好”的任务放置决策。为了弥补缺乏强一致性带来的问题，每个计算节点本身还负责对最终的放置决策进行校验。节点会执行严格的准入约束（hard admission constraint）：如果接受一个新的任务会使本地资源使用超过预设的警戒阈值，那么该节点就会拒绝这个新的任务。整体而言，这种设计避免了系统依赖中心化协调，因此 DSec 可以扩展到数百万个容器，而不会因为中央协调机制而形成性能瓶颈。

NUMA: Non-Uniform Memory Access，比如 NUMA1 的 CPU 访问 NUMA1 的内存快于 NUMA2 的内存

在 DSec 中，每一个 Worker VM 都被绑定到一个 NUMA domain (硬件绑定)

```
Physical Machine
      │
      ├── Worker VM 1
      │      ├── container
      │      ├── container
      │      └── container
      │
      ├── Worker VM 2
      │      ├── container
      │      ├── container
      │      └── container
      │
      └── Worker VM 3
             ├── container
             └── container
```

基于硬件绑定的分片系统可以降低 linux kernel 的锁互斥，提升并行数（更加激进）
同样，在内存超限的激进策略下，受影响的沙箱被限制到了一个 VM 下，整体的峰谷现象更弱，调度更稳定

> 定位类似物理机, 如果走阿里云的沙箱系统调度会更稳定

#### Controllable Reasoning Effort in RL

- 不同的 reasoning effort 通过一个数字传递到 sys-prompt 中
- 在训练过程中，对多个 reasoning effort 实现 responses 采样，但是在算 advantage 的时候，只能 group  $(x, b)$ 内部的样本进行计算，不可以横跨样本
- 对于一个轨迹 $r_{b,j}^{\text{len}}$，进行长度惩罚
	- $r_{b,j}^{\text{len}} = - \min \{ C_\max, k(b) \frac{l_{b,j}}{L_{norm}} \}$
		- Cmax 是最大惩罚的兜底
		- $k(b)$ 是一个随 $b$ 进行指数衰减的系数
			- b 越大，k(b) 越小，对应的惩罚强度就越小
			- 更关键的其实是一个 long2short 控制思维链的手段，重点惩罚小 reasoning-effort 下，控制其输出思维链的长度
		- $\frac{l_{b,j}}{L_{norm}}$ 是一个归一化后的 reasoning 长度

### MOPD

使用 domain-wise teacher 做 OPD
training infra 支持 OPD Settings 实时热更新
> 这一部分在 DeepSeek-V4 的 Tech Blog 里面有所更新

### PostTraining Infra

Agentic 任务的最大痛点就是**长尾问题**，会严重影响 rollout 阶段的效率
核心优化: **异步 rollout**

> To address this, we extend our post-training infrastructure to allow asynchronous generation of samples.

- sample-level dispatch: 当新完成的样本数量达到“下一个 prompt 所对应的 GRPO group size”时，就立即调度这个 prompt，而不关心这些已完成样本原本来自哪些 GRPO group。这样可以在整个训练过程中更稳定地维持 rollout 并发度。
	- 粒度优于 batch-level dispatch & prompt-level dispatch

- 在异步 RL 中，trainer 会不断收集 rollout 成功的样本并更新计算梯度做 weight update，因此部分样本的 rollout 轨迹可能涉及多个 moe expert 的激活。具体激活的专家数量会被存储 **(MoE RL 很容易被整成 off-policy rl)**

- weight-update 的过程往往会 interrupt 现在的 rollout 阶段，因此做好 token-level 的阻隔和状态维护。(e.g. KV-Cache, GC 时间)


####  Mitigating Length Bias and Off-Policy Effects

异步 RL 会带来的两个问题:
- RL 过程中，简单的 prompt 被优先释放，主导前期的训练过程
	- 调度器控制不同 training sample 出现的最大比例(避免全部 fill 成简单的题目)
	- 丢弃早期过短回答(离群点 discard)
- 异步更新导致的 off-policy 问题 (数据采集模型和更新的模型权重不同)
	- maximum off-policy ratio 控制 policy diff 不能太大 (在 GRPO 算法中 somehow 也对这一点有约束)
	- 做 loss-mask



