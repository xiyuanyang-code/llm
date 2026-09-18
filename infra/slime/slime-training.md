本文，我们将会重点梳理 Slime 中通用的训练组建和框架，包括他如何勾连 Megatron 在 GPU 上进行训练。重心将会放在 training 的一般抽象和构建中，具体的训练算法和一些额外的 trick 将会在后续具体章节给出。

## 入口文件

入口是 `train.py`,但它**不会直接导入 `actor.py`**。`MegatronTrainRayActor` 是经 `RayTrainGroup` 这条链、在**创建 Ray actor 的那一刻延迟导入**的。

完整的导入链:

```text
train.py(仓库根目录)
└── from slime.ray.placement_group import create_training_models    ① 入口只导入 placement_group
    │
    └── create_training_models(args, pgs, rollout_manager)         ② placement_group.py
        │
        └── allocate_train_group(...)                              ③ placement_group.py
            │
            └── RayTrainGroup(args=..., ...)                       ④ actor_group.py(构造训练组)
                │
                └── _allocate_gpus_for_actor(...)                  ⑤ actor_group.py(创建 actor 时)
                    │
                    └── from slime.backends.megatron_utils.actor import MegatronTrainRayActor  ⑥ 延迟导入!
                        │
                        └── ray.remote(...)(actor_impl)            ⑦ 包装成 Ray Actor 类
                            └── TrainRayActor.options(...).remote(rank)  ⑧ 每张 GPU spawn 一个 worker
```

继承关系:

```
MegatronTrainRayActor        slime/backends/megatron_utils/actor.py
    └── TrainRayActor        slime/ray/train_actor.py(通用训练 actor 逻辑)
        └── RayActor         slime/ray/ray_actor.py(最底层的 Ray actor 基类)
```

backend 层只实现 Megatron 特有部分(`init`、`train_actor`、`update_weights`、`save_model` 等),通用的 Ray 生命周期逻辑在基类里。(这部分会在 Ray 这个专题中进行讲述)

## 核心训练封装

我们来看最核心的 `MegatronTrainRayActor` 类。

核心的训练流程，可以抽象为两个部分:

- Rollout/Generations:
    - 模型不会进行梯度计算和权重更新，而是进行数据 rollout，拿到 log_probs 等 snapshot，存储起来作为训练数据
    - 前向计算的核心就是**谁 Rollout 什么数据**:
        - "什么数据为上游数据输入的准备"
        - 关键在于**模型权重的切换**: `ref_model`, `teacher_model`, `actor_model`
- Optimizations:
    - Training Forward: 开梯度前向计算，得到 log_probs
    - 计算 Loss (根据现有的数据，例如 log_probs, old_log_probs, advantages) 进行计算并且反向传比
    - 更新权重

Training 核心的函数在 `train` 中被定义 (`MegatronTrainRayActor`): 

```python
def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
    if self.args.debug_rollout_only:
        return None

    if self.args.offload_train:
        # 激活唤醒
        self.wake_up()

    with timer("data_preprocess"):
        # Rollout/Generations 
        # 主要执行数据预处理
        rollout_data = self._get_rollout_data(rollout_data_ref)

    # 核心的训练 & weight update
    if self.role == "critic":
        result = self.train_critic(rollout_id, rollout_data)
    else:
        self.train_actor(rollout_id, rollout_data, external_data=external_data)
        result = None

    # 清理数据，完成一轮训练
    if self.args.offload_train:
        del rollout_data
        self.sleep()

    return result
```

在核心训练中，会分成 Actor/Critic 两个不同的角色，这两个角色的优化目标不同:

- Actor 输出 `[Batch, Length, Vocab_Size]`，即当前 token 下模型输出的 logits (正常语言模型的输出)
- Critic 输出 `[Batch, Length, 1]`, 给出一个 token 级别的 reward 输出，即在当前状态下的 token 可以拿到的预期回报。

我们先来探寻 actor 训练的核心流程。

### Actor Training

核心来看 `train_actor` 这个函数，核心阶段是：

- 前向计算，拿到不同 model 的 log_probs 等参数，准备后续 loss 的计算
- 计算 advantages & returns 等 (在训练前完成)
- 核心 Megatron 训练:
    - Forward pass in torch (带梯度图)
    - Backward Weight Update
- 更新 & 备份模型参数
    
```python
def train_actor(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None) -> None:
    # Create data iterator for log_probs and train.
    # * 从 data_iterator 中读取 rollout 数据（输入是训练数据）
    data_iterator = get_data_iterator(rollout_data)
    num_microbatches = rollout_data["num_microbatches"]
    global_batch_sizes = rollout_data["global_batch_sizes"]

    # * Rollout Routing Replay 技巧
    # 论文 https://arxiv.org/pdf/2510.11370
    # 主要核心解决的问题是 MoE 语言模型在训练和 rollout 中路由器不一致的问题，使用 replay 技巧可以提升训练的稳定性
    if self.args.use_rollout_routing_replay:
        self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

    with inverse_timer("train_wait"), timer("train"):
        # 前向计算
        # 在前向计算中，核心就两件事情:
        # - 使用什么模型权重
        # - 使用什么计算，计算出什么
        if self.args.compute_advantages_and_returns:
            if "ref" in self.weights_backuper.backup_tags:
                if self.args.use_routing_replay:
                    os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                # 如果 ref 模型存在 (常用于 KL 散度的计算)
                # 更新到 ref 模型的权重 && 更新 data (前向 rollout data)
                self._switch_model("ref")
                rollout_data.update(
                    self.compute_log_prob(
                        data_iterator,
                        num_microbatches,
                        store_prefix="ref_",
                    )
                )

            # 如果 teacher 模型存在
            # 更新到 teacher 模型的权重 && 更新 data
            # * 这一个常用作 On Policy Distillation 算法中，需要存储 teacher 前向过程中的 log_probs
            # Forward teacher model to get teacher_log_probs for Megatron-based OPD
            if "teacher" in self.weights_backuper.backup_tags:
                if self.args.use_routing_replay:
                    os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                self._switch_model("teacher")
                rollout_data.update(
                    self.compute_log_prob(
                        data_iterator,
                        num_microbatches,
                        store_prefix="teacher_",
                    )
                )

            # actor 模型的 rollout 步骤
            # * 切换成 actor (需要被更新模型的参数)
            # * Off Policy Importance Sampling
            # 在 Off Policy 的策略中，actor 模型的策略和 rollout 模型的策略是两个不同的策略，因此需要加上重要性采样的步骤
            self._switch_model("old_actor" if self.args.keep_old_actor else "actor")

            # can_reuse_log_probs_in_loss 是一个核心的 tag，其代表的是模型是否能够省略一次无梯度的前向过程
            # * 后续会详细的解读这一个判断逻辑，为什么可以不跑 log_probs
            can_reuse_log_probs_in_loss = (
                len(num_microbatches) == 1
                and self.args.loss_type == "policy_loss"
                and self.args.kl_coef == 0
                and not self.args.use_rollout_logprobs
                and not self.args.get_mismatch_metrics
                and not self.args.use_critic
                and not self.args.keep_old_actor
                and not self.args.use_opd
                and not self.args.use_routing_replay
                and self.args.advantage_estimator != "gspo"
            )
            if (
                not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics
            ) and not can_reuse_log_probs_in_loss:
                if self.args.use_routing_replay:
                    if self.args.use_rollout_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                    else:
                        os.environ["ROUTING_REPLAY_STAGE"] = "record"
                # * 如果 can_reuse_log_probs_in_loss 被设置为 True，这一行将会被跳过
                # 注意，这里计算的是 actor 模型运行的无梯度的 prob
                rollout_data.update(
                    self.compute_log_prob(
                        data_iterator,
                        num_microbatches,
                        store_prefix="",
                    )
                )
                if self.args.use_rollout_routing_replay:
                    RoutingReplay.clear_all_forward()

            # 汇入 critic 数据
            if self.args.use_critic:
                if external_data is not None and mpu.is_pipeline_last_stage():
                    values = external_data.get("values")
                    if values is not None:
                        from slime.backends.megatron_utils.data import tensors_to_gpu

                        rollout_data["values"] = tensors_to_gpu(values)
            if self._active_model_tag != "actor":
                self._switch_model("actor")

            # Calculate adv and returns. Need to performed before training (instead of on the fly),
            # because we may need normalize the whole rollout.
            # * 核心函数: 在训练之前计算 return 和 advantages
            # 因为 GRPO 需要做组归一化，因此这个过程不可以在训练中执行
            compute_advantages_and_returns(self.args, rollout_data)

        # 数据后处理，主要是日志处理等，省略
        if self.rollout_data_postprocess is not None:
            self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

        log_rollout_data(
            rollout_id,
            self.args,
            rollout_data,
        )

        # Train
        # * 核心的训练 Weight Update 的过程
        # 核心的 train 过程出现在 `model.py` 中
        # 核心的训练过程被 megatron 封装，主要分成 forward_pass (带梯度的动态计算图) & backward weight update
        if self.args.use_routing_replay:
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
        with timer("actor_train"):
            train(
                rollout_id,
                self.model,
                self.optimizer,
                self.opt_param_scheduler,
                data_iterator,
                num_microbatches,
                global_batch_sizes,
            )

        self.prof.step(rollout_id=rollout_id)

    # 日志产物落盘 & 其他步骤
    train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

    if self.args.use_routing_replay:
        RoutingReplay.clear_all()

    # update the cpu actor weight to the latest model
    self.weights_backuper.backup("actor")

    # Update ref model if needed
    if (
        self.args.ref_update_interval is not None
        and (rollout_id + 1) % self.args.ref_update_interval == 0
        and "ref" in self.weights_backuper.backup_tags
    ):
        # 在某些算法中，ref model 的权重也需要被更新，在这一步执行
        with timer("ref_model_update"):
            if is_megatron_main_rank():
                logger.info(f"Updating ref model at rollout_id {rollout_id}")
            self.weights_backuper.backup("ref")

    log_perf_data(rollout_id, self.args, extra_metrics=self.weight_updater.pop_metrics())
```