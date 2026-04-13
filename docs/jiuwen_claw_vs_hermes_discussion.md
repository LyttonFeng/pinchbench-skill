# Jiuwen-Claw / Hermes 讨论

## 讨论什么

1. 要不要在 Jiuwen-Claw 上做一版类似 Hermes / Atropos 的闭环，把 `runtime + trajectory + reward + LoRA 在线训练` 串起来。
2. RL 和 skills 的边界怎么分，能不能做一个能力判断器，让系统决定任务该走 `skills` 自进化还是 `RL`。

## 我们已经做了什么

我们这边已经把一版自己的闭环跑通了：

- 用 `OpenClaw runtime` 跑真实任务
- 用 `self-judge + terminal reward` 做 reward
- 用 `veRL + LoRA` 做在线训练
- 保证训练和推理 runtime 一致
- 在 RL8 上验证了 `Qwen3-4B + LoRA` 相比 baseline 有提升

这说明我们不是只在讨论概念，而是已经有了可验证的工程闭环。

## Hermes 做到了什么

Hermes 这条线已经把几件事串起来了：

- **Hermes** 负责 agent runtime 和用户入口
- **Atropos** 负责 environment / trajectory / reward 编排
- **Tinker** 负责 LoRA 训练、sampling、optimizer step、checkpoint
- **Tinker-Atropos** 把环境和训练接成一个闭环

它的主训练路线偏 **GRPO + LoRA**，还有更密一点的 **OPD / on-policy distillation** 辅助分支。  
这说明它不是只讲概念，而是真的在做一套训练栈。

Hermes 也明确做了两类效果验证：

- `DeepHermes-ToolCalling-Specialist-Atropos`
  - Berkeley Function Calling Benchmark 上有明显提升
- `DeepHermes-Financial-Fundamentals-Prediction-Specialist-Atropos`
  - directional prediction accuracy 也有明显提升

## Hermes 没做好的地方

Hermes 目前还没有把下面这件事做成显式系统：

- 哪些任务走 `skills`
- 哪些任务走 `RL`

从代码上看，它是：

- `skills` 自己沉淀经验
- `RL` 自己做策略优化

但还没有一个统一的能力边界分类器，自动判断该走哪条路。

另外，虽然它借鉴了 OpenClaw-RL / OPD 的思路，但从公开代码和文档里看，**还没有看到足够可靠的大规模 live-user 在线 RL 验证**。  
它更像是把平台和闭环先搭出来，再逐步沉淀成训练体系。

从实现风格上看，它的算法路线明显是在借鉴 / 适配 OpenClaw-RL 的思路，而不是从零独立发明一套完全不同的范式。

## 对我们的启发

Hermes 给我们的直接启发是：

- `runtime + trajectory + reward + training` 这条闭环是能做出来的
- `skills` 和 `RL` 最好不要混着用，应该有明确分工

对我们来说，下一步可以讨论两件事：

1. 在 Jiuwen-Claw 上也做一版类似的训练闭环，验证自己的 runtime 能不能形成可训练、可回流、可持续提升的体系。
2. 设计一个“能力边界判断器”，让系统判断任务该走 `skills` 还是 `RL`，把这件事做得比 Hermes 更清楚。

## 一句话总结

Hermes 已经证明了“agent runtime + trajectory + reward + training” 这种闭环可以落地，但它还没有把 `skills` 和 `RL` 的边界做成显式系统。我们可以借鉴它的闭环形状，但在能力分流和 live-user 训练验证上继续往前做。