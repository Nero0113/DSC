# 已完成结果

所有结果使用 D1–D4 训练、D5 选择 best epoch、D6 独立终测。主指标为 macro-F1。

| 模型 | 参数量 | best epoch | D5 Acc | D5 Macro-F1 | D6 Acc | D6 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| StridedTemporalCNN | 47,427 | 26 | 64.06% | 62.32% | 61.92% | 60.80% |
| InceptionTemporalNet | 41,043 | 3 | 63.67% | 63.11% | 57.97% | 58.82% |
| RAD-TCN (EMA) | 54,252 | 42 | 63.57% | 61.87% | 未运行 | 未运行 |
| Inception final-fit (D1–D5) | 41,043 | 固定 3 | 架构选择已完成 | 架构选择已完成 | 61.81% | 63.46% |
| Frozen Inception + context25 v2 | 66,435（上下文头） | 20 | 71.40% | 68.10% | 未运行 | 未运行 |
| Context25 final-fit (D1–D5) | 66,435（上下文头） | 固定 20 | 上下文选择已完成 | 上下文选择已完成 | 75.03% | 77.38% |
| Context25 + causal EMA(alpha=0.05) | 同上 | D5 固定 | 76.62% | 72.39% | 72.88% | 74.98% |

## StridedTemporalCNN

D5 混淆矩阵（行是真值，列是预测）：

```text
[[364,  29,  54],
 [ 51, 167,  19],
 [165,  54, 132]]
```

D5 每类 F1：NORMAL 70.89%、AGGRESSIVE 68.58%、DROWSY 47.48%。

## InceptionTemporalNet

D5 混淆矩阵：

```text
[[300,  28, 119],
 [ 58, 170,   9],
 [ 62, 100, 189]]
```

D5 每类 F1：NORMAL 69.20%、AGGRESSIVE 63.55%、DROWSY 56.59%。

## RAD-TCN（固定验证方案）

此方案使用动态归一化/一阶差分双视图、多尺度 residual Conv1d、road 专家辅助头、
driver gradient reversal 和 EMA。只按要求运行到 D5，未查看 D6。

D5 混淆矩阵：

```text
[[335,  73,  39],
 [ 40, 190,   7],
 [141,  77, 133]]
```

D5 每类 F1：NORMAL 69.57%、AGGRESSIVE 65.86%、DROWSY 50.19%；road accuracy
82.80%。训练在 epoch 62 early stop，最佳 checkpoint 为 epoch 42。中途从 epoch 20
完整恢复，并按固定指令将后续 EMA decay 从 0.995 调整为 0.99。

## Inception final-fit（最终一次测试）

D5 已用于选择 InceptionTemporalNet 和固定训练轮数 3。随后合并 D1–D5 训练，训练
结束后只评估 D6 一次，未根据 D6 调整任何设置。

```text
[[179,  25, 150],
 [ 20, 121,  38],
 [ 96,   9, 247]]
```

D6 accuracy 61.81%、macro-F1 63.46%；每类 F1 为 NORMAL 55.16%、
AGGRESSIVE 72.46%、DROWSY 62.77%。

## 冻结 Inception + context25（仅验证）

冻结窗口编码器，在同一 trip 内使用过去 25 个窗口（约 140 秒覆盖范围）的因果 TCN。
修正 checkpoint 选择为“任何真实 F1 提升均保存”后，固定方案在 epoch 30 early stop，
选择 epoch 20：D5 accuracy 71.40%、macro-F1 68.10%；每类 F1 为 NORMAL 82.75%、
AGGRESSIVE 73.08%、DROWSY 48.47%。本实验明确未评估 D6。

## Context25 final-fit（最终一次测试）

D5 已选择 context=25 和固定轮数 20。随后使用 D1–D5 的冻结 final Inception
embedding 训练上下文头，完整训练结束后只评估 D6 一次，未根据测试结果调参。

```text
[[232,   0, 122],
 [ 37, 142,   0],
 [ 62,   0, 290]]
```

D6 accuracy 75.03%、macro-F1 77.38%；每类 F1 为 NORMAL 67.74%、
AGGRESSIVE 88.47%、DROWSY 75.92%。这是当前严格协议下的最佳独立测试结果，但仍未
达到 90% / 90% 目标。

额外的会话内因果 EMA 只用 D5 选择 `alpha=0.05`，但 D6 降至
accuracy 72.88%、macro-F1 74.98%，因此已淘汰，没有在 D6 上试其他 alpha。

## 结论

当前严格跨驾驶员结果没有达到 Acc >90%、macro-F1 >90%。多尺度模型在 D5 的
macro-F1 略高，但 D6 并未同步改善；这表明主要限制是未见驾驶员的分布漂移和只有
6 位驾驶员的数据规模，而不是简单扩大网络。不得用随机窗口切分替代跨驾驶员评估。
