# 2026-08-19 MultiScalePrefixTCN 固定验证报告

## 结论

在严格的 `D1–D4 train / D5 validation` 协议下，最佳结果来自 seed 2026：

- D5 Accuracy：**74.78%**
- D5 Macro-F1：**72.15%**
- 相对 Frozen Inception + context25 v2 基线：Accuracy **+3.38** 个百分点，Macro-F1 **+4.05** 个百分点
- 未达到预设的 Accuracy 80% / Macro-F1 80%
- 本轮完全未读取或评估 D6，`test_evaluated=false`

受硬时间边界限制，seed 2026 完整 early stop，seed 2027 保存到 epoch 29 后在安全 checkpoint
处暂停；没有启动 seed 2028。预注册条件要求三个 seed 全部完成后才做概率平均，因此本轮没有报告
两 seed 临时 ensemble，也没有据验证结果修改架构或超参。

## 固定协议

- 冻结 `InceptionTemporalNet(width=48, dropout=0.15)`，从既有 best checkpoint 提取每窗 128 维特征。
- 特征标准化统计量仅由 D1–D4 计算。
- local context 为同 trip 过去 25 个窗口，缺失历史只复制首窗；全局 prefix 为三尺度 EMA、累计均值和累计标准差。
- local TCN 为四个因果 depthwise-separable residual block，dilation `1/2/4/8`。
- 训练使用固定 trip×class 平衡 sampler、5% 一致 feature dropout 和三个固定 CE loss。
- 模型可训练参数：144,233，低于 150,000 参数预算。

## Seed 汇总

| Seed | 状态 | Best epoch | D5 Acc | D5 Macro-F1 | ΔAcc vs baseline | ΔF1 vs baseline | 用时 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2026 | epoch 15 patience 完成 | 3 | 74.78% | 72.15% | +3.38 pp | +4.05 pp | 992.8s（16分33秒） |
| 2027 | 时间边界暂停于 epoch 29 | 24 | 70.82% | 64.46% | -0.58 pp | -3.64 pp | 约1743s（29分03秒） |
| 2028 | 未启动 | — | — | — | — | — | — |

## Seed 2026 详细结果

混淆矩阵（行是真值，列是预测）：

```text
[[403,   3,  41],
 [  0, 222,  15],
 [ 84, 118, 149]]
```

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 82.75% | 90.16% | 86.30% | 447 |
| AGGRESSIVE | 64.72% | 93.67% | 76.55% | 237 |
| DROWSY | 72.68% | 42.45% | 53.60% | 351 |

| D5 trip | Acc |
|---|---:|
| NORMAL / MOTORWAY | 90.00% |
| AGGRESSIVE / MOTORWAY | 90.32% |
| DROWSY / MOTORWAY | 42.33% |
| NORMAL1 / SECONDARY | 97.01% |
| NORMAL2 / SECONDARY | 83.46% |
| AGGRESSIVE / SECONDARY | 100.00% |
| DROWSY / SECONDARY | 42.65% |

## Seed 2027 详细结果

混淆矩阵：

```text
[[437,   0,  10],
 [  0, 222,  15],
 [154, 123,  74]]
```

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 73.94% | 97.76% | 84.20% | 447 |
| AGGRESSIVE | 64.35% | 93.67% | 76.29% | 237 |
| DROWSY | 74.75% | 21.08% | 32.89% | 351 |

| D5 trip | Acc |
|---|---:|
| NORMAL / MOTORWAY | 95.56% |
| AGGRESSIVE / MOTORWAY | 90.32% |
| DROWSY / MOTORWAY | 0.00% |
| NORMAL1 / SECONDARY | 98.51% |
| NORMAL2 / SECONDARY | 100.00% |
| AGGRESSIVE / SECONDARY | 100.00% |
| DROWSY / SECONDARY | 54.41% |

## Checkpoint 与恢复

两个 seed 的 `best_model.pt` 和 `last_checkpoint.pt` 均核验包含：模型、optimizer、scheduler、
当前 epoch、best F1/accuracy/epoch、stale、history、Python/NumPy/Torch RNG、完整 config、
feature mean/std、encoder checkpoint、feature names 和 architecture contract。

seed 2027 可从 epoch 30 恢复：

```bash
PREFIX_CACHE=/private/tmp/prefix-luna-pycache \
PYTHONPYCACHEPREFIX=$PREFIX_CACHE \
.venv/bin/python \
  luna_temporal_classifier/experiments/2026-08-19_multiscale_prefix_tcn/train_multiscale_prefix_tcn.py \
  --data /absolute/path/to/uah_3class \
  --encoder-checkpoint \
  luna_temporal_classifier/experiments/2026-08-19_multiscale_prefix_tcn/artifacts/inception_w48_d1_d4_seed2026.pt \
  --out luna_temporal_classifier/runs/multiscale_prefix_tcn_seed2027_resume \
  --seed 2027 \
  --resume \
  luna_temporal_classifier/experiments/2026-08-19_multiscale_prefix_tcn/artifacts/multiscale_prefix_tcn_seed2027_last.pt
```

机器可读的指标和训练历史位于本实验目录的 `results/`。逐窗预测 CSV 含原始 trip 标识，因此未上传。
