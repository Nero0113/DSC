# UAH 三分类轻量时序模型

这是一个不复制原始数据的 PyTorch 训练包。输入为 20 秒车载传感器窗，目标为
`NORMAL / AGGRESSIVE / DROWSY`。现有结果是严格的跨驾驶员评估，不使用随机窗口切分。

## 手机端快速导航

- `model.py`：轻量 1D 时序模型定义
- `train.py`：训练、断点续训、验证、独立测试和逐窗结果导出
- `rad_tcn.py`：固定的 road-aware / domain-adversarial 多尺度 TCN
- `train_rad_tcn.py`：RAD-TCN 训练、EMA、辅助任务和完整断点续训
- `context_tcn.py` / `train_context_tcn.py`：冻结窗口编码器的同 trip 因果上下文验证
- `final_fit_inception.py` / `final_fit_context.py`：选型完成后的 D1–D5 固定轮数 final-fit
- `configs/`：可复现命令对应的 JSON 配置
- `RESULTS.md`：已完成实验、混淆矩阵和限制
- `DATA_AUDIT.md`：数据结构、分组和泄漏检查
- `runs/<run>/metrics.json`：机器可读的完整指标
- `runs/<run>/best_model.pt`：按 D5 macro-F1 选择的模型
- `runs/<run>/last_checkpoint.pt`：含优化器/调度器/RNG 的可恢复训练状态

## 数据契约

数据目录通过 `--data` 指定，必须包含 `train.npz`、`val.npz` 和 `test.npz`。
模型输入为 `float32 [batch, 8, 200]`，8 个通道的固定顺序见 `DATA_AUDIT.md`。
本包不会读取 `driver_id`、`trip_id`、`road` 或 `window_start_s` 作为模型特征。

## 运行

当前验证最好的框架是两级因果时序模型：

1. `InceptionTemporalNet(width=48, dropout=0.15)` 编码每个 20 秒窗口；
2. `HierarchicalCausalTCN(width=64, dropout=0.30, context=25)` 只聚合同一会话的
   过去 25 个窗口，约覆盖 140 秒；
3. 窗口编码可缓存，端侧每 5 秒只新算一个 embedding 和一次 context head。

固定选型命令（D1–D4 训练，D5 验证，不读 D6）：

```bash
.venv/bin/python luna_temporal_classifier/train_context_tcn.py \
  --data /absolute/path/to/uah_3class \
  --encoder-checkpoint luna_temporal_classifier/runs/inception_w48_seed2026/best_model.pt \
  --out luna_temporal_classifier/runs/context25
```

固定超参：`context=25`、`batch_size=256`、`AdamW(lr=1e-3,
weight_decay=1e-3)`、`CosineAnnealingLR`、`dropout=0.30`、`label_smoothing=0.03`、
`patience=10`、`seed=2026`。该配置在 D5 选中 epoch 20。

以下是单窗口基线命令，便于复现对照：

```bash
python3 -m venv .venv
.venv/bin/pip install -r luna_temporal_classifier/requirements.txt
.venv/bin/python luna_temporal_classifier/train.py \
  --data /absolute/path/to/uah_3class \
  --out luna_temporal_classifier/runs/example \
  --model strided --width 32 --batch-size 512 \
  --epochs 100 --patience 20 --lr 0.003
```

断点续训：

```bash
.venv/bin/python luna_temporal_classifier/train.py \
  --data /absolute/path/to/uah_3class \
  --out luna_temporal_classifier/runs/example \
  --model strided --width 32 --batch-size 512 \
  --epochs 100 --patience 20 --lr 0.003 \
  --resume luna_temporal_classifier/runs/example/last_checkpoint.pt
```

当前严格跨驾驶员最好 D6 结果为 accuracy 75.03%、macro-F1 77.38%，
尚未达到 90% 指标，详见 `RESULTS.md`。不能通过随机拆分高度重叠的窗口来
制造高分。
