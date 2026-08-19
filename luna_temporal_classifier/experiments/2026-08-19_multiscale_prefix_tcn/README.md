# MultiScalePrefixTCN — 2026-08-19

本目录独立保存 2026-08-19 的模型挑战，不修改项目根目录中的历史模型和结果。

## 内容

- `multiscale_prefix_tcn.py`：冻结 Inception1D 编码器之后的多时间尺度因果聚合模型。
- `train_multiscale_prefix_tcn.py`：严格 `D1–D4 train / D5 validation` 训练与完整断点续训。
- `evaluate_multiscale_prefix_checkpoint.py`：只在 D5 上复核 checkpoint，不读取 D6。
- `config.json`：本轮固定架构和训练设置。
- `RESULTS.md`：两个 seed 的结果、混淆矩阵、每类和每 trip 指标。
- `results/`：机器可读的 metrics 和 history。
- `artifacts/`：复现所需的 D1–D4 窗口编码器、最佳 checkpoint 和 seed 2027 完整续训点，已移除本机绝对路径。

## 环境

```bash
python3 -m venv .venv
.venv/bin/pip install -r luna_temporal_classifier/requirements.txt
```

## 复现训练

```bash
.venv/bin/python \
  luna_temporal_classifier/experiments/2026-08-19_multiscale_prefix_tcn/train_multiscale_prefix_tcn.py \
  --data /absolute/path/to/uah_3class \
  --encoder-checkpoint \
  luna_temporal_classifier/experiments/2026-08-19_multiscale_prefix_tcn/artifacts/inception_w48_d1_d4_seed2026.pt \
  --out luna_temporal_classifier/runs/multiscale_prefix_tcn_reproduction \
  --seed 2026
```

训练脚本只读取 `train.npz` 和 `val.npz`，不包含 D6 评估路径。最佳结果为
D5 accuracy 74.78%、macro-F1 72.15%；相对两阶段 TCN 基线提升 3.38/4.05 个百分点。

## 仅复核已上传的最佳 checkpoint

```bash
.venv/bin/python \
  luna_temporal_classifier/experiments/2026-08-19_multiscale_prefix_tcn/evaluate_multiscale_prefix_checkpoint.py \
  --data /absolute/path/to/uah_3class \
  --encoder-checkpoint \
  luna_temporal_classifier/experiments/2026-08-19_multiscale_prefix_tcn/artifacts/inception_w48_d1_d4_seed2026.pt \
  --checkpoint \
  luna_temporal_classifier/experiments/2026-08-19_multiscale_prefix_tcn/artifacts/multiscale_prefix_tcn_seed2026_best.pt \
  --out luna_temporal_classifier/runs/multiscale_prefix_tcn_checkpoint_check
```

原始数据、逐窗预测 CSV 和未选中的中间 checkpoint 不包含在仓库中。
