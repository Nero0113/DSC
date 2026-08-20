# InceptionTemporalNet + context TCN 部署到 i.MX93

这套代码把已训练的 PyTorch 两级模型转成适合 i.MX93 Ethos-U65-256 的
全整型 INT8 TFLite。部署不是单独跑 context head，而是完整保留了：

1. `InceptionTemporalNet` 的 96 维 average/max-pool embedding；
2. 原始窗口的 32 维 mean/std/delta 统计；
3. `feature_mean` / `feature_std` 归一化；
4. 同一 trip 内 25 个窗口的因果缓存；
5. D5 固定的 `alpha=0.05` 会话平滑。

PyTorch `Conv1d` 在转换时被等价映射为 NHWC `Conv2D(1×k)`，使 Vela
更容易将卷积、池化、BN/ReLU 和全连接层下沉到 NPU。统计特征和环形
缓存在 Cortex-A55 上处理，每 5 秒只需要各调用一次 encoder 和 context NPU 模型。

## 1. 主机上导出和 INT8 PTQ

建议在 x86-64 Linux 主机上建立独立环境：

```bash
python3 -m venv .venv-imx93
.venv-imx93/bin/pip install -r luna_temporal_classifier/deploy_imx93/requirements-convert.txt
```

使用 `train.npz` 做代表数据校准，使用 `val.npz` 做转换一致性检查：

```bash
.venv-imx93/bin/python luna_temporal_classifier/deploy_imx93/export_int8_tflite.py \
  --data /absolute/path/to/uah_3class \
  --encoder-checkpoint luna_temporal_classifier/runs/inception_final_d1_d5_seed2026/final_model.pt \
  --context-checkpoint luna_temporal_classifier/runs/hierarchical_context25_final_d1_d5_seed2026/final_context_model.pt \
  --out luna_temporal_classifier/deploy_imx93/build
```

导出脚本会先检查 PyTorch Conv1d 和 Keras Conv2D 的 FP32 逐层映射误差，再生成：

- `inception_encoder_int8.tflite`
- `context_tcn_int8.tflite`
- `preprocessing.npz`
- `manifest.json`，其中包含 INT8 误差和 top-1 agreement

量化校准不要使用 D6/test，以免破坏现有的独立测试契约。

## 2. 用 NXP Vela 编译

Vela 版本需与板子上的 NXP BSP / Ethos-U driver 匹配。NXP LF6.18.20_2.0.0
的参考工具链是 Vela 4.2.0。可以按对应 BSP tag 安装 NXP fork：

```bash
git clone https://github.com/nxp-imx/ethos-u-vela.git
cd ethos-u-vela
git checkout lf-6.18.20_2.0.0
python3 -m pip install .
vela --version
```

安装匹配版本后执行：

```bash
chmod +x luna_temporal_classifier/deploy_imx93/compile_vela.sh
luna_temporal_classifier/deploy_imx93/compile_vela.sh \
  luna_temporal_classifier/deploy_imx93/build \
  luna_temporal_classifier/deploy_imx93/build/vela
```

检查 Vela 日志中的 `CPU operators` 和 `NPU operators`。少量 shape/slice 算子留在
Cortex-A 可以接受；如果卷积或全连接大量回退 CPU，通常是 Vela 与 BSP 版本不匹配
或输入模型没有做全整型量化。

## 3. 复制到 i.MX93

把下列文件放在板端同一目录，例如 `/opt/driver_model`：

```text
manifest.json
preprocessing.npz
inception_encoder_int8.tflite
context_tcn_int8.tflite
inception_encoder_int8_vela.tflite
context_tcn_int8_vela.tflite
runtime_imx93.py
```

NXP Yocto BSP 需要包含 TensorFlow Lite Python runtime、`/dev/ethosu0` 和
`/usr/lib/libethosu_delegate.so`。可先检查：

```bash
ls -l /dev/ethosu0 /usr/lib/libethosu_delegate.so
```

## 4. 板端推理

CPU 先做功能验证：

```bash
python3 runtime_imx93.py \
  --model-dir /opt/driver_model \
  --input /opt/test_windows.npz \
  --delegate cpu \
  --encoder-model inception_encoder_int8.tflite \
  --context-model context_tcn_int8.tflite
```

Ethos-U65 推理：

```bash
python3 runtime_imx93.py \
  --model-dir /opt/driver_model \
  --input /opt/test_windows.npz
```

`test_windows.npz` 至少包含 `X: float32 [N,8,200]`。如果同时包含 `trip_id`，脚本会在
trip 变化时清空上下文和平滑状态。实际车载程序直接持有 `StreamingClassifier`，
每产生一个新的 20 秒窗口就调用一次 `push(window)`，会话边界调用 `reset()`。

## 5. 板端性能确认

NXP BSP 自带的 `vela-prof` 可用来查看 NPU 利用率、DRAM/SRAM 带宽和时延：

```bash
vela-prof -i /opt/driver_model/inception_encoder_int8.tflite
vela-prof -i /opt/driver_model/context_tcn_int8.tflite
```

最终验收时应同时比较 FP32 PyTorch、CPU TFLite 和 Ethos-U TFLite 的预测一致率，
并在真实流式输入上测量 P50/P95 时延。
