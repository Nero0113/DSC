# 数据审计

## 布局

| split | 驾驶员 | 窗口数 | NORMAL | AGGRESSIVE | DROWSY |
|---|---|---:|---:|---:|---:|
| train | D1–D4 | 4,126 | 1,758 | 1,125 | 1,243 |
| validation | D5 | 1,035 | 447 | 237 | 351 |
| test | D6 | 885 | 354 | 179 | 352 |

`X` 为 `float32 [N, 8, 200]`。采样率 10 Hz，窗长 20 秒，步长 5 秒；相邻窗
重叠 75%。通道顺序为：

1. `speed_kmh`
2. `course_delta_deg`
3. `acc_x_kf_g`
4. `acc_y_kf_g`
5. `acc_z_kf_g`
6. `roll_deg`
7. `pitch_deg`
8. `yaw_deg`

标签是 `0=NORMAL, 1=AGGRESSIVE, 2=DROWSY`。

## NPZ metadata

三个 split 均含 `X, y, driver_id, trip_id, road, label_name, window_start_s,
feature_names`。`scaler.npz` 含 `mean, std, feature_names`；现有 `X` 已使用训练集
统计量标准化。

## 泄漏审计

- train/validation/test 之间无驾驶员重叠。
- split 之间无 trip 重叠、无完全相同 X hash、无重复 `(driver, trip, start)` 键。
- 训练时只输入 X；metadata 只用于审计和导出结果。
- 因相邻窗重叠 75%，随机窗口拆分会产生严重泄漏，禁止用于报告泛化指标。
- D5 用于 early stopping/方案选择；D6 只作为独立终测，不能反向选超参。

## 已观察到的难点

D5/D6 在若干传感器通道上相对 D1–D4 存在明显分布漂移。每个 trip 的标签恒定，
窗口并非独立样本；窗口级指标应同时结合 trip 分组理解。
