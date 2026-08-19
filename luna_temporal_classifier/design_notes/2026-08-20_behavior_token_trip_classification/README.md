# 驾驶行为文本化与 Trip 风格分类方案

日期：2026-08-20
状态：设计讨论结论，尚未进行正式训练验证

## 1. 结论

将短时驾驶行为转换为类似文字的行为 Token，再将整段 Trip 建模为 Token 序列进行风格分类是可行的，而且比直接把每个 20 秒窗口继承为整段 Trip 的 `NORMAL / AGGRESSIVE / DROWSY` 硬标签更符合标签语义。

推荐的主方案不是把行为写成自然语言后送入 BERT，而是：

```text
20 秒 8 维传感器数据
        ↓
局部行为识别器 LocalEventNet
        ↓
软行为 Token Embedding + 连续隐藏特征
        ↓
整段 Trip 行为序列
        ↓
轻量因果 Event-TCN
        ↓
Trip 驾驶风格分类
```

这种方案可以称为：

> Driving Behavior Tokenization and Event-Sequence Style Classification

从数学形式上看，它与文本分类相似；从任务本质上看，它是结构化驾驶事件序列分类，而不是传统自然语言理解。

## 2. 为什么需要行为 Token 化

UAH-DriveSet 的 `NORMAL / AGGRESSIVE / DROWSY` 标签对应完整驾驶 Trip，而不是其中每一个 20 秒局部片段。把整段标签直接复制给所有窗口会引入明显标签噪声，例如：

- `AGGRESSIVE` Trip 中可能存在大量平稳驾驶窗口；
- `DROWSY` Trip 中的某个 20 秒片段可能没有可观察的漂移或蛇形行为；
- 一个窗口可能同时出现急加速、超速和急转向，无法用单个硬类别准确描述。

行为 Token 化的目标是先回答“这个窗口发生了哪些局部行为”，再回答“这些行为在长时间内以什么频率、强度和顺序出现，从而构成什么驾驶风格”。

## 3. 局部行为词表

第一版建议使用 9 个基础事件：

| Token | 中文含义 | 典型证据 |
|---|---|---|
| `STABLE` | 平稳驾驶 | 速度、加速度和方向变化稳定 |
| `HARD_ACCEL` | 急加速 | 纵向加速度、jerk 明显增大 |
| `HARD_BRAKE` | 急刹车 | 负向加速度、负 jerk 明显增大 |
| `ABRUPT_TURN` | 急转向 | 横向加速度、yaw 或 course 快速变化 |
| `OVERSPEED` | 超速 | 需要合法限速或道路上下文支持 |
| `CLOSE_FOLLOW` | 跟车过近 | 需要前车距离或视频语义数据支持 |
| `LANE_WEAVING` | 蛇形行驶 | 航向或横向位置反复摆动 |
| `LANE_DRIFT` | 车道漂移 | 横向位置持续偏移、延迟修正 |
| `SPEED_INSTABILITY` | 速度控制不稳定 | 长时间非交通诱发的速度波动 |

如果训练和部署阶段只有当前 8 维传感器输入，`OVERSPEED`、`CLOSE_FOLLOW` 和车道相关事件必须明确其可观测性。可以在离线标注时使用视频、车道和前车信息作为特权信息，但部署模型仍只使用 8 维信号。

仅依靠车辆运动数据不能证明驾驶员在生理意义上困倦，因此建议把最终类别理解为 `DROWSY_STYLE`，即“具有困倦式车辆控制特征”，而不是医学诊断意义上的困倦。

## 4. Token 表示方式

### 4.1 不使用单一硬 Token

一个窗口可能同时包含多个事件，因此不建议：

```python
token = argmax(event_probability)
```

推荐由局部模型输出多标签概率：

```python
event_probability = [
    0.20,  # STABLE
    0.80,  # HARD_ACCEL
    0.05,  # HARD_BRAKE
    0.65,  # ABRUPT_TURN
    0.30,  # OVERSPEED
    0.00,  # CLOSE_FOLLOW
    0.00,  # LANE_WEAVING
    0.00,  # LANE_DRIFT
    0.10,  # SPEED_INSTABILITY
]
```

### 4.2 软行为 Token Embedding

设行为词表大小为 `K`，Embedding 维度为 `D`：

```python
event_table = nn.Parameter(torch.randn(K, D))
soft_event_embedding = event_probability @ event_table
```

即窗口向量是所有行为词向量的概率加权和。它能够：

- 表示同一窗口中的多个事件；
- 保留局部模型的不确定性；
- 避免错误 `argmax` 导致的信息突变；
- 支持端到端反向传播。

第一版推荐 `K=9`、`D=32`。

### 4.3 因子化属性 Embedding

严重程度和持续时间不应导致词表组合爆炸，可以独立编码：

```python
token_embedding = (
    event_embedding
    + severity_embedding
    + duration_embedding
)
```

建议属性：

- 严重程度：`0 / 1 / 2 / 3`；
- 持续时间：`0 / 1-3s / 3-8s / >8s`；
- 标注置信度：`low / medium / high`。

## 5. 是否使用自然语言模型

可以把每个窗口写成：

```text
本窗口发生中度急加速和轻度急转向，持续约 4 秒。
```

再使用 BERT 或 SentenceTransformer 编码，但不建议作为主模型，原因包括：

- 通用词向量学习的是自然语言共现关系，不是车辆动力学关系；
- 行为词汇量很小，不需要数百维语言表示；
- 当前 Trip 数量不足以有效微调大型文本模型；
- BERT/BiTransformer 通常不是因果模型；
- 对 i.MX93 的全 INT8 部署明显复杂于 Conv1d 和 Embedding Lookup；
- 把传感器规则翻译成文字不会产生新的独立信息。

自然语言模型可以作为离线教师或消融实验，例如使用预训练文本向量初始化行为 Embedding，随后将其蒸馏为 32 维端侧向量。最终端侧不部署语言模型。

## 6. 推荐模型结构

### 6.1 LocalEventNet

输入：

```text
[B, 8, 200]，即 20 秒 × 10 Hz
```

输出：

```text
event_logits:     [B, 9]
event_severity:   [B, 9]
sensor_embedding: [B, 16 或 32]
```

局部模型优先使用轻量 Conv1d / depthwise-separable Conv1d，避免使用 2D-CNN、BiLSTM 或大型注意力网络。

### 6.2 混合窗口表示

只使用人工定义的行为 Token 可能丢失波形节奏、频率和车辆动力响应等信息，因此推荐：

```python
window_embedding = torch.cat(
    [soft_event_embedding, sensor_embedding],
    dim=-1,
)
```

例如：

```text
32 维软行为 Token + 32 维连续传感器隐藏特征 = 64 维窗口表示
```

### 6.3 Event-TCN

每个 Trip 由若干个不重叠 20 秒窗口组成：

```text
trip_embedding.shape = [B, N_windows, 64]
```

建议 Event-TCN：

```yaml
channels: 48
kernel_size: 3
dilations: [1, 2, 4, 8, 16]
dropout: 0.10
pooling: [mean, max, last]
classifier: 144 -> 48 -> 3
```

它可以学习：

- 激进事件的密度和最高严重程度；
- 急加速、急刹车、急转向的连续组合；
- 车道漂移或速度不稳定是否持续数分钟；
- 异常事件是零散出现还是形成稳定风格。

## 7. 标注方案

不需要逐个标注步长 5 秒、重叠 75% 的窗口。建议在连续 Trip 时间轴上标注事件起止时间，再自动映射到任意窗口。

第一版人工标注流程：

1. 使用同步视频、8 维信号曲线和语义候选查看完整 Trip；
2. 以 1～5 秒精度记录事件类型、起止时间、严重程度和置信度；
3. 自动计算每个非重叠 20 秒窗口内各事件的持续比例；
4. 生成多标签软目标；
5. 至少 20% 数据由第二位标注员独立复标；
6. 对冲突和低置信度区间进行仲裁或标记为 `UNCERTAIN`。

自动阈值或现有 DriveSafe 语义只能用于提出候选，不能未经人工复核直接作为 Ground Truth，否则局部模型可能只是在复现标注规则。

## 8. 训练流程

### 阶段一：局部事件预训练

```python
loss_event = BCEWithLogitsLoss(event_logits, event_target)
loss_event *= annotation_confidence
```

严重程度可以使用回归、序数分类或独立的多级分类损失。

### 阶段二：Trip 风格模型训练

局部模型对训练 Trip 产生事件概率和连续隐藏特征，Event-TCN 使用完整序列预测原始 Trip 标签：

```python
style_logits = event_tcn(predicted_window_embeddings)
loss_style = cross_entropy(style_logits, trip_label)
```

不要只用完美人工事件向量训练 Event-TCN，否则训练阶段与部署阶段的输入噪声不一致。建议使用训练数据内部的 out-of-fold 局部模型预测，或者对事件概率注入符合实际误差的噪声。

### 阶段三：端到端联合微调

```text
Loss =
1.0 × Trip 风格损失
+ 0.5 × 局部事件损失
+ 0.1 × 相邻窗口时间一致性损失
```

联合微调可以让 LocalEventNet 保留事件可解释性的同时，为最终 Trip 风格分类优化。

## 9. 小样本约束与强基线

Token 化解决的是“窗口标签与 Trip 标签错配”，不会增加独立 Trip 标签数量。当前 D1-D4 仍只有 28 个训练 Trip，因此 Event-TCN 必须保持小型，并与可解释统计基线比较。

统计基线可以从每类事件概率计算：

```text
均值、最大值、90% 分位数、阈值以上次数、
总持续时间、最长连续长度、事件切换次数
```

然后使用 Logistic Regression、Random Forest 或小型 MLP 预测 Trip 风格。如果 Event-TCN 不能在严格跨驾驶员验证中稳定超过这一基线，就不应增加序列模型复杂度。

## 10. 评估协议

所有拆分必须在原始 Trip/驾驶员层面完成，禁止先切窗口再随机划分。

推荐 nested leave-one-driver-out：

1. 外层保留一位驾驶员作为测试；
2. 剩余驾驶员中再保留一位用于架构和超参数选择；
3. 标准化、事件模型训练、out-of-fold 预测均只能使用对应训练驾驶员；
4. 报告六个外层驾驶员的 Accuracy、Macro-F1、每类 F1、均值和标准差。

局部和整体指标需要分开报告：

- 局部事件：每类 Precision、Recall、F1、mAP；
- Trip 风格：Accuracy、Macro-F1、每驾驶员和每道路结果；
- 在线能力：观察 2、5、10 分钟后的时间-性能曲线。

## 11. 必须完成的消融实验

| 实验 | 输入 | 目的 |
|---|---|---|
| A | 原始 8 维整段时序 | TripStyleTCN 直接建模基线 |
| B | 人工真实行为 Token | 判断行为词表的理论上限 |
| C | 局部模型预测 Token | 测量真实部署性能 |
| D | 预测 Token + 连续隐藏特征 | 验证混合表示是否减少信息损失 |
| E | 自定义 Token Embedding | 推荐的轻量文本化方案 |
| F | 预训练文本 Embedding | 判断通用语言语义是否真的有增益 |

结果解释：

- B 高、C 低：局部事件识别是主要瓶颈；
- B 也低：事件词表不足或 Trip 标签本身噪声过大；
- D 明显高于 C：人工 Token 存在信息瓶颈，连续传感器特征有必要；
- F 不高于 E：不需要自然语言预训练模型；
- Event-TCN 不超过统计基线：Trip 样本量不足以支持更复杂序列模型。

## 12. i.MX93 端侧约束

推荐只使用：

- Conv1d / depthwise Conv1d；
- ReLU；
- Embedding Lookup 或常量矩阵乘法；
- average/max pooling；
- Linear；
- 固定长度的因果状态缓存。

避免把 BERT、BiLSTM、动态 Attention、复杂 KAN 或需要未来窗口的双向结构作为最终端侧模型。行为词向量可以作为固定小矩阵部署，LocalEventNet 和 Event-TCN 可以联合量化为 INT8。

## 13. 最终推荐

主实验采用：

```text
LocalEventNet
  → 9 维多标签事件概率
  → 32 维软行为 Token Embedding
  + 32 维连续传感器 Embedding
  → 轻量因果 Event-TCN
  → Trip 风格
```

该方案保留了“将驾驶行为建模为语言序列”的核心思想，同时避免通用文本模型带来的无关复杂度。它兼顾标签语义、可解释性、端到端训练和 i.MX93 部署约束，是下一阶段最值得正式验证的架构方向。
