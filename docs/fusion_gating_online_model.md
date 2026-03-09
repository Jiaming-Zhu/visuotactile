# `train_fusion_gating_online.py` 模型结构说明

本文档对应脚本：`visuotactile/scripts/train_fusion_gating_online.py`

这个脚本并没有重新定义一个全新的网络骨架，而是**直接复用** `train_fusion_gating2.py` 中的 `FusionModel`，并在训练与评估流程上加入 **在线前缀（online prefix）机制**。换句话说，它的重点不是改模型“长什么样”，而是改模型“看多少触觉数据再做判断、以及训练时怎样模拟早期决策”。

因此可以把它理解为：

> `train_fusion_gating2.py` 的在线前缀训练版

它的核心目标是：

1. 让模型不必总看到完整触觉序列才学习分类；
2. 训练时随机截断到不同长度的触觉前缀；
3. 迫使模型在较早阶段也能给出合理预测；
4. 用统一的前缀评估接口分析不同接触进度下的性能。

---

## 1. 任务定义

输入仍然来自一次抓取 episode：

- 一张抓取前 RGB 图像 `image`
- 一段 24 通道触觉序列 `tactile`

模型同时预测三个离散物理属性：

- `mass`
- `stiffness`
- `material`

与 `train_fusion_gating2.py` 的差别不在输出形式，而在于：

- 训练时，模型不一定看到完整触觉序列；
- 评估时，可以强制只看前 `10% / 20% / 40% / ... / 100%` 的触觉前缀。

因此这个脚本关注的是 **prefix-aware learning**，而不是新的融合模块。

---

## 2. 复用的基础模型：`FusionModel`

这个脚本中的 `build_model()` 直接返回：

```text
FusionModel(...)
```

也就是 `train_fusion_gating2.py` 中定义的模型。其结构不变，仍包括：

1. 视觉分支：`ResNet18 + 1x1 Conv`
2. 触觉分支：3 层 `Conv1d`
3. 全局摘要：`v_global` 和 `t_global`
4. 样本级门控网络 `gate_mlp`
5. `t_null` 视觉替代向量
6. Transformer Encoder 融合
7. 三个主分类头
8. 三个触觉辅助头

因此从网络结构角度看，`train_fusion_gating_online.py` 和 `train_fusion_gating2.py` 是同一个模型家族。

---

## 3. 基础模型内部结构回顾

为了便于阅读，这里简要回顾一下它复用的 `FusionModel`。

### 3.1 视觉分支

视觉 backbone 为 ImageNet 预训练的 `ResNet18`，去掉最后两层后接一个 `1x1 Conv2d` 投影到融合维度 `fusion_dim=256`。

输入图像：

```text
(B, 3, 224, 224)
```

输出视觉 token：

```text
(B, 49, 256)
```

同时对视觉 token 取均值得到：

```text
v_global: (B, 256)
```

### 3.2 触觉分支

触觉序列先由 3 层 `Conv1d` 编码：

```text
24 -> 64 -> 128 -> 256
```

每层带：

- `BatchNorm1d`
- `ReLU`
- `stride=2`

对于默认长度 `3000` 的触觉序列，最终大约得到：

```text
t_tokens: (B, 375, 256)
```

如果有 padding，则只在有效 token 上求：

```text
t_global: (B, 256)
```

### 3.3 样本级门控

门控网络输入为：

```text
concat(v_global, t_global)
```

再通过：

```text
Linear(512, 256) -> ReLU -> Linear(256, 1) -> Sigmoid
```

得到：

```text
g: (B, 1)
```

然后对视觉 token 做连续替换：

```text
v_tokens_gated = g * v_tokens + (1 - g) * t_null
```

这里 `t_null` 是一个可学习的“无信息视觉先验”。

### 3.4 Transformer 融合

拼接序列为：

```text
[CLS] + gated visual tokens + tactile tokens
```

默认总长度：

```text
1 + 49 + 375 = 425
```

再送入 4 层 Transformer Encoder，并从 `[CLS]` 输出三个主任务分类结果。

### 3.5 辅助头

除了主任务头，模型还从 `t_global` 额外输出：

- `aux_mass`
- `aux_stiffness`
- `aux_material`

用于加强触觉分支的可分性，防止模型只靠视觉“偷懒”。

---

## 4. 这个脚本真正新增的东西：前缀机制

`train_fusion_gating_online.py` 的真正创新点不在 backbone，而在 **如何构造 `padding_mask`**。

它通过修改有效触觉长度，让同一个 `FusionModel` 在训练或评估时只看到一部分触觉时间序列。

核心思想是：

1. 原始数据集中每条序列可能有自己的真实有效长度 `valid_length`
2. 训练时随机采样一个前缀长度 `prefix_length`
3. 把 `prefix_length` 之后的所有时间步都视为 padding
4. 这样模型在前向时就只能使用前缀部分的信息

这相当于用 **mask 的方式模拟“当前只观察到了前缀”**。

---

## 5. 前缀掩码的构造方式

### 5.1 真实有效长度

脚本先根据原始 `padding_mask` 计算每个样本的真实有效长度：

```text
valid_lengths = (~padding_mask).sum(dim=1)
```

并强制至少为 1，避免极端情况。

### 5.2 构造前缀 mask

当给定某个前缀长度 `prefix_length` 时，脚本会构造：

```text
prefix_valid = positions < prefix_length
```

再和原始有效区域相交：

```text
effective_valid = original_valid & prefix_valid
```

最后取反得到新的 `padding_mask`：

```text
effective_padding_mask = ~(effective_valid)
```

因此，即使原始序列很长，模型在这个 batch 中也只会看到指定前缀之前的触觉数据。

---

## 6. 训练时的随机前缀采样

### 6.1 两个关键超参数

训练阶段使用前缀学习时，最重要的两个控制量是：

- `online_train_prob`
- `online_min_prefix_ratio`

外加一个长度下限：

- `min_prefix_len`

默认配置为：

```text
online_train_prob = 1.0
online_min_prefix_ratio = 0.2
min_prefix_len = 64
```

### 6.2 含义

对每个 batch 中的每个样本：

- 若启用前缀训练，则从 `[lower_bound, valid_length]` 之间随机采样一个前缀长度
- 其中：

```text
lower_bound = max(ceil(valid_length * online_min_prefix_ratio), min_prefix_len)
```

因此训练时不会只给模型极短的几帧噪声，而是保证至少有：

- 原始长度的某个比例
- 或者最少 `64` 个原始时间步

### 6.3 `online_train_prob`

如果 `online_train_prob < 1.0`，则并不是每个样本都用前缀训练：

- 一部分样本用随机前缀
- 另一部分样本仍用完整序列

这样可以在“学会早期预测”和“保留完整序列性能”之间折中。

当默认 `online_train_prob=1.0` 时，所有训练样本都按随机前缀方式训练。

---

## 7. 评估时的两种模式

这个脚本提供了三种运行模式：

- `train`
- `eval`
- `online_eval`

### 7.1 `eval`

`eval_split()` 中调用 `effective_padding_mask()` 时传入：

```text
fixed_ratio = None
train_mode = False
online_train_prob = 0.0
```

因此这里不会对评估集随机裁前缀，而是使用样本本来的完整有效长度。

所以：

- `eval` 仍然是完整序列评估
- 只是模型本身曾经接受过前缀训练

### 7.2 `online_eval`

`online_eval_split()` 则会遍历多个固定前缀比例，例如默认：

```text
0.1, 0.2, 0.4, 0.6, 0.8, 1.0
```

对于每个比例 `r`：

1. 计算 `prefix_length = ceil(valid_length * r)`
2. 构造固定比例的 prefix mask
3. 用同一个模型分别跑一遍评估

最后得到一条 `prefix_curves` 曲线，记录：

- `loss`
- `gate_score`
- `mass`
- `stiffness`
- `material`
- `average_accuracy`

这可以直观看出：随着接触信息逐渐增加，模型性能和 gate 如何变化。

---

## 8. 训练损失

训练损失与 `train_fusion_gating2.py` 基本一致：

```text
loss = ce_loss + lambda_reg * reg_loss + lambda_aux * aux_loss
```

### 8.1 主分类损失 `ce_loss`

三个主任务交叉熵之和：

```text
CE(mass) + CE(stiffness) + CE(material)
```

### 8.2 辅助损失 `aux_loss`

三个触觉辅助头交叉熵之和：

```text
CE(aux_mass) + CE(aux_stiffness) + CE(aux_material)
```

默认：

```text
lambda_aux = 0.5
```

### 8.3 Gate 正则 `reg_loss`

支持的 gate 正则类型和 `gating2` 完全一致：

- `polarization`
- `sparsity`
- `mean`
- `center`
- `entropy`
- `none`

默认：

```text
reg_type = entropy
lambda_reg = 0.1
```

同时支持：

- `gate_reg_warmup_epochs`
- `gate_reg_ramp_epochs`

用于在训练早期逐步引入 gate 正则，减少不稳定性。

---

## 9. 为什么前缀训练是有意义的

如果一个模型只在完整序列上训练，那么它有可能学会依赖：

- 序列末尾的强证据
- 更晚阶段才出现的接触模式
- 对早期部分几乎不敏感

这样一来，即使它在完整序列测试上表现很好，也不代表它能在真实在线场景中尽早给出判断。

前缀训练相当于强迫模型在训练时不断面对下面这种问题：

> “如果现在只看到了前 20% 或前 40% 的触觉过程，你还要尽量分类正确。”

这会鼓励模型：

- 更好利用早期接触信号
- 更快整合视觉先验与短触觉证据
- 在有限交互时间内完成判断

---

## 10. 默认张量尺寸

由于网络骨架没有变，默认张量尺寸与 `FusionModel` 相同：

| 模块 | 张量形状 |
|---|---|
| 图像输入 | `(B, 3, 224, 224)` |
| 视觉 token | `(B, 49, 256)` |
| 触觉输入 | `(B, 24, 3000)` |
| 触觉 token | `(B, 375, 256)` |
| `v_global` | `(B, 256)` |
| `t_global` | `(B, 256)` |
| `gate_score` | `(B,)` 或等价 `(B, 1)` |
| 拼接序列 | `(B, 425, 256)` |
| `cls_out` | `(B, 256)` |

需要注意的是，前缀机制不会改变张量总长度，而是通过 `padding_mask` 改变其中哪些 token 被视为有效。

也就是说：

- 张量形状仍然固定
- 有效信息长度是动态变化的

---

## 11. 与 `train_fusion_gating2.py` 的区别

这两个脚本的关系非常直接：

### `train_fusion_gating2.py`

- 使用完整序列训练
- gate 是样本级
- 结构重点在“跨模态冲突估计 + 连续视觉门控”

### `train_fusion_gating_online.py`

- 使用同一个 `FusionModel`
- 训练时随机只暴露触觉前缀
- 评估时可按固定前缀比例测试
- 重点在“如何让同一个模型适应在线早期决策”

因此可以说：

- `gating2` 解决的是“如何门控视觉”
- `gating_online` 解决的是“如何在只看到部分触觉时也学会预测”

---

## 12. 与 `train_fusion_gating_stream_transformer.py` 的区别

这两个脚本都和“在线”有关，但本质并不一样。

### `train_fusion_gating_online.py`

- 底层仍是离线 `FusionModel`
- 一次性前向整段张量
- 只是通过 prefix mask 模拟“只能看到前缀”
- gate 仍然是单个样本级分数

### `train_fusion_gating_stream_transformer.py`

- 结构本身就是严格流式
- 触觉按 chunk 逐步处理
- 每一步都会更新 `cls_state`
- 每一步都会重新计算 gate
- 使用 tactile KV cache 实现因果历史记忆

所以：

- `gating_online` 更像是“离线模型的在线训练策略”
- `stream_transformer` 更像是“真正的在线架构”

---

## 13. 设计动机总结

这个脚本非常适合作为从离线模型过渡到在线模型的中间方案，因为它保留了 `FusionModel` 的全部结构优势：

- 代码改动小
- 可直接复用已有 checkpoint 训练逻辑
- 与原实验结果对比容易

同时又引入了在线前缀训练的关键能力：

- 训练时随机模拟不同接触进度
- 测试时能够画出前缀性能曲线
- 可以研究“模型在什么时候开始足够自信、足够准确”

它的优点是实现简单、实验可控；限制则是：

- 内部结构仍是离线 Transformer
- gate 不是时间步级
- 没有显式的因果缓存或逐步状态更新

因此它更像一个 **在线学习 protocol**，而不是一个完全新的流式架构。

---

## 14. 一句话概括

`train_fusion_gating_online.py` 可以概括为：

> 在 `FusionModel` 不变的前提下，通过随机触觉前缀训练和固定比例前缀评估，让原本的离线门控融合模型具备更强的早期预测能力与在线分析能力。
