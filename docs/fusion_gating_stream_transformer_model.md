# `train_fusion_gating_stream_transformer.py` 模型结构说明

本文档对应脚本：`visuotactile/scripts/train_fusion_gating_stream_transformer.py`

这个脚本实现的是一个 **严格流式（strict streaming）的视觉-触觉门控融合 Transformer**。它延续了 `train_fusion_gating2.py` 里“根据触觉状态动态调节视觉权重”的思路，但把整段触觉一次性编码的方式，改成了 **按时间块 chunk 递进处理**。因此它更适合描述“接触过程逐步展开、模型需要边看边判断”的在线感知场景。

相比普通的离线融合模型，这个版本的核心变化是：

1. 把长触觉序列切成多个 chunk，逐步输入模型；
2. 每一步只看当前 chunk、视觉 token 和过去缓存的 tactile token；
3. 每一步根据 `visual_summary + 上一步 cls_state + 当前触觉摘要` 计算 gate；
4. 用 gate 对视觉 token 做连续抑制，再参与当前步注意力；
5. 每一步都输出分类结果，训练时对整条预测轨迹做加权监督；
6. 模型选择不只看最终准确率，还看“前期能不能尽早预测对”。

---

## 1. 任务定义

输入来自一次抓取 episode：

- 一张静态 RGB 图像 `image`
- 一段 24 通道触觉序列 `tactile`

模型同时预测三个离散物理属性：

- `mass`
- `stiffness`
- `material`

与 `train_fusion.py` 或 `train_fusion_gating2.py` 不同，这个脚本天然支持 **在线逐步输出**：

- 每处理完一个触觉 chunk，都可以得到一次中间预测
- 所有 chunk 处理结束后，再取最后一步输出作为最终预测

因此它本质上是一个 **流式多模态多任务分类模型**。

---

## 2. 输入与预处理

### 2.1 视觉输入

视觉输入是一张抓取前图像，预处理沿用融合模型的标准流程：

- `Resize(224, 224)`
- `ToTensor()`
- ImageNet 标准化

输入形状：

```text
(B, 3, 224, 224)
```

### 2.2 触觉输入

触觉由 24 个通道组成，来自 6 个关节的 4 类反馈：

- `joint_position`
- `joint_load`
- `joint_current`
- `joint_velocity`

经过 Z-score 标准化后，拼接为：

```text
(B, 24, L)
```

默认 `max_tactile_len=3000`。不足部分用零补齐，并由 `padding_mask` 标出无效时间步。

### 2.3 Chunk 切分

这个脚本的一个重要约束是：

```text
window_size == step_size == chunk_size
```

也就是说，在当前 `v1` 实现里：

- 每次输入一个长度为 `chunk_size` 的片段
- chunk 之间不重叠
- 流式步数约为 `ceil(L / chunk_size)`

默认：

```text
chunk_size = 256
```

因此长度为 `3000` 的序列大约会被分成 12 个流式步。

---

## 3. 模型总体结构

模型可以分成 6 个主要部分：

1. 视觉编码器
2. 触觉 chunk tokenizer
3. 样本级 step gate 估计器
4. 因果流式 Transformer 块
5. 主分类头与触觉辅助头
6. 多步监督与在线评估机制

整体信息流如下：

```text
image
  -> ResNet18 backbone
  -> visual tokens + visual summary -----------------------------\
                                                                 \
tactile sequence -> split into chunks -> chunk tokenizer -> current tactile tokens -> step gate -> gated visual tokens
                                                                                     \                     /
                                                                                      \                   /
                                                                                       -> stream transformer ->
                                                                                          update cls_state + KV cache
                                                                                                       |
                                                                                                  prediction heads
```

更具体地说，每个时间步都维护一个 `state`，其中保存：

- `visual_tokens`
- `visual_summary`
- `cls_state`
- `step_index`
- 每层的 tactile KV cache
- 上一步输出 logits
- 上一步 gate 分数

这让模型可以像 RNN 一样“带状态”运行，但内部更新规则仍然是 Transformer 风格的注意力计算。

---

## 4. 视觉分支

### 4.1 Backbone

视觉 backbone 使用 `ResNet18`：

- 默认可加载 ImageNet 预训练权重
- 去掉最后两层，只保留卷积特征图
- 接一个 `1x1 Conv2d(512 -> fusion_dim)` 做投影

默认 `fusion_dim=256`。

### 4.2 Token 化

当输入尺寸为 `224x224` 时，ResNet18 最终输出通常是 `7x7` 特征图，因此：

```text
视觉特征图: (B, 512, 7, 7)
投影后:     (B, 256, 7, 7)
展平后:     (B, 49, 256)
```

脚本中固定：

```text
VISUAL_TOKENS = 49
```

并为其加入可学习位置编码：

```text
visual_pos_emb: (1, 49, fusion_dim)
```

同时做均值池化得到：

```text
visual_summary: (B, 256)
```

它会作为后续 gate 计算的一部分。

---

## 5. 触觉 Chunk Tokenizer

### 5.1 结构

每个 tactile chunk 都会送入 `TactileChunkTokenizer`。它本质上是一个 3 层 `Conv1d` 编码器：

```text
24 -> 64 -> 128 -> fusion_dim
```

每层都带：

- `Conv1d`
- `BatchNorm1d`
- `ReLU`

三层步长都为 `stride=2`，因此时间维被压缩 8 倍。

### 5.2 默认输出长度

当 `chunk_size=256` 时，经过三次下采样后，chunk token 数大约为：

```text
256 -> 128 -> 64 -> 32
```

因此每个 chunk 约对应：

```text
current_tokens: (B, 32, 256)
```

脚本用 `output_length(chunk_size)` 动态推算这个长度，并记为：

```text
max_chunk_tokens
```

### 5.3 有效 token 掩码

因为最后一个 chunk 可能不足 `chunk_size`，脚本会：

1. 根据 `chunk_valid_len` 构造原始时间维 padding mask
2. 用 3 次 `max_pool1d` 将 mask 下采样到 token 维度
3. 得到：

- `token_valid_mask`
- `token_valid_len`

随后无效 token 会被置零，当前 chunk 摘要由有效 token 的 masked mean 得到：

```text
current_summary: (B, 256)
```

---

## 6. Step Gate：按时间步动态调视觉

这是这个脚本相对 `train_fusion_gating2.py` 的最关键变化。

### 6.1 Gate 输入

每个流式步都会拼接三个全局向量：

- `visual_summary`
- `cls_state`
- `current_summary`

也就是：

```text
gate_input = concat(visual_summary, cls_state, current_summary)
```

形状为：

```text
(B, 3 * fusion_dim)
```

### 6.2 Gate 网络

然后送入一个两层 MLP：

```text
Linear(3D, D) -> ReLU -> Linear(D, 1) -> Sigmoid
```

得到当前步 gate：

```text
current_gate: (B,)
```

这里 gate 依赖于：

- 视觉全局先验
- 上一步融合状态
- 当前触觉 chunk 的摘要

因此它不是静态样本级 gate，而是 **随时间更新的 step-level gate**。

### 6.3 `t_null` 替换

和 `train_fusion_gating2.py` 一样，脚本不直接把视觉 token 硬置零，而是使用一个可学习的 `t_null`：

```text
gated_visual = g * visual_tokens + (1 - g) * t_null
```

因此：

- `g -> 1` 时，保留视觉 token
- `g -> 0` 时，视觉 token 全部退化为统一先验 `t_null`

### 6.4 非活跃样本的处理

如果某个样本在当前 chunk 没有有效数据，则不会更新当前步输出，而是沿用上一步的：

- `last_outputs`
- `last_aux_outputs`
- `last_gate_score`

这是流式批处理里一个重要的稳健设计，避免不同样本步长不一致时出现无效更新。

---

## 7. 严格流式 Transformer 块

### 7.1 设计动机

普通离线 Transformer 会一次性看完整段视觉 token 和触觉 token。这个脚本则要求：

- 当前步只能看当前 chunk
- 可以看过去已缓存的 tactile token
- 不能提前看到未来 chunk

因此脚本实现了自定义的：

- `StreamMultiheadAttention`
- `CausalStreamTransformerBlock`

### 7.2 注意力可见范围

在某个时间步，查询来自：

```text
[prev_cls_state, current_tokens]
```

而 key/value 则由三部分拼接：

1. `gated visual tokens`
2. 过去所有有效 tactile cache
3. 当前 chunk 的 tactile tokens

也就是说，当前步可以访问：

- 全部视觉信息
- 过去触觉历史
- 当前触觉片段

但不能访问未来 chunk，因此是 **严格因果的流式注意力**。

### 7.3 每层独立缓存

每个 `CausalStreamTransformerBlock` 都维护自己的 tactile KV cache：

- `key`
- `value`
- `valid_len`

当前步执行完后，会把当前有效 tactile token 的 key/value 附加到缓存末尾，供后续步使用。

因此这套结构更接近“带 KV cache 的在线 Transformer”。

### 7.4 CLS 状态递推

每一步都把上一步的 `cls_state` 作为全局记忆带入，经过注意力和 FFN 更新后得到新的 `cls_state`：

```text
prev_cls_state -> current_cls -> next cls_state
```

最终分类头都是基于这个递推后的 `cls_state` 输出，因此它扮演了“当前整段接触进度下的融合摘要”的角色。

---

## 8. 位置编码与步编码

这个脚本对触觉流式建模引入了三种位置信息：

### 8.1 视觉位置编码

```text
visual_pos_emb: (1, 49, D)
```

用于标记视觉空间 token 的位置。

### 8.2 触觉 token 位置编码

```text
tactile_pos_emb: (1, max_chunk_tokens, D)
```

用于标记 chunk 内部 token 的局部顺序。

### 8.3 Step 编码

```text
step_emb: Embedding(max_steps + 1, D)
```

用于标记当前是第几个流式步。它会加到：

- 当前 chunk token
- 当前 `cls_state`

因此模型既知道：

- 当前 token 在 chunk 内的位置
- 当前 chunk 在整个 episode 中的相对进度

---

## 9. 输出头

### 9.1 主任务头

三个主分类头分别预测：

- `mass`
- `stiffness`
- `material`

结构一致：

```text
Linear(D, 128) -> GELU -> Dropout -> Linear(128, num_classes)
```

输入为当前步更新后的 `cls_state`。

### 9.2 触觉辅助头

脚本还保留了三个辅助头：

- `aux_mass`
- `aux_stiffness`
- `aux_material`

结构为：

```text
Linear(D, 64) -> GELU -> Linear(64, num_classes)
```

输入不是 `cls_state`，而是当前 chunk 的 `current_summary`。

作用与 `train_fusion_gating2.py` 中类似：

- 直接监督触觉局部表示
- 防止模型过度依赖视觉

---

## 10. 单步前向过程

一次 `forward_step()` 可以概括为：

1. 将当前 tactile chunk pad 到固定 `chunk_size`
2. 用 `TactileChunkTokenizer` 编码成 `current_tokens`
3. 计算 `current_summary`
4. 拼接 `visual_summary + cls_state + current_summary` 得到 gate
5. 用 gate 生成 `gated_visual`
6. 把 `[cls_state + step_emb]` 与当前 tactile tokens 一起送入每层流式 Transformer
7. 更新每层 KV cache
8. 用新的 `cls_state` 输出主任务 logits
9. 用 `current_summary` 输出辅助 logits
10. 更新 state，进入下一步

整段序列的 `forward()` 则是对所有 chunk 反复调用 `forward_step()`。

---

## 11. 默认张量尺寸

在默认配置下：

- `fusion_dim = 256`
- `chunk_size = 256`
- `max_tactile_len = 3000`
- `num_heads = 8`

主要张量大致如下：

| 模块 | 张量形状 |
|---|---|
| 图像输入 | `(B, 3, 224, 224)` |
| 视觉 token | `(B, 49, 256)` |
| `visual_summary` | `(B, 256)` |
| 当前 tactile chunk | `(B, 24, 256)` |
| 当前 tactile token | `(B, 32, 256)` |
| `current_summary` | `(B, 256)` |
| `current_gate` | `(B,)` |
| `cls_state` | `(B, 256)` |
| 每层 KV cache | `(B, num_heads, max_cache_tokens, 32)` |

其中：

- `max_steps = ceil(3000 / 256) = 12`
- `max_chunk_tokens ≈ 32`
- `max_cache_tokens = max_steps * max_chunk_tokens ≈ 384`

也就是说，每层最多缓存大约 384 个触觉 token。

---

## 12. 训练损失

这个脚本训练时不是只看最后一步输出，而是对整条在线预测轨迹施加监督。

总损失为：

```text
total_loss = ce_loss + lambda_aux * aux_loss + lambda_reg * reg_loss
```

### 12.1 多步分类损失 `ce_loss`

脚本会遍历每个样本的所有有效时间步，对每一步的主任务 logits 都计算交叉熵。

但不是简单平均，而是使用递增权重：

```text
weights = [1, 2, ..., T] / sum(1..T)
```

也就是说：

- 越靠后的时间步，权重越大
- 但早期时间步也会被监督

这很符合在线预测的需求：既希望模型尽早有用，也承认后期证据更充分。

### 12.2 辅助损失 `aux_loss`

辅助损失只对最终步输出的三个辅助头做交叉熵之和：

```text
CE(aux_mass) + CE(aux_stiffness) + CE(aux_material)
```

默认：

```text
lambda_aux = 0.5
```

### 12.3 Gate 正则 `reg_loss`

脚本支持多种 gate 正则类型：

- `polarization`
- `sparsity`
- `mean`
- `center`
- `entropy`
- `none`

正则的计算对象不是单个最终 gate，而是 **整条流式路径上所有有效 step gate**。

默认配置是：

```text
reg_type = entropy
gate_reg_warmup_epochs = 5
gate_reg_ramp_epochs = 10
```

即训练早期先不强制 gate，之后逐步引入正则，减少不稳定性。

---

## 13. 在线评估与 checkpoint 选择

这是这个脚本另一个很重要的特点。

### 13.1 Early Online AUC

脚本不会只看整段结束后的准确率，还会额外计算一个“早期在线表现”指标：

- 默认在 `25% / 50% / 75%` 进度位置取预测
- 检查这些前缀时刻能否已经预测正确
- 对每个样本求平均，再对整个数据集求平均

代码中把它记为：

```text
early_online_auc
```

虽然严格来说它不是传统 ROC-AUC，但表达的是“前缀预测质量随进度变化的平均表现”。

### 13.2 选择分数

验证集上，最佳 checkpoint 的选择指标不是纯平均准确率，而是：

```text
selection_score = 0.6 * average_accuracy + 0.4 * early_online_auc
```

这意味着模型选择时同时考虑：

- 最终准确率
- 是否能尽早做出正确判断

这个设计非常贴合流式感知任务。

### 13.3 `online_eval` 模式

脚本专门提供了 `online_eval` 模式，用于统计不同前缀比例下的性能曲线。默认比例为：

```text
0.25, 0.5, 0.75, 1.0
```

输出包括：

- 每个比例下三个任务的准确率
- 平均准确率
- 该前缀位置的平均 gate 分数

因此可以直观看到模型随着接触进展，性能和视觉依赖如何变化。

---

## 14. 与 `train_fusion_gating2.py` 的区别

这两个脚本都属于“视觉门控融合”路线，但侧重点不同。

`train_fusion_gating2.py`：

- 一次性看完整段触觉
- gate 是样本级
- gate 作用于整段视觉 token
- 用标准 Transformer 对整段 token 一次性融合

`train_fusion_gating_stream_transformer.py`：

- 按 chunk 逐步处理触觉
- gate 是时间步级
- 每一步都根据当前触觉状态重新评估视觉可信度
- 使用带 KV cache 的因果流式 Transformer
- 显式优化早期在线预测表现

可以把后者理解为：

> `gating2` 的“在线版本 + 严格因果版本 + 早期预测优化版本`

---

## 15. 设计动机总结

这个模型背后的动机很明确：

1. 触觉感知本来就是一个逐步积累信息的过程，不应强迫模型只在 episode 结束后输出结果。
2. 视觉在不同接触阶段的价值并不固定，因此 gate 也不应固定为单个样本常数。
3. 过去触觉历史很重要，但未来信息不应泄漏，因此需要显式的流式缓存机制。
4. 对机器人来说，“尽早判断对”往往比“最后判断对”更有价值，因此 checkpoint 选择也应纳入早期性能。

因此，这个脚本并不是单纯把离线模型拆成几段，而是把任务本身重新表述成：

- 视觉提供静态先验
- 触觉逐块到达
- 融合状态逐步更新
- 视觉可信度逐步重估
- 分类结果逐步收敛

---

## 16. 一句话概括

`train_fusion_gating_stream_transformer.py` 可以概括为：

> 用静态视觉 token 提供先验，用触觉 chunk 逐步更新因果 Transformer 状态，并在每个时间步根据当前触觉摘要动态门控视觉信息，从而实现可在线评估、可早期决策的流式多模态分类。
