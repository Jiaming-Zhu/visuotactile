# `train_fusion_gating2.py` 模型结构说明

本文档对应脚本：`visuotactile/scripts/train_fusion_gating2.py`

该模型是在基础双模态融合模型 `train_fusion.py` 之上加入 **连续门控（continuous gating）**、**无信息先验替代（uninformative prior）** 和 **触觉辅助监督（auxiliary tactile heads）** 的版本。它的核心目标是：

1. 保留视觉提供的外观先验；
2. 让触觉承担物理属性判断的主体责任；
3. 当视觉和触觉不一致，或视觉不可靠时，自动降低视觉 token 的影响；
4. 缓解模型只走视觉捷径、导致 OOD 泛化差的问题。

---

## 1. 任务定义

输入来自一次抓取 episode：

- 视觉输入：一张静态 RGB 图像 `image`
- 触觉输入：一段 24 通道时间序列 `tactile`

模型同时预测三个离散属性：

- `mass`
- `stiffness`
- `material`

输出不是单头分类，而是三个并行任务头，因此本质上是一个 **多任务多模态分类模型**。

---

## 2. 输入与预处理

### 2.1 视觉输入

图像预处理为：

- `Resize(224, 224)`
- `ToTensor()`
- ImageNet 标准化

输入张量形状：

```text
(B, 3, 224, 224)
```

### 2.2 触觉输入

触觉由 24 个通道组成，来自 6 个关节的 4 类反馈：

- `joint_position`
- `joint_load`
- `joint_current`
- `joint_velocity`

每类特征先按预先统计的均值和标准差做 Z-score 标准化，然后在通道维拼接，形成：

```text
(24, T)
```

其中默认会截断或补零到 `max_tactile_len=3000`，最终 batch 形状为：

```text
(B, 24, 3000)
```

同时生成一个原始时间维上的 `padding_mask`，用于标记补零区域。

---

## 3. 总体结构

模型可以分成 7 个部分：

1. 视觉编码器
2. 触觉编码器
3. 模态全局摘要
4. 跨模态门控网络
5. 连续模态 dropout / 无信息先验替代
6. Transformer 融合编码器
7. 主分类头 + 触觉辅助头

整体信息流如下：

```text
image
  -> ResNet18 backbone
  -> 1x1 Conv projection
  -> visual tokens
                  \
                   -> gate score g -> gated visual tokens --\
                  /                                          \
tactile -> 1D CNN -> tactile tokens -> tactile global -------> Transformer -> CLS
                                                           /                 |  |  |
                                       tactile global ----/                  |  |  |
                                                                              |  |  |
                                                                        mass head ...
                                                                        stiffness head
                                                                        material head

另外：
tactile global -> aux_mass / aux_stiffness / aux_material
```

---

## 4. 视觉分支

### 4.1 Backbone

视觉 backbone 使用 ImageNet 预训练的 `ResNet18`：

- 去掉最后两层，只保留卷积特征提取部分
- 默认 `freeze_visual=True`，即冻结 backbone 参数

对应实现：

- `self.vis_backbone = nn.Sequential(*list(resnet.children())[:-2])`
- `self.vis_proj = nn.Conv2d(512, fusion_dim, kernel_size=1)`

### 4.2 Token 化

当输入图像尺寸为 `224x224` 时，ResNet18 最终输出空间分辨率通常为 `7x7`。因此：

```text
视觉特征图: (B, 512, 7, 7)
投影后:     (B, 256, 7, 7)    # 默认 fusion_dim=256
展平后:     (B, 49, 256)
```

也就是说，视觉分支会产生 **49 个 visual tokens**。

---

## 5. 触觉分支

触觉编码器是一个 3 层 `Conv1d` 堆叠：

```text
24 -> 64 -> 128 -> fusion_dim
```

每层都包含：

- `Conv1d`
- `BatchNorm1d`
- `ReLU`

并且三层卷积的步长均为 `stride=2`，因此时间维会被下采样 8 倍。

若原始长度为 `3000`，则编码后大致为：

```text
输入:   (B, 24, 3000)
输出:   (B, 256, 375)
转置后: (B, 375, 256)
```

也就是说，触觉分支默认产生 **375 个 tactile tokens**。

---

## 6. 模态全局摘要与门控分数

这是该模型相对基础版 `FusionModel` 的关键改动。

### 6.1 全局摘要

模型先分别从两种模态的 token 序列中提取全局向量：

- `v_global = mean(v_tokens, dim=1)`
- `t_global = mean(t_tokens, dim=1)`，若存在 padding，则只在有效触觉 token 上求平均

得到：

```text
v_global: (B, 256)
t_global: (B, 256)
```

### 6.2 门控网络

然后把两个全局向量拼接起来：

```text
vt_global = concat(v_global, t_global)   # (B, 512)
```

再送入一个两层 MLP，输出单个 gate score：

```text
Linear(512, 256) -> ReLU -> Linear(256, 1) -> Sigmoid
```

因此：

```text
g in (0, 1), shape = (B, 1)
```

可以把它理解为当前样本中“视觉信息应被信任到什么程度”的连续权重：

- `g` 趋近 1：更依赖视觉 token
- `g` 趋近 0：更少依赖视觉 token

这里的 gate 是 **样本级** 的，而不是 token 级或通道级的。

---

## 7. 连续模态 Dropout 与 `t_null`

### 7.1 `t_null` 的作用

模型额外引入了一个可学习参数：

```text
self.t_null: (1, 1, fusion_dim)
```

它表示一种“无信息视觉先验”或“默认占位特征”。当模型判断视觉不可信时，不是简单把视觉 token 置零，而是把它们拉向这个可学习的先验向量。

### 7.2 Gated Visual Tokens

代码中的核心公式是：

```text
v_tokens_gated = g * v_tokens + (1 - g) * t_null
```

这意味着：

- 当 `g = 1` 时，保留原始视觉 token；
- 当 `g = 0` 时，视觉 token 全部被统一替换成 `t_null`；
- 当 `0 < g < 1` 时，视觉信息被连续衰减，而不是硬切换。

这种设计比直接 zero-out 更平滑，也更容易训练。

注意：当前实现中 **只对视觉 token 做门控**，触觉 token 会完整保留。

---

## 8. Token 拼接与 Transformer 融合

### 8.1 序列构成

模型使用一个可学习的 `cls_token` 作为全局汇聚 token，并将序列拼接为：

```text
[CLS] + gated visual tokens + tactile tokens
```

默认长度为：

- `1` 个 `CLS`
- `49` 个视觉 token
- `375` 个触觉 token

总长度：

```text
425 tokens
```

对应位置编码：

```text
self.pos_emb: (1, 425, fusion_dim)
```

### 8.2 Padding Mask

由于触觉原始序列可能比 `3000` 短，脚本会先在原始时间维构造 `padding_mask`，再通过 3 次 `max_pool1d` 下采样到触觉 token 对应长度，以便与 `Conv1d` 编码后的 token 一一对应。

最终的 `full_mask` 结构是：

```text
[CLS 和 visual token 全部为 False] + [tactile token 的有效/填充标记]
```

这样 Transformer 在自注意力中就不会关注无效的触觉 padding token。

### 8.3 Transformer 编码器

融合模块使用 `TransformerEncoder`：

- `num_layers = 4`
- `num_heads = 8`
- `d_model = fusion_dim = 256`
- `dim_feedforward = 1024`
- `dropout = 0.1`
- `activation = GELU`
- `batch_first = True`

Transformer 输出后，取：

```text
cls_out = x[:, 0, :]
```

作为三个主任务分类头的共享输入。

---

## 9. 输出头设计

### 9.1 主任务头

模型包含三个主分类头，分别预测：

- `mass`
- `stiffness`
- `material`

每个头结构相同：

```text
Linear(256, 128) -> GELU -> Dropout -> Linear(128, num_classes)
```

它们都基于 `cls_out` 进行预测，表示融合后的最终决策。

### 9.2 触觉辅助头

模型还额外定义了三个辅助头：

- `aux_mass`
- `aux_stiffness`
- `aux_material`

结构为：

```text
Linear(256, 64) -> GELU -> Linear(64, num_classes)
```

这些辅助头不是从 `cls_out` 读取，而是直接从 `t_global` 读取，也就是只基于触觉全局摘要做预测。

这样做的动机是：

- 防止模型在训练时过度依赖视觉分支；
- 迫使触觉分支本身具备可分性；
- 给触觉编码器提供更直接的监督信号。

代码注释把它称为 **Anti-Laziness** 设计，本质上就是避免融合模型“偷懒”，把触觉当成可有可无的补充。

---

## 10. 训练目标

训练时总损失由三部分组成：

```text
loss = ce_loss + lambda_reg * reg_loss + lambda_aux * aux_loss
```

### 10.1 主分类损失 `ce_loss`

三个主任务交叉熵之和：

```text
CE(mass) + CE(stiffness) + CE(material)
```

### 10.2 辅助损失 `aux_loss`

三个触觉辅助头的交叉熵之和：

```text
CE(aux_mass) + CE(aux_stiffness) + CE(aux_material)
```

默认 `lambda_aux = 0.5`。

### 10.3 门控正则 `reg_loss`

脚本支持多种 gate 正则化方式：

- `polarization`: 最小化 `g(1-g)`，鼓励 gate 接近 0 或 1
- `sparsity`: 最小化 `g.mean()`，鼓励整体更少使用视觉
- `mean`: 让 `g.mean()` 接近某个目标均值
- `center`: 让 `g` 靠近 0.5
- `entropy`: 鼓励高熵 gate，避免过早饱和
- `none`: 不使用 gate 正则

此外还支持：

- `gate_reg_warmup_epochs`
- `gate_reg_ramp_epochs`

用于在训练早期关闭或逐步增大 gate 正则，降低不稳定性。

---

## 11. 训练阶段的显式模态扰动

除了结构本身的 gate 外，训练循环里还支持两种外部模态干预。

### 11.1 模态 dropout

通过：

- `visual_drop_prob`
- `tactile_drop_prob`

可对整批样本中的某些视觉或触觉输入直接置零，用于提升鲁棒性。

这一步发生在送入模型之前，属于 **输入级随机扰动**。

### 11.2 模态阻断

通过：

- `--block_modality visual`
- `--block_modality tactile`

可以在训练或评估时强制屏蔽某一模态，做消融实验。

---

## 12. 默认张量尺寸汇总

在默认配置下（`fusion_dim=256`, `max_tactile_len=3000`, 输入图像 `224x224`）：

| 模块 | 张量形状 |
|---|---|
| 输入图像 | `(B, 3, 224, 224)` |
| 视觉特征图 | `(B, 512, 7, 7)` |
| 视觉 token | `(B, 49, 256)` |
| 输入触觉 | `(B, 24, 3000)` |
| 触觉 token | `(B, 375, 256)` |
| `v_global` | `(B, 256)` |
| `t_global` | `(B, 256)` |
| `gate_score` | `(B, 1)` |
| 拼接序列 | `(B, 425, 256)` |
| Transformer 输出 | `(B, 425, 256)` |
| `cls_out` | `(B, 256)` |

---

## 13. 与基础 `train_fusion.py` 的区别

相对于基础融合模型，这个 `gating2` 版本的主要新增点是：

1. 增加了 `gate_mlp`，基于视觉/触觉全局摘要计算样本级 gate；
2. 增加了 `t_null`，用于在低 gate 时替代视觉 token；
3. 视觉 token 先经过连续门控，再进入 Transformer；
4. 增加了三个基于 `t_global` 的辅助分类头；
5. 训练损失中加入了 `lambda_aux * aux_loss`；
6. 训练损失中加入了可配置的 gate 正则项；
7. 提供了 gate 正则 warmup / ramp 机制。

可以把它理解为：

- 基础版：直接把视觉 token 和触觉 token 拼起来做融合；
- `gating2` 版：先估计“这张图现在值不值得信”，再决定视觉 token 以多大程度参与融合。

---

## 14. 设计动机总结

该模型背后的直觉是：

- 视觉适合提供抓取前的外观先验，但容易学习颜色、纹理、形状等捷径；
- 触觉更直接对应质量、硬度、材料等物理属性，但单独使用时可能缺少外观上下文；
- 因此最理想的策略不是静态平均融合，而是让模型在每个样本上动态判断视觉信息是否可靠。

`train_fusion_gating2.py` 的实现正是在这个方向上的一个简单而直接的方案：

- 用 `v_global` 和 `t_global` 估计模态一致性；
- 用连续 gate 衰减视觉 token；
- 用 `t_null` 避免粗暴置零；
- 用触觉辅助监督防止模型只依赖视觉。

如果后续还要继续升级，这个版本自然可以往下面几个方向扩展：

- 从样本级 gate 改成 token 级 gate；
- 不只门控视觉，也门控触觉；
- 让 gate 由时序触觉状态逐步更新；
- 把 `t_null` 扩展为多原型先验，而不是单一向量。
