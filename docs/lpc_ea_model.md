# `train_lpc_ea.py` 模型结构说明

本文档对应脚本：`visuotactile/scripts/train_lpc_ea.py`

`train_lpc_ea.py` 实现的是一个面向流式触觉感知的多模态模型：**LPC-EA（Latent Predictive Coding with Evidential Accumulation）**。它不是把整段触觉序列一次性压成一个向量再分类，而是把触觉切成时间窗口，逐步编码成潜变量状态，并通过“预测未来是否准确”来判断当前视觉先验是否可信，最后以 **证据累积（evidential accumulation）** 的形式输出分类结果和不确定性。

这个模型的主线可以概括为：

1. 用单张抓取前图像编码出一个静态视觉先验 `z_vis`；
2. 用滑动窗口方式从触觉序列中提取一串时序潜变量 `z_kin^(t)`；
3. 用视觉条件化的前向动力学模型预测未来潜变量；
4. 通过预测轨迹与真实轨迹之间的“惊讶度（surprise）”生成门控分数 `g`；
5. 用 `g` 缩放视觉证据，再与流式触觉证据相加；
6. 将总证据转成 Dirichlet 参数，完成分类并给出不确定性。

---

## 1. 任务定义

每个样本来自一次抓取 episode，输入包括：

- 一张视觉锚点图像 `image`
- 一段 24 通道触觉/本体感觉时序 `tactile`

模型同时预测三个离散物理属性：

- `mass`
- `stiffness`
- `material`

与标准 softmax 分类不同，这个脚本最终输出的是每个任务的 **Dirichlet 参数 `alpha`**、类别概率 `prob` 和不确定性 `uncertainty`，因此它是一个 **带不确定性估计的多任务流式多模态分类模型**。

---

## 2. 输入与预处理

### 2.1 视觉输入

视觉输入是一张抓取前 RGB 图像，预处理为：

- `Resize(224, 224)`
- `ToTensor()`
- ImageNet 均值方差归一化

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

四类特征分别做 Z-score 标准化后，在通道维拼接，得到：

```text
(24, T)
```

默认会截断或补零到 `max_tactile_len=3000`，因此 batch 输入通常是：

```text
(B, 24, 3000)
```

同时构造 `padding_mask`，标记补零区域，后续用于计算每个样本真正可用的流式步数。

---

## 3. 整体结构

模型由 5 个核心组件组成：

1. 视觉先验编码器 `VisualPriorEncoder`
2. 流式触觉编码器 `TransformerProprioceptiveEncoder`
3. 潜变量前向动力学模型 `LatentForwardDynamics`
4. 惊讶度门控模块 `surprise -> gate`
5. 证据头 `EvidenceHeads` 与 EDL 分类

整体信息流如下：

```text
image
  -> VisualPriorEncoder
  -> z_vis -----------------------------\
                                         \
                                          -> visual evidence --\
touch sequence -> sliding windows -> TransformerProprioEncoder -> z_kin(t) -> kin evidence --+--> alpha -> prob / uncertainty
                                   \                         /
                                    -> forward rollout -----/
                                            |
                                    compare predicted vs observed trajectory
                                            |
                                        surprise r
                                            |
                                    gate g = exp(-gamma * r)
                                            |
                             scale visual evidence by g
```

和 `train_fusion_gating2.py` 的最大区别在于，这里不直接在 token 层做视觉/触觉融合，而是：

- 先分别编码视觉与流式触觉潜变量；
- 再用未来预测误差估计当前视觉先验是否可靠；
- 最后在“证据层”而不是“token 层”做门控和累积。

---

## 4. 视觉先验编码器

### 4.1 结构

视觉编码器 `VisualPriorEncoder` 使用 ImageNet 预训练的 `ResNet18`：

- 将 `fc` 替换为 `Identity()`
- 取 backbone 的全局视觉特征
- 再经过一个线性投影映射到潜变量维度

对应结构：

```text
ResNet18 -> Linear(512, latent_dim) -> LayerNorm(latent_dim)
```

默认 `latent_dim=256`，因此：

```text
输入图像: (B, 3, 224, 224)
backbone 输出: (B, 512)
z_vis: (B, 256)
```

### 4.2 冻结视觉分支

脚本默认 `freeze_visual_backbone=True`，也就是：

- ResNet 权重不参与训练
- `train()` 状态下强制让视觉 backbone 保持 `eval()`，避免 BatchNorm 统计量漂移

这说明这个分支被设计成一个 **稳定的静态先验编码器**，而不是一个会被流式触觉任务强烈反向塑形的联合编码器。

---

## 5. 流式触觉编码器

### 5.1 滑动窗口构造

原始触觉序列不是一次性整段送入模型，而是先切成若干滑动窗口：

```text
window_size = 128
step_size = 20
```

通过 `unfold()` 构造：

```text
tactile: (B, 24, L)
windows: (B, T, 128, 24)
```

其中 `T` 取决于有效长度 `valid_len`：

```text
T = floor((valid_len - window_size) / step_size) + 1
```

如果有效长度不足一个窗口，则该样本没有有效流式步。

### 5.2 单窗口编码

每个窗口 `(w, 24)` 会被编码成一个潜变量向量。编码器 `TransformerProprioceptiveEncoder` 分为两段：

1. 1D CNN 做局部时间特征提取
2. 小型 Transformer 做窗口内时序汇聚

CNN 结构为：

```text
Conv1d(24, 64, k=7, s=2)
-> Conv1d(64, 128, k=5, s=2)
-> Conv1d(128, latent_dim, k=3, s=2)
```

每层后面都是：

- `BatchNorm1d`
- `ReLU`

由于 3 次 `stride=2` 下采样，窗口长度 `128` 会变成大约 `16` 个时间 token。然后再：

- 加一个可学习 `cls_token`
- 加位置编码 `pos_emb`
- 送入 2 层 Transformer Encoder
- 取 `[CLS]` 位置作为窗口潜变量

因此每个窗口最终变成一个向量：

```text
z_kin^(t) in R^latent_dim
```

对整段流式窗口而言：

```text
z_kin: (B, T, 256)
```

---

## 6. 潜变量前向动力学模型

模型定义了一个前向预测器 `LatentForwardDynamics`，用于根据：

- 静态视觉先验 `z_vis`
- 当前流式状态 `z_t`

预测下一时刻潜变量：

```text
hat(z_{t+1}) = F([z_vis, z_t])
```

实现上是一个简单的 MLP：

```text
Linear(2D, 2D) -> GELU -> Linear(2D, D) -> LayerNorm(D)
```

其中 `D = latent_dim`。

### 6.1 为什么要 `detach(z_vis)`

在前向动力学里，脚本使用的是 `z_vis.detach()`，也就是把视觉先验从这条预测损失路径上截断梯度。

这样做的目的在代码注释里写得很明确：

- 避免视觉分支被前向预测任务“污染”
- 保持视觉先验更稳定

也就是说，前向模型把视觉看作条件，而不是要被动力学误差直接更新的对象。

---

## 7. 惊讶度门控：LPC-EA 的核心

这是整个脚本最关键的部分。

### 7.1 基本思想

如果视觉先验 `z_vis` 与当前触觉演化过程是一致的，那么基于 `z_vis` 和当前状态 `z_t` 去预测未来潜变量，应该比较准确。

反过来，如果视觉先验和真实接触过程不一致，那么预测的未来轨迹就会和实际观测轨迹偏离较大。

因此模型用：

- “预测轨迹”
- “真实轨迹”

之间的距离，定义一个 **惊讶度 `r`**。

再根据：

```text
g = exp(-gate_gamma * r)
```

得到视觉门控分数 `g`：

- `r` 小，说明视觉先验与触觉过程一致，`g` 接近 1
- `r` 大，说明视觉先验不可信，`g` 接近 0

### 7.2 轨迹采样方式

对于每个样本，脚本会先选择一个起点 `t0`，然后：

1. 取 `z_kin[t0]` 作为起始潜变量 `z0`
2. 用前向模型滚动预测 `rollout_k` 步，得到 `z_pred`
3. 从真实 `z_kin` 中取出后续 `rollout_k` 步，得到 `z_true`

默认参数：

- `rollout_k = 8`
- 训练时 `t0_strategy = random`
- 评估时 `t0_strategy = middle`

### 7.3 惊讶度度量方式

脚本支持两种度量：

#### 方式一：`l2`

对归一化后的潜变量序列做逐步平方误差平均：

```text
r = mean_t || z_pred(t) - z_true(t) ||_2^2
```

#### 方式二：`soft_dtw`

使用 `Soft-DTW` 比较两段潜变量轨迹，允许时间轴上的柔性对齐。

脚本里用纯 PyTorch 实现了一个 **diagonal wavefront** 版本的 batched Soft-DTW 动态规划。虽然仍是 `O(k^2)`，但避免了双重 Python 循环。

默认：

```text
surprise_metric = soft_dtw
soft_dtw_gamma = 0.1
```

这更适合流式接触过程可能存在速度差、相位差的情况。

### 7.4 有效样本保护

如果某个样本可用流式步数不足 `rollout_k + 1`，则脚本不会对它执行门控估计，而是直接返回：

```text
r = 0
g = 1
```

即跳过 gating，避免短序列导致不稳定。

---

## 8. 证据头与 Evidential Accumulation

### 8.1 证据头结构

`EvidenceHeads` 为每个任务都定义了两套 MLP：

- `vis[task]`：从 `z_vis` 生成视觉证据
- `kin[task]`：从 `z_kin` 生成每个流式时刻的触觉证据

每个头的结构相同：

```text
Linear(latent_dim, 128) -> GELU -> Dropout -> Linear(128, num_classes)
```

### 8.2 证据生成

原始输出经过 `softplus` 转成非负证据：

```text
e_vis = softplus(head_vis(z_vis))
e_kin = softplus(head_kin(z_kin))
```

但视觉证据会先乘上门控分数：

```text
e_vis_gated = e_vis * g
```

这里 `g` 是 **样本级** 标量，所以它会整体缩放该样本的视觉证据。

触觉证据则会对所有有效时间步累加：

```text
e_kin_sum = sum_t e_kin(t)
```

### 8.3 Dirichlet 参数

对每个任务，最终 Dirichlet 参数定义为：

```text
alpha = 1 + e_vis_gated + e_kin_sum
```

其中：

- `1` 是非信息先验
- `e_vis_gated` 是经 gate 调整后的视觉证据
- `e_kin_sum` 是整个流式过程累积得到的触觉证据

然后计算：

```text
prob = alpha / sum(alpha)
uncertainty = K / sum(alpha)
```

这里 `K` 是类别数。

因此：

- 总证据越大，`sum(alpha)` 越大，不确定性越低
- 总证据越少，不确定性越高

这正是 Evidential Deep Learning 的核心思想。

---

## 9. 训练目标

总损失定义为：

```text
loss_total = loss_edl + lambda_nce * loss_nce
```

### 9.1 EDL 分类损失

脚本实现的是 Sensoy et al. (2018) 的 evidential classification loss。对每个任务：

```text
L = A + anneal * beta_kl * KL(Dir(tilde_alpha) || Dir(1))
```

其中：

- `A` 是基于 `digamma` 的分类项
- `KL` 项约束不确定情况下不要产生过强错误证据
- `tilde_alpha` 会去除真实类别的证据，用于构造 EDL 中常见的正则形式

脚本会对 `mass / stiffness / material` 三个任务分别计算后再求和。

默认超参数：

- `beta_kl = 1e-3`
- `kl_anneal_epochs = 10`

也就是前几个 epoch 会逐渐打开 KL 正则，减少训练初期不稳定。

### 9.2 InfoNCE 动力学损失

为了让前向动力学模型更有辨识力，脚本额外加入了一个 InfoNCE 损失：

1. 从每条序列中采样若干 `(z_t, z_{t+1})` 正样本对
2. 用 `forward_model(z_vis_detached, z_t)` 预测 `hat(z_{t+1})`
3. 与真实 `z_{t+1}` 做批内对比学习

实现形式：

```text
logits = normalize(hat_next) @ normalize(z_next)^T / temperature
loss_nce = CrossEntropy(logits, diagonal_labels)
```

默认超参数：

- `lambda_nce = 1.0`
- `temperature = 0.1`
- `nce_pairs_per_seq = 1`

这个损失只在训练时启用，评估时关闭。

---

## 10. 关键张量形状

在默认配置下：

- `latent_dim = 256`
- `window_size = 128`
- `step_size = 20`
- `max_tactile_len = 3000`

主要张量形状如下：

| 模块 | 张量形状 |
|---|---|
| 图像输入 | `(B, 3, 224, 224)` |
| `z_vis` | `(B, 256)` |
| 触觉输入 | `(B, 24, 3000)` |
| 滑动窗口 | `(B, T, 128, 24)` |
| 单窗口编码输出 | `(B*T, 256)` |
| `z_kin` | `(B, T, 256)` |
| 视觉证据 | `(B, K)` |
| 每步触觉证据 | `(B, T, K)` |
| 累积触觉证据 | `(B, K)` |
| `alpha_task` | `(B, K)` |
| `prob_task` | `(B, K)` |
| `unc_task` | `(B,)` |
| `surprise` | `(B,)` |
| `gate` | `(B,)` |

其中 `T` 由有效序列长度动态决定，`K` 则取决于当前任务的类别数。

---

## 11. 训练与评估流程

### 11.1 训练阶段

每个 epoch 中：

1. 编码图像得到 `z_vis`
2. 将触觉序列切窗并编码成 `z_kin`
3. 计算 surprise 与 gate
4. 生成视觉证据与触觉证据
5. 累积成 `alpha`
6. 计算 EDL 损失与 InfoNCE 损失
7. 反向传播，做梯度裁剪 `max_norm=1.0`

优化器与学习率策略沿用了本项目常见配置：

- `AdamW`
- 线性 warmup
- 余弦退火

### 11.2 评估阶段

评估时：

- 默认从 checkpoint 自动同步结构参数，如 `latent_dim`、`window_size`、`step_size`、`rollout_k`
- 不再计算 `loss_nce`
- 除分类准确率外，还会统计：
  - `gate` 平均值
  - 每个任务的平均不确定性
  - 正确/错误样本上的平均不确定性

这使得该脚本除了分类，还可以用于分析模型“什么时候不自信”。

---

## 12. 与基础融合模型的区别

相对 `train_fusion.py` 或 `train_fusion_gating2.py`，`train_lpc_ea.py` 的思路有几个本质差别：

1. 它不是整段触觉一次编码，而是滑动窗口流式编码。
2. 它不是 token 级融合，而是潜变量级建模与证据级融合。
3. 视觉的作用不是直接参与注意力融合，而是作为静态先验与动力学条件。
4. 视觉可靠性不是通过跨模态拼接后学习一个 gate，而是通过“未来预测是否正确”间接估计。
5. 输出不是普通 logits，而是 Dirichlet 参数，因此天然带有不确定性估计。
6. 训练损失不只是分类，还加入了前向动力学的对比学习约束。

可以把它理解为：

- `FusionModel` 更像是“静态多模态表征融合”
- `LPC-EA` 更像是“视觉条件下的流式潜变量建模 + 证据累积决策”

---

## 13. 设计动机总结

这个模型试图解决两个问题：

### 13.1 视觉捷径问题

视觉分支在 OOD 条件下可能不可靠，因此不能让它始终以固定权重参与决策。`LPC-EA` 的做法是：

- 不直接相信视觉
- 而是先看“它是否能帮助解释接下来触觉状态如何演化”
- 如果解释不了，就自动降低视觉证据

这比直接基于单帧外观做门控更接近因果直觉。

### 13.2 流式决策与不确定性问题

物理属性感知往往是一个随接触过程逐步积累证据的过程，而不是单步判断。脚本中的 EDL 证据累积机制正对应这一点：

- 每个触觉窗口产生一部分证据
- 证据随时间累积
- 不确定性随着证据总量变化

因此它更适合做“接触越多，越确定”的决策建模。

---

## 14. 实现备注

从当前脚本实现来看，还有几个值得注意的点：

1. `Soft-DTW` 虽然做了向量化，但本质仍是 `O(k^2)`，因此 `rollout_k` 不宜过大。
2. 门控公式实际使用的是模型中的可学习参数 `model.gate_gamma`；命令行里虽然定义了 `--gamma`，但当前前向路径并没有直接使用这个参数。
3. 对短序列样本，脚本会跳过 gating，默认 `g=1`，这是一个稳健性保护策略。
4. 视觉证据只按样本级 gate 缩放，没有做到时间步级或类别级门控。

---

## 15. 一句话概括

`train_lpc_ea.py` 可以概括为：

> 用单帧视觉提供静态先验，用流式触觉窗口编码接触过程，用潜变量未来预测误差评估视觉是否可信，再把视觉和触觉证据累积成带不确定性的 Dirichlet 分类结果。
