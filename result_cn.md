# 视触觉融合模型 - 实验结果汇总

## 1. 模型概览

本文档汇总了 **5 个模型** 在机器人抓取物理属性预测任务（质量、刚度、材质）上的实验结果。

### 1.1 模型索引

| 编号 | 模型名称 | 训练脚本 | 模型架构 | 输入模态 |
|---|---|---|---|---|
| A | **融合模型（双模态）** | `scripts/train_fusion.py` | FusionModel | 视觉 + 触觉 |
| B | **纯视觉模型（独立）** | `scripts/train_vision.py` | VisionOnlyModel | 仅视觉 |
| C | **纯触觉模型（独立）** | `scripts/train_tactile.py` | TactileOnlyModel | 仅触觉 |
| D | **融合模型（屏蔽触觉）** | `scripts/train_fusion.py --block_modality tactile` | FusionModel | 仅视觉（触觉置零） |
| E | **融合模型（屏蔽视觉）** | `scripts/train_fusion.py --block_modality visual` | FusionModel | 仅触觉（视觉置零） |

### 1.2 架构详情

#### 模型 A / D / E：FusionModel（`train_fusion.py`）

- **视觉编码器**：ResNet18（ImageNet 预训练，默认冻结）→ 1×1 Conv2d 投影（512 → fusion_dim）
- **触觉编码器**：3 层 Conv1d（24 → 64 → 128 → fusion_dim），stride=2，含 BatchNorm + ReLU
- **Token 序列**：[CLS] + 49 个视觉 token（7×7）+ 375 个触觉 token（3000/8）= **425 个 token**
- **融合层**：Transformer Encoder（4 层，8 头，d_model=256，FFN=1024，GELU，dropout=0.1）
- **分类头**：3 个 MLP（Linear(256,128) → GELU → Dropout → Linear(128, n_classes)）
  - 质量：4 类 | 刚度：4 类 | 材质：5 类
- **位置编码**：可学习参数，形状 (1, 425, 256)，硬编码
- **填充掩码**：通过 3 次 max_pool1d 下采样处理变长触觉序列

#### 模型 B：VisionOnlyModel（`train_vision.py`）

- **视觉编码器**：与 FusionModel 相同（ResNet18 + 1×1 Conv2d）
- **Token 序列**：[CLS] + 49 个视觉 token = **50 个 token**
- **Transformer + 分类头**：与 FusionModel 架构完全相同
- **位置编码**：可学习参数，形状 (1, 50, 256)，通过 dummy forward 动态推断
- **无填充掩码**（视觉输入长度固定）
- **额外功能**：支持可配置的 `--image_size`（默认 224）

#### 模型 C：TactileOnlyModel（`train_tactile.py`）

- **触觉编码器**：与 FusionModel 相同（3 层 Conv1d）
- **Token 序列**：[CLS] + 375 个触觉 token = **376 个 token**
- **Transformer + 分类头**：与 FusionModel 架构完全相同
- **位置编码**：可学习参数，形状 (1, 376, 256)，通过 dummy forward 动态推断
- **填充掩码**：与 FusionModel 处理方式相同（3 次 max_pool1d 下采样）

### 1.3 统一训练策略

所有 5 个模型使用完全相同的训练超参数，以确保公平对比：

| 参数 | 值 |
|---|---|
| 优化器 | AdamW（lr=1e-4, weight_decay=0.01） |
| 学习率调度 | 线性 Warmup（5 个 epoch）+ 余弦退火 |
| 损失函数 | CrossEntropyLoss（3 个任务等权相加） |
| 梯度裁剪 | max_norm=1.0 |
| 训练轮数 | 50 |
| 批大小 | 16 |
| 随机种子 | 42 |
| 最佳模型选择 | 3 个任务验证集平均准确率最高 |

---

## 2. 主要结果

### 2.1 Test 集（分布内，68 个样本）

| 模型 | 质量 Acc | 刚度 Acc | 材质 Acc | **平均 Acc** | 平均 Macro F1 | Loss |
|---|---|---|---|---|---|---|
| **A. 融合（双模态）** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | 0.051 |
| B. 纯视觉 | 91.18% | 91.18% | 97.06% | 93.14% | 0.9177 | 0.597 |
| C. 纯触觉 | 94.12% | 94.12% | 91.18% | 93.14% | 0.9041 | 1.034 |
| D. 融合（屏蔽触觉） | 95.59% | 95.59% | 98.53% | 96.57% | 0.9586 | 0.666 |
| E. 融合（屏蔽视觉） | 94.12% | 94.12% | 89.71% | 92.65% | 0.8939 | 0.666 |

### 2.2 OOD Test 集（分布外，120 个样本）

| 模型 | 质量 Acc | 刚度 Acc | 材质 Acc | **平均 Acc** | 平均 Macro F1 | Loss |
|---|---|---|---|---|---|---|
| **A. 融合（双模态）** | **97.50%** | **96.67%** | **96.67%** | **96.94%** | **0.8262** | 0.253 |
| B. 纯视觉 | 22.50% | 23.33% | 22.50% | 22.78% | 0.1215 | 15.168 |
| C. 纯触觉 | 80.83% | 80.83% | 80.83% | 80.83% | 0.6671 | 2.997 |
| D. 融合（屏蔽触觉） | 25.00% | 25.00% | 25.00% | 25.00% | 0.1312 | 20.233 |
| E. 融合（屏蔽视觉） | 77.50% | 77.50% | 77.50% | 77.50% | 0.6136 | 2.476 |

### 2.3 训练收敛情况

| 模型 | 最佳 Epoch | 最佳验证集平均 Acc | 最终训练 Loss |
|---|---|---|---|
| A. 融合（双模态） | 6 / 50 | 100.00% | 0.0010 |
| B. 纯视觉 | 14 / 50 | 100.00% | 0.0042 |
| C. 纯触觉 | 33 / 50 | 93.63% | 0.0083 |
| D. 融合（屏蔽触觉） | 36 / 50 | 100.00% | 0.0021 |
| E. 融合（屏蔽视觉） | 16 / 50 | 93.14% | 0.0087 |

> **注意**：模型 D（融合模型屏蔽触觉）在训练过程中出现严重的不稳定性，验证集准确率在 ~19% 和 ~100% 之间剧烈震荡。这是由于 375 个全零触觉 token 对 Transformer 自注意力机制造成了干扰。最佳 epoch（36）是在短暂的稳定窗口中达到的。

---

## 3. 逐任务分析（OOD Test 集）

### 3.1 质量分类（4 类）

| 模型 | very_low | low | medium | high | 准确率 |
|---|---|---|---|---|---|
| A. 融合 | P:0.94 R:1.00 | P:1.00 R:0.90 | P:0.97 R:1.00 | P:1.00 R:1.00 | **97.50%** |
| B. 纯视觉 | P:0.00 R:0.00 | P:0.00 R:0.00 | P:0.45 R:0.90 | P:0.00 R:0.00 | 22.50% |
| C. 纯触觉 | P:0.57 R:1.00 | P:1.00 R:0.23 | P:1.00 R:1.00 | P:1.00 R:1.00 | 80.83% |

### 3.2 刚度分类（4 类）

| 模型 | very_soft | soft* | medium | rigid | 准确率 |
|---|---|---|---|---|---|
| A. 融合 | P:1.00 R:1.00 | P:0.00 R:0.00 | P:1.00 R:0.87 | P:0.97 R:1.00 | **96.67%** |
| B. 纯视觉 | P:0.00 R:0.00 | P:0.00 R:0.00 | P:0.00 R:0.00 | P:0.32 R:0.47 | 23.33% |
| C. 纯触觉 | P:1.00 R:1.00 | P:0.00 R:0.00 | P:1.00 R:0.23 | P:1.00 R:1.00 | 80.83% |

> *\*soft 类在 OOD 测试集中无样本（support=0）*

### 3.3 材质分类（5 类）

| 模型 | sponge | foam | wood | hollow_c* | filled_c | 准确率 |
|---|---|---|---|---|---|---|
| A. 融合 | P:1.00 R:1.00 | P:1.00 R:0.87 | P:0.94 R:1.00 | P:0.00 R:0.00 | P:0.97 R:1.00 | **96.67%** |
| B. 纯视觉 | P:0.00 R:0.00 | P:0.00 R:0.00 | P:0.45 R:0.90 | P:0.00 R:0.00 | P:0.00 R:0.00 | 22.50% |
| C. 纯触觉 | P:1.00 R:1.00 | P:1.00 R:0.23 | P:1.00 R:1.00 | P:0.00 R:0.00 | P:0.64 R:1.00 | 80.83% |

> *\*hollow_container 类在 OOD 测试集中无样本（support=0）*

---

## 4. 核心发现

### 4.1 融合模型在分布内和分布外均取得最佳性能

- 分布内（Test）：100%（完美）vs 93.14%（最佳单模态）
- 分布外（OOD）：96.94% vs 80.83%（最佳单模态，纯触觉）
- OOD 上的提升幅度（+16.1 个百分点）远大于分布内的提升（+6.9 个百分点），表明融合在面对未见过的物体时提供了最大的泛化优势。

### 4.2 视觉过拟合物体外观，在 OOD 上完全失效

- 视觉在分布内表现优异（93.14%），但在 OOD 上崩溃至随机水平（22.78%）。
- 独立纯视觉模型（22.78%）和融合模型屏蔽触觉（25.00%）均出现此崩溃，证实这是视觉模态本身的属性，而非模型结构的问题。
- 在 OOD 上，视觉模型将所有样本预测为单一类别（如质量全部预测为 "medium"，材质全部预测为 "wood"），表明视觉特征完全无法泛化到未见过的物体。

### 4.3 触觉提供稳健的泛化能力但存在局限

- 纯触觉模型在 OOD 上维持 80.83%（vs 分布内 93.14%），展现出有意义但不完美的泛化能力。
- 触觉在 OOD 上持续在以下类别上表现不佳：**低质量 low**（Recall 23%）、**中等刚度 medium**（Recall 23%）、**泡沫材质 foam**（Recall 23%），将它们与手感相似的类别混淆。

### 4.4 协同互补效应：1 + 1 > 2

- 纯视觉 OOD：22.78%（独立使用无效）
- 纯触觉 OOD：80.83%
- 融合 OOD：96.94%

一个独立使用在 OOD 上完全无效的模态，加入后却大幅提升了整体性能。这证明了**协同互补（synergistic complementarity）**——视觉特征虽然不足以独立分类未见过的物体，但能通过 Transformer 中的跨模态注意力机制，为触觉的模糊判断提供消歧线索。

具体来说，融合模型精确修复了触觉失败的类别：

| 任务 | 失败类别 | 纯触觉 Recall | 融合 Recall | 提升 |
|---|---|---|---|---|
| 质量 | low | 23% | **90%** | +67pp |
| 刚度 | medium | 23% | **87%** | +64pp |
| 材质 | foam | 23% | **87%** | +64pp |

### 4.5 Token 数量不平衡带来结构性偏置

融合模型中视觉 token 49 个 vs 触觉 token 375 个（比例约 1:7.7），在 Transformer 的自注意力机制中产生了天然的结构性偏置，使模型更倾向于依赖触觉信息。这可能部分解释了为何融合模型在 OOD 上的行为模式更接近纯触觉模型，以及视觉的贡献可能被低估。

---

## 5. 训练命令

```bash
# 模型 A：融合模型（双模态）
python visuotactile/scripts/train_fusion.py --mode train --save_dir visuotactile/outputs/fusion_model_clean --no_live_plot

# 模型 B：纯视觉模型（独立）
python visuotactile/scripts/train_vision.py --mode train --save_dir visuotactile/outputs/vision_model_clean --no_live_plot

# 模型 C：纯触觉模型（独立）
python visuotactile/scripts/train_tactile.py --mode train --save_dir visuotactile/outputs/tactile_model_clean --no_live_plot

# 模型 D：融合模型（屏蔽触觉，仅视觉）
python visuotactile/scripts/train_fusion.py --mode train --block_modality tactile --save_dir visuotactile/outputs/fusion_model_visualOnly --no_live_plot

# 模型 E：融合模型（屏蔽视觉，仅触觉）
python visuotactile/scripts/train_fusion.py --mode train --block_modality visual --save_dir visuotactile/outputs/fusion_model_tactileOnly --no_live_plot
```

---

## 6. 输出目录

| 模型 | Checkpoint 与训练历史 | Test 评估结果 | OOD Test 评估结果 |
|---|---|---|---|
| A. 融合 | `outputs/fusion_model_clean/` | `outputs/fusion_model_clean/eval_test/` | `outputs/fusion_model_clean/eval_ood_test/` |
| B. 纯视觉 | `outputs/vision_model_clean/` | `outputs/vision_model_clean/eval_test/` | `outputs/vision_model_clean/eval_ood_test/` |
| C. 纯触觉 | `outputs/tactile_model_clean/` | `outputs/tactile_model_clean/eval_test/` | `outputs/tactile_model_clean/eval_ood_test/` |
| D. 融合（屏蔽触觉） | `outputs/fusion_model_visualOnly/` | `outputs/fusion_model_visualOnly/eval_test_block_tactile/` | `outputs/fusion_model_visualOnly/eval_ood_test_block_tactile/` |
| E. 融合（屏蔽视觉） | `outputs/fusion_model_tactileOnly/` | `outputs/fusion_model_tactileOnly/eval_test_block_visual/` | `outputs/fusion_model_tactileOnly/eval_ood_test_block_visual/` |
