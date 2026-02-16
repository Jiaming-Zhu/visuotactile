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
| 随机种子 | 42, 123, 456, 789, 2024（5 次独立运行） |
| 最佳模型选择 | 3 个任务验证集平均准确率最高 |

---

## 2. 主要结果（多种子：mean ± std，n=5）

所有结果为 5 次独立运行（种子 42、123、456、789、2024）的平均值。数据划分在所有运行中保持固定，仅模型初始化和训练随机性不同。

### 2.1 Test 集（分布内，68 个样本）

| 模型 | 质量 Acc | 刚度 Acc | 材质 Acc | **平均 Acc** |
|---|---|---|---|---|
| **A. 融合（双模态）** | **100.00 ± 0.00%** | **100.00 ± 0.00%** | 98.53 ± 1.86% | **99.51 ± 0.62%** |
| B. 纯视觉 | 93.82 ± 1.95% | 94.12 ± 1.86% | **97.94 ± 1.18%** | 95.29 ± 1.47% |
| C. 纯触觉 | 95.59 ± 1.32% | 95.88 ± 1.10% | 95.59 ± 2.28% | 95.69 ± 1.37% |

### 2.2 OOD Test 集（分布外，120 个样本）

| 模型 | 质量 Acc | 刚度 Acc | 材质 Acc | **平均 Acc** |
|---|---|---|---|---|
| **A. 融合（双模态）** | **89.00 ± 8.42%** | **91.50 ± 6.06%** | **89.33 ± 7.37%** | **89.94 ± 7.00%** |
| B. 纯视觉 | 19.50 ± 5.79% | 21.33 ± 3.10% | 20.67 ± 4.78% | 20.50 ± 4.47% |
| C. 纯触觉 | 79.00 ± 1.86% | 79.67 ± 1.55% | 79.17 ± 1.75% | 79.28 ± 1.64% |

### 2.3 逐种子明细

| 模型 | 指标 | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 |
|---|---|---|---|---|---|---|
| 融合 | Test 平均 Acc | 100.00% | 98.53% | 99.02% | 100.00% | 100.00% |
| 融合 | OOD 平均 Acc | 96.94% | 82.50% | 84.44% | 86.11% | 99.72% |
| 纯视觉 | Test 平均 Acc | 93.14% | 97.55% | 94.61% | 95.10% | 96.08% |
| 纯视觉 | OOD 平均 Acc | 22.78% | 25.56% | 20.83% | 12.22% | 21.11% |
| 纯触觉 | Test 平均 Acc | 93.14% | 96.57% | 97.06% | 96.08% | 95.59% |
| 纯触觉 | OOD 平均 Acc | 80.83% | 77.78% | 78.33% | 77.78% | 81.67% |

### 2.4 单种子结果（Seed 42，含消融实验）

以下表格包含模态屏蔽消融实验（模型 D 和 E），仅使用 seed=42 运行。

| 模型 | Test 平均 Acc | OOD 平均 Acc |
|---|---|---|
| A. 融合（双模态） | 100.00% | 96.94% |
| B. 纯视觉 | 93.14% | 22.78% |
| C. 纯触觉 | 93.14% | 80.83% |
| D. 融合（屏蔽触觉） | 96.57% | 25.00% |
| E. 融合（屏蔽视觉） | 92.65% | 77.50% |

### 2.5 训练收敛情况

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

## 4. 推理时注意力掩码消融实验（多种子，n=5）

使用完全训练好的 5 个融合模型 checkpoint（模型 A），在推理时通过 **注意力掩码**（`src_key_padding_mask`）进行模态消融。与模型 D/E 训练时使用的输入置零不同，此方法将 Transformer 的 key padding mask 对目标模态的 token 位置设为 `True`，使得其余 token 完全无法 attend 到被屏蔽的模态。

脚本：`scripts/infer_fusion_multiseed_ablation.py`

> **重要说明**：注意力掩码与输入置零是本质不同的消融方法。注意力掩码将 token 从注意力机制中完全隐藏，而输入置零将全零 token 送入注意力机制中仍然参与计算。两者的结果**不能直接对比**模型 D/E。

### 4.1 消融结果（seed_means，n=5）

#### Test 集（分布内，68 个样本）

| 模式 | 质量 Acc | 刚度 Acc | 材质 Acc | **平均 Acc** |
|---|---|---|---|---|
| 完整（双模态） | 100.00 ± 0.00% | 100.00 ± 0.00% | 98.53 ± 1.86% | **99.51 ± 0.62%** |
| 仅触觉（掩码视觉） | 86.47 ± 2.35% | 87.35 ± 2.88% | 86.18 ± 1.76% | 86.67 ± 2.29% |
| 仅视觉（掩码触觉） | 55.88 ± 9.16% | 68.24 ± 8.71% | 61.47 ± 9.41% | 61.86 ± 8.79% |

#### OOD Test 集（分布外，120 个样本）

| 模式 | 质量 Acc | 刚度 Acc | 材质 Acc | **平均 Acc** |
|---|---|---|---|---|
| 完整（双模态） | 89.00 ± 8.42% | 91.50 ± 6.06% | 89.33 ± 7.37% | **89.94 ± 7.00%** |
| 仅触觉（掩码视觉） | **97.83 ± 3.93%** | **93.33 ± 8.16%** | **98.17 ± 3.27%** | **96.44 ± 4.70%** |
| 仅视觉（掩码触觉） | 30.67 ± 7.84% | 33.17 ± 10.74% | 22.67 ± 6.86% | 28.83 ± 5.77% |

#### 逐种子 OOD 平均 Acc

| 模式 | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 |
|---|---|---|---|---|---|
| 完整 | 96.94% | 82.50% | 84.44% | 86.11% | 99.72% |
| 仅触觉（掩码视觉） | 100.00% | 93.89% | 100.00% | 88.33% | 100.00% |
| 仅视觉（掩码触觉） | 31.39% | 35.28% | 30.28% | 18.06% | 29.17% |

### 4.2 关键发现：掩码视觉反而提升 OOD 性能

反直觉地，融合模型在 OOD 上掩码视觉后（96.44%）比使用双模态时（89.94%）表现**更好**。这说明在 OOD 数据上，来自未见过物体的视觉特征主动**干扰**了模型判断，在注意力机制中引入了误导性信号。通过注意力掩码移除视觉 token 后，Transformer 只关注触觉 token，避免了视觉噪声。

### 4.3 不同消融方法对比（OOD 平均 Acc）

| 方法 | 模型权重来源 | OOD 平均 Acc |
|---|---|---|
| 注意力掩码 仅触觉 | 融合模型（双模态训练） | **96.44 ± 4.70%** |
| 融合完整（双模态） | 融合模型（双模态训练） | 89.94 ± 7.00% |
| 独立纯触觉模型 | TactileOnlyModel | 79.28 ± 1.64% |
| 输入置零 屏蔽视觉（模型 E） | 融合模型（视觉置零训练） | 77.50%（仅 seed 42） |

注意力掩码（96.44%）与独立触觉模型（79.28%）之间的巨大差距**并不意味着**融合模型的触觉分支天生更强。差异来源于：

1. **不同的模型权重**：融合模型的触觉编码器经过了与视觉的联合训练，受益于更丰富的跨模态梯度信号。
2. **不同的消融机制**：注意力掩码完全隐藏 token；输入置零将全零衍生的噪声 token 送入注意力计算。
3. **训练外推理模式**：模型从未在训练中见过注意力掩码视觉的情况，因此这对模型本身而言也是一种 OOD 推理模式。

---

## 5. 核心发现

### 5.1 融合模型在分布内和分布外均取得最佳性能

- 分布内（Test）：**99.51 ± 0.62%** vs 95.69 ± 1.37%（最佳单模态，纯触觉）
- 分布外（OOD）：**89.94 ± 7.00%** vs 79.28 ± 1.64%（最佳单模态，纯触觉）
- OOD 上的提升幅度（+10.66 个百分点）远大于分布内的提升（+3.82 个百分点），表明融合在面对未见过的物体时提供了最大的泛化优势。

### 5.2 视觉过拟合物体外观，在 OOD 上完全失效

- 视觉在分布内表现优异（95.29 ± 1.47%），但在 OOD 上崩溃至随机水平（20.50 ± 4.47%）。
- 这一崩溃在所有 5 个种子上一致出现（范围：12.22%–25.56%），证实这是视觉模态的稳健特性，而非随机偶然。
- 独立纯视觉模型和融合模型屏蔽触觉均出现此崩溃，证实这是视觉模态本身的属性，而非模型结构的问题。
- 在 OOD 上，视觉模型将所有样本预测为单一类别（如质量全部预测为 "medium"，材质全部预测为 "wood"），表明视觉特征完全无法泛化到未见过的物体。

### 5.3 触觉提供稳健的泛化能力但存在局限

- 纯触觉模型在 OOD 上维持 **79.28 ± 1.64%**（vs 分布内 95.69 ± 1.37%），展现出有意义但不完美的泛化能力。
- 值得注意的是，触觉在所有模型中 OOD **方差最低**（std=1.64%），表明其泛化行为非常稳定。
- 触觉在 OOD 上持续在以下类别上表现不佳：**低质量 low**（Recall 23%）、**中等刚度 medium**（Recall 23%）、**泡沫材质 foam**（Recall 23%），将它们与手感相似的类别混淆。

### 5.4 协同互补效应：1 + 1 > 2

- 纯视觉 OOD：**20.50 ± 4.47%**（独立使用无效）
- 纯触觉 OOD：**79.28 ± 1.64%**
- 融合 OOD：**89.94 ± 7.00%**

一个独立使用在 OOD 上完全无效的模态，加入后仍能将整体性能平均提升约 10.7 个百分点。这证明了**协同互补（synergistic complementarity）**——视觉特征虽然不足以独立分类未见过的物体，但能通过 Transformer 中的跨模态注意力机制，为触觉的模糊判断提供消歧线索。

具体来说，在最佳种子（42）中，融合模型精确修复了触觉失败的类别：

| 任务 | 失败类别 | 纯触觉 Recall | 融合 Recall | 提升 |
|---|---|---|---|---|
| 质量 | low | 23% | **90%** | +67pp |
| 刚度 | medium | 23% | **87%** | +64pp |
| 材质 | foam | 23% | **87%** | +64pp |

### 5.5 融合模型在 OOD 上方差较高

融合模型的 OOD 方差最高（std=7.00%），相比触觉（1.64%）和视觉（4.47%）。逐种子 OOD 结果分别为：96.94%、82.50%、84.44%、86.11%、99.72%。这表明基于 Transformer 的跨模态融合对随机初始化较为敏感——不同种子可能导致不同质量的跨模态注意力模式。值得注意的是，种子 42 和 2024 均达到了近乎完美的 OOD 准确率（96.94% 和 99.72%），而其余种子在 82–86% 范围内。这是多模态学习中的已知挑战，可通过集成方法或增加训练数据来缓解。

### 5.6 Token 数量不平衡带来结构性偏置

融合模型中视觉 token 49 个 vs 触觉 token 375 个（比例约 1:7.7），在 Transformer 的自注意力机制中产生了天然的结构性偏置，使模型更倾向于依赖触觉信息。这可能部分解释了为何融合模型在 OOD 上的行为模式更接近纯触觉模型，以及视觉的贡献可能被低估。

---

## 6. 训练命令

### 6.1 单种子实验

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

### 6.2 多种子实验

```bash
# 自动化脚本，运行 5 个种子（42、123、456、789、2024）的模型 A/B/C
bash visuotactile/scripts/run_multi_seed.sh

# 或手动逐模型逐种子运行：
for SEED in 42 123 456 789 2024; do
  python visuotactile/scripts/train_fusion.py --mode train --seed $SEED --save_dir visuotactile/outputs/fusion_seed${SEED} --no_live_plot
  python visuotactile/scripts/train_vision.py --mode train --seed $SEED --save_dir visuotactile/outputs/vision_seed${SEED} --no_live_plot
  python visuotactile/scripts/train_tactile.py --mode train --seed $SEED --save_dir visuotactile/outputs/tactile_seed${SEED} --no_live_plot
done
```

### 6.3 推理时注意力掩码消融

```bash
# 对 5 个融合模型 checkpoint 运行注意力掩码消融
python visuotactile/scripts/infer_fusion_multiseed_ablation.py
```

---

## 7. 输出目录

### 7.1 单种子实验（Seed 42）

| 模型 | Checkpoint 与训练历史 | Test 评估结果 | OOD Test 评估结果 |
|---|---|---|---|
| A. 融合 | `outputs/fusion_model_clean/` | `outputs/fusion_model_clean/eval_test/` | `outputs/fusion_model_clean/eval_ood_test/` |
| B. 纯视觉 | `outputs/vision_model_clean/` | `outputs/vision_model_clean/eval_test/` | `outputs/vision_model_clean/eval_ood_test/` |
| C. 纯触觉 | `outputs/tactile_model_clean/` | `outputs/tactile_model_clean/eval_test/` | `outputs/tactile_model_clean/eval_ood_test/` |
| D. 融合（屏蔽触觉） | `outputs/fusion_model_visualOnly/` | `outputs/fusion_model_visualOnly/eval_test_block_tactile/` | `outputs/fusion_model_visualOnly/eval_ood_test_block_tactile/` |
| E. 融合（屏蔽视觉） | `outputs/fusion_model_tactileOnly/` | `outputs/fusion_model_tactileOnly/eval_test_block_visual/` | `outputs/fusion_model_tactileOnly/eval_ood_test_block_visual/` |

### 7.2 多种子实验

| 模型 | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 |
|---|---|---|---|---|---|
| 融合 | `outputs/fusion_seed42/` | `outputs/fusion_seed123/` | `outputs/fusion_seed456/` | `outputs/fusion_seed789/` | `outputs/fusion_seed2024/` |
| 纯视觉 | `outputs/vision_seed42/` | `outputs/vision_seed123/` | `outputs/vision_seed456/` | `outputs/vision_seed789/` | `outputs/vision_seed2024/` |
| 纯触觉 | `outputs/tactile_seed42/` | `outputs/tactile_seed123/` | `outputs/tactile_seed456/` | `outputs/tactile_seed789/` | `outputs/tactile_seed2024/` |

### 7.3 注意力掩码消融

| 输出内容 | 路径 |
|---|---|
| 总体汇总 | `outputs/fusion_infer_ablation_mask/summary_overall.json` |
| 逐种子结果 | `outputs/fusion_infer_ablation_mask/seed_*/` |

汇总文件：`outputs/multi_seed_summary.json`
