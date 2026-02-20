# Visuotactile Fusion Project Context (For AI Agents)

**目标读者**：此文档专为 AI 编程助手（Agent）设计，旨在提供高信息密度的项目背景、架构设计、数据处理流程和核心实验结论。在阅读具体代码前，请务必先浏览本文档以建立全局视野。

---

## 1. 项目概述 (Project Overview)

本项目研究如何使用低成本机械臂完成物体物理属性的感知。
- **核心任务**：基于机械臂的单次抓取动作，融合**视觉**（抓取前的静态 RGB 图像）和**触觉**（抓取过程中的动态时间序列数据），联合预测物体的三个物理属性：
  1. **质量 (Mass)**：4 个类别（very_low, low, medium, high）
  2. **硬度 (Stiffness)**：4 个类别（very_soft, soft, medium, rigid）
  3. **材料 (Material)**：5 个类别（sponge, foam, wood, hollow_container, filled_container）
- **核心创新**：未使用昂贵的专用触觉传感器（如 GelSight），而是直接利用低成本舵机（Feetech STS3215）的内部反馈参数（位置、负载、电流、速度）作为隐式（Proprioceptive）的触觉信号。

---

## 2. 核心模型架构 (Model Architecture)

在 `visuotactile/scripts/train_fusion.py` 中定义的 `FusionModel` 包含三个主要部分：

### 2.1 视觉分支 (Visual Encoder)
- **输入**：RGB 图像 `(Batch, 3, 224, 224)`。
- **结构**：ImageNet 预训练的 **ResNet18**（去除了最后两层：AdaptiveAvgPool 和 FC层）。
- **投影**：接一个 `1x1 Conv2d` 将通道数从 512 降至 `fusion_dim` (默认 256)。
- **输出**：拉平后得到 **49** 个视觉 Token (`7x7` 空间分辨率)。

### 2.2 触觉分支 (Tactile Encoder)
- **输入**：抓取时间序列 `(Batch, 24, SequenceLength)`。
- **结构**：3 层 **1D-CNN** (`Conv1d -> BatchNorm1d -> ReLU`)。每层的 `stride=2`，从而在时间维度上进行了 $2^3 = 8$ 倍的下采样。
- **输出**：对于长度为 3000 的原始序列，下采样后得到 **375** 个触觉 Token（`3000 / 8`）。

### 2.3 融合模块与分类头 (Fusion & Heads)
- **输入序列重组**：将 1 个 `[CLS]` Token，49 个视觉 Token，以及 375 个触觉 Token 拼接（共 425 个 Token），并加上可学习的位置编码（Positional Encoding）。
- **Transformer Encoder**：4 层，8 头，`d_model=256`，激活函数为 GELU。
  - **Mask 处理**：由于触觉序列长度在不同 episode 间可能不同，会对短序列使用 0 填充。在送入 Transformer 前，通过 3 次 `max_pool1d` 生成对应的 `src_key_padding_mask`。
- **分类头**：提取 Transformer 输出序列中的 `[CLS]` Token（即 `x[:, 0, :]`），分别送入三个独立的 MLP 网络（`Linear -> GELU -> Dropout -> Linear`），预测 `mass`, `stiffness` 和 `material`。

---

## 3. 代码仓库结构导航 (Repository Map)

| 目录/文件 | 描述 | Agent 处理指南 |
|---|---|---|
| `scripts/train_fusion.py` | **主训练脚本**，包含双模态融合模型的定义和训练循环。 | **核心代码**。支持通过 `--block_modality` 参数进行模态消融。 |
| `scripts/train_vision.py` | 纯视觉基线模型（VisionOnlyModel）。 | 基于 ResNet18 的独立视觉模型。 |
| `scripts/train_tactile.py` | 纯触觉基线模型（TactileOnlyModel）。 | 基于 1D-CNN + Transformer 的独立触觉模型。 |
| `collect_custom_multimodal.py` | 机械臂数据采集主脚本。 | 在真机环境执行，记录摄像头画面和舵机状态。 |
| `interactive_control_oop.py` | 机械臂交互式控制脚本。 | 提供用于遥操作或调试的控制接口。 |
| `scripts/clean_dataset_ui.py` | 基于 Streamlit 的数据集清洗 UI。 | 用于可视化检查录制数据并剔除异常数据。 |
| `outputs/` | 模型训练日志、检查点 (Checkpoints) 和指标图表。 | 请勿将此目录视为源码，仅供分析实验结果使用。 |

---

## 4. 数据处理与格式 (Data Pipeline)

### 4.1 数据集结构 (`Plaintextdataset/`)
训练和验证集共享统一的目录结构：
```
Plaintextdataset/train/
├── physical_properties.json   # 包含物体的属性标签和 ID 映射字典
├── WoodBlock_Native/          # 物体类别级目录
│   ├── episode_001/           # 单次抓取的数据
│   │   ├── visual_anchor.jpg  # 抓取前的静态照片
│   │   ├── tactile_data.pkl   # 包含时间戳和触觉阵列的 Pickle
│   │   └── metadata.json      # 额外元数据
```

### 4.2 触觉数据通道 (Tactile Channels)
送入模型的触觉数据为 24 维的时序特征，由 6 个关节的 4 种反馈构成：
- `0-5`: Joint Positions (关节位置)
- `6-11`: Joint Loads (关节负载)
- `12-17`: Joint Currents (关节电流)
- `18-23`: Joint Velocities (关节速度)

### 4.3 数据归一化
- 视觉数据：使用标准的 ImageNet 统计量（`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`）标准化。
- 触觉数据：使用预先计算的全局统计量（Z-score），常量定义在 `train_fusion.py` 中的 `TACTILE_STATS` 字典中。

---

## 5. 核心实验结论 (Key Findings - Agent 必读)

在对模型进行迭代、修复或优化时，必须理解以下实验结论（基于 `result.md`）：

1. **强悍的 OOD 泛化能力**：
   - 融合模型在面对**训练中未见过的物体**（Out-Of-Distribution, OOD）时，取得了最佳的性能（平均准确率 ~90%）。纯触觉约为 ~79%，纯视觉仅为 ~20%。
2. **视觉模态存在极强的 "Simplicity Bias"（过度依赖简单特征）**：
   - 视觉模型在分布内（In-Distribution）表现很好（~95%），但在 OOD 数据上完全崩溃（退化为随机猜测 ~20%）。
   - 这表明模型在训练时“走捷径”，过度记住了物体的颜色/纹理（比如看到红色就认为是某种特定的木块），而没有学习到真正的物理属性。
3. **Transformer 掩码消融的惊人发现**：
   - 使用训练好的双模态融合模型在 OOD 测试集上推理时，如果**通过 Attention Mask 故意屏蔽掉所有视觉 Token**，整体平均准确率反而**提升了**（从 89.9% 提升到 96.4%）。
   - **结论**：在遇到未见过的物体时，视觉信息反而构成了“噪声”，误导了模型的决策。
   - **Agent 指导建议**：如果你需要改进模型架构，考虑引入**动态模态路由（Dynamic Modality Routing）** 或 **模态门控（Modality Gating）**，让模型学会在不确定的视觉输入下主动降低视觉分支的权重。