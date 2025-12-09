# 多任务分类模型架构详解

本文档详细介绍了项目中两个训练脚本的模型架构和数据处理流程。

---

## 目录

1. [任务定义](#任务定义)
2. [视觉分类器 (Visual ResNet18)](#视觉分类器-visual-resnet18)
3. [触觉分类器 (Tactile Transformer)](#触觉分类器-tactile-transformer)
4. [数据处理对比](#数据处理对比)
5. [训练策略对比](#训练策略对比)
6. [模型导出](#模型导出)

---

## 任务定义

两个模型共享相同的多任务分类目标，同时预测 4 个属性：

| 任务 | 类别数 | 取值 |
|------|--------|------|
| **Class** | 10 | WoodBlock_Native, WoodBlock_Foil, WoodBlock_Red, YogaBrick_Native, YogaBrick_Blue, CardboardBox_Hollow, CardboardBox_SpongeFilled, CardboardBox_RockFilled, CardboardBox_RockFilled_Red, Sponge_Blue |
| **Mass** | 4 | very_low, low, medium, high |
| **Stiffness** | 4 | very_soft, soft, medium, rigid |
| **Material** | 5 | sponge, foam, wood, hollow_container, filled_container |

### 物理属性映射表

```
物体名称                      质量        硬度         材料
─────────────────────────────────────────────────────────────
WoodBlock_Native              medium     rigid        wood
WoodBlock_Foil                medium     rigid        wood
WoodBlock_Red                 medium     rigid        wood
YogaBrick_Native              low        medium       foam
YogaBrick_Blue                low        medium       foam
CardboardBox_Hollow           very_low   soft         hollow_container
CardboardBox_SpongeFilled     very_low   soft         filled_container
CardboardBox_RockFilled       high       rigid        filled_container
CardboardBox_RockFilled_Red   high       rigid        filled_container
Sponge_Blue                   very_low   very_soft    sponge
```

---

## 视觉分类器 (Visual ResNet18)

**脚本**: `train_visual_classifier.py`

### 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                    MultiTaskResNet                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: RGB Image (3, 224, 224)                             │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────┐                   │
│  │     ResNet-18 Backbone              │                   │
│  │     (ImageNet 预训练)                │                   │
│  │                                     │                   │
│  │  Conv1 → BN → ReLU → MaxPool        │                   │
│  │        ↓                            │                   │
│  │  Layer1 (64 channels, 2 blocks)     │                   │
│  │        ↓                            │                   │
│  │  Layer2 (128 channels, 2 blocks)    │                   │
│  │        ↓                            │                   │
│  │  Layer3 (256 channels, 2 blocks)    │                   │
│  │        ↓                            │                   │
│  │  Layer4 (512 channels, 2 blocks)    │                   │
│  │        ↓                            │                   │
│  │  AdaptiveAvgPool2d → Flatten        │                   │
│  └─────────────────────────────────────┘                   │
│                    │                                        │
│                    ▼                                        │
│           Feature Vector (512-dim)                          │
│                    │                                        │
│     ┌──────┬──────┼──────┬──────┐                          │
│     ▼      ▼      ▼      ▼      ▼                          │
│  ┌─────┐┌─────┐┌─────┐┌─────┐                              │
│  │Class││Mass ││Stiff││Mater│  ← 4 个独立线性分类头         │
│  │Head ││Head ││Head ││Head │                              │
│  │(10) ││(4)  ││(4)  ││(5)  │                              │
│  └─────┘└─────┘└─────┘└─────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 关键实现细节

```python
class MultiTaskResNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        # 加载 ImageNet 预训练权重
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 获取特征维度 (512)
        num_features = self.backbone.fc.in_features
        
        # 移除原始分类头，替换为 Identity
        self.backbone.fc = nn.Identity()
        
        # 多任务分类头 (简单线性层)
        self.head_class = nn.Linear(num_features, num_classes)      # 512 → 10
        self.head_mass = nn.Linear(num_features, 4)                 # 512 → 4
        self.head_stiffness = nn.Linear(num_features, 4)            # 512 → 4
        self.head_material = nn.Linear(num_features, 5)             # 512 → 5
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 512]
        return {
            'class': self.head_class(features),
            'mass': self.head_mass(features),
            'stiffness': self.head_stiffness(features),
            'material': self.head_material(features),
        }
```

### 数据处理流程

#### 输入数据
- **来源**: `Plaintextdataset/<ClassName>/<EpisodeID>/visual_anchor.jpg`
- **格式**: RGB 图像，抓取时刻的静态快照
- **数量**: 678 张图像 (10 类)

#### 数据增强 (训练集)

```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 统计量
])
```

#### 数据划分
- **比例**: 80% 训练 / 20% 验证
- **方法**: `random_split` with `seed=42`
- **样本数**: 542 训练 / 136 验证

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 0.001 |
| 损失函数 | CrossEntropyLoss |
| Batch Size | 32 |
| Epochs | 10 |

### 损失函数加权

```python
loss = loss_class + 0.5 * (loss_mass + loss_stiffness + loss_material)
```

主任务 (class) 权重为 1.0，辅助任务权重为 0.5。

---

## 触觉分类器 (Tactile Transformer)

**脚本**: `train_tactile_transformer.py`

### 模型架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    MultiTaskTransformer                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Tactile Sequence (B, T=256, D=24)                        │
│         6 joints × 4 channels (position, load, current, velocity)│
│                    │                                             │
│                    ▼                                             │
│  ┌────────────────────────────────┐                             │
│  │   Input Projection (Linear)   │                             │
│  │        24 → 128 (d_model)     │                             │
│  └────────────────────────────────┘                             │
│                    │                                             │
│                    ▼                                             │
│  ┌────────────────────────────────┐                             │
│  │    Prepend [CLS] Token        │  ← 可学习的分类 token         │
│  │    (B, 256, 128) → (B, 257, 128)                             │
│  └────────────────────────────────┘                             │
│                    │                                             │
│                    ▼                                             │
│  ┌────────────────────────────────┐                             │
│  │   Sinusoidal Positional Enc   │  ← 正弦位置编码              │
│  │      + Dropout (0.1)          │                             │
│  └────────────────────────────────┘                             │
│                    │                                             │
│                    ▼                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Transformer Encoder (4 layers)                 │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  Multi-Head Self-Attention (4 heads)                 │  │ │
│  │  │  d_model=128, d_k=d_v=32                             │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  │                         ↓                                   │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  Feed-Forward Network                                │  │ │
│  │  │  128 → 256 → 128 (GELU activation)                   │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  │                    (× 4 layers)                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                    │                                             │
│                    ▼                                             │
│           Extract [CLS] Token Output                             │
│                    │                                             │
│                    ▼                                             │
│  ┌────────────────────────────────┐                             │
│  │       Layer Normalization     │                             │
│  └────────────────────────────────┘                             │
│                    │                                             │
│     ┌──────┬──────┼──────┬──────┐                               │
│     ▼      ▼      ▼      ▼      ▼                               │
│  ┌─────┐┌─────┐┌─────┐┌─────┐                                   │
│  │Class││Mass ││Stiff││Mater│  ← 4 个 MLP 分类头                │
│  │Head ││Head ││Head ││Head │                                   │
│  └─────┘└─────┘└─────┘└─────┘                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 分类头结构

**Class Head** (主任务，更大容量):
```
Linear(128, 128) → GELU → Dropout(0.1) → Linear(128, 10)
```

**Mass / Stiffness / Material Head** (辅助任务):
```
Linear(128, 64) → GELU → Dropout(0.1) → Linear(64, num_classes)
```

### 关键实现细节

#### 1. 位置编码 (Sinusoidal Positional Encoding)

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度: cos
        
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

#### 2. [CLS] Token 机制

```python
# 可学习的 CLS token
self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
nn.init.trunc_normal_(self.cls_token, std=0.02)

# Forward 中拼接
cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
x = torch.cat([cls, x], dim=1)          # (B, 1+T, d_model)
```

### 数据处理流程

#### 输入特征

| 通道组 | 维度 | 描述 |
|--------|------|------|
| joint_position_profile | (T, 6) | 6 个关节的位置 (度) |
| joint_load_profile | (T, 6) | 6 个关节的负载 |
| joint_current_profile | (T, 6) | 6 个关节的电流 (mA) |
| joint_velocity_profile | (T, 6) | 6 个关节的速度 |

**总特征维度**: 6 joints × 4 channels = **24 维**

**关节顺序**: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

#### 滑动窗口采样 (关键技术)

```
原始序列长度: ~2800+ 时间步
窗口长度 (context_len): 256
滑动步长 (stride): 64
每个 episode 生成窗口数: ~40 个
```

```
Episode 数据 (T=2900):
├─────────────────────────────────────────────────────────────┤
│▓▓▓▓▓▓▓▓│                                                    │ Window 0: [0:256]
│    │▓▓▓▓▓▓▓▓│                                               │ Window 1: [64:320]
│        │▓▓▓▓▓▓▓▓│                                           │ Window 2: [128:384]
│            ...                                              │ ...
│                                              │▓▓▓▓▓▓▓▓│    │ Window N: [2644:2900]
├─────────────────────────────────────────────────────────────┤
```

#### ⚠️ 关键：Episode 级别数据划分 (防止数据泄漏)

```python
# ❌ 错误做法：先切窗口，再划分
all_windows = generate_windows(all_episodes)
train, val = random_split(all_windows)  # 同一 episode 的重叠窗口可能进入不同集合！

# ✅ 正确做法：先划分 episode，再分别切窗口
train_episodes, val_episodes = random_split(all_episodes, [0.8, 0.2])
train_windows = generate_windows(train_episodes)  # 仅来自训练 episode
val_windows = generate_windows(val_episodes)      # 仅来自验证 episode
```

#### 数据预加载

所有触觉数据在训练开始前预加载到内存，避免训练时频繁读取磁盘：

```python
def _preload_all_data(self):
    self.cached_features = []
    for pkl_path, _ in tqdm(self.episodes, desc="Preloading"):
        features = self._load_and_extract_features(pkl_path)
        self.cached_features.append(features)
```

- 训练集内存占用: ~140 MB
- 验证集内存占用: ~35 MB

#### 数据标准化

使用训练集的全局统计量进行 Z-score 标准化：

```python
mean = all_train_features.mean(axis=0)  # (24,)
std = all_train_features.std(axis=0) + 1e-8
normalized = (features - mean) / std
```

**重要**: 验证集使用训练集的 mean/std，防止信息泄漏。

#### 数据增强 (训练集)

```python
def _augment_window(self, window: np.ndarray) -> np.ndarray:
    # 1. 随机高斯噪声 (50% 概率)
    if random.random() < 0.5:
        noise_scale = 0.02 * (np.abs(window).mean() + 1e-8)
        noise = np.random.randn(*window.shape) * noise_scale
        window = window + noise
    
    # 2. 随机幅度缩放 (30% 概率)
    if random.random() < 0.3:
        scale = random.uniform(0.95, 1.05)
        window = window * scale
    
    # 3. 随机通道 dropout (10% 概率，模拟传感器故障)
    if random.random() < 0.1:
        drop_idx = random.randint(0, 23)
        window[:, drop_idx] = 0
    
    return window
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| Weight Decay | 0.01 |
| 损失函数 | CrossEntropyLoss |
| Batch Size | 32 (推荐 512 以提高 GPU 利用率) |
| Epochs | 50 |
| 梯度裁剪 | max_norm=1.0 |
| 学习率调度 | OneCycleLR (with warmup) |

### 评估指标

#### Window-Level Accuracy
对每个滑动窗口独立计算准确率。

#### Episode-Level Accuracy (推荐)
对同一 episode 的所有窗口预测进行**多数投票**，得到 episode 级别的最终预测：

```python
def compute_episode_accuracy(all_preds, all_labels, all_ep_indices):
    episode_preds = defaultdict(list)
    
    # 按 episode 分组
    for i, ep_idx in enumerate(all_ep_indices):
        episode_preds[ep_idx].append(all_preds[i])
    
    # 多数投票
    for ep_idx, preds in episode_preds.items():
        majority_pred = Counter(preds).most_common(1)[0][0]
        # 比较 majority_pred 与真实标签
```

Episode-Level Accuracy 更能反映实际部署场景下的性能。

---

## 数据处理对比

| 特性 | Visual ResNet18 | Tactile Transformer |
|------|-----------------|---------------------|
| **输入类型** | 静态图像 | 时间序列 |
| **输入维度** | (3, 224, 224) | (256, 24) |
| **每 episode 样本数** | 1 | ~40 (滑动窗口) |
| **总样本数** | 678 | ~27,000 |
| **预处理** | ImageNet 归一化 | Z-score 标准化 |
| **数据增强** | 裁剪、翻转、颜色抖动 | 噪声、缩放、通道 dropout |
| **划分方式** | Episode 级别 | Episode 级别 (窗口前划分) |

---

## 训练策略对比

| 特性 | Visual ResNet18 | Tactile Transformer |
|------|-----------------|---------------------|
| **预训练** | ImageNet 权重 | 从头训练 |
| **优化器** | Adam | AdamW |
| **学习率** | 1e-3 | 1e-4 |
| **调度器** | 无 | OneCycleLR |
| **梯度裁剪** | 无 | max_norm=1.0 |
| **正则化** | Dropout (backbone 内置) | Dropout + Weight Decay |
| **参数量** | ~11.2M | ~577K |

---

## 模型导出

两个脚本都支持导出为 ONNX 格式：

```bash
# 视觉模型
python train_visual_classifier.py --eval-only --multitask --export-onnx --checkpoint <path>

# 触觉模型
python train_tactile_transformer.py --eval-only --export-onnx --checkpoint <path>
```

### ONNX 输出格式

**输入**:
- Visual: `input_image` (batch, 3, 224, 224)
- Tactile: `tactile_sequence` (batch, 256, 24)

**输出** (4 个 logits):
- `class_logits` (batch, 10)
- `mass_logits` (batch, 4)
- `stiffness_logits` (batch, 4)
- `material_logits` (batch, 5)

---

## 命令行参数速查

### train_visual_classifier.py

```bash
python train_visual_classifier.py \
    --dataset_root Plaintextdataset \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --output_dir learn_PyBullet/outputs/visual_resnet \
    --multitask \           # 启用多任务学习
    --no-augment \          # 禁用数据增强
    --eval-only \           # 仅评估模式
    --checkpoint <path> \   # 加载检查点
    --export-onnx           # 导出 ONNX
```

### train_tactile_transformer.py

```bash
python train_tactile_transformer.py \
    --dataset_root Plaintextdataset \
    --epochs 50 \
    --batch_size 512 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --output_dir learn_PyBullet/outputs/tactile_transformer \
    --context_len 256 \     # 窗口长度
    --d_model 128 \         # Transformer 隐藏维度
    --n_heads 4 \           # 注意力头数
    --n_layers 4 \          # Transformer 层数
    --no-augment \          # 禁用数据增强
    --resume \              # 恢复训练
    --eval-only \           # 仅评估模式
    --checkpoint <path> \   # 加载检查点
    --export-onnx           # 导出 ONNX
```

---

## 附录：模型参数统计

### Visual ResNet18 (MultiTask)

| 组件 | 参数量 |
|------|--------|
| ResNet-18 Backbone | 11,176,512 |
| Class Head (512→10) | 5,130 |
| Mass Head (512→4) | 2,052 |
| Stiffness Head (512→4) | 2,052 |
| Material Head (512→5) | 2,565 |
| **Total** | **~11.2M** |

### Tactile Transformer

| 组件 | 参数量 |
|------|--------|
| Input Projection (24→128) | 3,200 |
| CLS Token | 128 |
| Transformer Encoder (4 layers) | ~530,000 |
| Layer Norm | 256 |
| Class Head (128→128→10) | 17,674 |
| Mass Head (128→64→4) | 8,516 |
| Stiffness Head (128→64→4) | 8,516 |
| Material Head (128→64→5) | 8,581 |
| **Total** | **~577K** |

---

*文档更新日期: 2025-12-09*

