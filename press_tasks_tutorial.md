# 海绵 vs 木块分类与时序动作预测：从零搭建到训练的完整讲解

本文用“教学生”的方式，带你从数据到模型，理解并上手本仓库中两类任务：
- 帧级二分类（海绵 sponge vs 木块 woodblock）
- 基于低维状态的时序 Transformer 行为克隆（预测动作）

相关代码文件：
- 帧级分类
  - 模型：`learn_PyBullet/models/mlp_classifier.py`
  - 数据：`learn_PyBullet/datasets/parquet_frame_dataset.py`
  - 训练：`learn_PyBullet/train_press_classifier.py`
- 时序 Transformer
  - 模型：`learn_PyBullet/models/press_transformer.py`
  - 训练：`learn_PyBullet/train_press_transformer.py`

---

## 1. 背景与目标

你的数据已按 LeRobot v2.1 规范保存：
- 每个数据集根目录下有 `meta/`（包含 `info.json` 等）和 `data/`（包含 `episode_*.parquet`）。
- 我们关心的“低维状态”特征包括：`observation.position`、`observation.velocity`、`observation.load`、`observation.current` 等（每个都是固定长度向量）。

两个任务：
1) 帧级分类：给出单帧的低维状态，判断是“海绵”还是“木块”。
2) 时序动作预测：给出一个时间窗口内的状态序列，预测当前步的动作（模仿学习）。

---

## 2. 数据结构速览（LeRobot v2.1）

以一个数据集目录为例：
```
<dataset_root>/
 ├─ meta/
 │   ├─ info.json          # 特征信息、fps、路径模板等
 │   └─ ...
 └─ data/
     └─ chunk-000/
         ├─ episode_000000.parquet
         ├─ episode_000001.parquet
         └─ ...
```
- 每个 parquet 的一行就是一帧，列名如 `observation.position`，值是定长浮点列表。
- `meta/info.json` 里的 `features` 告诉我们各个键的 shape，用来计算输入维度。

---

## 3. 两种读取方式（本教程选用第2种）

- 方式A：`LeRobotDataset`（优点：功能丰富；缺点：本地/权限环境不一致时可能触发缓存权限问题）
- 方式B：`pyarrow` 直接读 parquet（本教程分类任务采用）。

我们在 `learn_PyBullet/datasets/parquet_frame_dataset.py` 实现了 `ParquetFrameDataset`：
- 初始化时扫描所有 `episode_*.parquet` 文件；
- 预读每个文件行数，建立前缀和以支持按全局索引访问；
- `__getitem__` 只读取所需列与单行，返回：
  - `x`: 拼接后的状态向量（`float32`）
  - `label`: 类别 id（`long`）

这样设计的好处：
- 轻量、稳定，避免第三方缓存权限；
- 可以灵活组合多个数据源（例如 sponge=0，woodblock=1）。

---

## 4. 帧级二分类任务（基线 MLP）

目标：用低维状态（单帧）判断是海绵还是木块。

步骤拆解：

1) 准备数据（读取 + 打标签）
- 代码：`ParquetFrameDataset`（`learn_PyBullet/datasets/parquet_frame_dataset.py`）
- 构造 `sources=[(sponge_root, label=0), (woodblock_root, label=1)]`
- 选择要用的状态键（默认四个：position/velocity/load/current）并拼接为输入向量。

2) 划分训练/验证集
- 代码：`train_press_classifier.py` 中的 `train_val_split`（随机打乱+按比例切分）

3) 批处理 `collate`
- 将 `list[dict]` 合成为批次张量：
  - `x`: `[B, D]`，`y`: `[B]`

4) 模型（MLP 基线）
- 代码：`MLPClassifier`（`learn_PyBullet/models/mlp_classifier.py`）
- 结构：LayerNorm → 多层（Linear + GELU + Dropout）→ Linear 输出 `num_classes`
- 简洁稳定，适合快速验证可行性。

5) 训练循环
- 损失：交叉熵（CrossEntropyLoss）
- 优化器：AdamW，含轻度权重衰减
- 训练过程：前向 → 计算损失 → 反向传播 → 梯度裁剪 → 更新参数
- 验证：每个 epoch 后计算准确率，若更好则保存最优权重

6) 运行命令示例
```
# 默认目录就是你已有的数据集
python learn_PyBullet/train_press_classifier.py --epochs 5 --batch_size 512

# 显式指定数据集目录
python learn_PyBullet/train_press_classifier.py \
  --sponge_dir data/so101_press/sponge_20251107_201409 \
  --woodblock_dir data/so101_press/woodBlock_20251107_204040

# 自定义输入键
python learn_PyBullet/train_press_classifier.py \
  --state_keys observation.position observation.velocity
```

训练完成后，最佳模型会保存在：`outputs/press_classifier.pt`

---

## 5. 时序 Transformer（预测动作）

目标：输入一个时间窗口内的状态序列，预测当前（最后一帧）的动作。

关键概念：
- 时间窗口（上下文）：长度 `T`，索引为 `[-T+1, ..., 0]`
- `delta_timestamps`：将相对索引转成相对时间（秒），供 `LeRobotDataset` 按时间对齐抽取序列
- `EpisodeAwareSampler`：丢弃每个 episode 末尾的 `T-1` 帧，防止索引越界

模型：`PressTransformer`（`learn_PyBullet/models/press_transformer.py`）
- 输入：`(B, T, state_dim)`，其中 `state_dim` 为各状态键拼接后的长度
- 结构：线性投影 → 正弦位置编码 → TransformerEncoder（多层）→ 取最后 token → MLP 头输出动作
- 损失：MSE（监督模仿学习）

训练脚本：`learn_PyBullet/train_press_transformer.py`
- 从 `meta/info.json` 读取 `fps` 和 `features`，构造 `delta_timestamps`
- 通过 `LeRobotDataset` 输出带时间上下文的序列样本
- 用 `EpisodeAwareSampler` 避免跨 episode 窗口越界

运行示例：
```
python learn_PyBullet/train_press_transformer.py \
  --dataset_dir data/so101_press/sponge_20251107_201409 \
  --epochs 5 --batch_size 64 --context 16
```

---

## 6. 超参数与技巧

- 输入键选择：从 `meta/info.json` 的 `features` 中选择（保持两数据集形状一致）。
- 学习率：`1e-3` 对 MLP 基线通常足够；Transformer 可从 `1e-3 ~ 3e-4` 试起。
- 批大小：低维向量内存开销小，可适当加大；受限于 CPU I/O 时可减小。
- 正则化：适当 `Dropout` 与 `WeightDecay` 对泛化有帮助。
- 归一化：当前示例未做全局统计归一化；如任务难度提升，可对各状态维度按训练集均值方差标准化。

---

## 7. 常见问题（FAQ）

- PermissionError（HF 数据集缓存）：
  - 解决：已将训练脚本默认使用工作区本地缓存 `outputs/hf_cache`，或直接使用 `ParquetFrameDataset`。
- 类别不平衡：
  - 解决：加权损失（`CrossEntropyLoss(weight=...)`），或在 `Sampler` 层面做均衡采样。
- 性能瓶颈（I/O）：
  - 解决：适当增大批量、降低 num_workers 争用；若磁盘成为瓶颈，可考虑分片缓存或合并文件。

---

## 8. 练习与扩展

- 帧级改为“时间窗口分类”：将 `ParquetFrameDataset` 扩展为按窗口拼接序列后再做分类。
- 加入更多状态键：如 `observation.velocity_derived`、`observation.raw_present_speed` 等，比较性能差异。
- 可视化：绘制学习曲线、混淆矩阵；对错误样本做对比分析。
- 实时推理：加载 `outputs/press_classifier.pt`，在新数据上做在线分类或日志分析。

---

## 9. 文件速查

- `learn_PyBullet/models/mlp_classifier.py`：MLP 分类器实现。
- `learn_PyBullet/datasets/parquet_frame_dataset.py`：直接读取 parquet 的帧级数据集实现。
- `learn_PyBullet/train_press_classifier.py`：分类训练脚本（读取 → 划分 → 训练 → 验证 → 保存）。
- `learn_PyBullet/models/press_transformer.py`：序列 Transformer（低维状态 → 动作）。
- `learn_PyBullet/train_press_transformer.py`：时序训练脚本（构造时间窗、采样器与训练循环）。

祝学习顺利！如果你想把分类器也改成“时间窗口 + Transformer”的序列分类版，我可以基于现有组件快速扩展并给出示例代码。

