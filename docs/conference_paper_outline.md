# 会议论文大纲：基于低成本机械臂本体触觉的在线视触觉门控融合

## 0. 写作定位

本文大纲基于以下现有材料整理：

- `docs/result_cn.md`
- `docs/fusion_gating_online_v2_analysis.md`
- `docs/fusion_gating_online_model.md`
- `../Tikz/build/paper/main.pdf`
- `outputs/fusion_gating_online_v2/*`
- `outputs/fusion_gating_online_v2_multiseed/meta/multi_seed_summary_gating_online_v2.json`
- `outputs/fusion_gating_online_v2/latency_benchmark/latency_report.md`

其中，`Tikz/build/paper/main.pdf` 提供了一个值得借鉴的 IEEE 风格会议短文骨架，尤其有三个优点：

1. 引言把问题清晰地表述为视觉 `simplicity bias` 与物理验证感知之间的矛盾。
2. 方法部分不只讲网络结构，也讲标准化操作原语，如 `grasping` 和 `lifting`。
3. 评估部分强调对抗式验证，而不仅是普通 train/test 划分。

建议将论文主线统一为：

- 任务设定采用当前结果最完整的版本：`mass=3 类, stiffness=4 类, material=5 类`
- 方法主角采用 `G3 / fusion_gating_online_v2`
- 基线命名尽量继承旧 draft 的顺序：`A=Vision-only`, `B=Tactile-only`, `C=Vanilla Early Fusion`，`G1 / G2 / G3` 作为逐步增强版本

建议论文核心叙事为：

1. 低成本机械臂可以仅依赖相机和舵机内部反馈完成物体物理属性识别。
2. 视觉模态在 OOD 场景中存在严重的 shortcut / simplicity bias，必须进行动态抑制。
3. 通过门控融合、触觉辅助监督和前缀在线训练，可以同时获得高 OOD 泛化和实时在线决策能力。

同时，建议整体写作风格参考 `main.pdf`：

- 摘要和引言使用“问题缺口 -> 物理动机 -> 方法 -> 结果”四段式推进。
- 方法章节把“数据采集动作设计”与“神经网络结构设计”并列呈现。
- 实验章节除常规准确率外，增加“针对视觉误导的对抗评测”表述。
- 行文上保留旧 draft 的两组关键词：`Hypothesis Generator / Physical Verifier` 与 `Physical Override / Cross-Modal Arbitration`。

## 1. 论文题目候选

### 英文题目候选 1

**Prefix-Aware Visuotactile Gating for Real-Time Physical Property Recognition with Low-Cost Robotic Proprioception**

### 英文题目候选 2

**Online Visuotactile Property Recognition via Dynamic Visual Gating and Proprioceptive Tactile Learning**

### 英文题目候选 3

**From Kinematic-Visual Fusion to Prefix-Aware Visuotactile Gating for Robotic Physical-Property Recognition**

### 中文题目候选

**基于低成本机械臂本体触觉的前缀感知视触觉门控融合实时物理属性识别**

## 2. 摘要应包含的核心信息

### 第一部分：问题背景

- 机器人在抓取过程中需要快速判断物体的质量、刚度和材质。
- 高性能触觉系统通常依赖昂贵专用传感器，限制了部署性。
- 本文使用低成本机械臂舵机内部反馈信号作为隐式触觉，与视觉共同建模。

### 第二部分：核心问题

- 直接的视触觉融合虽然在分布内效果很好，但在分布外场景中容易受到视觉 shortcut 的干扰。
- 视觉分支会过拟合物体外观，导致 OOD 泛化不稳定。

### 第三部分：方法

- 提出一个前缀感知的在线视触觉门控框架。
- 通过标准化 `grasping` 与 `lifting` 操作原语，将可观测的信号变化尽可能归因于物体属性而非控制扰动。
- 使用样本级视觉门控抑制不可靠视觉 token。
- 使用触觉辅助监督防止模态懒惰，保证触觉分支始终学习可迁移物理特征。
- 使用随机前缀训练，使模型在抓取尚未结束时即可提前预测。

### 第四部分：结果

- `G2` 在 OOD 上达到 `99.72 ± 0.35%` 的平均准确率。
- `G3` 在多随机种子下达到 `97.61 ± 3.68%` 的 OOD 平均准确率。
- `G3` 在 `40%` 触觉前缀时可达到 `90.56%`，在 `60%` 前缀时达到 `99.72%`。
- 最坏前缀下单次推理平均仅 `2.486 ms`，约 `402.3 Hz`。

### 第五部分：结论

- 低成本本体触觉可作为有效物理属性感知信号。
- 动态门控和辅助监督能够显著提升多模态 OOD 泛化。
- 该方法具备真实机器人闭环部署潜力。

## 3. 引言（Introduction）

### 3.1 研究背景

- 机器人操控不仅需要“看见物体”，还需要理解物体的物理属性。
- 质量、刚度、材质等属性直接影响抓取策略、力控和后续操作。
- 现有工作往往依赖高成本触觉传感器，而大量实际机器人平台只具备基础视觉与本体反馈。

### 3.2 问题动机

- 机械臂舵机的 `position / load / current / velocity` 反馈包含丰富的接触动态信息。
- 如果能够将其作为隐式触觉使用，可以大幅降低系统成本。
- 但简单融合视觉与触觉并不能自然解决 OOD 泛化问题。
- 与旧 draft 一样，建议把方法动机表述为：从依赖精确解析建模的 `analytical robotics`，转向依赖表征学习的 `learning-based kinematic-visual fusion`。

这一段的行文可以参考 `main.pdf` 的表达方式：

- 把视觉定义为一种“提出假设”的感知。
- 把触觉 / 本体反馈定义为一种“验证假设”的感知。
- 顺势引出本文的核心任务不是普通多模态分类，而是跨模态冲突下的物理仲裁。

### 3.3 关键观察

- 纯视觉模型在 OOD 上几乎退化为随机猜测，平均准确率约 `18.00%`。
- 纯触觉模型更稳定，OOD 平均准确率约 `88.89%`。
- 基础融合模型达到 `89.89%`，但仍存在明显方差与视觉误导现象。
- 进一步的推理时消融发现：在部分设置下，屏蔽视觉反而提升 OOD 表现，说明视觉信息可能是噪声源。

### 3.4 本文目标

- 构建一种在低成本硬件上可部署的在线视触觉感知框架。
- 解决视觉 shortcut 导致的 OOD 泛化崩溃。
- 支持抓取过程中的前缀式早期预测。
- 在论文层面回答一个更基础的问题：标准舵机本体反馈是否已经足以承担“物理验证感知”的角色。

### 3.5 本文贡献

1. 提出一种基于低成本机械臂本体反馈的视触觉物理属性识别框架，无需专用高价触觉传感器。
2. 提出视觉门控与触觉辅助监督结合的方法，显著抑制视觉 shortcut 并提升 OOD 稳定性。
3. 提出前缀感知在线训练策略，使模型在部分触觉序列下即可做出高精度预测。
4. 在真实数据集上验证了该方法兼具高泛化能力与实时控制可行性。

## 4. 相关工作（Related Work）

### 4.1 机器人中的物理属性感知

- 介绍基于视觉、触觉和多模态的物理属性推断任务。
- 强调质量、刚度、材质分类在机器人操作中的价值。

### 4.2 低成本触觉与本体感觉式触觉

- 回顾通过关节电流、力矩、负载等内部反馈估计接触状态的研究。
- 强调本文采用普通舵机反馈构建触觉表征，与专用触觉硬件路线不同。

### 4.3 多模态融合与动态模态路由

- 回顾视觉-触觉融合、gating、mixture-of-experts、uncertainty-aware fusion 等方向。
- 引出本文方法的独特性：不是简单加权，而是面向 OOD 风险的动态视觉抑制。

### 4.4 在线 / 流式机器人感知

- 介绍 prefix prediction、early classification、streaming inference 在机器人中的意义。
- 说明本文不是严格增量缓存式结构，而是前缀重算式在线预测。

## 5. 问题定义与系统设置（Problem Formulation and System Setup）

### 5.1 输入输出定义

- 输入：
  - 一张抓取前 RGB 图像 `image`
  - 一段抓取过程中的 24 通道触觉时间序列 `tactile`
- 输出：
  - `mass`：3 类
  - `stiffness`：4 类
  - `material`：5 类

### 5.2 机器人平台与感知来源

- 相机提供抓取前静态外观信息。
- 6 个关节的内部反馈构成 24 维触觉通道：
  - position
  - load
  - current
  - velocity

### 5.3 数据划分与 OOD 定义

- 训练集、验证集和测试集按 episode 组织。
- OOD 测试集由训练中未见过的物体组成。
- 需要明确强调：本文关注的是对象级泛化，而不是随机样本级泛化。

### 5.4 挑战

- 视觉分支容易记住颜色、纹理和形状等外观线索，而非物理属性。
- 小数据场景下，多模态训练容易形成“谁强学谁”，导致触觉分支偷懒。
- 在线场景要求模型在不完整触觉序列下提前决策。
- 若不先标准化动作原语，舵机信号中的变化会混入控制轨迹差异，削弱物理属性的可学习性。

### 5.5 建议补充：标准化操作原语

参考 `main.pdf`，正文中建议不要只写“收集了抓取数据”，而要明确说明为什么操作动作本身是标准化的。

可以拆成两个子段：

- **Standardized Grasping**
  - 说明采用位置控制或阻抗式抓取而非复杂力控。
  - 强调接触阶段的电流、负载、位置误差随物体刚度变化而变化。
  - 给出“刚度信息体现在时序演化而不是单帧值”的解释。
- **Standardized Lifting**
  - 说明抓取后采用统一的抬升动作与速度轮廓。
  - 强调匀速抬升阶段的额外负载与物体质量相关。
  - 将“质量可观测性”从数据层面说清楚。

这一部分很重要，因为它能让评审相信：

- 触觉信号并非偶然可用；
- 物理属性确实被编码在舵机反馈里；
- 你的学习问题是有物理可观测性支撑的。

### 5.6 建议显式写出的 Methodology Overview

这一小段最好直接继承 `main.pdf` 的逻辑，用一句核心假设统领后文：

- 为了学习 `f: X -> Y`，需要让信号方差尽可能来自物体属性，而不是来自机械臂控制策略本身。
- 因此，标准化动作原语不是实验细节，而是让学习问题成立的必要条件。
- 这样后文的 `gating`、`auxiliary supervision`、`prefix-aware learning` 才不是孤立技巧，而是建立在“物理可观测 + 统计可学习”前提上的三步增强。

## 6. 方法（Method）

### 6.1 Methodology Overview 与基础融合骨架

- 视觉分支：冻结的 `ResNet18 + 1x1 Conv`，输出 49 个视觉 token。
- 触觉分支：3 层 `Conv1d + BN + ReLU`，作为可学习的时序 tokenizer，输出触觉 token 序列。
- 融合主干：`[CLS] + visual tokens + tactile tokens` 输入 4 层 Transformer Encoder。
- 输出头：分别预测 `mass / stiffness / material`。
- 推荐在文中明确指出：冻结视觉 backbone 的一个副作用是保留其对外观 shortcut 的偏好，从而迫使融合模块学会在冲突样本中执行 `Physical Override`。

这一节可借鉴 `main.pdf` 的表述方式，把两种模态赋予不同角色：

- 视觉分支是 `Hypothesis Generator`
- 触觉分支是 `Physical Verifier`

这种写法的好处是：

- 它自然衔接后文的 gating 设计；
- 它能强化“为什么视觉要被动态抑制”这一研究动机；
- 它比单纯写 encoder 名称更有论文叙事张力。

### 6.2 视觉 shortcut 问题分析

- 说明基础模型在分布内接近满分，但 OOD 下视觉分支会误导决策。
- 将该现象定义为视觉 shortcut 或 simplicity bias。
- 由此引出需要动态调节视觉贡献，而不是固定融合。
- 可以借用旧 draft 的措辞：目标不是简单“多模态加权”，而是学习一种 `cross-modal arbitration` 机制，在视觉与触觉冲突时优先信任物理验证信号。

### 6.3 样本级视觉门控

- 从视觉全局摘要 `v_global` 和触觉全局摘要 `t_global` 估计门控分数 `g`。
- 用 `g` 对视觉 token 进行连续缩放。
- 当视觉不可靠时，用可学习的 `t_null` 代替部分视觉信息。

建议突出以下解释：

- `g` 不是简单二值开关，而是连续抑制系数。
- 门控对象是视觉 token，而不是最终 logits。
- 这样可以在 token 级别减少视觉噪声对 Transformer 融合的污染。

### 6.4 触觉辅助监督

- 仅使用门控仍会遇到 `modality laziness`。
- 由于视觉在训练早期更强，模型可能让触觉编码器停止学习深层物理特征。
- 因此增加三个触觉辅助头：
  - `aux_mass`
  - `aux_stiffness`
  - `aux_material`
- 使用辅助损失强制触觉分支保持可分性。

这一节建议明确表达：

- G1 的问题不是“门控无效”，而是“门控没有可靠触觉特征可用”。
- G2 的核心价值是让门控在 OOD 条件下有真正可依赖的触觉表征。
- 这一段和旧 draft 的 Baseline B 可以自然衔接：`tactile-only` 不一定语义最强，但它证明了物理属性在本体反馈中是可观测、可分类的。

### 6.5 前缀感知在线训练

- 在 G2 的基础上加入 prefix-aware training，形成 G3。
- 训练时随机截断触觉有效长度，仅保留当前前缀。
- 评估时固定测量 `10% / 20% / 40% / 60% / 80% / 100%` 前缀。

关键超参数可以写为：

- `online_train_prob = 0.6`
- `online_min_prefix_ratio = 0.4`
- `min_prefix_len = 64`
- `lambda_reg = 0.1`
- `visual_drop_prob = 0.1`

### 6.6 训练目标

- 主任务分类损失：三任务交叉熵之和。
- 触觉辅助损失：三任务辅助分类损失。
- 门控正则：基于 entropy 的门控约束，避免门控过早饱和。

### 6.7 方法章节推荐结构（更贴近 `main.pdf`）

如果要按 IEEE 风格压缩成 4 到 6 页短文，推荐方法章节按下面顺序排：

1. Methodology Overview
2. Standardized Grasping
3. Standardized Lifting
4. Data Acquisition
5. Data Processing
6. Architecture & Baselines
7. Gated Fusion with Auxiliary Supervision
8. Prefix-Aware Online Inference
9. Implementation Details

如果希望更强地继承旧 draft 的语言风格，可以在 `Architecture & Baselines` 中显式保留下面三个小标题：

1. Baseline A: Vision-only
2. Baseline B: Proprioceptive-only
3. Baseline C: Early Fusion

这样做的优点是：

- 不会让论文看起来只有“换了个网络”；
- 能把机器人实验设计和模型设计绑在一起；
- 更符合机器人会议对“系统方法完整性”的期待。

## 7. 实验设置（Experiments）

### 7.1 对比方法

- `A`：纯视觉模型（Vision-only）
- `B`：纯触觉模型（Tactile-only / Proprioceptive-only）
- `C`：基础双模态融合（Vanilla Early Fusion）
- `G1`：门控融合（仅 entropy 正则）
- `G2`：门控融合 + 触觉辅助监督
- `G3`：在线前缀门控模型

这样命名的好处是：

- 与 `main.pdf` 的 baseline 顺序基本一致；
- 阅读结果表时更容易形成“单模态诊断 -> 早期融合 -> 门控增强”的递进关系；
- 可以减少同一篇论文里 baseline 编号前后含义不一致的问题。

### 7.2 训练细节

- 优化器：`AdamW`
- 学习率：`1e-4`
- 权重衰减：`0.01`
- Warmup + Cosine 学习率调度
- 多随机种子评估：`42, 123, 456, 789, 2024`

### 7.3 评估指标

- 各任务准确率
- 三任务平均准确率
- 多随机种子均值和标准差
- 平均 gate score
- 前缀曲线下的准确率变化
- 推理延迟和等效 Hz

### 7.4 消融实验设计

- 模态阻断：visual blocked / tactile blocked
- 逐模型对比：A, B, C, G1, G2, G3
- 前缀比例分析
- 多随机种子稳定性分析
- 延迟压测

### 7.5 建议加入：对抗式评测叙事

`main.pdf` 的一个很强的写法是，不把评测只写成普通分类，而是写成“对视觉误导的认知鲁棒性测试”。

你可以沿用这个思路，但不必完全照搬它的命名。建议用下面三类表述：

- **Anchor samples**
  - 训练分布内、视觉与物理属性一致的样本。
- **Trap samples**
  - 外观容易诱导错误先验，但物理属性相反的样本。
- **Conflict samples**
  - 视觉极像某一类对象，但触觉强烈指向另一类属性的样本。

可以把三者在正文里的作用分别写清楚：

- `Anchor` 用于建立视觉先验，说明模型在常规条件下具备基本识别能力。
- `Trap` 用于检测 `spurious correlation`，例如颜色或纹理与质量标签之间的偶然共现。
- `Conflict` 用于测试 `cross-modal arbitration`，这是门控方法最该赢下来的样本类型。

如果当前数据集还没有显式标注这三类，也可以在论文中用更保守的说法：

- “OOD objects with deceptive appearance”
- “cross-modal conflict cases”
- “visually biased failure cases”

这样可以保留 `main.pdf` 的优点，同时避免引入未正式定义的新数据集名。

### 7.6 建议加入：鲁棒性指标

参考 `main.pdf` 中的 `Deception Gap` 思路，本文可以增加一个辅助指标，例如：

- `Visual Deception Gap = Acc_anchor_like - Acc_trap_like`

或者更贴近当前实验的写法：

- 正常融合准确率与 `block_visual` 后准确率之间的差值
- 视觉阻断前后在 OOD 上的性能变化
- 冲突子集上 `block_visual` 与正常融合之间的性能差值，用于衡量模型是否已经学会“主动不信视觉”

这一指标不一定要成为主指标，但可以增强论文对“视觉误导是否被克服”的论证力度。

## 8. 结果（Results）

### 8.1 基础诊断结果：视觉 shortcut 的存在

应先给出 A / B / C 的对比结果：

- `A` 纯视觉：OOD 平均准确率 `18.00 ± 6.16%`
- `B` 纯触觉：OOD 平均准确率 `88.89 ± 2.26%`
- `C` 基础融合：OOD 平均准确率 `89.89 ± 5.69%`

应强调的结论：

- 视觉模态在分布内有效，但在未见物体上严重失效。
- 触觉模态提供主要的泛化能力。
- 直接融合虽然优于单模态，但仍未根治视觉误导和性能方差。
- 这一组结果最好沿用旧 draft 的话语：`A` 暴露 `simplicity bias`，`B` 证明 `physical observability`，`C` 则说明“早期交互”虽然有效，但还缺少稳定的模态仲裁机制。

### 8.2 从 G1 到 G2：解决模态懒惰

关键对比结果：

- `G1`：OOD 平均准确率 `94.94%`，标准差 `± 9.15%`
- `G2`：OOD 平均准确率 `99.72%`，标准差 `± 0.35%`

建议解读：

- 单纯门控确实能提升 OOD 均值，但不稳定。
- 加入触觉辅助监督后，系统同时提升性能和稳定性。
- 这说明触觉分支的独立可分性是成功门控的前提。

这里的写法可以参考 `main.pdf` 中“不是简单提高准确率，而是提升跨模态仲裁能力”的叙事。建议把结果解释为：

- G1 证明了“关闭错误视觉”是有价值的；
- G2 证明了“要让关闭视觉后仍然能做对，触觉分支必须先学好”。

### 8.3 G3：在线前缀模型的完整序列性能

多随机种子结果：

- `test`：`99.71 ± 0.59%`
- `ood_test`：`97.61 ± 3.68%`

单次最佳运行结果：

- `test`：`100.00%`
- `ood_test`：`99.17%`

应强调：

- G3 在引入在线前缀训练后，几乎没有牺牲完整序列性能。
- OOD 表现仍显著高于基础融合模型。

### 8.4 模态诊断：健康的“触觉主导，视觉辅助”

`fusion_gating_online_v2` 在 OOD 上的模态阻断结果：

- 正常融合：`99.17%`
- 屏蔽视觉：`94.44%`
- 屏蔽触觉：`32.50%`

建议结论：

- OOD 条件下真正可靠的是触觉。
- 视觉不再是主导模态，而成为受控辅助信息。
- 这说明模型已摆脱“视觉常开”状态。
- 如果加一段 case study，这里最适合借用旧 draft 的术语：模型在 `conflict cases` 上开始表现出稳定的 `Physical Override`。

### 8.5 在线前缀曲线

推荐直接展示下列关键点：

- `0.1` 前缀：`30.83%`
- `0.2` 前缀：`34.72%`
- `0.4` 前缀：`90.56%`
- `0.6` 前缀：`99.72%`
- `1.0` 前缀：`99.17%`

建议叙事：

- 模型不是“极早期一触即知”，而是“中段迅速收敛”。
- `40%` 前缀已经进入可用区间。
- `60%` 前缀基本完成决策。

### 8.6 实时性结果

延迟压测结果建议单独成段：

- 最坏前缀 `1.0`：`2.486 ms`，约 `402.3 Hz`
- `0.4` 前缀：`2.012 ms`，约 `497.0 Hz`
- 全部前缀调度总平均时延：`12.600 ms`

应强调：

- 虽然模型不是严格增量式 streaming，但前缀重算的计算代价足够低。
- 该延迟远快于典型 `100 Hz` 机器人控制周期。

### 8.7 结果章节推荐小标题

如果希望整体风格更接近 `main.pdf`，结果部分可以考虑采用下面的小标题：

1. Baseline Comparison
2. OOD Robustness Against Visual Deception
3. Effect of Gating and Tactile Auxiliary Supervision
4. Prefix-Aware Online Prediction
5. Real-Time Latency Benchmark

## 9. 讨论（Discussion）

### 9.1 为什么视觉会在 OOD 中失效

- 视觉更容易利用外观相关 shortcut。
- 小样本和固定对象集合会进一步放大这种偏置。
- 物理属性并不总能由外观稳定决定。

### 9.2 为什么 gating 必须结合 auxiliary loss

- 若触觉分支没有被充分训练，门控即使想关闭视觉，也没有强触觉特征可用。
- 因此“会关视觉”与“关完还能正确预测”是两个不同问题。

### 9.3 为什么 G3 的前缀曲线在 `40%` 附近出现跃迁

- 当前训练配置主要暴露了 `40%-100%` 区间的前缀样本。
- 模型被塑造成“中段可用型”在线预测器，而非极短前缀预测器。

### 9.4 局限性

- 当前在线实现是 prefix re-computation，而非状态缓存式增量推理。
- 当前主结果建立在 `mass=3 类` 设定上。
- OOD 仍局限于当前数据集对象集合，尚未覆盖更大规模跨场景变化。

### 9.5 后续工作

- 研究真正的增量式在线推理结构。
- 扩展到更复杂的操作任务，而不仅是单次抓取属性分类。
- 评估跨机器人平台、跨相机、跨抓取策略的迁移能力。

## 10. 结论（Conclusion）

建议结论部分用三层结构：

### 10.1 回答研究问题

- 低成本机械臂是否能在无专用触觉传感器条件下识别物理属性？
- 回答：可以，而且在合理建模下可达到很强 OOD 泛化。

### 10.2 总结方法价值

- 动态视觉门控抑制了不可靠视觉 shortcut。
- 触觉辅助监督解决了模态懒惰问题。
- 前缀在线训练让模型具备抓取中途提前决策能力。

### 10.3 强调部署意义

- 方法在真实机器人数据上验证有效。
- 推理速度满足实时控制需求。
- 具备低成本机器人系统落地价值。

## 11. 建议图表安排

### 图 1：系统总览图

- 机械臂抓取
- 抓取前图像输入
- 舵机反馈形成触觉序列
- 输出三类物理属性

### 图 2：方法结构图

- 视觉分支
- 触觉分支
- gate `g`
- `t_null`
- Transformer 融合
- 三个主任务头和三个辅助头

### 图 3：方法演进图或问题诊断图

- 从 A/B/C 到 G1/G2/G3 的演化逻辑
- 突出 shortcut、modality laziness、prefix learning 三个问题

如果篇幅更紧，图 3 也可以替换为旧 draft 风格的“对抗评测三层示意图”：

- Anchor：视觉与物理一致
- Trap：视觉先验误导
- Conflict：视觉与触觉冲突
- 这样能更直接服务于 `Deception Gap` 和 `Physical Override` 的结果分析

### 图 3 备选方案：标准化操作原语图

如果篇幅允许，建议优先考虑加入一张与 `main.pdf` 风格相近的“动作-信号-属性”示意图：

- 抓取阶段对应刚度线索
- 抬升阶段对应质量线索
- 触觉反馈来源于舵机内部状态

这类图对机器人方向评审通常比抽象网络框图更有说服力。

### 图 4：OOD 前缀准确率曲线

- 可直接复用：
  - `docs/figures/fusion_gating_online_v2/online_prefix_multiseed_average_accuracy.png`

### 图 5：精度-延迟权衡图

- 可直接复用：
  - `docs/figures/fusion_gating_online_v2/prefix_accuracy_latency_tradeoff.png`

### 表 1：任务定义与数据划分

- 输入模态
- 标签空间
- 训练 / 测试 / OOD 测试设置

### 表 2：主结果表

- A / B / C / G1 / G2 / G3 全部纳入
- 展示 test / OOD 平均准确率和标准差
- 在表注里明确：`A=Vision-only`, `B=Tactile-only`, `C=Vanilla Early Fusion`

### 表 3：G3 前缀结果表

- `0.1 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0`
- 对应准确率、gate score、延迟

### 表 4：模态阻断与诊断表

- 正常融合
- block visual
- block tactile

## 12. 写正文时的建议顺序

建议按以下顺序扩写全文：

1. 先写引言和贡献点，明确论文问题。
2. 再写方法部分，统一术语：`shortcut`、`cross-modal arbitration`、`gating`、`auxiliary supervision`、`prefix-aware learning`。
3. 然后写实验设置，固定主设定为 `mass=3 类`。
4. 结果部分按 `A/B/C -> G1/G2 -> G3 -> prefix -> latency` 顺序展开，其中 `A/B/C` 明确对应 `vision-only / tactile-only / vanilla fusion`。
5. 最后补 related work、讨论和结论。

如果最终目标是模仿 `main.pdf` 的 5 页左右会议稿结构，也可以采用下面的压缩版组织：

1. Introduction
2. Methodology
3. Experimental Results
4. Discussion
5. Conclusion

其中：

- `Methodology` 内部再细分动作设计、数据处理、网络结构、在线前缀机制；
- `Experimental Results` 内部再细分主结果、对抗 OOD 结果、前缀曲线和延迟分析。

## 13. 可直接扩写成摘要的结果句

- We show that low-cost proprioceptive feedback from standard robot servos can serve as an effective tactile signal for physical property recognition.
- Our gated visuotactile model suppresses misleading visual evidence under distribution shift and achieves near-perfect OOD generalization.
- With prefix-aware training, the model reaches `90.56%` average accuracy at `40%` tactile progress and `99.72%` at `60%`, while maintaining sub-`2.5 ms` inference latency.
