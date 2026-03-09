

# Tactile-Guided Visual Retrieval and Gated Recurrent Fusion

## 触觉引导的视觉读取与门控递归融合模型

该模型面向**实时多模态物理属性识别**任务。核心思想是：

1. 用静态视觉图像提供先验表观信息；
2. 用因果触觉编码器对流式触觉序列进行在线建模；
3. 在每个时刻由当前触觉状态主动查询视觉 token，读取与当前接触阶段最相关的视觉上下文；
4. 再通过门控机制抑制不可信的视觉线索；
5. 最后用递归融合状态持续整合视觉—触觉证据，并输出当前时刻的属性预测。

它本质上是一个**触觉主导、视觉条件读取、门控抑制视觉捷径、支持 anytime prediction 的因果多模态模型**。

---

# 1. 任务定义

给定一次抓取 episode，其输入由两部分组成：

* 一张视觉锚点图像 (x^{(v)})
* 一段随时间到达的触觉序列 (x^{(t)}_{1:T})

其中：

[
x^{(v)} \in \mathbb{R}^{3 \times H \times W}
]

[
x^{(t)}_{1:T} = {x^{(t)}_1, x^{(t)}_2, \dots, x^{(t)}_T}, \quad x^{(t)}_t \in \mathbb{R}^{C_t}
]

模型需要联合预测三个离散物理属性：

* mass：(C_m) 类
* stiffness：(C_s) 类
* material：(C_r) 类

记标签为：

[
y = (y^{(m)}, y^{(s)}, y^{(r)})
]

与传统离线模型不同，本模型在每个时刻 (t) 都输出一个当前预测：

[
\hat y_t = f\big(x^{(v)}, x^{(t)}_{1:t}\big)
]

因此这是一个**在线因果分类**问题，而不是只能在完整 episode 结束后才输出结果的离线分类问题。

---

# 2. 输入与预处理

## 2.1 视觉输入

视觉输入为抓取前或抓取开始时采集的单帧图像：

[
x^{(v)} \in \mathbb{R}^{3 \times H \times W}
]

通常采用如下预处理：

[
\text{Resize} \rightarrow \text{ToTensor} \rightarrow \text{Normalize}
]

在实现上，可令 (H=W=224)，并采用 ImageNet 标准归一化。

视觉输入在整个 episode 内通常视为静态不变，因此只需在 episode 开始时编码一次，后续实时推理中直接复用。

---

## 2.2 触觉输入

触觉输入是一个随时间到达的多通道传感器序列。设原始每个时刻观测为：

[
x_t^{(t)} \in \mathbb{R}^{C_t}
]

其中 (C_t) 表示触觉观测维度，例如可由多组机器人本体传感器拼接而成：

* joint position
* joint load
* joint current
* joint velocity

每个通道先独立做 Z-score 标准化：

[
\tilde x = \frac{x-\mu}{\sigma+\varepsilon}
]

然后按通道拼接形成多维触觉输入。

与离线模型不同，在线模型不要求将序列统一填充到固定长度 (T_{\max})。在实时推理时，系统仅处理当前已经到达的前缀：

[
x^{(t)}_{1:t}
]

因此模型天然支持变长前缀输入。

---

# 3. 模型总体结构

整个模型由 6 个模块组成：

1. 视觉编码器（Visual Encoder）
2. 因果触觉编码器（Causal Tactile Encoder, 例如 Conv+Mamba）
3. 触觉引导的视觉读取模块（Tactile-guided Visual Retrieval）
4. 视觉门控抑制模块（Gated Visual Suppression）
5. 递归融合模块（Recurrent Fusion Module）
6. 多任务分类头（Multi-task Prediction Heads）

其信息流可以概括为：

[
x^{(v)} \rightarrow V
]

[
x^{(t)}_{1:t} \rightarrow s_t
]

[
(s_t, V) \rightarrow c_t
]

[
(c_t, s_t) \rightarrow g_t
]

[
(\tilde c_t, s_t, h_{t-1}) \rightarrow h_t
]

[
h_t \rightarrow (o_t^{(m)}, o_t^{(s)}, o_t^{(r)})
]

其中：

* (V) 是视觉 token 序列
* (s_t) 是当前时刻的触觉状态
* (c_t) 是触觉引导读取出的视觉上下文
* (g_t) 是当前视觉可信度门值
* (h_t) 是融合后的递归状态

---

# 4. 视觉编码器

设视觉编码器由一个卷积主干网络 (F_v) 和一个线性投影层 (P_v) 构成。

输入图像：

[
x^{(v)} \in \mathbb{R}^{B \times 3 \times H \times W}
]

首先通过视觉 backbone 提取高层特征图：

[
H_v = F_v(x^{(v)}) \in \mathbb{R}^{B \times C_v \times H' \times W'}
]

再通过 (1\times1) 卷积或线性映射投影到融合维度 (d)：

[
H'_v = P_v(H_v) \in \mathbb{R}^{B \times d \times H' \times W'}
]

然后将空间维展平，得到视觉 token 序列：

[
V = {V_1, V_2, \dots, V_N}, \quad V_j \in \mathbb{R}^d
]

其中：

[
N = H'W'
]

例如使用 ResNet18 截断主干时，常见设置为：

[
H'=W'=7,\quad N=49
]

因此：

[
V \in \mathbb{R}^{B \times 49 \times d}
]

视觉编码只在 episode 开始时执行一次，并在后续所有时刻复用。

---

# 5. 因果触觉编码器

触觉编码器的目标是在时刻 (t) 将前缀序列 (x^{(t)}_{1:t}) 编码为一个当前状态 (s_t)，并且必须满足**因果性**：

[
s_t \text{ 仅依赖 } x^{(t)}*{1:t}, \text{ 不依赖未来信息 } x^{(t)}*{t+1:T}
]

为了同时兼顾局部模式提取与长程时序建模，可以采用 **Conv stem + Mamba** 结构。

---

## 5.1 局部卷积前端

首先用一维卷积前端处理原始触觉序列，以提取局部模式并进行时间降采样。设输入为：

[
X^{(t)} \in \mathbb{R}^{B \times C_t \times L}
]

经过若干层因果卷积后，得到中间特征：

[
Z \in \mathbb{R}^{B \times d \times L'}
]

其中 (L' < L) 为下采样后的时间长度。
卷积前端的作用包括：

1. 提取局部动态模式；
2. 平滑高频噪声；
3. 将原始长序列压缩为更短的 token 序列，减轻后续时序建模负担。

---

## 5.2 Mamba 时序建模

将卷积输出转置为 token 序列形式：

[
\bar Z \in \mathbb{R}^{B \times L' \times d}
]

再输入 Mamba 模块，得到因果建模后的触觉状态序列：

[
S = \text{Mamba}(\bar Z) \in \mathbb{R}^{B \times L' \times d}
]

在实时推理时，Mamba 以递推方式维护内部状态，因此当前时刻的输出表示为：

[
s_t \in \mathbb{R}^{B \times d}
]

这里 (s_t) 可理解为当前时刻的**触觉 belief state**，用于总结到当前为止的触觉证据。

与 LSTM 类似，Mamba 维护因果状态；但与普通 RNN/LSTM 不同，Mamba 对长序列通常具有更强的建模能力与更稳定的长程记忆能力。

---

# 6. 触觉引导的视觉读取

该模块的目标是在每个时刻 (t)，利用当前触觉状态 (s_t) 从视觉 token 序列 (V) 中主动读取当前最相关的视觉证据。

这一步不是将视觉 token 与触觉 token 进行对称双向融合，而是采用**单向条件读取**：由触觉发起 query，视觉提供被检索的信息。

---

## 6.1 查询、键和值映射

给定当前触觉状态：

[
s_t \in \mathbb{R}^{B \times d}
]

先通过线性映射生成 query：

[
q_t = W_q s_t \in \mathbb{R}^{B \times d}
]

对视觉 token 生成 key 和 value：

[
k_j = W_k V_j,\quad v_j = W_v V_j
]

其中：

[
V_j \in \mathbb{R}^{B \times d},\quad k_j,v_j \in \mathbb{R}^{B \times d}
]

---

## 6.2 触觉条件化视觉注意力

对每个视觉 token 计算相似度分数：

[
e_{t,j} = \frac{q_t^\top k_j}{\sqrt{d}}
]

再通过 softmax 得到注意力权重：

[
\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{l=1}^{N}\exp(e_{t,l})}
]

因此：

[
\alpha_t \in \mathbb{R}^{B \times N}
]

其中 (N) 为视觉 token 个数，例如 49。

---

## 6.3 视觉上下文聚合

根据注意力权重对视觉 value 做加权求和，得到当前时刻的视觉上下文：

[
c_t = \sum_{j=1}^{N}\alpha_{t,j} v_j
]

因此：

[
c_t \in \mathbb{R}^{B \times d}
]

该向量可理解为：

> 在当前触觉状态 (s_t) 条件下，从图像中读取出的最相关视觉证据。

这使得模型不再依赖全局平均视觉表示，而是能够在不同接触阶段动态地读取不同的图像区域。

---

# 7. 视觉门控抑制机制

为了避免视觉捷径，模型不直接使用 (c_t)，而是进一步通过门控机制评估这份视觉证据当前是否可信。

---

## 7.1 门值计算

将当前视觉上下文 (c_t) 与触觉状态 (s_t) 拼接：

[
z_t = [c_t; s_t] \in \mathbb{R}^{B \times 2d}
]

通过两层 MLP 和 sigmoid 生成门值：

[
g_t = \sigma\left(W_2 \phi(W_1 z_t + b_1) + b_2\right)
]

其中：

* (\phi) 为非线性函数，如 ReLU 或 GELU；
* (g_t \in \mathbb{R}^{B \times 1})

该门值是一个样本级标量，表示当前时刻视觉信息的可信度。

语义上：

* (g_t \approx 1)：当前读取到的视觉证据较可信
* (g_t \approx 0)：当前视觉证据不可信，应更多依赖触觉

---

## 7.2 可学习空视觉先验

为了避免在门值较小时简单将视觉置零，模型引入一个可学习的空视觉先验：

[
t_{\text{null}} \in \mathbb{R}^{1 \times d}
]

则门控后的视觉表示为：

[
\tilde c_t = g_t \cdot c_t + (1-g_t)\cdot t_{\text{null}}
]

其中：

[
\tilde c_t \in \mathbb{R}^{B \times d}
]

这意味着当视觉不可信时，模型不是完全删除视觉分支，而是将其替换为一个可学习的中性先验表示。

因此，该结构是一个**非对称门控机制**：

* 只对视觉进行抑制；
* 触觉状态 (s_t) 始终保留；
* 体现“触觉主导、视觉可抑制”的设计原则。

---

# 8. 递归融合模块

门控视觉表示 (\tilde c_t) 与触觉状态 (s_t) 还需要进一步整合为当前时刻的融合状态。为支持在线推理，模型采用递归更新形式，而不是对整段序列重复全局重算。

记上一时刻融合状态为：

[
h_{t-1} \in \mathbb{R}^{B \times d_h}
]

将当前门控视觉和触觉状态拼接：

[
u_t = [\tilde c_t; s_t] \in \mathbb{R}^{B \times 2d}
]

然后使用 GRUCell 更新融合状态：

[
h_t = \text{GRUCell}(u_t, h_{t-1})
]

其中：

[
h_t \in \mathbb{R}^{B \times d_h}
]

该递归状态 (h_t) 表示当前时刻的多模态决策状态，用于整合：

* 过去时刻累积的视觉—触觉证据；
* 当前时刻新读出的视觉上下文；
* 当前时刻的触觉状态。

采用 GRUCell 的原因在于：

1. 它能够显式维护跨时刻融合记忆；
2. 更新代价较小，适合实时部署；
3. 相比纯 MLP，能够利用历史融合状态；
4. 相比再引入一个复杂时序模型，结构更轻、更稳定。

---

# 9. 多任务输出头

从当前融合状态 (h_t) 出发，模型输出三个任务的 logits：

[
o_t^{(m)} = f_m(h_t)
]

[
o_t^{(s)} = f_s(h_t)
]

[
o_t^{(r)} = f_r(h_t)
]

其中：

* (o_t^{(m)} \in \mathbb{R}^{B \times C_m})
* (o_t^{(s)} \in \mathbb{R}^{B \times C_s})
* (o_t^{(r)} \in \mathbb{R}^{B \times C_r})

每个输出头可采用两层 MLP：

[
\text{Linear}(d_h \rightarrow d')
\rightarrow \text{GELU}
\rightarrow \text{Dropout}
\rightarrow \text{Linear}(d' \rightarrow C)
]

这样模型在每个时刻都能给出对质量、刚度、材质的当前预测。

---

# 10. 可选触觉辅助头

为了进一步保证模型不会过度依赖视觉，可以在触觉状态 (s_t) 上额外加三个辅助分类头：

[
o_{t,\text{aux}}^{(m)} = f_m^{\text{aux}}(s_t)
]

[
o_{t,\text{aux}}^{(s)} = f_s^{\text{aux}}(s_t)
]

[
o_{t,\text{aux}}^{(r)} = f_r^{\text{aux}}(s_t)
]

这样可以显式约束触觉分支单独具备判别能力，从而进一步降低视觉捷径风险。

---

# 11. 损失函数

模型训练可采用多时刻监督。设监督时刻集合为 (\mathcal{S})，例如若干前缀时刻或均匀采样时刻。

---

## 11.1 主任务损失

在每个监督时刻，对三个主任务计算交叉熵：

[
\mathcal{L}_{\text{main}}
=========================

\sum_{t\in \mathcal{S}} w_t
\Big[
\text{CE}(o_t^{(m)}, y^{(m)})
+
\text{CE}(o_t^{(s)}, y^{(s)})
+
\text{CE}(o_t^{(r)}, y^{(r)})
\Big]
]

其中 (w_t) 为时刻权重，可用于强调后期或均匀监督。

---

## 11.2 辅助触觉损失

若启用触觉辅助头，则对应损失为：

[
\mathcal{L}_{\text{aux}}
========================

\sum_{t\in \mathcal{S}} w_t
\Big[
\text{CE}(o_{t,\text{aux}}^{(m)}, y^{(m)})
+
\text{CE}(o_{t,\text{aux}}^{(s)}, y^{(s)})
+
\text{CE}(o_{t,\text{aux}}^{(r)}, y^{(r)})
\Big]
]

---

## 11.3 门控正则

为了约束门值行为，可引入门控正则项，例如：

### 稀疏正则

鼓励整体降低视觉依赖：

[
\mathcal{L}*{g}^{\text{sparse}} = \frac{1}{|\mathcal{S}|}\sum*{t\in\mathcal{S}} g_t
]

### 极化正则

鼓励门值接近 0 或 1：

[
\mathcal{L}*{g}^{\text{polar}} = \frac{1}{|\mathcal{S}|}\sum*{t\in\mathcal{S}} g_t(1-g_t)
]

### 目标均值正则

约束门值均值靠近给定目标 (\tau)：

[
\mathcal{L}*{g}^{\text{mean}} =
\left(
\frac{1}{|\mathcal{S}|}\sum*{t\in\mathcal{S}} g_t - \tau
\right)^2
]

---

## 11.4 时间平滑正则

为了避免在线门控在相邻时刻剧烈抖动，还可加入时间平滑项：

[
\mathcal{L}*{\text{smooth}} =
\sum*{t=2}^{T} |g_t-g_{t-1}|_2^2
]

这有助于提升在线推理稳定性。

---

## 11.5 总损失

总损失定义为：

[
\mathcal{L}
===========

\mathcal{L}*{\text{main}}
+
\lambda*{\text{aux}}\mathcal{L}*{\text{aux}}
+
\lambda_g \mathcal{L}*g
+
\lambda*{\text{smooth}}\mathcal{L}*{\text{smooth}}
]

其中 (\lambda_{\text{aux}}, \lambda_g, \lambda_{\text{smooth}}) 为权重超参数。

---

# 12. 训练方式

为了让模型真正具备实时前缀预测能力，训练时不能只用完整序列末端监督，而应采用**前缀训练**或**多时刻监督训练**。

---

## 12.1 前缀训练

随机采样一个前缀长度 (\tau)，仅输入到该时刻为止的触觉数据：

[
x^{(t)}_{1:\tau}
]

并要求模型预测最终标签：

[
\hat y_\tau = f(x^{(v)}, x^{(t)}_{1:\tau})
]

这样模型被迫学习在不完整证据下进行预测。

---

## 12.2 多时刻监督

更强的做法是在多个时间点同时监督：

[
t_1 < t_2 < \cdots < t_K
]

使模型学会：

* 早期基于有限证据给出粗预测；
* 中期随着触觉积累不断修正；
* 后期输出稳定、高置信的最终结果。

---

# 13. 推理行为

在在线部署时，模型按照如下方式运行：

### 初始化阶段

1. 输入视觉图像 (x^{(v)})
2. 提取并缓存视觉 token (V)
3. 初始化 Mamba 状态与 GRU 融合状态 (h_0)

### 每个时刻 (t)

1. 接收当前触觉观测 (x_t^{(t)})
2. 更新触觉状态 (s_t)
3. 由 (s_t) 查询视觉 token，得到 (c_t)
4. 计算视觉门值 (g_t)
5. 得到门控视觉表示 (\tilde c_t)
6. 更新融合状态 (h_t)
7. 输出当前三个任务的预测

因此模型具备以下性质：

* **严格因果**：时刻 (t) 的输出不依赖未来触觉；
* **实时可部署**：每步只做状态更新，不需要重复重跑整段序列；
* **随时间逐步修正**：支持 anytime prediction；
* **鲁棒性更强**：当视觉不可靠时，可由门控机制自动降低视觉影响。

---

# 14. 模型特征总结

该模型与原始对称多模态 Transformer 的本质区别在于：

## 1. 非对称性

不是视觉—触觉对等交互，而是：

* 触觉主导当前状态估计；
* 触觉主动查询视觉；
* 视觉再经过门控抑制。

## 2. 条件化视觉读取

不是固定使用全局视觉均值，而是在不同时间步根据当前触觉状态动态读取不同视觉区域。

## 3. 在线递归融合

不是每个时刻重跑整段融合，而是维护融合状态 (h_t) 做增量更新。

## 4. 面向视觉捷径抑制

通过门控与可学习 null prior，显式允许模型在 OOD 或视觉误导场景下降低视觉依赖。

---

# 15. 一句标准化概括

可以将该模型概括为：

> 一个面向实时物理属性识别的因果多模态模型。该模型首先利用因果触觉编码器对流式触觉输入进行在线状态建模，然后以当前触觉状态为查询，从静态视觉 token 中读取与当前接触阶段最相关的视觉上下文；随后通过样本级门控机制抑制不可信视觉证据，并借助递归融合状态持续整合视觉—触觉信息，最终在任意时间步输出对 mass、stiffness 和 material 的联合预测。


