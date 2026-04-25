# train_fusion_gating_online_reliable 开发记录

日期：2026-04-12

## 目标

这次新建脚本的目标，是把前面讨论的两件事真正落地：

1. 修正 `best checkpoint` 过早选中的问题
2. 给 `gate score` 增加更明确的“视觉不可靠”监督

对应新脚本：

- `visuotactile/scripts/train_fusion_gating_online_reliable.py`

这个脚本是在原版：

- `visuotactile/scripts/train_fusion_gating_online.py`

的基础上扩展出来的，核心模型骨架、`eval` 和 `online_eval` 口径保持不变，只改训练阶段的选模和监督逻辑。

## 实现内容

### 1. 双 checkpoint 机制

新脚本不再只保存一个 `best_model.pth`，而是同时保存：

- `best_acc.pth`
  - 纯按 `val avg acc` 选择
  - 作用：保留和旧逻辑最接近的结果，方便回溯
- `best_reliable.pth`
  - 只有在 `reliable_selection_start_epoch` 之后才允许参与竞争
  - 默认自动设为 `gate_reg_warmup_epochs + 1`
  - 作用：避免在 gate 正则还没真正启动时，就把早期高 gate 的模型保存成最终模型
- `best_model.pth`
  - 作为主导出 checkpoint
  - 默认优先指向 `best_reliable.pth`
  - 如果当前训练还没有产生 `best_reliable.pth`，则自动回退到 `best_acc.pth`

额外还会输出：

- `checkpoint_selection_summary.json`

里面会记录：

- 当前主 checkpoint 是哪一个
- `best_acc` 的 epoch 和路径
- `best_reliable` 的 epoch 和路径
- `best_reliable` 从第几个 epoch 开始允许竞争

### 2. 视觉错配 gate 监督

训练时新加了一个视觉错配分支，逻辑是：

- 保持 tactile、padding mask 和标签不变
- 只把视觉图像替换成 batch 里别的样本的图像
- 优先选择标签组合不同的样本做替换
- 对这些错配样本单独做一次前向传播
- 对错配样本的 gate score 增加：
  - `BCE(g_mismatch, 0)`

也就是显式告诉模型：

- 当视觉和触觉不一致时，`g` 应该变小

这里刻意没有加 `BCE(g_clean, 1)`，原因是：

- 干净视觉不等于“视觉一定该被强信任”
- 我们真正确定的是“错配视觉是坏的”
- 所以第一版只做负监督，更符合“可靠性学习”的方向

### 3. 训练日志中新增的可观测指标

除了原来的：

- `loss`
- `reg_loss`
- `gate_score`
- `mass / stiffness / material`

现在还会记录：

- `mismatch_gate_loss`
- `mismatch_gate_score`
- `clean_gate_on_mismatch`
- `mismatch_gate_gap`
- `val_mismatch.gate_loss`
- `val_mismatch.gate_score`
- `val_mismatch.clean_minus_mismatch_gap`
- `lambda_reg_eff`
- `reliable_candidate`
- `best_acc_updated`
- `best_reliable_updated`

这样后面分析时，可以直接回答这些问题：

- gate 正则当前到底有没有生效
- 模型有没有学会把错配视觉的 gate 压下去
- `best_reliable` 是在哪个阶段开始出现的

### 4. 验证集可靠性评估

每个 epoch 结束后，除了常规 `val` 外，还会额外跑一轮：

- `val_mismatch`

它会在验证集上人为制造视觉错配，然后记录：

- 错配视觉下的平均 gate
- 干净视觉与错配视觉的 gate 差值
- `BCE(g_mismatch, 0)` 的均值

`best_reliable` 的选择会综合看：

1. `avg_val_acc`
2. `val_mismatch.gate_loss`
3. `val_mismatch.clean_minus_mismatch_gap`
4. `val reg_loss`
5. `val loss`
6. epoch

也就是说：

- 先保证主任务精度不掉
- 再优先选“对错配视觉更敏感”的 checkpoint

## 新增参数

新脚本在原版参数之外，增加了这几个：

- `--visual_mismatch_prob`
  - 训练时每个样本被拿去做视觉错配监督的概率
  - 默认：`0.25`
- `--lambda_mismatch_gate`
  - 错配 gate 监督权重
  - 默认：`0.5`
- `--visual_mismatch_eval_prob`
  - 验证集错配评估时的错配比例
  - 默认：`1.0`
- `--reliable_selection_start_epoch`
  - 从第几个 epoch 开始允许保存 `best_reliable`
  - 默认：`0`
  - 含义：自动推导为 `gate_reg_warmup_epochs + 1`
- `--primary_checkpoint`
  - `best_model.pth` 指向哪个 checkpoint
  - 可选：`reliable` / `acc`
  - 默认：`reliable`

## 默认行为解读

默认配置下：

- gate 正则 warmup 是 `5`
- 所以 `best_reliable` 默认从第 `6` 个 epoch 开始才允许参与竞争
- 如果训练很短，或者在前几轮就结束，那么 `best_reliable` 可能还不存在
- 这种情况下，脚本会自动回退，让 `best_model.pth` 指向 `best_acc.pth`

这个回退是故意设计的，避免短训练直接报错。

## 推荐使用方式

### 推荐训练命令

```bash
conda run -n Y3 python /home/jiaming/Y3_Project/visuotactile/scripts/train_fusion_gating_online_reliable.py \
  --mode train \
  --data_root /home/jiaming/Y3_Project/Plaintextdataset \
  --save_dir /home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_v1 \
  --device cuda \
  --no_live_plot
```

### 如果你想保守一点

先从下面这组超参数开始：

```bash
--visual_mismatch_prob 0.25
--lambda_mismatch_gate 0.5
--primary_checkpoint reliable
```

这组是当前脚本的默认值，不需要额外改。

## 已完成验证

### 1. 语法检查

已通过：

```bash
conda run -n Y3 python -m py_compile /home/jiaming/Y3_Project/visuotactile/scripts/train_fusion_gating_online_reliable.py
```

### 2. CLI 检查

已通过：

```bash
conda run -n Y3 python /home/jiaming/Y3_Project/visuotactile/scripts/train_fusion_gating_online_reliable.py --help
```

### 3. 1 epoch 训练 smoke test

已在 RTX 5090 上完成：

```bash
conda run -n Y3 python /home/jiaming/Y3_Project/visuotactile/scripts/train_fusion_gating_online_reliable.py \
  --mode train \
  --data_root /home/jiaming/Y3_Project/Plaintextdataset \
  --save_dir /tmp/fusion_gating_online_reliable_smoke \
  --epochs 1 \
  --batch_size 8 \
  --num_workers 0 \
  --save_every 1 \
  --no_live_plot \
  --device cuda
```

这个 smoke test 验证了几件事：

- 训练循环可以正常跑通
- `val_mismatch` 会正常执行
- `best_acc.pth` 会被正常保存
- 因为只训练了 1 个 epoch，而 `best_reliable` 默认要从 epoch 6 才开始竞争，所以：
  - `best_reliable` 不会生成
  - `best_model.pth` 会自动回退指向 `best_acc.pth`
- 训练结束后的自动 `test / ood_test` 评估也能正常走通

smoke test 末尾的关键日志是：

- `best acc epoch: 1`
- `best reliable epoch: none`
- `primary checkpoint: best_acc.pth`

这说明新的 checkpoint 回退逻辑是正常工作的。

## 下一步建议

真正值得跑的正式实验，不是 1 epoch smoke，而是至少这两组：

### 实验 A

- 原版 online gating
- 作为旧 baseline

### 实验 B

- 新脚本 reliable 版
- 默认参数

然后统一比较：

- `best_acc` 和 `best_reliable` 哪个更好
- `ood_test avg_gate_score` 是否下降
- `online_eval_ood_test` 在 `0.4 / 0.6 / 0.8 / 1.0` 是否提升
- 是否至少不低于 tactile-only

如果这一步有效，再去调：

- `visual_mismatch_prob`
- `lambda_mismatch_gate`
- `reliable_selection_start_epoch`

## 产出文件

- 脚本：
  - `visuotactile/scripts/train_fusion_gating_online_reliable.py`
- 开发记录：
  - `visuotactile/docs/train_fusion_gating_online_reliable_devlog_2026-04-12.md`
