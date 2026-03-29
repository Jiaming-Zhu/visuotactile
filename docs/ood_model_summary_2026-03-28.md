# OOD Results Summary

更新时间：2026-03-28

本文档汇总了本轮最新 `ood_test` 评估结果，对应的数据版本已经包含类别 `Cardbox_hollow_noise`，其属性映射与 `CardboardBox_Hollow` 一致：

- `mass = low`
- `stiffness = soft`
- `material = hollow_container`

本次只统计 4 组模型：

- `vision-only`
- `tactile-only`
- `standard fusion`
- `fusion gating online v2`

所有数字均来自各模型目录下的 `eval_ood_test/evaluation_results.json`。文中的 `OOD Avg` 指 `mass / stiffness / material` 三个任务准确率的平均值。除 seed 明细外，其余结果均写为 `mean ± std`。

## Overall Comparison

| Model | OOD Avg | Mass | Stiffness | Material |
| --- | ---: | ---: | ---: | ---: |
| vision-only | 18.04 ± 3.49% | 23.87 ± 3.69% | 15.07 ± 5.41% | 15.20 ± 4.12% |
| tactile-only | 81.02 ± 1.98% | 100.00 ± 0.00% | 71.60 ± 1.77% | 71.47 ± 5.00% |
| standard fusion | 78.80 ± 4.57% | 92.40 ± 4.19% | 73.73 ± 5.78% | 70.27 ± 4.29% |
| fusion gating online v2 | 84.84 ± 2.87% | 99.87 ± 0.27% | 76.53 ± 5.95% | 78.13 ± 2.75% |

## Quick Read

- `vision-only` 仍然明显失败，`OOD Avg` 只有 `18.04%`，基本接近随机水平。
- `tactile-only` 仍然是很强的 OOD 基线，`OOD Avg = 81.02%`，其中 `mass` 继续保持 `100%`。
- `standard fusion` 这轮重新回到 `78.80%`，整体仍明显强于纯视觉，但没有超过最好的门控模型。
- `fusion gating online v2` 仍然是这 4 组里最新 OOD 表现最好的，`OOD Avg = 84.84%`，同时 `mass` 基本满分。

## Per-Seed Results

### vision-only

| Seed | OOD Avg | Mass | Stiffness | Material |
| --- | ---: | ---: | ---: | ---: |
| 42 | 20.67% | 21.33% | 20.67% | 20.00% |
| 123 | 21.33% | 26.00% | 19.33% | 18.67% |
| 456 | 20.44% | 29.33% | 16.00% | 16.00% |
| 789 | 12.67% | 24.00% | 5.33% | 8.67% |
| 2024 | 15.11% | 18.67% | 14.00% | 12.67% |

### tactile-only

| Seed | OOD Avg | Mass | Stiffness | Material |
| --- | ---: | ---: | ---: | ---: |
| 42 | 79.33% | 100.00% | 69.33% | 68.67% |
| 123 | 82.89% | 100.00% | 71.33% | 77.33% |
| 456 | 78.00% | 100.00% | 70.67% | 63.33% |
| 789 | 82.44% | 100.00% | 72.00% | 75.33% |
| 2024 | 82.44% | 100.00% | 74.67% | 72.67% |

### standard fusion

| Seed | OOD Avg | Mass | Stiffness | Material |
| --- | ---: | ---: | ---: | ---: |
| 42 | 72.67% | 86.00% | 66.67% | 65.33% |
| 123 | 84.00% | 97.33% | 78.67% | 76.00% |
| 456 | 82.89% | 96.67% | 78.67% | 73.33% |
| 789 | 74.22% | 90.67% | 66.67% | 65.33% |
| 2024 | 80.22% | 91.33% | 78.00% | 71.33% |

### fusion gating online v2

| Seed | OOD Avg | Mass | Stiffness | Material |
| --- | ---: | ---: | ---: | ---: |
| 42 | 86.44% | 100.00% | 80.00% | 79.33% |
| 123 | 86.44% | 100.00% | 79.33% | 80.00% |
| 456 | 86.22% | 99.33% | 80.00% | 79.33% |
| 789 | 86.00% | 100.00% | 78.67% | 79.33% |
| 2024 | 79.11% | 100.00% | 64.67% | 72.67% |

## Notes

- 本文档只总结 `OOD`，不包含 `test` 或 `val`。
- `fusion gating online v2` 这里使用的是 `eval_ood_test` 的完整 OOD 评估结果，而不是前缀在线曲线中的某个中间 ratio。
- 如果后续继续改动 `ood_test` 数据集，本文档里的数字也需要跟着重算。
