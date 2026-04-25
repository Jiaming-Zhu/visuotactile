# SAM Mask Workflow File Index

这份文档整理的是这一条工作流相关的文件：

- 手工 prompt 标注
- SAM mask 生成
- 自动传播
- 全量 review
- 基于 reviewed SAM masks 的 object/background counterfactual

## 1. 最重要的结果文件

### 1.1 可靠性模型主报告

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/analysis_report_2026-04-12.md`
  - `reliable` 主模型正式实验报告

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/reliable_mechanism_report_2026-04-12.md`
  - `reliable` 机制实验总报告

### 1.2 reviewed SAM masks 重跑后的 object/background 反事实结果

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/object_background_counterfactual_sam_reviewed/object_background_counterfactual.json`
  - 使用 reviewed SAM masks 重跑后的正式结果 JSON

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/object_background_counterfactual_sam_reviewed/analysis_report_2026-04-13.md`
  - 这次 reviewed SAM 版本的简短分析报告

### 1.3 旧版粗分割结果

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/object_background_counterfactual_ood_only/object_background_counterfactual.json`
  - 旧版 `GrabCut` 粗分割的 OOD-only 反事实结果
  - 现在主要作为对照参考，不再作为最可信版本

## 2. 手工标注与 SAM 输出

根目录：

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13`

这个目录是整条 SAM 工作流的中枢目录。

### 2.1 手工种子样本

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/manifest.json`
  - 最初那 `40` 张手工种子样本的 manifest

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/images`
  - 拷贝出来给你做手工标注的原图

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/annotations`
  - 40 张手工 prompt 标注结果
  - 每个 `json` 里存 `box / positive / negative`

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/previews`
  - 40 张手工 prompt 的可视化预览图

### 2.2 手工种子生成的 SAM masks

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/sam_outputs/masks`
  - 40 张手工种子样本生成的二值 mask

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/sam_outputs/mask_previews`
  - 40 张手工种子样本的 SAM 叠加预览图

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/sam_outputs/mask_metadata`
  - 40 张手工种子样本的 SAM 元数据
  - 每个 `json` 里有：
    - `source_image_path`
    - `mask_path`
    - `overlay_path`
    - `score`

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/sam_outputs/sam_generation_summary.json`
  - 40 张手工种子 SAM 输出的汇总

## 3. 自动传播与全量 review

自动传播根目录：

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation`

这一层覆盖的是剩余的 `test + ood_test` 样本。

### 3.1 自动传播结果

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/auto_propagation_summary.json`
  - 自动传播总摘要
  - 包含每张自动样本的：
    - `record_id`
    - `source_image_path`
    - `score`
    - `mask_area_ratio`
    - `review_needed`

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/auto_prompt_annotations`
  - 自动生成的 prompt 标注
  - 你 review 之后，这里的内容已经被修正过

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/masks`
  - 自动传播样本的最终 mask
  - 你 review 后重新生成的 mask 也在这里

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/mask_previews`
  - 自动传播样本的最终 mask 叠加图

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/mask_metadata`
  - 自动传播样本的最终元数据

### 3.2 review 清单

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/review_needed/review_needed_manifest.json`
  - 机器筛出来建议优先复查的样本清单

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/review_needed/review_needed_summary.md`
  - 上述复查样本的人类可读摘要

### 3.3 全量 review manifest

- `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/review_all_manifest.json`
  - 给浏览器标注器使用的“全量自动样本 review manifest”
  - 包含 `186` 张自动传播样本
  - 你后来 review 全部自动样本时，用的就是这个

## 4. 相关脚本

### 4.1 标注与 SAM 相关脚本

- `/home/jiaming/Y3_Project/visuotactile/scripts/prepare_reliable_mask_prompt_subset.py`
  - 从 `reliable` 模型结果里抽取最初的 `40` 张手工种子样本

- `/home/jiaming/Y3_Project/visuotactile/scripts/annotate_mask_prompts_streamlit.py`
  - 浏览器版标注器
  - 当前正式使用版本

- `/home/jiaming/Y3_Project/visuotactile/scripts/sam_prompt_mask_utils.py`
  - SAM 推理工具函数
  - 包含 `get_cached_sam / predict_mask / save_mask_outputs`

- `/home/jiaming/Y3_Project/visuotactile/scripts/generate_sam_masks_from_prompts.py`
  - 对手工 prompt 批量生成 SAM mask

- `/home/jiaming/Y3_Project/visuotactile/scripts/propagate_sam_prompts_from_seed.py`
  - 用手工种子 prompt 传播到剩余样本

- `/home/jiaming/Y3_Project/visuotactile/scripts/build_auto_propagation_review_manifest.py`
  - 为全部自动传播样本生成 review manifest

- `/home/jiaming/Y3_Project/visuotactile/scripts/launch_auto_review_all.sh`
  - 一键启动“全量自动样本 review”页面

### 4.2 旧的或辅助脚本

- `/home/jiaming/Y3_Project/visuotactile/scripts/annotate_mask_prompts_cv2.py`
  - 早期 OpenCV GUI 标注器
  - 当前环境是 headless OpenCV，不建议再用

- `/home/jiaming/Y3_Project/visuotactile/scripts/annotate_mask_prompts_web.py`
  - 早期纯 Web 版尝试
  - 当前主要流程不再依赖它

### 4.3 反事实实验脚本

- `/home/jiaming/Y3_Project/visuotactile/scripts/diagnose_object_background_counterfactual.py`
  - object/background counterfactual 主脚本
  - 现在已经改成优先读取 reviewed SAM masks
  - 不再默认依赖粗分割

## 5. 其它机制实验文件

这些文件不属于 SAM 标注本身，但属于 `reliable` 机制分析主线。

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/visual_residual_prefix_scan/visual_residual_prefix_scan.json`
  - prefix 因果消融原始结果

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/visual_residual_prefix_scan/visual_residual_prefix_scan_summary.json`
  - prefix 因果消融摘要

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/visual_delta_analysis/visual_delta_analysis.json`
  - original vs force-gate-zero 的细拆结果

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/visual_delta_analysis/visual_delta_analysis_summary.json`
  - 任务级/类别级视觉增益摘要

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/corruption_scan_ood_only/corruption_scan_results.json`
  - OOD 视觉 corruption 扫描

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/causal_saliency_validation/causal_saliency_summary.json`
  - 因果式 patch deletion / insertion 摘要

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/causal_saliency_validation/test/causal_saliency_curves.png`
  - `test` saliency 曲线

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/causal_saliency_validation/ood_test/causal_saliency_curves.png`
  - `ood_test` saliency 曲线

## 6. 推荐查看顺序

如果你现在只想最快定位关键材料，按这个顺序看：

1. `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/reliable_mechanism_report_2026-04-12.md`
2. `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/object_background_counterfactual_sam_reviewed/analysis_report_2026-04-13.md`
3. `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/review_all_manifest.json`
4. `/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/mask_previews`
5. `/home/jiaming/Y3_Project/visuotactile/scripts/diagnose_object_background_counterfactual.py`

## 7. 当前最可信的 object/background 结论

目前最该引用的是 reviewed SAM 版本，不是旧的粗分割版本。

原因是：

- reviewed SAM masks 覆盖了 `226 / 226` 个 `test + ood_test` 样本
- `fallback_count = 0`
- `background_only` 在 `ood_test` 上掉回 tactile baseline
- `object_only` 保留了大部分视觉增益

所以当前这条主结论应基于：

- `/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/object_background_counterfactual_sam_reviewed/object_background_counterfactual.json`
