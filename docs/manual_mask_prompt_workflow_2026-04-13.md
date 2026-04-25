# Manual Mask Prompt Workflow

## 目的

这套流程不是让你手工描完整 mask，而是先为一小批关键样本提供：

- 目标物体的 bounding box
- 少量 positive points
- 少量 negative points

这些提示现在已经可以直接接到 `SAM` 上，生成比 `GrabCut` 更可靠的前景 mask。

## 已准备好的脚本

### 1. 抽样导出待标图片

脚本：

- `scripts/prepare_reliable_mask_prompt_subset.py`

默认会从 `reliable` 主模型相关样本中导出：

- `ood_test` top gate: `20`
- `ood_test` random: `10`
- `test` top gate: `5`
- `test` random: `5`

默认输出到：

- `outputs/mask_prompt_annotation_2026-04-13`

### 2. 浏览器标注工具

脚本：

- `scripts/annotate_mask_prompts_streamlit.py`

这是一个基于 `Streamlit + streamlit-drawable-canvas` 的本地标注页，支持：

- 拖框
- 点 positive / negative prompt
- 直接调用 `SAM` 生成 mask 预览
- 保存当前样本
- 上一张 / 下一张

### 3. 批量生成 SAM mask

脚本：

- `scripts/generate_sam_masks_from_prompts.py`

这个脚本会读取 `annotations/` 中已经保存好的提示，批量生成：

- 二值 mask
- 叠加预览图
- mask 元数据

## 使用步骤

### 第一步：生成待标子集

```bash
conda run -n Y3 python visuotactile/scripts/prepare_reliable_mask_prompt_subset.py
```

完成后会生成：

- `manifest.json`
- `images/`
- `annotations/`
- `previews/`

### 第二步：启动标注工具

```bash
conda run -n Y3 streamlit run /home/jiaming/Y3_Project/visuotactile/scripts/annotate_mask_prompts_streamlit.py -- --manifest /home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/manifest.json
```

默认会读取：

- `outputs/mask_prompt_annotation_2026-04-13/manifest.json`

启动后在浏览器中打开终端里显示的本地地址，一般是：

- `http://localhost:8501`

## 标注建议

每张图建议这样做：

1. 先画一个尽量包住目标物体的框
2. 在物体主体上点 `1-3` 个 positive points
3. 如果背景容易混进来，再补 `1-2` 个 negative points

不用追求很多点。重点是：

- 框基本包住物体
- 正点落在稳定的物体本体上
- 负点落在最容易误分进来的背景区域上

## 操作方式

- 通过顶部 `Mode` 切换 `box / positive / negative`
- 通过 `Prev / Next` 切换样本
- 通过 `Save Annotation` 保存
- 通过 `Save and Next` 保存并进入下一张
- 通过 `Generate Mask Preview` 直接在页面中调用 `SAM`
- 通过 `Save Annotation + Mask` 同时保存提示和 `SAM` mask
- 通过 `Clear All` 清空当前画布

## 输出格式

每张图会在 `annotations/` 下保存一个 JSON，包含：

- `bbox_xyxy`
- `positive_points`
- `negative_points`
- 样本来源信息

同时会在 `previews/` 下保存一张叠加预览图，方便快速检查。

## 批量导出所有 SAM mask

如果你完成了一批提示标注，想一次性导出全部 mask，运行：

```bash
conda run -n Y3 python /home/jiaming/Y3_Project/visuotactile/scripts/generate_sam_masks_from_prompts.py --manifest /home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/manifest.json --output_dir /home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/sam_outputs --device cuda
```

输出会保存在：

- `sam_outputs/masks/`
- `sam_outputs/mask_previews/`
- `sam_outputs/mask_metadata/`

## 下一步

完成这一批 prompt 标注并生成 `SAM` mask 后，下一步应当做的是：

1. 用这些更高质量的 mask 构造：
   - `object_only`
   - `background_only`
   - `background_swapped`
2. 导出：
   - object/background 反事实输入
3. 重跑 object/background 反事实实验

`OpenCV` 版本的本地弹窗标注器在当前环境里不推荐使用，因为 `Y3` 中的 OpenCV 是 headless 构建，无法弹出 GUI 窗口。
