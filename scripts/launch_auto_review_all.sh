#!/usr/bin/env bash
set -euo pipefail

conda run -n Y3 streamlit run \
  /home/jiaming/Y3_Project/visuotactile/scripts/annotate_mask_prompts_streamlit.py \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.port 8765 \
  -- \
  --manifest /home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/review_all_manifest.json
