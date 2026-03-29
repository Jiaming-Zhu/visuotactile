# `newOutput` Test vs OOD Comparison

Source summaries:
- `/home/martina/Y3_Project/visuotactile/newOutput/meta/multi_seed_summary_single_modal.json`
- `/home/martina/Y3_Project/visuotactile/newOutput/meta/multi_seed_summary_standard.json`
- `/home/martina/Y3_Project/visuotactile/newOutput/fusion_gating_online_v2_multiseed/meta/multi_seed_summary_gating_online_v2.json`

This document compares the four newly trained models under the current label schema:
- `stiffness`: `soft / medium / rigid`
- `material`: `sponge / foam / wood / container`

All metrics below are multi-seed results over seeds `42 / 123 / 456 / 789 / 2024`.

## Overall Summary

| Model | Test Avg | OOD Avg | Test -> OOD Drop | Notes |
| --- | --- | --- | --- | --- |
| Vision Only | `97.06 ± 0.62%` | `26.62 ± 2.65%` | `70.44%` | Severe OOD collapse, especially on `material` |
| Tactile Only | `97.16 ± 0.37%` | `89.47 ± 1.06%` | `7.69%` | Strongest non-gated baseline on OOD |
| Standard Fusion | `99.41 ± 0.57%` | `80.53 ± 8.01%` | `18.88%` | Better ID than single-modal, but weaker OOD than tactile-only |
| Fusion Gating Online v2 | `99.22 ± 0.50%` | `89.91 ± 6.07%` | `9.30%` | Best OOD average, but only slightly above tactile-only |

For `Fusion Gating Online v2`, the average gate score is:
- `Test`: `88.08 ± 5.43%`
- `OOD`: `90.52 ± 4.06%`

## Task-Level Comparison

### Vision Only

| Task | Test | OOD | Drop |
| --- | --- | --- | --- |
| Mass | `94.12 ± 0.93%` | `24.27 ± 3.39%` | `69.85%` |
| Stiffness | `98.53 ± 1.32%` | `38.80 ± 3.08%` | `59.73%` |
| Material | `98.53 ± 0.93%` | `16.80 ± 4.45%` | `81.73%` |

Interpretation:
- `Vision Only` is nearly perfect on ID, but fails badly on OOD.
- The weakest task by far is `material`, which falls to `16.80%`.
- This confirms that the current OOD shift is dominated by appearance/domain mismatch rather than pure ID classification difficulty.

### Tactile Only

| Task | Test | OOD | Drop |
| --- | --- | --- | --- |
| Mass | `100.00 ± 0.00%` | `100.00 ± 0.00%` | `0.00%` |
| Stiffness | `100.00 ± 0.00%` | `99.60 ± 0.80%` | `0.40%` |
| Material | `91.47 ± 1.10%` | `68.80 ± 2.87%` | `22.67%` |

Interpretation:
- `Tactile Only` is extremely robust on `mass` and `stiffness`.
- The only real OOD weakness is `material`.
- Under the current label schema, tactile is the main source of OOD robustness.

### Standard Fusion

| Task | Test | OOD | Drop |
| --- | --- | --- | --- |
| Mass | `99.71 ± 0.59%` | `87.87 ± 8.78%` | `11.84%` |
| Stiffness | `100.00 ± 0.00%` | `86.53 ± 8.21%` | `13.47%` |
| Material | `98.53 ± 1.61%` | `67.20 ± 7.34%` | `31.33%` |

Interpretation:
- `Standard Fusion` is very strong on ID.
- On OOD it underperforms `Tactile Only`, despite using both modalities.
- The main issue is again `material`, plus noticeably larger seed variance than `Tactile Only`.

### Fusion Gating Online v2

| Task | Test | OOD | Drop |
| --- | --- | --- | --- |
| Mass | `100.00 ± 0.00%` | `97.33 ± 5.33%` | `2.67%` |
| Stiffness | `100.00 ± 0.00%` | `96.40 ± 7.20%` | `3.60%` |
| Material | `97.65 ± 1.50%` | `76.00 ± 5.73%` | `21.65%` |

Interpretation:
- `Fusion Gating Online v2` gives the best OOD average among the four models.
- It keeps `mass` and `stiffness` very high on OOD.
- `Material` is still the bottleneck, but it is better than both `Tactile Only` and `Standard Fusion`.

## Ranking

### Test Ranking by Average Accuracy

1. `Standard Fusion`: `99.41 ± 0.57%`
2. `Fusion Gating Online v2`: `99.22 ± 0.50%`
3. `Tactile Only`: `97.16 ± 0.37%`
4. `Vision Only`: `97.06 ± 0.62%`

### OOD Ranking by Average Accuracy

1. `Fusion Gating Online v2`: `89.91 ± 6.07%`
2. `Tactile Only`: `89.47 ± 1.06%`
3. `Standard Fusion`: `80.53 ± 8.01%`
4. `Vision Only`: `26.62 ± 2.65%`

### OOD Ranking by Material Accuracy

1. `Fusion Gating Online v2`: `76.00 ± 5.73%`
2. `Tactile Only`: `68.80 ± 2.87%`
3. `Standard Fusion`: `67.20 ± 7.34%`
4. `Vision Only`: `16.80 ± 4.45%`

## Seed-Level Average Accuracy

### Test Average Accuracy by Seed

| Model | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 |
| --- | --- | --- | --- | --- | --- |
| Vision Only | `96.08%` | `97.06%` | `97.06%` | `97.06%` | `98.04%` |
| Tactile Only | `97.06%` | `97.06%` | `96.57%` | `97.55%` | `97.55%` |
| Standard Fusion | `99.51%` | `99.02%` | `100.00%` | `100.00%` | `98.53%` |
| Fusion Gating Online v2 | `98.53%` | `100.00%` | `99.51%` | `99.02%` | `99.02%` |

### OOD Average Accuracy by Seed

| Model | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 |
| --- | --- | --- | --- | --- | --- |
| Vision Only | `28.89%` | `30.22%` | `24.89%` | `26.22%` | `22.89%` |
| Tactile Only | `88.44%` | `88.22%` | `90.44%` | `89.33%` | `90.89%` |
| Standard Fusion | `82.22%` | `72.89%` | `83.11%` | `71.11%` | `93.33%` |
| Fusion Gating Online v2 | `92.89%` | `93.11%` | `93.33%` | `77.78%` | `92.44%` |

Observations:
- `Tactile Only` is the most stable model on OOD.
- `Standard Fusion` and `Fusion Gating Online v2` both show a weak seed at `789`.
- `Fusion Gating Online v2` is the best average OOD model, but not the most stable one.

## Seed-Level OOD Material Accuracy

| Model | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 |
| --- | --- | --- | --- | --- | --- |
| Vision Only | `24.00%` | `19.33%` | `12.67%` | `12.00%` | `16.00%` |
| Tactile Only | `67.33%` | `64.67%` | `71.33%` | `68.00%` | `72.67%` |
| Standard Fusion | `68.67%` | `60.00%` | `67.33%` | `60.00%` | `80.00%` |
| Fusion Gating Online v2 | `78.67%` | `79.33%` | `80.00%` | `64.67%` | `77.33%` |

This table makes the central pattern very clear:
- OOD weakness is still dominated by `material`.
- `Fusion Gating Online v2` gives the best `material` accuracy overall.
- `Vision Only` is unusable for the current OOD `material` setting.

## Takeaways

1. `Vision Only` is not a viable OOD solution for the current setup. Its `material` generalization collapses.
2. `Tactile Only` is the strongest baseline for OOD and remains very stable across seeds.
3. `Standard Fusion` improves ID performance but does not translate that gain to OOD; in fact it is clearly worse than `Tactile Only` on OOD.
4. `Fusion Gating Online v2` is the best overall OOD model in `newOutput`, mainly because it improves OOD `material` prediction while keeping `mass` and `stiffness` high.
5. The margin between `Fusion Gating Online v2` and `Tactile Only` on OOD is small: `89.91%` vs `89.47%`. So the practical gain is real but modest.
