# Wan TI2V Pre-Context Validation Artifacts

This directory contains local copies of visual validation outputs for the
Wan2.2 TI2V pre-context action adapter run:

```text
wan-pre-context-v6e64-full-gbs512-fresh-scratch-ckpt100-east1d-20260616-191526
```

Directories:

- `wan_ti2v_pre_context_validation_step_013300`
- `wan_ti2v_pre_context_validation_step_017200`

Each checkpoint directory contains:

- `config.json`
- `summary.json` and `summary.csv`
- `comparison_midframe_contact_sheet.png`
- `contact_frames/frame_*.png`
- four sample directories with `sample.mp4`, `ground_truth.mp4`,
  `comparison_gt_top_pred_bottom.mp4`, `metrics.json`, and `meta.json`

