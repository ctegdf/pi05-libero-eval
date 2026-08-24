# LIBERO official-vs-base PPT report

This directory contains a reproducible, data-rich comparison of the official
`pi05_libero` fine-tuned checkpoint and `pi05_base` parameters paired with the
official LIBERO assets and normalization statistics.

Outputs:

- `OpenPI_pi05_LIBERO_权重对比分析.pptx`: PowerPoint deck;
- `OpenPI_pi05_LIBERO_权重对比分析.pdf`: fixed-layout review copy;
- `PPT分析报告.md`: slide-by-slide Chinese analysis and speaker notes;
- `figures/`: PNG and SVG chart assets;
- `data/`: JSON and CSV source tables;
- `preview/`: rendered slide previews and contact sheet.

Regenerate:

```bash
python3 generate_ppt_report.py
```

The generator validates the paired 2,000-episode matrices, unique IDs,
terminal outcomes, infrastructure-error separation, and 1:1 non-empty video
inventory before writing any report.
