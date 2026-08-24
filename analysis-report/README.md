# Evaluation analysis report

This directory contains a reproducible Chinese analysis of the archived OpenPI
π0.5 LIBERO, LIBERO-Plus, and LIBERO-Pro evaluation results.

Open `report_zh.html` for the visual report or `report_zh.md` for the plain
Markdown version. Derived tables live under `data/`, and charts are standalone
SVG files under `figures/`.

Regenerate and verify the report from the archived episode JSONL files:

```bash
python3 generate_report.py
python3 -m unittest -v test_generate_report.py
```

The generator uses only the Python standard library. It refuses to generate a
report unless the final audit has passed and all four evaluated matrices have
the expected number of unique terminal episode and attempt IDs, no recorded
infrastructure errors, and written video status.

The public checkout contains the generated report and derived tables, but not
the raw archive directories used by the generator. To run these commands,
mount the completed evaluation archive at the layout expected by
`generate_report.py`; the large video tree is intentionally kept in the
separate Hugging Face release directory.

Statistical notes:

- Reported 95% confidence intervals use the Wilson score interval at episode
  level.
- LIBERO-Plus has one trial per generated variant.
- LIBERO-Pro has 50 trials per source, so source-level distributions are
  reported alongside episode-level rates.
- The unavailable 2,000 LIBERO-Pro `env` episodes and the incompatible
  Base-native protocol are N/A, never failures.
- Cross-benchmark overall rates have different task compositions and should not
  be read as a paired ranking.
