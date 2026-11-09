## N. Experiment sweep: what moves validation - 09 November 2026

`notebooks/08_experiments.ipynb`, 40 epochs per run, SegFormer 2.3 M unless stated. Every row shares one recipe; only the named thing changes.

| run | channels | loss | epochs | val mIoU (9) | val (6 main) | test (9) | test (6 main) |
|---|---|---|---|---|---|---|---|
| temporal_mstat | 9 | wce | 40 | 54.3% | 80.9% | 43.5% | 65.0% |
| temporal_t3 | 9 | wce | 40 | 42.1% | 62.6% | 27.2% | 40.8% |
| loss_ce_lovasz | 3 | ce_lovasz | 40 | 39.0% | 58.1% | 29.5% | 44.2% |
| loss_focal | 3 | focal | 40 | 38.1% | 56.5% | 27.5% | 41.2% |
| sampling_oversample | 3 | wce | 40 | 38.1% | 56.5% | 29.5% | 44.2% |
| baseline | 3 | wce | 40 | 36.6% | 54.3% | 27.4% | 41.2% |
| loss_ce_dice | 3 | ce_dice | 40 | 36.4% | 53.9% | 29.3% | 44.0% |

Baseline on the current tiles: **36.6%** (9 classes), 54.3% (6 main). Best: **temporal_mstat**, +17.7 points.