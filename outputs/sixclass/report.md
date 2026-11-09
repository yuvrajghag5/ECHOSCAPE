## N. Six classes instead of nine - 09 November 2026

`notebooks/09_sixclass.ipynb`. segformer (2,280,071 parameters), 60 epochs, notebook 05's recipe unchanged. Labels shrub & scrub, flooded vegetation, snow & ice become unlabelled, so the network has 7 output channels and those pixels leave both the loss and the score.

Tiles kept: train 6,228, val 1,800, test 2,100 (of 7,164 / 1,800 / 2,100; a tile with under 10% of its pixels still labelled is dropped).

| | 9-class, 6 main | 6-class | change |
|---|---|---|---|
| val, plain | 75.7% | 75.4% | -0.3 |
| val, year+tta | 84.4% | 82.9% | -1.5 |
| test, plain | 47.3% | 49.1% | +1.8 |
| test, year+tta | 49.8% | 52.3% | +2.5 |

Per-class validation IoU (plain): water 90.3%, built-up 84.4%, trees 58.8%, cropland 63.5%, grass 63.8%, bare 91.7%

Per-class test IoU (plain): water 58.1%, built-up 70.6%, trees 64.8%, cropland 39.5%, grass 12.0%, bare 49.3%