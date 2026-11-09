## N. Six classes instead of nine - 09 November 2026

`notebooks/09_sixclass.ipynb`. segformer (2,280,071 parameters), 60 epochs, notebook 05's recipe unchanged. Labels shrub & scrub, flooded vegetation, snow & ice become unlabelled, so the network has 7 output channels and those pixels leave both the loss and the score.

Tiles kept: train 5,400, val 1,800, test 2,100 (of 5,400 / 1,800 / 2,100; a tile with under 10% of its pixels still labelled is dropped).

| | 9-class, 6 main | 6-class | change |
|---|---|---|---|
| val, plain | 75.7% | 75.4% | -0.3 |
| val, year+tta | 84.4% | 81.2% | -3.2 |
| test, plain | 47.3% | 49.8% | +2.5 |
| test, year+tta | 49.8% | 50.6% | +0.8 |

Per-class validation IoU (plain): water 94.4%, built-up 83.9%, trees 63.1%, cropland 64.6%, grass 62.3%, bare 83.9%

Per-class test IoU (plain): water 62.8%, built-up 70.4%, trees 66.0%, cropland 35.5%, grass 13.3%, bare 50.9%