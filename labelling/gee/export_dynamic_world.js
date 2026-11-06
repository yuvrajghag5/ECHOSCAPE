/*
 * Export a 2021 Dynamic World land-cover map for each ECHOSCAPE region.
 *
 * Paste this whole file into the Earth Engine Code Editor
 * (https://code.earthengine.google.com) and press Run. It creates one export
 * task per region in the "Tasks" tab on the right; click RUN on each task.
 * They export to a Google Drive folder called "echoscape_dw". When they finish,
 * download every file into the project's data/dw_raw/ folder and run:
 *
 *     .venv\Scripts\python.exe labelling\import_labels.py
 *
 * WHAT IS EXPORTED - three bands, all uint8, at 10 m:
 *
 *   1  label       Dynamic World class + 1, so 1..9.
 *                  0 means NO DATA. The +1 is deliberate: Dynamic World's
 *                  own class 0 is WATER, and Earth Engine writes masked pixels
 *                  as 0 too. Without the shift, every pixel with no cloud-free
 *                  view would silently arrive as water.
 *   2  confidence  the winning class's mean probability over 2021, x100.
 *   3  count       how many cloud-free Dynamic World images saw the pixel.
 *
 * WHY THE MEAN OF PROBABILITIES, NOT THE MOST COMMON LABEL
 * Dynamic World is one map per Sentinel-2 pass, not an annual product. Our
 * labels are annual, so the year has to be collapsed. Averaging the nine
 * probability bands and then taking the winner keeps how sure each pass was;
 * taking the most frequent label throws that away and lets a run of uncertain
 * passes outvote a few confident ones.
 *
 * WHY 10 m AND NOT STRAIGHT TO 50 m
 * The radar defines the project's grid, and keeping labels exactly on it is
 * the invariant everything depends on. Doing the 10 m -> 50 m majority vote
 * locally, in import_labels.py, keeps that step in code check_labels.py can
 * verify - rather than trusting Earth Engine to land on our exact pixels.
 *
 * The region boxes below are config.bbox_for() for every region - checked
 * identical, to the last decimal, to the bounds of the radar files - plus a
 * small margin so the edges are fully covered. Do not edit them by hand.
 */

var YEAR_START = '2021-01-01';
var YEAR_END = '2022-01-01';
var DRIVE_FOLDER = 'echoscape_dw';
var MARGIN = 0.005;   // degrees, ~500 m, so the edge pixels have neighbours

var CLASS_NAMES = [
  'water', 'trees', 'grass', 'flooded_vegetation', 'crops',
  'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'];

// west, south, east, north - from config.bbox_for()
var REGIONS = {
  ijsselmeer:   [   5.177641,  52.565253,    5.622359,  52.834747],
  mekong:       [ 105.663175,   9.865253,  105.936825,  10.134747],
  nile_delta:   [  30.843128,  30.665253,   31.156872,  30.934747],
  randstad:     [   4.380645,  51.965253,    4.819355,  52.234747],
  cairo:        [  31.094329,  29.915253,   31.405671,  30.184747],
  delhi:        [  76.946527,  28.465253,   77.253473,  28.734747],
  finland:      [  25.411084,  62.065253,   25.988916,  62.334747],
  oregon:       [-122.687320,  43.865253, -122.312680,  44.134747],
  amazon:       [ -60.134944,  -3.234747,  -59.865056,  -2.965253],
  pampas:       [ -60.662534, -34.134747,  -60.337466, -33.865253],
  punjab:       [  75.642964,  30.765253,   75.957036,  31.034747],
  iowa:         [ -93.781035,  41.765253,  -93.418965,  42.034747],
  nsw:          [ 147.235303, -35.234747,  147.564697, -34.965253],
  mongolia:     [ 105.802424,  46.865253,  106.197576,  47.134747],
  kenya:        [  36.665219,  -1.434747,   36.934781,  -1.165253],
  atacama:      [ -69.646933, -23.634747,  -69.353067, -23.365253],
  egypt_desert: [  28.850081,  25.865253,   29.149919,  26.134747],
  arabia:       [  44.852501,  23.865253,   45.147499,  24.134747],
  mannheim:     [   8.392860,  49.285253,    8.807140,  49.554747],
  chesapeake:   [ -76.421465,  38.065253,  -76.078535,  38.334747],
  paris:        [   2.145187,  48.725253,    2.554813,  48.994747],
  black_forest: [   7.947245,  48.215253,    8.352755,  48.484747],
  beauce:       [   1.447839,  48.065253,    1.852161,  48.334747],
  serengeti:    [  34.765114,  -2.734747,   35.034886,  -2.465253],
  namib:        [  15.051920, -24.634747,   15.348080, -24.365253],
  lake_geneva:  [   6.354428,  46.315253,    6.745572,  46.584747],
  tokyo:        [ 139.534073,  35.565253,  139.865927,  35.834747],
  congo:        [  24.365246,   0.465253,   24.634754,   0.734747],
  kansas:       [ -98.771938,  38.265253,  -98.428062,  38.534747],
  ireland:      [  -7.626000,  53.265253,   -7.174000,  53.534747],
  death_valley: [-117.067409,  36.265253, -116.732591,  36.534747]
};

// Which regions to create export tasks for. The first 19 were exported on
// 3 Nov 2026; these are the validation and test regions added on 5 Nov.
// Re-exporting the others would only spend Earth Engine quota again.
var EXPORT_ONLY = [
  'chesapeake', 'paris', 'black_forest', 'beauce',
  'serengeti', 'namib', 'lake_geneva', 'tokyo',
  'congo', 'kansas', 'ireland', 'death_valley'
];


function annualComposite(region) {
  var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterBounds(region)
    .filterDate(YEAR_START, YEAR_END);

  var meanProb = dw.select(CLASS_NAMES).mean();

  // winner = index of the largest mean probability, 0..8
  var winner = meanProb.toArray().arrayArgmax().arrayGet([0]);
  var confidence = meanProb.reduce(ee.Reducer.max());
  var count = dw.select('label').count();

  return ee.Image.cat([
      winner.add(1).rename('label'),                   // 1..9, 0 = no data
      confidence.multiply(100).round().rename('confidence'),
      count.min(255).rename('count')
    ])
    .unmask(0)                                         // masked -> 0, explicitly
    .toUint8()
    .clip(region);
}


EXPORT_ONLY.forEach(function (name) {
  var b = REGIONS[name];
  var region = ee.Geometry.Rectangle(
    [b[0] - MARGIN, b[1] - MARGIN, b[2] + MARGIN, b[3] + MARGIN], null, false);

  Export.image.toDrive({
    image: annualComposite(region),
    description: 'dw2021_' + name,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'dw2021_' + name,
    region: region,
    scale: 10,
    crs: 'EPSG:4326',
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
});


// A quick look at one region before you start 19 exports - check it looks
// like land cover, not noise. Change the name to inspect another.
var LOOK_AT = 'kansas';
var look = REGIONS[LOOK_AT];
var lookRegion = ee.Geometry.Rectangle(look, null, false);
var preview = annualComposite(lookRegion);
var PALETTE = ['000000',                          // 0 no data
  '419bdf', '397d49', '88b053', '7a87c6', 'e49635',
  'dfc35a', 'c4281b', 'a59b8f', 'b39fe1'];
Map.centerObject(lookRegion, 11);
Map.addLayer(preview.select('label'), {min: 0, max: 9, palette: PALETTE},
             'Dynamic World 2021 - ' + LOOK_AT);
Map.addLayer(preview.select('confidence'), {min: 0, max: 100},
             'confidence', false);

print('Export tasks created: ' + EXPORT_ONLY.length +
      '. Open the Tasks tab and click RUN on each.');
