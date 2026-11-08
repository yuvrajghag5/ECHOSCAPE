/*
 * Monthly Dynamic World class shares for every ECHOSCAPE region, 2021.
 *
 * The CHECK for notebooks/04_monthly_change.ipynb. The notebook says how the
 * land cover of each region changes month by month, as the radar model sees
 * it. This exports the same numbers as the OPTICAL Dynamic World sees them, so
 * the two curves can be put side by side. Radar and optical are independent
 * instruments - if both show cropland rising in spring, the change is real.
 *
 * Paste this whole file into the Earth Engine Code Editor
 * (https://code.earthengine.google.com) and press Run. It creates ONE export
 * task in the Tasks tab; click RUN. It writes a small CSV (31 regions x 12
 * months, a few tens of kB) to the Drive folder "echoscape_dw". Download it to
 *
 *     outputs/dw_monthly_2021.csv
 *
 * (outputs/, not data/, so it travels with git to the data lab).
 *
 * WHAT EACH ROW HOLDS - one region, one month:
 *
 *   region, month      e.g. kansas, 2021-07
 *   images             how many Dynamic World images touched the box that month
 *   coverage           share of the box with at least one cloud-free view, 0-1.
 *                      Low in cloudy months (tropics, northern winter): the
 *                      shares below then describe only the part that was seen,
 *                      and the notebook leaves those months out of the check.
 *   water ... snow_and_ice
 *                      share of the SEEN part of the box, 0-1, per class
 *
 * HOW A PIXEL'S MONTHLY CLASS IS DECIDED - the same way the annual labels were:
 * mean of the nine class probabilities over that month's images, then the
 * winner. Counted at 50 m, the radar's resolution. No confidence threshold
 * and no cropland rule: this is Dynamic World as it is, a reference, not
 * training labels.
 */

var YEAR = 2021;
var DRIVE_FOLDER = 'echoscape_dw';
var SCALE = 50;          // metres - the radar's grid

var CLASS_NAMES = [
  'water', 'trees', 'grass', 'flooded_vegetation', 'crops',
  'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'];

// west, south, east, north - config.bbox_for(), the same boxes as
// export_dynamic_world.js. No margin here: shares are of the box itself.
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


function monthRow(name, month) {
  var b = REGIONS[name];
  var box = ee.Geometry.Rectangle(b, null, false);
  var start = ee.Date.fromYMD(YEAR, month, 1);
  var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterBounds(box)
    .filterDate(start, start.advance(1, 'month'));

  // A month with no image at all (possible in the cloudiest tropics) would
  // give an image with no bands and stop the whole export; it becomes a fully
  // masked image instead - coverage 0, shares empty.
  var nothing = ee.Image.constant([0, 0, 0, 0, 0, 0, 0, 0, 0])
    .rename(CLASS_NAMES).updateMask(0);
  var meanProb = ee.Image(ee.Algorithms.If(
    dw.size().gt(0), dw.select(CLASS_NAMES).mean(), nothing));
  var winner = meanProb.toArray().arrayArgmax().arrayGet([0]);   // 0..8, masked if unseen

  // one 0/1 band per class; their means over the seen pixels are the shares
  var indicators = ee.Image.cat(CLASS_NAMES.map(function (c, i) {
    return winner.eq(i).rename(c);
  }));
  var shares = indicators.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: box, scale: SCALE,
    maxPixels: 1e9, tileScale: 4
  });
  var coverage = winner.mask().gt(0).unmask(0).rename('coverage').reduceRegion({
    reducer: ee.Reducer.mean(), geometry: box, scale: SCALE,
    maxPixels: 1e9, tileScale: 4
  });

  return ee.Feature(null, shares)
    .set('region', name)
    .set('month', start.format('YYYY-MM'))
    .set('images', dw.size())
    .set('coverage', coverage.get('coverage'));
}


var rows = [];
Object.keys(REGIONS).forEach(function (name) {
  for (var m = 1; m <= 12; m++) {
    rows.push(monthRow(name, m));
  }
});
var table = ee.FeatureCollection(rows);

Export.table.toDrive({
  collection: table,
  description: 'dw_monthly_2021',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'dw_monthly_2021',
  fileFormat: 'CSV',
  selectors: ['region', 'month', 'images', 'coverage'].concat(CLASS_NAMES)
});

// A quick look before exporting: Kansas, month by month, in the Console.
print('Kansas, monthly Dynamic World shares:',
      table.filter(ee.Filter.eq('region', 'kansas')));
print('One export task created. Open the Tasks tab and click RUN.');
