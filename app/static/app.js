/* ECHOSCAPE prototype - the page. Talks to app/server.py. */
"use strict";

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const PLACES = [
  { name: "Mannheim", lat: 49.42, lon: 8.6 },
  { name: "Kansas", lat: 38.4, lon: -98.6 },
  { name: "Munich", lat: 48.14, lon: 11.58 },
  { name: "Po Valley", lat: 44.8, lon: 11.0 },
  { name: "Dhaka", lat: 23.78, lon: 90.4 },
  { name: "Nile at Luxor", lat: 25.7, lon: 32.65 },
];

const state = {
  info: null, result: null, key: null,
  month: 0, playing: false,
  sweep: 0,              // clock degrees, 0 = top, clockwise
  scanning: false,
};

/* ------------------------------------------------------------------ geometry */
function boxAround(lat, lon, km) {                 // as pipeline.box_around
  const halfLat = (km / 2) / 111.32;
  const halfLon = halfLat / Math.max(Math.cos(lat * Math.PI / 180), 0.01);
  return [lon - halfLon, lat - halfLat, lon + halfLon, lat + halfLat];
}
const overlaps = (a, b) => !(a[2] <= b[0] || b[2] <= a[0] || a[3] <= b[1] || b[3] <= a[1]);

function familiarity(bbox) {
  for (const r of state.info.regions) if (overlaps(bbox, r.bbox)) return { status: r.split, region: r.name };
  return { status: "unseen", region: null };
}
const TAG_TEXT = { unseen: "never seen", test: "test region", validation: "validation region", training: "training region" };
function tag(f) { return `<span class="tag ${f.status}">${TAG_TEXT[f.status]}</span>`; }

/* ------------------------------------------------------------------ point */
const SIZE_KM = 20;                    // the box the radar is fetched for

function setPoint(lat, lon) {
  lat = Math.max(-80, Math.min(80, +lat)); lon = ((+lon + 540) % 360) - 180;
  $("lat").value = lat.toFixed(3); $("lon").value = lon.toFixed(3);
  const f = familiarity(boxAround(lat, lon, SIZE_KM));
  $("familiar").innerHTML = f.region
    ? `${tag(f)} overlaps <b>${f.region.replace("_", " ")}</b>`
    : `${tag(f)} the model never trained, tuned or tested here`;
}

/* ------------------------------------------------------------------ dial */
const C = 180, R0 = 64, R1 = 150, RT = 165;
function polar(r, clockDeg) {
  const a = (clockDeg - 90) * Math.PI / 180;
  return [C + r * Math.cos(a), C + r * Math.sin(a)];
}
function wedge(ra, rb, a, b) {
  const [x1, y1] = polar(ra, a), [x2, y2] = polar(rb, a), [x3, y3] = polar(rb, b), [x4, y4] = polar(ra, b);
  const large = b - a > 180 ? 1 : 0;
  return `M${x1},${y1} L${x2},${y2} A${rb},${rb} 0 ${large} 1 ${x3},${y3} L${x4},${y4} A${ra},${ra} 0 ${large} 0 ${x1},${y1}Z`;
}
const radiusAt = (f) => Math.sqrt(R0 * R0 + f * (R1 * R1 - R0 * R0));   // area-true stacking
function el(name, attrs, parent) {
  const n = document.createElementNS(SVGNS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}

function drawDialFrame(monthLabels) {
  const svg = $("dial");
  svg.innerHTML = "";
  const defs = el("defs", {}, svg);
  const hatch = el("pattern", { id: "hatch", width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
  el("line", { x1: 0, y1: 0, x2: 0, y2: 5, stroke: "#56616b", "stroke-width": 1.4 }, hatch);
  const grad = el("linearGradient", { id: "sweepFade", x1: "0", y1: "0", x2: "1", y2: "0" }, defs);
  el("stop", { offset: "0", "stop-color": "#d3246f", "stop-opacity": "0" }, grad);
  el("stop", { offset: "1", "stop-color": "#d3246f", "stop-opacity": "0.32" }, grad);

  el("circle", { cx: C, cy: C, r: R1, class: "ring" }, svg);
  el("circle", { cx: C, cy: C, r: R0, class: "ring" }, svg);
  const slots = el("g", { id: "slots" }, svg);
  for (let i = 0; i < 12; i++) {
    el("path", { d: wedge(R0, R1, i * 30 + 1, i * 30 + 29), class: "slot", "data-i": i }, slots);
  }
  el("g", { id: "sectors" }, svg);
  const ticks = el("g", { id: "ticks" }, svg);
  for (let i = 0; i < 12; i++) {
    const [x, y] = polar(RT, i * 30 + 15);
    const t = el("text", { x, y, class: "tick", "data-i": i }, ticks);
    t.textContent = monthLabels[i];
  }
  // the sweep: a line pointing up plus a fading trail behind it, rotated as one
  const sweep = el("g", { id: "sweep" }, svg);
  el("path", { d: wedge(R0 - 6, R1 + 4, -26, 0), fill: "url(#sweepFade)" }, sweep);
  const [ax, ay] = polar(R0 - 8, 0), [bx, by] = polar(R1 + 8, 0);
  el("line", { x1: ax, y1: ay, x2: bx, y2: by, stroke: "#d3246f", "stroke-width": 2, "stroke-linecap": "round" }, sweep);
  setSweep(state.sweep);
}

function setSweep(deg) {
  state.sweep = ((deg % 360) + 360) % 360;
  const s = $("sweep");
  if (s) s.setAttribute("transform", `rotate(${state.sweep} ${C} ${C})`);
}

function drawSectors(r) {
  const g = $("sectors");
  g.innerHTML = "";
  $("slots").innerHTML = "";
  r.shares.forEach((row, i) => {
    const sec = el("g", { class: "sector", "data-i": i }, g);
    const total = row.reduce((a, b) => a + b, 0) || 1;
    let cum = 0;
    row.forEach((v, c) => {
      if (v <= 0) return;
      const a = radiusAt(cum / total), b = radiusAt((cum + v) / total);
      cum += v;
      el("path", { d: wedge(a, b, i * 30 + 1, i * 30 + 29), fill: r.classes[c].colour }, sec)
        .appendChild(Object.assign(document.createElementNS(SVGNS, "title"),
          { textContent: `${SHORT[+r.months[i].slice(5) - 1]}: ${r.classes[c].name} ${v.toFixed(1)}%` }));
    });
    sec.addEventListener("click", () => { stop(); selectMonth(i); });
  });
}

/* ------------------------------------------------------------------ months */
function monthName(m) { return SHORT[+m.slice(5) - 1]; }

function buildMonthRow(months, passes) {
  const row = $("monthRow");
  row.innerHTML = "";
  months.forEach((m, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.disabled = !state.result;
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<b>${monthName(m)}</b><small>${passes && passes[m] != null ? passes[m] + "× pass" : "'" + m.slice(2, 4)}</small>`;
    b.addEventListener("click", () => { stop(); selectMonth(i); });
    row.appendChild(b);
  });
}

function imageUrl(name) { return `/api/image/${encodeURIComponent(state.key)}/${name}`; }

function showPictures(i) {
  const m = state.result.months[i], label = `${monthName(m)} ${m.slice(0, 4)}`;
  for (const [img, kind] of [[$("imgRadar"), "radar"], [$("imgClasses"), "classes"]]) {
    img.src = imageUrl(`${kind}_${m}.png`);
    img.alt = `${kind === "radar" ? "Radar" : "Land cover"}, ${label}`;
    img.hidden = false;
  }
  $("capRadar").textContent = label;
  $("capClasses").textContent = label;
}

function clearPictures(scanning) {
  for (const id of ["imgRadar", "imgClasses"]) { $(id).hidden = true; $(id).removeAttribute("src"); }
  for (const id of ["picRadar", "picClasses"]) $(id).classList.toggle("scanning", scanning && !REDUCED);
  document.querySelectorAll(".placeholder").forEach((p) => (p.hidden = false));
  $("capRadar").textContent = "—"; $("capClasses").textContent = "—";
}

function selectMonth(i, fromSweep) {
  const r = state.result;
  if (!r) return;
  state.month = i;
  if (!fromSweep) setSweep(i * 30 + 15);
  showPictures(i);
  document.querySelectorAll("#monthRow button").forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
  document.querySelectorAll("#sectors .sector").forEach((s, j) => s.classList.toggle("dim", j !== i && !state.playing));
  document.querySelectorAll("#ticks .tick").forEach((t, j) => t.classList.toggle("on", j === i));
  const m = r.months[i];
  $("dialMonth").textContent = monthName(m);
  $("dialYear").textContent = m.slice(0, 4);
  renderMonth(i);
}

/* ------------------------------------------------------------------ readout */
function shownClasses(r) {
  return r.classes.map((c, k) => ({ ...c, k, peak: Math.max(...r.shares.map((s) => s[k])) }))
    .filter((c) => c.peak >= 0.5);
}

function sparkline(values, active, colour) {
  const max = Math.max(...values, 1), w = 58, h = 20;
  const pts = values.map((v, i) => [i * (w / 11), h - 2 - (v / max) * (h - 4)]);
  const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
  const [ax, ay] = pts[active];
  return `<svg viewBox="0 0 ${w} ${h}" aria-hidden="true"><path d="${d}" fill="none" stroke="${colour}" stroke-width="1.5"/>` +
    `<circle cx="${ax}" cy="${ay}" r="2.6" fill="#d3246f"/></svg>`;
}

function renderMonth(i) {
  const r = state.result;
  const m = r.months[i];
  $("monthTitle").textContent = `${monthName(m)} ${m.slice(0, 4)}`;
  $("monthMeta").innerHTML = `${r.passes[m]} radar pass${r.passes[m] === 1 ? "" : "es"}<br>model confidence ${r.confidence[i].toFixed(0)}%`;
  const rows = shownClasses(r).sort((a, b) => r.shares[i][b.k] - r.shares[i][a.k]);
  $("shares").innerHTML = rows.map((c) => {
    const v = r.shares[i][c.k];
    const d = i ? r.change[i - 1][c.k] : null;
    const delta = d == null ? "" : `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d).toFixed(1)}`;
    const cls = d == null || Math.abs(d) < 0.05 ? "" : d > 0 ? "up" : "down";
    return `<li class="share">
      <span class="sw" style="background:${c.colour}"></span>
      <span class="name">${c.name}</span>
      <span class="pct">${v.toFixed(1)}%</span>
      <span class="delta ${cls}" title="change since the month before, in percentage points">${delta}</span>
      ${sparkline(r.shares.map((s) => s[c.k]), i, c.colour)}
      <span class="bar"><i style="width:${v}%;background:${c.colour}"></i></span>
    </li>`;
  }).join("");
}

function renderYear(r) {
  const nf = r.model.noise_floor;
  const swings = shownClasses(r).map((c) => {
    const vals = r.shares.map((s) => s[c.k]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    return { ...c, lo, hi, loM: r.months[vals.indexOf(lo)], hiM: r.months[vals.indexOf(hi)],
             swing: hi - lo, total: r.total_change[c.k] };
  }).sort((a, b) => b.swing - a.swing).slice(0, 5);
  $("swings").innerHTML = swings.map((s) => {
    let verdict = "", vc = "noise";
    if (nf) {
      if (s.swing >= 2 * nf) { verdict = "clear change"; vc = "clear"; }
      else if (s.swing >= nf) { verdict = "above the wobble"; vc = "clear"; }
      else verdict = "within the wobble";
    }
    const path = s.loM < s.hiM                       // in the order it happened

      ? `${s.lo.toFixed(0)}% in ${monthName(s.loM)} → ${s.hi.toFixed(0)}% in ${monthName(s.hiM)}`
      : `${s.hi.toFixed(0)}% in ${monthName(s.hiM)} → ${s.lo.toFixed(0)}% in ${monthName(s.loM)}`;
    return `<li><span class="sw" style="background:${s.colour}"></span>
      <span class="what">${s.name} <small>${path}</small></span>
      <span class="verdict ${vc}">${verdict}</span></li>`;
  }).join("");
  $("noiseNote").textContent = nf
    ? `The wobble is the model's own: on the test regions, built-up land — which cannot appear or vanish in a year — still swings by about ${nf.toFixed(1)} points from month to month. A swing smaller than that is not evidence of change. Much of what does change is the season: a bare winter field can read as grass or cropland.`
    : "";

  const f = r.familiarity, checks = [];
  if (f.status === "unseen") checks.push(["", "Never seen by the model in training, tuning or testing — this is a genuine test."]);
  else if (f.status === "training") checks.push(["warn", `Overlaps ${f.region.replace("_", " ")}, a training region: the model memorised this ground's ${f.same_year ? "2021 " : ""}label, so expect flattering, flat results.`]);
  else checks.push(["", `Overlaps ${f.region.replace("_", " ")}, a ${f.status} region the model never trained on.`]);
  const passes = Object.values(r.passes), thin = r.months.filter((m) => r.passes[m] <= 1);
  checks.push([thin.length ? "warn" : "", `${Math.min(...passes)}–${Math.max(...passes)} radar passes per month on orbit track ${r.track.relative_orbit} (${r.track.direction})` +
    (thin.length ? `; ${thin.map(monthName).join(", ")} rest on one pass, so they are noisier.` : ".")]);
  checks.push([r.measured_pct < 90 ? "warn" : "", `Shares are counted on the ${r.measured_pct}% of the box the radar saw in all 12 months, so every month is measured on the same ground.`]);
  if (!r.months.every((m) => m.startsWith("2021"))) checks.push(["warn", "Outside 2021, the year it trained on. After December 2021 Sentinel-1B failed, so months carry fewer passes."]);
  const conf = r.confidence, lowI = conf.indexOf(Math.min(...conf));
  checks.push(["", `Model confidence ${(conf.reduce((a, b) => a + b, 0) / 12).toFixed(0)}% on average, lowest in ${monthName(r.months[lowI])} (${conf[lowI].toFixed(0)}%).`]);
  checks.push(["", `Ran ${r.model.name}, which reads ${r.model.features === "month" ? "one month of radar at a time" : "each month plus the year's mean and deviation"}.`]);
  if (r.model.test_miou != null) checks.push(["", `Scores mean agreement with Dynamic World: ${(100 * r.model.test_miou).toFixed(0)}% mIoU over ${r.classes.length} classes on unseen test regions.`]);
  $("checks").innerHTML = checks.map(([c, t]) => `<li class="${c}">${t}</li>`).join("");
}

/* ------------------------------------------------------------------ play */
let raf = null, last = 0, stepTimer = null;
const DEG_PER_MS = 30 / 1100;          // one month every 1.1 s

function play() {
  if (!state.result) return;
  state.playing = true;
  $("playIcon").setAttribute("d", "M4 2.5h3v11H4zM9 2.5h3v11H9z");
  $("play").setAttribute("aria-label", "Pause");
  document.querySelectorAll("#sectors .sector").forEach((s) => s.classList.remove("dim"));
  if (REDUCED) {
    stepTimer = setInterval(() => selectMonth((state.month + 1) % 12), 1100);
    return;
  }
  last = performance.now();
  const tick = (now) => {
    setSweep(state.sweep + (now - last) * DEG_PER_MS);
    last = now;
    const i = Math.floor(state.sweep / 30) % 12;
    if (i !== state.month) selectMonth(i, true);
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}
function stop() {
  state.playing = false;
  cancelAnimationFrame(raf); clearInterval(stepTimer);
  $("playIcon").setAttribute("d", "M4 2.5v11l9-5.5z");
  $("play").setAttribute("aria-label", "Play the year");
}
$("play").addEventListener("click", () => (state.playing ? stop() : play()));

function scan(on) {                     // the sweep circles while radar is fetched
  state.scanning = on;
  if (!on || REDUCED) return;
  let t0 = performance.now();
  const spin = (now) => {
    if (!state.scanning) return;
    setSweep(state.sweep + (now - t0) * 0.09); t0 = now;
    requestAnimationFrame(spin);
  };
  requestAnimationFrame(spin);
}

/* ------------------------------------------------------------------ run */
let busy = false, pollTimer = null;

async function run(ev) {
  ev.preventDefault();
  if (busy) return;
  stop();
  const body = { lat: +$("lat").value, lon: +$("lon").value, start: $("start").value, size_km: SIZE_KM, model: $("model").value };
  setPoint(body.lat, body.lon);
  busy = true; $("go").disabled = true; $("go").textContent = "Mapping…";
  $("status").hidden = false; $("error").hidden = true;
  state.result = null; $("results").hidden = true; $("empty").hidden = false;
  clearPictures(true);
  const months = Array.from({ length: 12 }, (_, i) => {
    const [y, m] = body.start.split("-").map(Number);
    const t = y * 12 + m - 1 + i;
    return `${Math.floor(t / 12)}-${String(t % 12 + 1).padStart(2, "0")}`;
  });
  drawDialFrame(months.map(monthName));
  buildMonthRow(months, null);
  $("pips").innerHTML = months.map(() => "<b></b>").join("");
  $("dialMonth").textContent = "…"; $("dialYear").textContent = "listening";
  scan(true);
  try {
    const res = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const s = await res.json();
    if (!res.ok) throw new Error(s.detail || "The server refused the request.");
    state.key = s.key;
    follow(s);
  } catch (e) { fail(e.message); }
}

function follow(s) {
  const order = ["searching", "downloading", "mapping", "done"];
  const at = order.indexOf(s.stage);
  document.querySelectorAll(".ladder li").forEach((li) => {
    const k = order.indexOf(li.dataset.stage);
    li.classList.toggle("on", k === at);
    li.classList.toggle("done", at > k || s.stage === "done");
  });
  const pips = $("pips").children;
  s.months.forEach((m, i) => {
    const got = s.months_done.includes(m);
    if (pips[i]) pips[i].classList.toggle("got", got);
    const slot = document.querySelector(`#slots [data-i="${i}"]`);
    if (slot) slot.classList.toggle("got", got);
  });
  $("statusText").textContent = s.message;
  $("clock").textContent = `${Math.floor(s.elapsed / 60)}:${String(Math.floor(s.elapsed % 60)).padStart(2, "0")}`;
  if (s.stage === "failed") return fail(s.error);
  if (s.stage === "done") return finish();
  pollTimer = setTimeout(async () => {
    try {
      const res = await fetch(`/api/job/${encodeURIComponent(state.key)}`);
      if (!res.ok) throw new Error((await res.json()).detail);
      follow(await res.json());
    } catch (e) { fail(e.message); }
  }, 900);
}

function fail(msg) {
  scan(false); busy = false;
  $("go").disabled = false; $("go").textContent = "Map 12 months";
  $("error").hidden = false; $("error").textContent = msg || "Something stopped the run.";
  $("dialMonth").textContent = "—"; $("dialYear").textContent = "";
  clearPictures(false);
}

async function finish() {
  const res = await fetch(`/api/result/${encodeURIComponent(state.key)}`);
  const r = await res.json();
  state.result = r;
  scan(false); busy = false;
  $("go").disabled = false; $("go").textContent = "Map 12 months";
  // warm the browser cache so playing the year never waits on an image
  for (const m of r.months) for (const k of ["classes", "radar"]) new Image().src = `/api/image/${encodeURIComponent(r.key)}/${k}_${m}.png`;
  clearPictures(false);
  document.querySelectorAll(".placeholder").forEach((p) => (p.hidden = true));
  $("boxNote").textContent = `Each picture covers a ${r.size_km} × ${r.size_km} km box centred on ${r.point.lat.toFixed(3)}, ${r.point.lon.toFixed(3)}, at 50 m per pixel. Dark radar is smooth ground or water; bright is rough or built.`;
  drawDialFrame(r.months.map(monthName));
  drawSectors(r);
  buildMonthRow(r.months, r.passes);
  $("play").disabled = false;
  $("empty").hidden = true; $("results").hidden = false;
  renderYear(r);
  selectMonth(0);
  $("statusText").textContent = `Done in ${Math.round(r.elapsed_s)} s` + (r.elapsed_s < 2 ? " (from the cache)" : "");
}

/* ------------------------------------------------------------------ controls */
["lat", "lon"].forEach((id) => $(id).addEventListener("change", () => setPoint($("lat").value, $("lon").value)));
$("start").addEventListener("change", () => {
  $("startHint").textContent = $("start").value.startsWith("2021")
    ? "The model trained on 2021."
    : "Outside 2021: fine to try, but it is not the year the model learned.";
});
/* The model card and the hint follow the dropdown, so what the page claims is
   always the model it is about to run - not the one it opened on. */
function describeModel(name) {
  const m = (state.info.models || []).find((r) => r.name === name) || state.info.model;
  const pct = (v) => (v == null ? "not measured" : `${(100 * v).toFixed(1)}%`);
  const params = m.parameters ? `${(m.parameters / 1e6).toFixed(2)} M parameters` : "";
  $("modelHint").textContent =
    `${m.classes - 1} classes, reads ${m.features === "month" ? "one month of radar" : "the month plus the year's mean and deviation"}.`
    + (name === state.info.default_model ? " Best validation score - the default." : "");
  $("modelcard").innerHTML =
    `Model <strong>${m.name}</strong>${params ? ", " + params : ""}, trained from scratch on Sentinel-1 alone. `
    + `Agreement with Dynamic World: <strong>${pct(m.val_miou)}</strong> validation, `
    + `<strong>${pct(m.test_miou)}</strong> on unseen test regions (mIoU over ${m.classes - 1} classes). `
    + `Validation regions are each dominated by one class, so the test figure is the honest one.`;
}

$("model").addEventListener("change", () => describeModel($("model").value));
$("form").addEventListener("submit", run);

/* ------------------------------------------------------------------ start */
(async function init() {
  const res = await fetch("/api/info");
  state.info = await res.json();
  const i = state.info;
  $("start").innerHTML = i.start_months.slice().reverse().map((m) =>
    `<option value="${m}" ${m === i.default_start ? "selected" : ""}>${SHORT[+m.slice(5) - 1]} ${m.slice(0, 4)} – ${SHORT[(+m.slice(5) + 10) % 12]} ${+m.slice(0, 4) + (m.slice(5) === "01" ? 0 : 1)}</option>`).join("");
  /* The class count is in every label on purpose: a six-class mIoU and a
     nine-class mIoU are not the same measurement, and side by side without it
     the smaller class list just looks like a better model. */
  $("model").innerHTML = (i.models || [i.model]).map((m) => {
    const val = m.val_miou == null ? "no score" : `${(100 * m.val_miou).toFixed(1)}% val`;
    return `<option value="${m.name}" ${m.name === i.default_model ? "selected" : ""}>`
      + `${m.name} — ${m.classes - 1} classes, ${val}</option>`;
  }).join("");
  describeModel($("model").value);
  $("places").innerHTML = PLACES.map((p, k) => {
    const f = familiarity(boxAround(p.lat, p.lon, 20));
    return `<button type="button" class="chip" data-k="${k}">${p.name}<i class="${f.status}">${TAG_TEXT[f.status]}</i></button>`;
  }).join("");
  $("places").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => {
    const p = PLACES[+b.dataset.k];
    if (!busy) setPoint(p.lat, p.lon);
  }));
  drawDialFrame(SHORT);
  buildMonthRow(SHORT.map((_, k) => `2021-${String(k + 1).padStart(2, "0")}`), null);
  setPoint(49.42, 8.6);
})().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin", `<p class="error" style="padding:12px">Could not reach the server: ${e.message}. Start it with app\\server.py.</p>`);
});
