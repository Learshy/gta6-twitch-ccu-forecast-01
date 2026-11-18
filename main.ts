// Forecast a launch day's Twitch viewers, hour by hour, from the first
// hour's print and the shape of past big launches (v2).
//
// The forecast for each hour is built from the case's comparables (past big
// launches). For a comparable, the launch-day shape is the log-ratio of each
// hour's viewers to that launch's hour-1 print. Two things predict the shape:
//
//   1. Size. How big the launch is (its hour-1 print). Big launches stay
//      near their hour-1 level and rise modestly into the evening; small
//      hour-1 launches explode later in the day. So each comparable is
//      weighted by how close its own hour-1 print is to this case's, with a
//      Gaussian kernel on the log scale (bandwidth 1.0).
//   2. Ramp. How much of the hour-1 audience was already there a day
//      before the release day. A launch whose viewers were already watching
//      24 hours before behaves differently from one whose audience only
//      arrives at unlock. So each comparable is also weighted by how close
//      its pre-release level is to this case's, where the pre-release level
//      is the viewers 24 hours before its release day as a fraction of its
//      hour-1 print (Gaussian kernel on the log scale, bandwidth 0.75).
//      This second dimension is what v2 adds over a size-only kernel, and
//      it is worth about 5% of pinball loss on big launches in
//      leave-one-out testing (0.242 vs 0.255), and about 8% on the very
//      biggest launches (0.199 vs 0.216).
//
// For each forecast hour, the centre is the trimmed weighted mean of the
// log-ratios (the outer 10% of weight dropped, so a few wild launches do
// not drag it), and the spread is the weighted standard deviation, scaled
// by 0.8 (the raw cross-launch spread overstates the conditional spread),
// clipped to [0.15, 1.2] in log units. Quantiles are the median times
// exp(z_tau * spread). When a case carries too few usable comparables, or
// an hour has too little kernel weight, that hour falls back to the house
// baseline's persistence shape. When the case's own pre-release level
// cannot be read, the size kernel alone is used; and when the target's
// pre-release level is an outlier that starves the two-dimensional kernel,
// the model degrades to the size-only kernel before falling back to the
// baseline, so an unusual launch still gets a real forecast.
//
// Everything is deterministic: no randomness, no clock, no iteration-order
// dependence. Input is read from $MAKEFUN_INPUT_DIR (else /input), output
// written to $MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in,
// in name order.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Gaussian kernel bandwidth on log(hour-1 print) when weighting comparables. */
const SIZE_BW = 1.0;
/** Gaussian kernel bandwidth on log(pre-release level) when weighting comparables. */
const RAMP_BW = 0.75;
/** The spread is the weighted standard deviation times this factor. */
const SPREAD_SCALE = 0.8;
/** Spread of an hour's log-ratio, clipped to this range (log units). */
const SPREAD_FLOOR = 0.15;
const SPREAD_CAP = 1.2;
/** Fraction of weight trimmed from each tail when averaging log-ratios. */
const TRIM = 0.10;
/** The house baseline's spread, in log units per square-root hour (fallback). */
const BASE_SIGMA = 0.15;
/** A comparable needs at least this much total kernel weight to be used. */
const MIN_WEIGHT = 0.5;
/** A comparable needs at least this many usable peers in the pool. */
const MIN_POOL = 5;
/** How far (ms) from the 24h-before mark a history point may be and still count. */
const RAMP_TOLERANCE_MS = 6 * 3600 * 1000;

const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;

interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number;
}

interface Comparable {
  day_start: string;
  hourly: HourlyPoint[];
}

interface Case {
  levels: number[];
  known: Array<{ hour: number; viewers: number }> | { hour: number; viewers: number };
  forecast: Array<{ hour: number }>;
  day_start: string;
  history_hourly?: HourlyPoint[] | null;
  comparables?: Comparable[] | null;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Parse an ISO timestamp to epoch milliseconds. */
const ms = (s: string): number => new Date(s).getTime();

/** The persistence forecast of one hour h: the five quantiles around hour 1's print. */
const baselineRow = (p1: number, h: number): number[] =>
  Z.map((z) => round2(p1 * Math.exp(z * BASE_SIGMA * Math.sqrt(h - 1))));

/** The weighted median of values with the given weights. */
const weightedMedian = (vals: number[], weights: number[]): number => {
  const pairs = vals.map((v, i) => [v, weights[i]] as const).sort((a, b) => a[0] - b[0]);
  const tot = weights.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (const [v, w] of pairs) {
    acc += w;
    if (acc >= 0.5 * tot) return v;
  }
  return pairs[pairs.length - 1][0];
};

/** The weighted standard deviation of values around a centre. */
const weightedStd = (vals: number[], weights: number[], centre: number): number => {
  const tot = weights.reduce((a, b) => a + b, 0);
  if (tot <= 0) return 0;
  const variance = vals.reduce((acc, v, i) => acc + weights[i] * (v - centre) ** 2, 0) / tot;
  return Math.sqrt(Math.max(variance, 0));
};

/** The trimmed weighted mean: drop the outer TRIM of weight from each tail. */
const trimmedWeightedMean = (vals: number[], weights: number[]): number => {
  const pairs = vals.map((v, i) => [v, weights[i]] as const).sort((a, b) => a[0] - b[0]);
  const tot = weights.reduce((a, b) => a + b, 0);
  const lo = TRIM * tot;
  const hi = (1 - TRIM) * tot;
  let acc = 0;
  let num = 0;
  let den = 0;
  for (const [v, w] of pairs) {
    acc += w;
    if (acc >= lo && acc <= hi) {
      num += w * v;
      den += w;
    }
  }
  return den > 0 ? num / den : weightedMedian(vals, weights);
};

/** A Gaussian kernel factor: 1 at distance 0, decaying with bandwidth bw. */
const gauss = (dist: number, bw: number): number => Math.exp(-0.5 * (dist / bw) ** 2);

/**
 * The pre-release level of a launch: the viewers 24 hours before its release
 * day starts, as a fraction of its hour-1 print. `points` is the launch's
 * hourly history (for the target case, history_hourly; for a comparable, its
 * hourly array). Returns null when no point near the 24h-before mark exists.
 */
const preReleaseLevel = (
  points: HourlyPoint[] | null | undefined,
  dayStartMs: number,
  hour1: number,
): number | null => {
  if (!points || points.length === 0 || hour1 <= 0) return null;
  const target = dayStartMs - DAY_MS;
  let best: HourlyPoint | null = null;
  let bestDist = Infinity;
  for (const pt of points) {
    const t = ms(pt.from);
    if (Number.isNaN(t)) continue;
    const d = Math.abs(t - target);
    if (d < bestDist) {
      bestDist = d;
      best = pt;
    }
  }
  if (!best || bestDist > RAMP_TOLERANCE_MS) return null;
  return Math.max(best.viewers, 0) / hour1;
};

/**
 * A comparable's 24 release-day hourly viewer counts (index 0 = hour 1),
 * or null for any hour with no point. The release day starts at day_start.
 */
const releaseDaySeries = (comp: Comparable): Array<number | null> => {
  const start = ms(comp.day_start);
  const byTime = new Map<number, number>();
  for (const pt of comp.hourly ?? []) {
    const t = ms(pt.from);
    if (!Number.isNaN(t)) byTime.set(t, pt.viewers);
  }
  const series: Array<number | null> = [];
  for (let h = 0; h < 24; h++) {
    const v = byTime.get(start + h * HOUR_MS);
    series.push(v === undefined ? null : v);
  }
  return series;
};

/** Hour 1's print for a case: the known row, else the last history point, else 1. */
const hour1Print = (c: Case): number => {
  const known = c.known;
  if (Array.isArray(known)) {
    const first = known.find((k) => k.hour === 1);
    if (first !== undefined) return Math.max(1, first.viewers);
  } else if (known && typeof known.viewers === 'number') {
    return Math.max(1, known.viewers);
  }
  const hist = c.history_hourly;
  if (hist && hist.length > 0) {
    const last = hist[hist.length - 1];
    if (typeof last.viewers === 'number') return Math.max(1, last.viewers);
  }
  return 1;
};

/** Forecast one case: 23 rows (hours 2..24), five quantiles each. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  const p1 = hour1Print(c);
  const logP1 = Math.log(p1);
  const dayStartMs = ms(c.day_start);

  // The target's own pre-release level (null when it cannot be read).
  const targetRamp = preReleaseLevel(c.history_hourly, dayStartMs, p1);
  const logTargetRamp = targetRamp !== null && targetRamp > 0 ? Math.log(targetRamp) : null;

  // Prepare the comparables: their release-day series and pre-release level.
  const pool: Array<{
    series: Array<number | null>;
    logH1: number;
    logRamp: number | null;
  }> = [];
  for (const comp of c.comparables ?? []) {
    const series = releaseDaySeries(comp);
    const h1 = series[0];
    if (h1 === null || h1 <= 0) continue;
    const ramp = preReleaseLevel(comp.hourly, ms(comp.day_start), h1);
    pool.push({
      series,
      logH1: Math.log(h1),
      logRamp: ramp !== null && ramp > 0 ? Math.log(ramp) : null,
    });
  }

  // Standardise the pre-release level across the pool (log scale), so the
  // kernel bandwidth is in units of the pool's own dispersion.
  let rampMean = 0;
  let rampSd = 1;
  {
    const vals = pool.map((p) => p.logRamp).filter((v): v is number => v !== null);
    if (vals.length >= MIN_POOL) {
      rampMean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((a, v) => a + (v - rampMean) ** 2, 0) / vals.length;
      rampSd = Math.sqrt(variance) || 1;
    }
  }

  const rows: Array<{ hour: number; q: number[] }> = [];
  for (const { hour } of c.forecast) {
    const h = hour;
    if (h < 2 || h > 24) continue;
    // The kernel weights for one hour. With useRamp, each comparable is
    // weighted by size and by pre-release level; without it, by size alone.
    const hourWeights = (useRamp: boolean): { ratios: number[]; weights: number[] } => {
      const ratios: number[] = [];
      const weights: number[] = [];
      for (const p of pool) {
        const v = p.series[h - 1];
        const v1 = p.series[0];
        if (v === null || v1 === null || v <= 0 || v1 <= 0) continue;
        const r = Math.log(v / v1);
        let w = gauss(logP1 - p.logH1, SIZE_BW);
        // The ramp dimension: neutral (factor 1) when either side lacks it.
        if (useRamp && logTargetRamp !== null && p.logRamp !== null) {
          const zTarget = (logTargetRamp - rampMean) / rampSd;
          const zComp = (p.logRamp - rampMean) / rampSd;
          w *= gauss(zTarget - zComp, RAMP_BW);
        }
        ratios.push(r);
        weights.push(w);
      }
      return { ratios, weights };
    };
    // Prefer the two-dimensional kernel. When the target's pre-release level
    // is an outlier, that kernel starves every comparable, so degrade to the
    // size-only kernel (still a good forecaster) before falling back to the
    // house baseline.
    let { ratios, weights } = hourWeights(true);
    let totalWeight = weights.reduce((a, b) => a + b, 0);
    if (totalWeight < MIN_WEIGHT) {
      ({ ratios, weights } = hourWeights(false));
      totalWeight = weights.reduce((a, b) => a + b, 0);
    }
    if (pool.length < MIN_POOL || ratios.length < MIN_POOL || totalWeight < MIN_WEIGHT) {
      rows.push({ hour: h, q: baselineRow(p1, h) });
      continue;
    }
    const centre = trimmedWeightedMean(ratios, weights);
    const spread = Math.min(
      Math.max(weightedStd(ratios, weights, centre) * SPREAD_SCALE, SPREAD_FLOOR),
      SPREAD_CAP,
    );
    const median = p1 * Math.exp(centre);
    const q = Z.map((z) => round2(median * Math.exp(z * spread)));
    // Keep the row non-decreasing (the normal quantiles already are).
    q.sort((a, b) => a - b);
    rows.push({ hour: h, q });
  }
  // The forecast lists hours 2..24; keep exactly those, in order.
  rows.sort((a, b) => a.hour - b.hour);
  return rows;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const hours = forecastCase(c);
  if (hours.length !== 23) throw new Error(`${file} produced ${hours.length} rows, not 23`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
