// Forecast a launch day's Twitch viewers, hour by hour, from the first hour's print.
//
// Model: the launch-day shape is learned from the case's comparables (past big
// launches). For each forecast hour, the log-ratio of that hour's viewers to hour
// 1's print is estimated across the comparables, weighting each comparable by how
// close its own hour-1 print is to this case's (Gaussian kernel on log hour-1,
// bandwidth 1.0). That conditioning is the whole point: big launches stay near
// their hour-1 level and rise modestly into the evening, while small hour-1
// launches explode later in the day, so an unweighted median shape would be badly
// wrong for a huge launch like GTA VI. The weighted median log-ratio is the
// centre; the weighted spread of log-ratios is the spread (clipped to [0.15, 1.2]
// in log units). Quantiles are the median times exp(z_tau * spread). When a case
// carries too few usable comparables, or an hour has too little kernel weight,
// that hour falls back to the house baseline's persistence shape.
//
// Everything is deterministic: no randomness, no clock, no iteration-order
// dependence. Input is read from $MAKEFUN_INPUT_DIR (else /input), output written
// to $MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in, in name order.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Gaussian kernel bandwidth on log(hour-1 print) when weighting comparables.
const KERNEL_BW = 1.0;
// Spread of an hour's log-ratio, clipped to this range (log units).
const SPREAD_FLOOR = 0.15;
const SPREAD_CAP = 1.2;
// Minimum total kernel weight for an hour before its comparables are trusted.
const MIN_WEIGHT = 0.5;
// A comparable counts as a usable shape only with hour 1 and at least this many
// of the release day's 24 hours present.
const MIN_SHAPE_HOURS = 20;
// The house baseline's persistence spread, per square-root hour (fallback).
const BASE_SIGMA = 0.15;
// Output bounds from the rules.
const MAX_VIEWERS = 1000000000;
const MAX_BYTES = 65536;

interface HourlyPoint {
  from: string;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  id: string;
  levels?: number[];
  known?: unknown;
  forecast?: Array<{ hour: number }>;
  history?: Array<{ at: string; viewers: number }>;
  history_hourly?: Array<{ from: string; viewers: number }>;
  comparables?: Comparable[];
}

/** Horner evaluation of a polynomial given by its coefficients, highest first. */
function horner(coef: number[], x: number): number {
  let r = 0;
  for (let i = 0; i < coef.length; i++) r = r * x + coef[i];
  return r;
}

/** The standard normal's quantile at p (Acklam's rational approximation). */
function normQuantile(p: number): number {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return horner(c, q) / (horner(d, q) * q + 1);
  }
  if (p <= pHigh) {
    const q = p - 0.5;
    const r = q * q;
    return (horner(a, r) * q) / (horner(b, r) * r + 1);
  }
  const q = Math.sqrt(-2 * Math.log(1 - p));
  return -horner(c, q) / (horner(d, q) * q + 1);
}

/** x rounded to 2 decimals (half up), as the house baseline rounds its quantiles. */
function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

/** x clamped into [lo, hi]. */
function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/** Weighted median of (value, weight) pairs; pairs are sorted by value first. */
function weightedMedian(pairs: Array<[number, number]>): number {
  const sorted = pairs.slice().sort((a, b) => a[0] - b[0]);
  let total = 0;
  for (const [, w] of sorted) total += w;
  let acc = 0;
  for (const [v, w] of sorted) {
    acc += w;
    if (acc >= total / 2) return v;
  }
  return sorted[sorted.length - 1][0];
}

/** Weighted mean and standard deviation of (value, weight) pairs. */
function weightedMeanSd(pairs: Array<[number, number]>): [number, number] {
  let total = 0;
  let mean = 0;
  for (const [v, w] of pairs) {
    total += w;
    mean += v * w;
  }
  mean /= total;
  let variance = 0;
  for (const [v, w] of pairs) variance += w * (v - mean) * (v - mean);
  variance /= total;
  return [mean, Math.sqrt(Math.max(variance, 0))];
}

/**
 * A comparable's release-day hours, as a map from hour (1..24, relative to its
 * day_start) to viewers. Points outside the day are dropped; a viewer count
 * below 1 counts as 1, since the score takes its log.
 */
function releaseDayHours(c: Comparable): Map<number, number> {
  const out = new Map<number, number>();
  const start = Date.parse(c.day_start);
  if (Number.isNaN(start) || !Array.isArray(c.hourly)) return out;
  for (const p of c.hourly) {
    const t = Date.parse(p.from);
    if (Number.isNaN(t)) continue;
    const h = Math.floor((t - start) / 3600000) + 1;
    if (h >= 1 && h <= 24 && typeof p.viewers === 'number' && Number.isFinite(p.viewers)) {
      out.set(h, Math.max(1, p.viewers));
    }
  }
  return out;
}

/**
 * Hour 1's print for a case: the `known` row for hour 1 when it is there (an
 * array or a single object), else the last whole-hour mean before the
 * checkpoint, else the last history point, else 1.
 */
function hourOnePrint(c: CaseFile): number {
  const known = c.known;
  if (Array.isArray(known)) {
    for (const row of known) {
      const r = row as { hour?: unknown; viewers?: unknown };
      if (r && r.hour === 1 && typeof r.viewers === 'number' && Number.isFinite(r.viewers)) {
        return Math.max(1, r.viewers);
      }
    }
  } else if (known && typeof known === 'object') {
    const r = known as { viewers?: unknown };
    if (typeof r.viewers === 'number' && Number.isFinite(r.viewers)) return Math.max(1, r.viewers);
  }
  const hourly = c.history_hourly;
  if (Array.isArray(hourly) && hourly.length > 0) {
    const last = hourly[hourly.length - 1];
    if (last && typeof last.viewers === 'number' && Number.isFinite(last.viewers)) {
      return Math.max(1, last.viewers);
    }
  }
  const hist = c.history;
  if (Array.isArray(hist) && hist.length > 0) {
    const last = hist[hist.length - 1];
    if (last && typeof last.viewers === 'number' && Number.isFinite(last.viewers)) {
      return Math.max(1, last.viewers);
    }
  }
  return 1;
}

/** The house baseline's persistence quantiles for one hour (the fallback shape). */
function baselineQuantiles(p1: number, h: number, z: number[]): number[] {
  const s = BASE_SIGMA * Math.sqrt(h - 1);
  return z.map((zi) => round2(p1 * Math.exp(zi * s)));
}

/** Forecast one case: 23 rows, hours 2..24, five quantiles each. */
function forecastCase(c: CaseFile): { hours: Array<{ hour: number; q: number[] }> } {
  const levels = Array.isArray(c.levels) && c.levels.length > 0 ? c.levels : [0.05, 0.25, 0.5, 0.75, 0.95];
  const z = levels.map((t) => normQuantile(clamp(t, 0.000001, 0.999999)));
  const p1 = hourOnePrint(c);
  const logP1 = Math.log(p1);

  // Usable comparable shapes: hour 1 present and at least MIN_SHAPE_HOURS of
  // the release day's 24 hours present.
  const shapes: Array<{ logH1: number; hours: Map<number, number> }> = [];
  if (Array.isArray(c.comparables)) {
    for (const comp of c.comparables) {
      if (!comp || typeof comp.day_start !== 'string') continue;
      const hours = releaseDayHours(comp);
      if (!hours.has(1) || hours.size < MIN_SHAPE_HOURS) continue;
      shapes.push({ logH1: Math.log(hours.get(1) as number), hours });
    }
  }

  const hours: Array<{ hour: number; q: number[] }> = [];
  // The output is exactly hours 2..24. The case's forecast array lists them;
  // if it ever listed anything else, fall back to the full range so the output
  // stays valid.
  const listed = Array.isArray(c.forecast)
    ? c.forecast.map((f) => f.hour).filter((h) => Number.isInteger(h) && h >= 2 && h <= 24).sort((a, b) => a - b)
    : [];
  const want = listed.length === 23 && listed[0] === 2 && listed[22] === 24
    ? listed
    : Array.from({ length: 23 }, (_, i) => i + 2);
  const seen = new Set<number>();
  for (const h of want) {
    if (seen.has(h)) continue;
    seen.add(h);
    let q: number[];
    // Kernel-weighted log-ratios for this hour across the usable shapes.
    const pairs: Array<[number, number]> = [];
    for (const s of shapes) {
      const v = s.hours.get(h);
      if (v === undefined) continue;
      const w = Math.exp(-0.5 * Math.pow((s.logH1 - logP1) / KERNEL_BW, 2));
      pairs.push([Math.log(v / (s.hours.get(1) as number)), w]);
    }
    let totalW = 0;
    for (const [, w] of pairs) totalW += w;
    if (pairs.length === 0 || totalW < MIN_WEIGHT) {
      q = baselineQuantiles(p1, h, z);
    } else {
      const med = weightedMedian(pairs);
      const [, sd] = weightedMeanSd(pairs);
      const spread = clamp(sd, SPREAD_FLOOR, SPREAD_CAP);
      q = z.map((zi) => round2(p1 * Math.exp(med + zi * spread)));
    }
    // Enforce the output contract: finite, above 0, at most 1e9, non-decreasing.
    const clean: number[] = [];
    let prev = 1;
    for (const v of q) {
      const x = clamp(Number.isFinite(v) ? v : p1, 1, MAX_VIEWERS);
      const y = Math.max(x, prev);
      clean.push(y);
      prev = y;
    }
    hours.push({ hour: h, q: clean });
  }
  return { hours };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const out = forecastCase(c);
  const text = `${JSON.stringify({ hours: out.hours })}\n`;
  if (text.length > MAX_BYTES) throw new Error(`${file} forecast exceeds ${MAX_BYTES} bytes`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), text);
}
