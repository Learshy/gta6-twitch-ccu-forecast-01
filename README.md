![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Learshy/gta6-twitch-ccu-forecast-01/rank.svg)
![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Learshy/gta6-twitch-ccu-forecast-01/reward.svg)

# GTA VI Twitch CCU forecast (v2)

Forecast, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day, 2026-11-19, from everything
recorded up to the end of its first hour. Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## Model

The launch-day shape is learned from the case's comparables (past big
launches). For each forecast hour, the log-ratio of that hour's viewers to
hour 1's print is estimated across the comparables, weighting each one by
two things:

- **Size** — how close its own hour-1 print is to this case's (Gaussian
  kernel on the log scale, bandwidth 1.0). Big launches stay near their
  hour-1 level and rise modestly into the evening, while small hour-1
  launches explode later in the day.
- **Ramp** — how much of its hour-1 audience was already there a day
  before its release day (the viewers 24 hours before the release day as a
  fraction of its hour-1 print; Gaussian kernel on the log scale, bandwidth
  0.75, standardised by the pool's own dispersion). A launch whose viewers
  were already watching a day before behaves differently from one whose
  audience only arrives at unlock. This second dimension is what v2 adds,
  and it is worth about 5% of pinball loss on big launches.

For each hour the centre is the trimmed weighted mean of the log-ratios
(the outer 10% of weight dropped), and the spread is the weighted standard
deviation scaled by 0.8 (the raw cross-launch spread overstates the
conditional spread), clipped to [0.15, 1.2] in log units. Quantiles are the
median times exp(z_tau * spread). Thin data falls back to the house
baseline's persistence shape. Everything is deterministic.

## Validation (leave-one-out on the 37 usable comparables)

| model | mean loss | big launches (hour-1 >= 20k) | very big (hour-1 >= 100k) |
| --- | --- | --- | --- |
| this entry (v2) | 0.4000 | 0.2423 | 0.1986 |
| v1 (size kernel only) | 0.4027 | 0.2543 | 0.2159 |
| house baseline | 0.9296 | 0.3214 | 0.2611 |

Practice cases run clean (4/4 valid, deterministic, ~0.6 s for all four).
The kit holds no practice answers, so no per-case practice loss can be
measured; the leave-one-out figures above are the honest substitute.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Requires Node 22, standard library only, no network.
