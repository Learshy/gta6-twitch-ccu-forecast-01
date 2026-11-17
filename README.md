![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Learshy/gta6-twitch-ccu-forecast-01/rank.svg)
![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Learshy/gta6-twitch-ccu-forecast-01/reward.svg)

# GTA VI Twitch CCU forecast

Forecast, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day, 2026-11-19, from everything
recorded up to the end of its first hour. Entry for the
[make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## Model

The launch-day shape is learned from the case's comparables (past big launches).
For each forecast hour, the log-ratio of that hour's viewers to hour 1's print is
estimated across the comparables, weighting each comparable by how close its own
hour-1 print is to this case's (Gaussian kernel on log hour-1, bandwidth 1.0).
That conditioning is the whole point: big launches stay near their hour-1 level
and rise modestly into the evening, while small hour-1 launches explode later in
the day, so an unweighted median shape would be badly wrong for a huge launch
like GTA VI. The weighted median log-ratio is the centre; the weighted spread of
log-ratios is the spread (clipped to [0.15, 1.2] in log units). Quantiles are the
median times exp(z_tau * spread). When a case carries too few usable comparables,
or an hour has too little kernel weight, that hour falls back to the house
baseline's persistence shape.

## Run

Node 22, standard library only, no network:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` in the input directory (in name order) and writes one
`<id>.out` per case: `{"hours": [...]}` with 23 rows (hours 2–24), five
quantiles each. Deterministic: two runs write byte-identical outputs.

## Verification (leave-one-out on the kit's comparables)

Each of the 37 usable comparables was held out in turn, forecast from the other
36, and scored against its real hours 2–24 with the pinball loss on the log
scale. The program was run end to end for this, not a prototype:

| model | mean loss | typical miss | 90% coverage |
| --- | --- | --- | --- |
| this entry | 0.4027 | 83.6% | 0.880 |
| house baseline | 0.9297 | 213.3% | 0.624 |

On big hour-1 launches only (hour-1 print ≥ 20,000 — the regime GTA VI will be
in): this entry 0.2543 vs house baseline 0.3214. The house baseline scores
about 0.387 on the practice cases; the practice answers are held by the server,
so no per-case practice loss can be computed locally — the leave-one-out figure
above is the honest estimate instead.

Runtime is about 0.3 s for all four practice cases (limit 8 s per case), and
outputs are about 1.5 KB each (limit 65,536 bytes).
