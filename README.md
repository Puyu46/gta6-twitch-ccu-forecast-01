# gta6-twitch-ccu-forecast-01 — Neighbour-Curve Forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Puyu46/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![make.fun reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Puyu46/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

A program for the [make.fun](https://make.fun/c/gta6-twitch-ccu-forecast-01)
`gta6-twitch-ccu-forecast-01` competition: after the first hour of Grand Theft
Auto VI's release day (2026-11-19 UTC) it forecasts Twitch concurrent viewers
in the GTA VI category for each of hours 2–24, as five quantiles per hour
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the log scale.

## Model

The kit ships 38 past big launches with their full release-day hourly viewer
curves. Two facts about those curves drive the design:

- **Hour 1 spans four orders of magnitude (164 to 875k viewers), while the
  release-day peak of almost every launch sits in a narrow band (~50k to
  1.2M).** A launch whose hour 1 was already near its peak stays flat; a
  launch whose hour 1 was pre-unlock hype ramps 10–100× by the evening.
  So the level of each hour is predicted from the *distribution* of similar
  launches, never by scaling hour 1 with one global curve.
- **The shape of the ramp depends on how the game unlocked.** A rolling
  midnight-local launch climbs in steps as timezones unlock (US East around
  05:00Z); a single global unlock at 14:00Z stays dead until hour 15 and
  then explodes; an early-access launch is already at full audience in
  hour 1. The unlock text shipped with each comparable — and the target
  case's own notes — is parsed for that signal.

For each forecast hour, every comparable gets a weight: a Gaussian kernel on
log hour-1 viewers (bandwidth 1.0) times an unlock-kind weight (1 for the
same kind, 0.10 for a different known kind, 1 when either side is unknown;
two global unlocks also decay with the gap between their unlock hours) times
0.5 for the kit's "weak" tier. Two weighted distributions are formed — the
log ratios `log(v_h / v1)` and the absolute levels `log(v_h)` — each
summarised by its weighted median and interquartile range, giving a
Gaussian-style log spread `sigma = max(IQR / 1.349, 0.25)`. Quantiles are
`median + z_tau * 0.85 * sigma`, blended in log space as 75% ratio forecast
(anchored to the target's own hour-1 print) + 25% absolute forecast (which
stops a tiny hour-1 print from extrapolating to an absurd peak).

| Parameter | Value |
| --- | --- |
| Neighbour kernel bandwidth (log v1) | 1.0 |
| Unlock-kind mismatch weight | 0.10 |
| Global-unlock hour bandwidth | 4 h |
| Weak-tier weight | 0.5 |
| Spread multiplier | 0.85 |
| Log-spread floor | 0.25 |
| Ratio / absolute blend | 0.75 / 0.25 |
| Fallback (no usable comparable) | persistence, sigma 0.15·√(h−1) |

## Validation (offline)

The kit's scorer is server-only, so validation is leave-one-out
cross-validation over the comparables: each comparable in turn plays the
target (its hour-1 print + the other 37 comparables), scored with the exact
pinball loss against its real release-day curve.

| Model | Mean LOO loss (37 comparables) |
| --- | --- |
| Persistence baseline (house starter) | 0.930 |
| Plain median log-ratio curve | 0.794 |
| **This model** | **0.376** |

These are end-to-end numbers: the shipped `main.py` itself was run on
pseudo-cases built from each comparable in turn (its hour-1 print + the
other 37 comparables) and scored with the exact pinball loss — 0.376
against 0.930 for the starter run through the same harness, 0.14 s per
case including interpreter startup.

On the large hour-1 launches (v1 ≥ 20k — the regime the practice targets
and GTA VI sit in) the loss is ~0.23, and on rolling-midnight launches
~0.26. The remaining error is dominated by launch-to-launch variation in
peak size and in the exact unlock hour, which no hour-1 print can remove.
For the final case the neighbour set grows to 42 comparables including the
four practice launches themselves — all rolling-midnight console launches,
GTA VI's closest analogues.

## Running

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Pure Python 3.13 standard library, offline, deterministic (no randomness,
no clock, no network, no files outside the case directory). Reads every
`<id>.in` in the input directory and writes `<id>.out` with 23 rows, hours
2–24, five quantiles each (positive, at most 1e9, non-decreasing). Runtime
is well under a second per case.

## Files

- `main.py` — the whole entry point and model.
- `README.md` — this file.
