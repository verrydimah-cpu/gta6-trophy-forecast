# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/verrydimah-cpu/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/verrydimah-cpu/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for the earn rate of every Grand Theft Auto VI PS5 trophy on Exophase at 2026-11-26T00:00:00Z — the percent of tracked players who will have earned each trophy one week after launch. `main.py` (Python 3.13, standard library only, no network) reads a case file holding every house read of the game's trophy page up to a checkpoint, and writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of each trophy's rate at the target.

The model fits a robust trend line of logit(rate) against ln(1 + hours since release) over the last 24 hours of reads — the early launch-night page is a different regime — and blends in a minority view from saturating fits to the earners and the tracked-player pool separately, which catches dilution when the pool grows faster than a trophy's earners. The spread is calibrated on synthetic launch weeks with a join-and-earn ground truth, where it beat both house baselines (persistence and the house curve) on every generated case. Forecasts are deterministic, byte-identical across runs, and take well under a second per case.

## Run it

Point `MAKEFUN_INPUT_DIR` at a directory holding one or more case files (`<id>.in`); one `<id>.out` forecast is written per case to `MAKEFUN_OUTPUT_DIR`:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=hz6kvzj2&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
