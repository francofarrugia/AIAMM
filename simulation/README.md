# AIAMM — Reproducibility Package

**Deep Reinforcement Learning for Concentrated Liquidity Management**
Franco Farrugia — MCAST Research Fellowship Scheme
Paper: *A Simulation-Based Empirical Benchmark of Reinforcement Learning Agents for Concentrated Liquidity Management* (accepted, *Academia AI and Applications*, December 2026 edition)

This package contains the evaluation code and dataset needed to reproduce the paper's
headline results, for the journal's data/code availability requirement.

---

## Contents

| File | Description |
|---|---|
| `13_AIAMM_V13.py` | Main evaluation script: AMM environment, DDQN/PPO/SAC agents, curriculum training, EEC bisection, GBM sensitivity sweep, and statistical testing. Running it end-to-end reproduces every table in the paper. |
| `binance_eth_usdc_hourly.npz` | 38,495 real hourly ETH/USDC candles (open, high, low, close, volume) sourced from Binance. This is the price series the main benchmark replays against. |
| `fetch_uniswap_data.py` | Script used to (re)download the raw hourly candle data from Binance's public REST API (no key required) and rebuild `binance_eth_usdc_hourly.npz`. Despite the filename, it pulls from Binance, not Uniswap — kept as-is for provenance. Optional: the `.npz` above is already included, so this is only needed to refresh or verify the data. |
| `results_reported_in_paper.txt` | Full console output from the run that produced the paper's numbers (per-seed values, Tables II–V, paper-ready summary). Included so reviewers can check the script's output format matches the manuscript without re-running the full ~20–45 minute job themselves. |

---

## What the code does

- **Main benchmark**: DDQN, PPO, and SAC agents manage a concentrated liquidity
  position (Uniswap v3-style) over real historical ETH/USDC price paths, evaluated
  against a passive (full-range) baseline. 5 seeds × 1000 curriculum episodes
  (T=100 → 300 → 500), Bonferroni-corrected one-sample t-tests vs. zero alpha.
- **GBM sensitivity sweep**: the same DDQN agent retrained and evaluated under
  synthetic Geometric Brownian Motion price paths at five volatility levels
  (σ = 0.005, 0.010, 0.015, 0.030, 0.050), 5 seeds × 100 evaluation episodes each.
  This is the analysis the paper's simulation-bounded framing and headline
  volatility-dependence result rest on.
- **Earliest Effective Cutoff (EEC)**: bisection across T ∈ {50, 100, 200, 300, 500}
  to locate the minimum deployment horizon.

Reward weights, range half-width, cooldown, and reward clipping are hard-coded at
the top of the script and match the values reported in the Methods section.

## Reproducing the results

```bash
pip install numpy torch scipy matplotlib
python 13_AIAMM_V13.py
```

Console output reproduces the tables in `results_reported_in_paper.txt` (modulo
Monte Carlo noise from the RNG — seeds are fixed and logged, but PyTorch's CUDA
kernels are not bit-exact across hardware). Full run time is roughly 20–45 minutes
on a single modern CPU core.

To confirm quickly that the environment and data load correctly without waiting
for the full run, reduce `num_episodes`/`n_seeds` when instantiating
`AIAMMSimulation` at the bottom of the script.

## Headline results (main benchmark, real ETH/USDC data, 5 seeds × 1000 episodes)

| Agent | Mean alpha vs. passive ($) | 95% CI | Win rate | Sortino |
|---|---|---|---|---|
| DDQN | −6.29 | [−6.50, −6.09] | 12.4% | −1.311 |
| PPO | −7.09 | [−7.23, −6.95] | 6.6% | −1.735 |
| SAC | −7.11 | [−7.27, −6.95] | 8.4% | −1.627 |

All three significantly underperform the passive baseline (p_adj < 0.001). Under the
GBM sensitivity sweep, DDQN alpha is negative at low/moderate volatility and turns
positive above σ≈0.015 (see `results_reported_in_paper.txt`, Table V, for the full
breakdown) — this volatility-conditional result, framed as a simulation-bounded
upper limit rather than a live-deployment claim, is the paper's central finding.

## Notes for reviewers

- The main benchmark uses **real market data**, not simulation, for the price path;
  GBM is used only for the controlled sensitivity sweep in Table V and the
  illustrative Table IVb episode. The paper's title and framing describe the
  overall study as simulation-bounded because live execution effects (gas cost
  surges, MEV, toxic order flow) are not modelled in either setting — see the
  Limitations section.
- Known deviations between the Methods section wording and this code, disclosed to
  the journal already: state space is 15-dimensional (12 core + 3 real-data candle
  features), hidden layers are 128/64, target network uses hard periodic sync every
  500 updates, and PPO/SAC use learning rate 3e-4 (DDQN uses 1e-3).
