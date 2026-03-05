# AIAMM V10 - AI-Assisted Automated Market Maker

**Deep Reinforcement Learning for Concentrated Liquidity Management**

> Franco Farrugia - Department of Computer Science and Artificial Intelligence, University of Malta

---

## Overview

AIAMM is a research framework that applies three deep reinforcement learning (DRL) architectures to the problem of automated liquidity range management in Uniswap v3 concentrated AMMs. At each time step, an agent decides whether to hold, expand, contract, or shift its active price range - balancing fee income, impermanent loss, and gas costs.

The framework uses **42,421 real hourly ETH/USDC candles (May 2021 - Mar 2026)** sourced from Binance as the price environment, replacing synthetic GBM simulation.

---

## Results (IEEE Experimental Protocol: 5 seeds x 500 episodes)

| Agent | Alpha vs. Passive | Sortino | Win Rate (T=500) | Significance |
|-------|-------------------|---------|------------------|--------------|
| **DDQN** | +$54.80 (CI: 41.19, 68.41) | 2.128 | 41.2% | p < 0.001 *** |
| **PPO**  | - | - | - | p = 0.009 ** |
| **SAC**  | - | - | - | non-significant |

- Bonferroni-corrected hypothesis testing across all three architectures
- Earliest Effective Cutoff (EEC): T* in (300, 500) steps

---

## Repository Structure

```
AIAMM_V10_IEEE.py              Main model - DDQN, PPO, SAC agents + AMM environment
fetch_uniswap_data.py          Binance ETH/USDC hourly data fetcher
binance_eth_usdc_hourly.npz    42,421 hourly candles (open, high, low, close, volume)
price_feed.npy                 Close prices array
volume_feed.npy                Volume array
AIAMM_V10_Paper.md             Full IEEE-format paper (Markdown)
AIAMM_V10_Paper.docx           IEEE paper (Word)
AIAMM_V8FRA2.py                Prior version (V8)
gen_v75.py                     Utility / generation script
```

---

## Key Contributions

1. **MDP Formulation** - cAMM LP management formalised as a Markov Decision Process with a reward function decomposing fee income, impermanent loss, and transaction costs.
2. **Multi-Architecture Benchmark** - DDQN, PPO, and SAC trained on identical environment instances under a curriculum-scheduled, multi-seed protocol.
3. **Earliest Effective Cutoff (EEC)** - bisection method identifying the minimum deployment horizon T* for positive expected alpha.
4. **Real Market Data** - 42,421 real ETH/USDC hourly candles replace GBM simulation; data gap (Oct 2022 - Mar 2023) patched with ETHUSDT data.

---

## Setup

```bash
pip install numpy torch gymnasium scipy
python fetch_uniswap_data.py   # download fresh data (optional, .npz included)
python AIAMM_V10_IEEE.py
```

---

## Citation

```
Franco Farrugia, "Deep Reinforcement Learning for Automated Liquidity Management
in Concentrated Automated Market Makers: A Multi-Architecture Empirical Study",
University of Malta, 2025.
```
