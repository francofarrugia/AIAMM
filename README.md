# AIAMM – Reinforcement-learning agents for concentrated-liquidity AMMs

Code, data and evidence for the MCAST Research Fellowship Scheme 2025 project
**"Integrating AI Agents with Automated Market Makers (AMMs) for Smarter Liquidity Management" (AIAMM)**
and the paper:

> Farrugia, F. & Deguara, C. *Reinforcement Learning Agents for Adaptive Liquidity Provisioning in Concentrated
> Decentralised Exchanges: A Comparative Architecture Study.* MCAST Journal of Applied Research and Practice
> (accepted 27 October 2026, December 2026 issue).

Released under the MIT licence (see `LICENSE`). Principal investigator: Franco Farrugia (franco.farrugia@mcast.edu.mt).

## What is in this repository

| Folder | Contents |
|---|---|
| `simulation/` | Final evaluation code `13_AIAMM_V13.py` (Uniswap v3-style AMM simulation, DDQN / PPO / SAC agents, curriculum training, Earliest Effective Cutoff bisection, GBM volatility sweep, Bonferroni-corrected tests), the dataset `binance_eth_usdc_hourly.npz` (38,495 hourly ETH/USDC candles, 3 May 2021 – 6 March 2026, Binance public API), the download script `fetch_uniswap_data.py` (despite its name it queries the Binance API) and the console output of the final run (`results_reported_in_paper.txt`, 22 May 2026). |
| `supplementary_oct2026/` | Scripts and summary of the October 2026 supplementary analysis on the unchanged V13 code: ROI, fee earnings, capital utilisation, Value-at-Risk, return variability, market-phase breakdown, sentiment-input ablation and permutation feature importance. |
| `testnet/` | Kit for the 28 October 2026 demonstration of the trained DDQN agent on the Ethereum Sepolia testnet with the official Uniswap v3 contracts, including the run outputs and the reconciliation of the run log against the chain. See `testnet/README.md`. |

## Main findings (simulation, V13)

On real ETH/USDC prices (5 seeds × 1,000 episodes) none of the three agents beats passive full-range liquidity
provision after costs: DDQN mean alpha −$6.29 per episode (95% CI [−6.50, −6.09]), PPO −$7.09, SAC −$7.11,
all p_adj < 0.001. In a controlled GBM volatility sweep DDQN's alpha rises with volatility: −$9.65 at σ = 0.005,
≈ break-even at σ = 0.015, +$53.62 at σ = 0.030 and +$240.87 at σ = 0.050 per episode. Active management by
these agents adds value only under high volatility, and that benefit has been shown in simulation only.

The Sepolia run is a single 48-step operational demonstration on test tokens; it shows that the agent's actions
execute end-to-end on public blockchain infrastructure and is **not** a performance test.

## Reproduce

```bash
pip install numpy scipy matplotlib
cd simulation
python 13_AIAMM_V13.py          # several hours on a laptop; prints Tables II–V
```

For the testnet kit see `testnet/README.md` (needs only free Sepolia test ETH).

## Limitations (declared in the project's final report)

No NLP-based sentiment model was built (the agents use a price-derived momentum indicator); the evaluation uses
Binance exchange prices rather than on-chain pool data; slippage, capital turnover and arbitrage profitability
are not measured; there has been no mainnet or live-market deployment.
