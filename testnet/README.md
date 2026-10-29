# AIAMM — Sepolia Testnet Deployment Demonstration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Testnet: Sepolia](https://img.shields.io/badge/testnet-Ethereum%20Sepolia-5a67d8.svg)](https://sepolia.etherscan.io/)

Reproducibility kit for the on-chain deployment demonstration of the AIAMM Double Deep Q-Network (DDQN) liquidity-management agent on the Ethereum Sepolia public testnet. The agent, trained in the simulation study, autonomously manages a Uniswap v3 concentrated-liquidity position end-to-end on real public blockchain infrastructure.

Companion repository to the research paper:

> Farrugia, F. & Deguara, C. **Reinforcement Learning Agents for Adaptive Liquidity Provisioning in Concentrated Decentralised Exchanges: A Comparative Architecture Study.** *(see §2.7 "On-chain Deployment Protocol" and §3.5 "On-chain Execution Evidence".)*

Everything in this kit runs on **test tokens with no monetary value**. You need only free Sepolia test ETH to pay gas (≈ 0.01 test ETH is enough for the full 48-step run plus deployment). Total runtime: about one hour, most of it waiting on-chain.

---

## What this kit does

A single Python pipeline that:

1. Creates a new Sepolia test wallet.
2. Deploys two bespoke ERC-20 test tokens (tETH, tUSDC) and a 0.30% Uniswap v3 pool through the **official Uniswap v3 Sepolia contracts**, then mints three positions (background market-maker, passive full-range benchmark, DDQN agent with a ±40% concentrated range).
3. Replays 48 hours of real ETH/USDC prices into the pool. At each step the trained DDQN agent reads on-chain state, selects one of five actions (Hold, Increase / Decrease liquidity, Shift range ±3%) and executes any non-Hold action as real transactions.
4. Produces an evidence report with every transaction, its Etherscan link, the step-by-step state log, and a chart of price vs. agent/passive portfolio values.

## Prerequisites

- Python 3.9 or newer (Windows, macOS or Linux).
- Any public Ethereum Sepolia RPC endpoint (the kit ships with three defaults — PublicNode, dRPC, sepolia.org — and tries them in order).
- Free Sepolia test ETH from a public faucet (Google Cloud Web3 is the fastest; PoW faucet works without an account).

> **Windows note.** If `python` is not on your PATH, use `py` instead everywhere below. The `.bat` wrappers assume `python`; invoking the `.py` files directly with `py` works on any Python 3 install from python.org.

## Install

```bash
pip install --only-binary=:all: -r requirements.txt
```

The `--only-binary=:all:` flag ensures `bitarray` and `ckzg` install from pre-built wheels rather than compiling from source (no C/C++ toolchain required).

## Quick start

```bash
# 1. Create a Sepolia test wallet (writes wallet.json locally)
python 00_create_wallet.py

# 2. Fund the printed 0x… address at a Sepolia faucet (0.05 test ETH is plenty)
#    https://cloud.google.com/application/web3/faucet/ethereum/sepolia  (0.05 ETH, Google sign-in)
#    https://sepolia-faucet.pk910.de/                                    (PoW, no account)

# 3. Deploy tokens, pool and three positions (≈10 transactions, 2–5 min)
python 01_deploy.py

# 4. Run the agent for 48 hourly steps (≈20–40 min, resumable)
python 02_run_agent.py --steps 48

# 5. Produce the evidence files in output/
python 03_report.py
```

Convenience `.bat` wrappers for Windows (`STEP 1 - …` through `STEP 4 - …`) run the same commands.

## What you get in `output/`

- `deployment.json` — contract addresses (pool, tokens, Uniswap v3 Factory / NPM / Router), position NFT token IDs, start price, initial-value accounting and every deployment transaction with its gas and Etherscan URL.
- `state.json` — per-step execution state (checkpointed after every step so the run resumes if interrupted).
- `run.log` — timestamped line-by-line log of every on-chain action.
- `testnet_report.md` — human-readable evidence summary with network, contract addresses, dates, results and every agent transaction.
- `testnet_transactions.csv` — every transaction (deployment + market replay + agent) with Etherscan links.
- `testnet_steps.csv` — the agent's decision and the position values at each step.
- `testnet_chart.png` — real price and agent-vs-passive portfolio values.

## Repository contents

```
AIAMM_Testnet_Demo/
├── 00_create_wallet.py          # Generate a Sepolia test wallet
├── 01_deploy.py                 # Deploy tokens + pool + three positions
├── 02_run_agent.py              # 48-step replay with the trained DDQN agent (patched for SPL on public RPCs)
├── 03_report.py                 # Produce evidence files (markdown, CSVs, chart)
├── tn_common.py                 # Shared helpers (chain, pair, Uniswap maths, agent loader)
├── requirements.txt             # Python dependencies
├── wallet.example.json          # Template showing wallet.json structure (keys zeroed)
├── abi/                         # Uniswap v3 + ERC-20 + NPM + SwapRouter02 ABIs and bytecode
├── model/                       # Trained V13 DDQN weights (ddqn_v13_seed42.npz)
├── AIAMM_data/                  # Hourly ETH/USDC data (binance_eth_usdc_hourly.npz) + V13 agent code
├── docs/                        # Appendix material (progress log, deployment log, paper figures, screenshots)
├── STEP 1-4 … .bat              # Convenience wrappers for Windows users
├── LICENSE                      # MIT
└── README.md                    # This file
```

## Scope and limitations

- Testnet pools have **no organic trading volume**: fee income is driven exclusively by the replay swaps, so the fee term is not comparable to live-mainnet fee flow.
- The agent is the DDQN trained in the simulation study (seed 42, 1,000 episodes) running in evaluation mode (ε = 0.05) with **no additional training on-chain**.
- The on-chain exercise is a **single-episode operational-feasibility demonstration (N = 1, T = 48 steps)**, not a statistical performance test. The authoritative performance evidence remains the multi-seed, multi-episode simulation evaluation in the main paper.
- One known RPC pitfall is handled by the runner: when a Sepolia public RPC returns a receipt-query timeout on a market-replay swap that in fact succeeded on-chain, a retry would re-emit the same swap and Uniswap v3 would revert with "SPL" because the pool price is already at the target. The runner catches this, re-reads pool state, and records the move as successful when the price is within 1% of the target.

## Reproducibility

The deterministic components of the demonstration are the trained agent weights (`model/ddqn_v13_seed42.npz`), the replayed hourly ETH/USDC data (`AIAMM_data/binance_eth_usdc_hourly.npz`), the Uniswap v3 contract set (official Sepolia deployment), and the discrete action-to-transaction mapping in `02_run_agent.py`. The stochastic components are Sepolia block timing, gas-fee auctions and the particular transaction hashes produced. Re-running the pipeline with a freshly funded wallet reproduces the same action sequence and the same qualitative outcome; the exact alpha value will differ marginally with gas-price variance.

## Citation

If you use this kit or extend it, please cite the companion paper:

```bibtex
@article{farrugia_deguara_aiamm,
  title   = {Reinforcement Learning Agents for Adaptive Liquidity Provisioning in Concentrated Decentralised Exchanges: A Comparative Architecture Study},
  author  = {Farrugia, Franco and Deguara, Cedric},
  year    = {2026},
  note    = {Companion on-chain deployment demonstration: \url{https://github.com/francofarrugia/AIAMM}}
}
```

## Security

**Never share `wallet.json`.** The file holds the private key for the Sepolia test wallet. Even though the wallet only ever handles test ETH, re-using the key on mainnet would be unsafe. `wallet.json` is gitignored by default; a `wallet.example.json` with zero-filled values is provided to show the schema.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

AIAMM is a research project conducted under the **MCAST Research Fellowship Scheme 2025**. The research author is Franco Farrugia (MCAST). Technical coordination and dissemination have been supported by Web3 Media Ltd throughout the project.
