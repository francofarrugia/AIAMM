# AIAMM testnet deployment – evidence summary

**Network:** Ethereum Sepolia testnet (chain ID 11155111)  
**Run period (chain time, UTC):** 2026-10-28 11:28:24 – 2026-10-28 13:11:12  
**Wallet:** https://sepolia.etherscan.io/address/0xDC3FCf4f645355f6e36cF0f10D26B0ccF5b5ab31  
**Replayed market data:** 2026-02-26 02:00 – 2026-02-28 01:00 UTC (real hourly ETH/USDC prices, Binance)

## Smart contracts
| Contract | Address |
|---|---|
| Uniswap v3 Factory (official) | https://sepolia.etherscan.io/address/0x0227628f3F023bb0B980b67D528571c95c6DaC1c |
| Uniswap v3 NonfungiblePositionManager (official) | https://sepolia.etherscan.io/address/0x1238536071E1c677A632429e3655c799b22cDA52 |
| Uniswap SwapRouter02 (official) | https://sepolia.etherscan.io/address/0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E |
| Test token tETH (no monetary value) | https://sepolia.etherscan.io/address/0xa1f8b8246Fd9b9373190f77B7D2A781C60e7416B |
| Test token tUSDC (no monetary value) | https://sepolia.etherscan.io/address/0xE2836bDAf42Ce9E940ac9301B52626aEB26EFA56 |
| Uniswap v3 pool tETH/tUSDC, 0.30% | https://sepolia.etherscan.io/address/0x407333868c4B779c12D9F159A109B8535Fb21692 |

Positions (Uniswap v3 position NFTs): passive benchmark #233912, AI agent initial #233913, background market liquidity #233911.

## Results
| Measure | Value |
|---|---|
| Steps completed | 48 |
| Agent decisions | Hold: 44; Decrease liquidity: 4 |
| Final value – AI agent (net of model gas cost $0.60) | $1,603.02 |
| Final value – passive full-range | $1,934.07 |
| Alpha (agent − passive) | $-331.05 |
| Fees earned – agent / passive | $0.9092 / $0.3473 |
| Transactions – deployment / market replay / agent | 10 / 44 / 8 |
| Gas used in total | 57,289,425 gas (0.00573 test ETH) |

## Agent transactions
| Step | Chain time (UTC) | Action | Transaction |
|---|---|---|---|
| 11 | 2026-10-28 11:39:48 | AGENT: decrease liquidity | https://sepolia.etherscan.io/tx/0x26dc783c6175839da2dc81846a19466d56c0894cac47d56ce853811563e641e7 |
| 11 | 2026-10-28 11:40:00 | AGENT: collect | https://sepolia.etherscan.io/tx/0x7ddd3b9ad97bac02f0ee43e3c1c80c58ded6b72914e4d15a4e3709a0eb1428d5 |
| 22 | 2026-10-28 11:42:36 | AGENT: decrease liquidity | https://sepolia.etherscan.io/tx/0xb534b179e01cbbf267649cea0ea23a4fa4af35c41478c42e69995af964050c46 |
| 22 | 2026-10-28 11:43:12 | AGENT: collect | https://sepolia.etherscan.io/tx/0x5fe44e721922f9fbc877e8995e8fb775a6a7e4b654c02310ca95334785aa69f9 |
| 33 | 2026-10-28 12:07:00 | AGENT: decrease liquidity | https://sepolia.etherscan.io/tx/0x221660b2ce1a618fbf86e4888137fbfaf85142a2d7f44c540e508e6d62c35c16 |
| 33 | 2026-10-28 12:07:36 | AGENT: collect | https://sepolia.etherscan.io/tx/0xe41d2b7c79c20d8a77f62e693d7b8ffb0aa3265d8054eeaa1afd0d45e91ea66b |
| 44 | 2026-10-28 13:10:24 | AGENT: decrease liquidity | https://sepolia.etherscan.io/tx/0x8081fc007119439965f936bf1aad94978831ae794bc270a38aacd86d08fc28e6 |
| 44 | 2026-10-28 13:10:36 | AGENT: collect | https://sepolia.etherscan.io/tx/0x04a3903933f782edad9113467f92f82853e94c531a48390d8682581b996638b7 |

All transactions are listed in testnet_transactions.csv; the step-by-step log is in testnet_steps.csv.

## Scope and limitations
- Test tokens and a dedicated pool were used, because testnet assets have no market value. The market is represented by swaps that move the pool to each replayed real ETH/USDC price; fee income therefore comes from these swaps only and is not representative of live trading volume.
- The agent is the DDQN trained in the V13 simulation (seed 42, 1,000 episodes), run with ε = 0.05, without further training.
- This is a technical deployment demonstration carried out in October 2026, after the project period; it is not a statistically meaningful performance test.