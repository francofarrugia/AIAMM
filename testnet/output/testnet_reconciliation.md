# AIAMM Sepolia run – reconciliation against the blockchain

Reconciled on 2026-10-29 19:02 UTC via https://ethereum-sepolia-rpc.publicnode.com. Read-only; no transactions sent.

## Unrecorded on-chain collect transactions
| Step | Block | Tx | tETH | tUSDC |
|---|---|---|---|---|
| 22 | 12012088 | 0xcb72409aac14a3ba04f49ef338b11bdb449b50f31945060116341d77c555ab40 | 0.035390 | 96.7616 |
| 33 | 12012210 | 0x06a8291cce5c271cb3e2b3aedfaaa4ee932c2f5c56c05eb57812bc273f45cb9e | 0.037919 | 91.5656 |

## All unrecorded transactions sent by the wallet
| Step | Block | Label | Tx | Status |
|---|---|---|---|---|
| 22 | 12012088 | AGENT: collect (successful first attempt; unrecorded – reconciled) | 0xcb72409aac14a3ba04f49ef338b11bdb449b50f31945060116341d77c555ab40 | success |
| 33 | 12012101 | market move (successful first attempt; unrecorded – reconciled) | 0xa92131cacdacbe0fee7e2ede86c7882a6f3d1409f9871df57519e08e7222c2ea | success |
| 33 | 12012210 | AGENT: collect (successful first attempt; unrecorded – reconciled) | 0x06a8291cce5c271cb3e2b3aedfaaa4ee932c2f5c56c05eb57812bc273f45cb9e | success |
| 34 | 12012213 | market move (successful first attempt; unrecorded – reconciled) | 0x10c8f1d40f6b19954dd0e85e755e6d91c0779616670f9831c5bb34c0b0dae87b | success |
| 41 | 12012518 | market move (successful first attempt; unrecorded – reconciled) | 0xb918b260e44bd33e6492b148277117add5fbd8a8abb221601469a4cdfe363d12 | success |

## Final result (step 48)
| Measure | As recorded | Reconciled |
|---|---|---|
| Agent value (net of model gas $0.60) | $1,603.02 | $1,932.43 |
| Passive value | $1,934.07 | $1,934.07 |
| Alpha | $-331.05 | $-1.64 |
| Agent value recomputed independently from chain state | – | $1,932.43 |
| Transactions deployment / market / agent / total | 10 / 44 / 8 / 62 | 10 / 47 / 10 / 67 |
| Gas used | 57,289,425 | 58,046,144 (0.00580 test ETH) |

## Per-step alpha (recorded → reconciled)
| Step | Action | Alpha recorded | Alpha reconciled | From chain |
|---|---|---|---|---|
| 1 | Hold | +0.01 | +0.01 | +0.01 |
| 2 | Hold | +0.01 | +0.01 | +0.01 |
| 3 | Hold | +0.01 | +0.01 | +0.01 |
| 4 | Hold | +0.05 | +0.05 | +0.05 |
| 5 | Hold | +0.07 | +0.07 | +0.07 |
| 6 | Hold | +0.08 | +0.08 | +0.08 |
| 7 | Hold | +0.08 | +0.08 | +0.08 |
| 8 | Hold | +0.08 | +0.08 | +0.08 |
| 9 | Hold | +0.13 | +0.13 | +0.13 |
| 10 | Hold | +0.15 | +0.15 | +0.15 |
| 11 | Decrease liquidity | +169.53 | +169.53 | +0.05 |
| 12 | Hold | +0.02 | +0.02 | +0.02 |
| 13 | Hold | +0.05 | +0.05 | +0.05 |
| 14 | Hold | +0.02 | +0.02 | +0.02 |
| 15 | Hold | +0.41 | +0.41 | +0.41 |
| 16 | Hold | -0.37 | -0.37 | -0.37 |
| 17 | Hold | -0.13 | -0.13 | -0.13 |
| 18 | Hold | +0.14 | +0.14 | +0.14 |
| 19 | Hold | +0.10 | +0.10 | +0.10 |
| 20 | Hold | +0.16 | +0.16 | +0.16 |
| 21 | Hold | +0.17 | +0.17 | +0.17 |
| 22 | Decrease liquidity | -168.52 | -0.00 | -0.00 |
| 23 | Hold | -168.15 | -0.08 | -0.08 |
| 24 | Hold | -168.14 | -0.07 | -0.07 |
| 25 | Hold | -168.45 | +0.01 | +0.01 |
| 26 | Hold | -169.02 | +0.13 | +0.13 |
| 27 | Hold | -168.93 | +0.13 | +0.13 |
| 28 | Hold | -168.93 | +0.13 | +0.13 |
| 29 | Hold | -168.72 | +0.10 | +0.10 |
| 30 | Hold | -168.61 | +0.09 | +0.09 |
| 31 | Hold | -168.61 | +0.09 | +0.09 |
| 32 | Hold | -168.57 | +0.17 | +0.17 |
| 33 | Decrease liquidity | -334.08 | -0.53 | -0.53 |
| 34 | Hold | -333.00 | -0.82 | -0.82 |
| 35 | Hold | -332.87 | -0.85 | -0.85 |
| 36 | Hold | -333.07 | -0.79 | -0.79 |
| 37 | Hold | -331.92 | -1.15 | -1.15 |
| 38 | Hold | -332.23 | -1.04 | -1.04 |
| 39 | Hold | -331.08 | -1.46 | -1.46 |
| 40 | Hold | -331.33 | -1.35 | -1.35 |
| 41 | Hold | -330.58 | -1.66 | -1.65 |
| 42 | Hold | -330.74 | -1.58 | -1.58 |
| 43 | Hold | -330.85 | -1.53 | -1.53 |
| 44 | Decrease liquidity | -330.90 | -1.72 | -1.72 |
| 45 | Hold | -330.90 | -1.72 | -1.72 |
| 46 | Hold | -331.32 | -1.53 | -1.53 |
| 47 | Hold | -331.46 | -1.47 | -1.47 |
| 48 | Hold | -331.05 | -1.64 | -1.64 |

## Wallet cross-check
Wallet balances now: tETH 999999949.102223, tUSDC 9999999902035.6738.
Agent idle per run state: tETH 0.307831, tUSDC 436.2815; passive idle [0.0, 0.0].

## All on-chain events of the agent position
| Block | Event | Tx | tETH | tUSDC | In original log |
|---|---|---|---|---|---|
| 12012025 | IncreaseLiquidity | 0xf341256803a1e6078470018c87fe55710b551347a985c8a7dbe28d6ee151889b | 0.252328 | 750.0000 | yes |
| 12012073 | DecreaseLiquidity | 0x26dc783c6175839da2dc81846a19466d56c0894cac47d56ce853811563e641e7 | 0.033041 | 101.2453 | yes |
| 12012074 | Collect | 0x7ddd3b9ad97bac02f0ee43e3c1c80c58ded6b72914e4d15a4e3709a0eb1428d5 | 0.033093 | 101.3819 | yes |
| 12012087 | DecreaseLiquidity | 0xb534b179e01cbbf267649cea0ea23a4fa4af35c41478c42e69995af964050c46 | 0.035287 | 96.6425 | yes |
| 12012088 | Collect | 0xcb72409aac14a3ba04f49ef338b11bdb449b50f31945060116341d77c555ab40 | 0.035390 | 96.7616 | NO |
| 12012090 | Collect | 0x5fe44e721922f9fbc877e8995e8fb775a6a7e4b654c02310ca95334785aa69f9 | 0.000000 | 0.0000 | yes |
| 12012209 | DecreaseLiquidity | 0x221660b2ce1a618fbf86e4888137fbfaf85142a2d7f44c540e508e6d62c35c16 | 0.037849 | 91.5083 | yes |
| 12012210 | Collect | 0x06a8291cce5c271cb3e2b3aedfaaa4ee932c2f5c56c05eb57812bc273f45cb9e | 0.037919 | 91.5656 | NO |
| 12012212 | Collect | 0xe41d2b7c79c20d8a77f62e693d7b8ffb0aa3265d8054eeaa1afd0d45e91ea66b | 0.000000 | 0.0000 | yes |
| 12012525 | DecreaseLiquidity | 0x8081fc007119439965f936bf1aad94978831ae794bc270a38aacd86d08fc28e6 | 0.041251 | 84.8701 | yes |
| 12012526 | Collect | 0x04a3903933f782edad9113467f92f82853e94c531a48390d8682581b996638b7 | 0.041313 | 84.8996 | yes |