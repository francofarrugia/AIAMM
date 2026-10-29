"""Step 3 – produce the evidence files for the Post-Award Panel:
  output/testnet_report.md         summary: network, contracts, dates, results
  output/testnet_transactions.csv  every transaction with its explorer link
  output/testnet_steps.csv         step-by-step agent decisions and values
  output/testnet_chart.png         price, agent vs passive value
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tn_common import *

dep = load_json(OUT / 'deployment.json'); st = load_json(OUT / 'state.json')
if not dep or not st or not st['rows']:
    sys.exit('Nothing to report yet – run 01_deploy.py and 02_run_agent.py first.')
rows = st['rows']; ex = dep['explorer']
link = (lambda kind, a: f'{ex}/{kind}/{a}') if dep['network'] != 'local' else (lambda kind, a: a)

alltx = [dict(step=0, **t) for t in dep['txs']] + [dict(step=r['step'], **t) for r in rows for t in r['txs']]
with open(OUT / 'testnet_transactions.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['step', 'label', 'tx', 'block', 'time_utc', 'gas_used', 'eth_fee', 'url'])
    w.writeheader(); [w.writerow({k: t.get(k) for k in w.fieldnames}) for t in alltx]
cols = ['step', 'replay_hour_utc', 'chain_time_utc', 'block', 'market_price', 'pool_price', 'action', 'agent_value',
        'passive_value', 'alpha_usd', 'agent_fees_usd', 'passive_fees_usd', 'liq_ratio', 'in_range', 'model_cost_usd', 'gas_used']
with open(OUT / 'testnet_steps.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({k: r[k] for k in cols}) for r in rows]

last = rows[-1]
acts = {}
for r in rows: acts[r['action']] = acts.get(r['action'], 0) + 1
agent_tx = [t for r in rows for t in r['txs'] if t['label'].startswith('AGENT')]
market_tx = [t for r in rows for t in r['txs'] if t['label'].startswith('market')]
gas_total = sum(t['gas_used'] for t in alltx); eth_total = sum(t['eth_fee'] for t in alltx)

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
x = [r['step'] for r in rows]
ax[0].plot(x, [r['market_price'] for r in rows], color='#1F4E79'); ax[0].set_ylabel('ETH price (USD)')
ax[0].set_title('AIAMM DDQN agent on Uniswap v3 – Ethereum Sepolia testnet')
for r in rows:
    if r['action'] != 'Hold': ax[0].axvline(r['step'], color='#C00000', alpha=0.25)
ax[1].plot(x, [r['agent_value'] for r in rows], label='AI agent (DDQN)', color='#C00000')
ax[1].plot(x, [r['passive_value'] for r in rows], label='Passive full-range', color='#7F7F7F')
ax[1].set_ylabel('Position value (USD)'); ax[1].set_xlabel('Step (one replayed hour)'); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT / 'testnet_chart.png', dpi=200)

L = []
L.append('# AIAMM testnet deployment – evidence summary\n')
L.append(f"**Network:** {'Ethereum Sepolia testnet' if dep['network'] == 'sepolia' else dep['network']} (chain ID {dep['chain_id']})  ")
L.append(f"**Run period (chain time, UTC):** {dep['txs'][0]['time_utc']} – {last['chain_time_utc']}  ")
L.append(f"**Wallet:** {link('address', dep['wallet'])}  ")
L.append(f"**Replayed market data:** {rows[0]['replay_hour_utc']} – {last['replay_hour_utc']} UTC (real hourly ETH/USDC prices, Binance)\n")
L.append('## Smart contracts')
L.append('| Contract | Address |\n|---|---|')
for nm, a in [('Uniswap v3 Factory (official)', dep['uniswap']['factory']), ('Uniswap v3 NonfungiblePositionManager (official)', dep['uniswap']['npm']),
              ('Uniswap SwapRouter02 (official)', dep['uniswap']['router']), ('Test token tETH (no monetary value)', dep['tETH']),
              ('Test token tUSDC (no monetary value)', dep['tUSDC']), ('Uniswap v3 pool tETH/tUSDC, 0.30%', dep['pool'])]:
    L.append(f'| {nm} | {link("address", a)} |')
L.append(f"\nPositions (Uniswap v3 position NFTs): passive benchmark #{dep['passive_token_id']}, AI agent initial #{dep['agent_token_id']}"
         f"{', final #' + str(st['token_id']) if st['token_id'] != dep['agent_token_id'] else ''}, background market liquidity #{dep['background_token_id']}.\n")
L.append('## Results')
L.append('| Measure | Value |\n|---|---|')
L.append(f"| Steps completed | {len(rows)} |")
L.append(f"| Agent decisions | " + '; '.join(f'{k}: {v}' for k, v in acts.items()) + ' |')
L.append(f"| Final value – AI agent (net of model gas cost ${last['model_cost_usd']:.2f}) | ${last['agent_value']:,.2f} |")
L.append(f"| Final value – passive full-range | ${last['passive_value']:,.2f} |")
L.append(f"| Alpha (agent − passive) | ${last['alpha_usd']:+,.2f} |")
L.append(f"| Fees earned – agent / passive | ${last['agent_fees_usd']:.4f} / ${last['passive_fees_usd']:.4f} |")
L.append(f"| Transactions – deployment / market replay / agent | {len(dep['txs'])} / {len(market_tx)} / {len(agent_tx)} |")
L.append(f"| Gas used in total | {gas_total:,} gas ({eth_total:.5f} test ETH) |\n")
L.append('## Agent transactions')
L.append('| Step | Chain time (UTC) | Action | Transaction |\n|---|---|---|---|')
for r in rows:
    for t in r['txs']:
        if t['label'].startswith('AGENT'):
            L.append(f"| {r['step']} | {t['time_utc']} | {t['label']} | {link('tx', t['tx'])} |")
L.append('\nAll transactions are listed in testnet_transactions.csv; the step-by-step log is in testnet_steps.csv.\n')
L.append('## Scope and limitations')
L.append('- Test tokens and a dedicated pool were used, because testnet assets have no market value. The market is represented by '
         'swaps that move the pool to each replayed real ETH/USDC price; fee income therefore comes from these swaps only and is not '
         'representative of live trading volume.')
L.append('- The agent is the DDQN trained in the V13 simulation (seed 42, 1,000 episodes), run with ε = 0.05, without further training.')
L.append('- This is a technical deployment demonstration carried out in October 2026, after the project period; it is not a '
         'statistically meaningful performance test.')
(OUT / 'testnet_report.md').write_text('\n'.join(L), encoding='utf-8')
print('\n'.join(L))
print('\nFiles written to', OUT)
