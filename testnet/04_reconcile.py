"""Step 4 – reconcile the agent's accounting against the blockchain.

Why this exists
---------------
During the 28 October 2026 Sepolia run, two agent "collect" transactions (steps 22 and 33)
succeeded on-chain, but the public RPC then failed while the script was reading their receipts
("Block with id ... not found"). The old retry logic re-sent the call; the second collect found
nothing left to collect and returned zero, so the tokens withdrawn by the first collect were never
credited to the agent's idle balance. The recorded agent value (and alpha) is therefore too low
from step 22 onwards, and the two successful collect transactions are missing from the log.

What this script does
---------------------
1. Reads every Collect / DecreaseLiquidity / IncreaseLiquidity event of the agent position(s)
   directly from the Uniswap v3 NonfungiblePositionManager on Sepolia.
2. Finds the on-chain Collect transactions that are not in the recorded log.
3. Re-values the agent at every step = recorded value + value of the unrecorded collected tokens
   (at that step's market price). Where the RPC serves historical state, it also re-values the
   position independently from the chain at each step's block, as a cross-check.
4. Writes reconciled evidence files next to the originals (originals are left untouched):
     output/testnet_steps_reconciled.csv
     output/testnet_transactions_reconciled.csv
     output/testnet_chart_reconciled.png
     output/testnet_reconciliation.md     <- send this file back to Claude

Run:   py 04_reconcile.py        (needs internet; read-only, sends no transactions, costs no gas)
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tn_common import *

dep = load_json(OUT / 'deployment.json'); st = load_json(OUT / 'state.json')
if not dep or not st or not st.get('rows'):
    sys.exit('deployment.json / state.json not found in output/.')
rows = st['rows']
ch = Chain(dep['network'])
pair = Pair(dep['tETH'], dep['tUSDC'])
npm = ch.contract(dep['uniswap']['npm'], 'NonfungiblePositionManager')
pool = ch.contract(dep['pool'], 'UniswapV3Pool')

first_block = dep['txs'][0]['block']
last_block = max(t['block'] for r in rows for t in r['txs']) + 5
token_ids = sorted({dep['agent_token_id'], st['token_id']})
recorded = {t['tx'].lower() for r in rows for t in r['txs']} | {t['tx'].lower() for t in dep['txs']}


def logs(event, tid):
    out, a = [], first_block
    while a <= last_block:                          # chunked to respect public-RPC range limits
        b = min(a + 4999, last_block)
        out += event.get_logs(argument_filters={'tokenId': tid}, from_block=a, to_block=b)
        a = b + 1
    return out


events = []
for tid in token_ids:
    for name in ('Collect', 'DecreaseLiquidity', 'IncreaseLiquidity'):
        for e in logs(getattr(npm.events, name)(), tid):
            h = '0x' + e['transactionHash'].hex().removeprefix('0x')
            eth, usdc = pair.split(e['args']['amount0'] / DEC, e['args']['amount1'] / DEC)
            events.append({'event': name, 'token_id': tid, 'block': e['blockNumber'], 'tx': h,
                           'eth': eth, 'usdc': usdc, 'recorded': h.lower() in recorded})
events.sort(key=lambda e: (e['block'], e['event']))
missing = [e for e in events if e['event'] == 'Collect' and not e['recorded'] and (e['eth'] > 0 or e['usdc'] > 0)]

# --- every transaction the wallet actually sent: scan the run's blocks (same "first attempt
#     succeeded but was not recorded" problem also affected some market-replay swaps)
missing_collect_tx = {e['tx'].lower() for e in missing}
n_sent = ch.w3.eth.get_transaction_count(ch.addr)
print(f'Wallet has sent {n_sent} transactions in total; recorded in the log: {len(recorded)}. Scanning blocks '
      f'{first_block}–{last_block} for unrecorded ones (a few minutes on a public RPC)...')
miss_tx, b = [], first_block
while b <= last_block:
    try:
        blk = ch.w3.eth.get_block(b, full_transactions=True)
    except Exception:
        time.sleep(2); continue                       # transient RPC error: retry the same block
    for tx in blk.transactions:
        if tx['from'].lower() != ch.addr.lower():
            continue
        h = '0x' + tx['hash'].hex().removeprefix('0x')
        if h.lower() in recorded:
            continue
        rc = ch.w3.eth.get_transaction_receipt(h)
        step = next((r['step'] for r in rows if r['block'] >= blk.number), rows[-1]['step'])
        kind = ('AGENT: collect (successful first attempt; unrecorded – reconciled)' if h.lower() in missing_collect_tx
                else 'market move (successful first attempt; unrecorded – reconciled)' if tx['to'] and tx['to'].lower() == dep['uniswap']['router'].lower()
                else f"unrecorded transaction to {tx['to']}")
        miss_tx.append({'step': step, 'label': kind, 'tx': h, 'block': blk.number,
                        'time_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(blk.timestamp)),
                        'gas_used': rc.gasUsed, 'eth_fee': rc.gasUsed * rc.get('effectiveGasPrice', 0) / 1e18,
                        'url': ch.explorer_tx(h), 'status': rc.status})
    b += 1
miss_tx = [t for t in miss_tx if t['status'] == 1] + [t for t in miss_tx if t['status'] != 1]
print(f'Found {len(miss_tx)} unrecorded transactions ({sum(1 for t in miss_tx if t["status"] == 1)} successful).')


# --- independent re-valuation from chain state at each step's block (needs an archive-capable RPC)
def chain_value(block, tid, idle):
    try:
        p = npm.functions.positions(tid).call(block_identifier=block)
        L, tl, tu = p[7], p[5], p[6]
        sqrt_x96 = pool.functions.slot0().call(block_identifier=block)[0]
        a0, a1 = amounts_for_liquidity(L, sqrt_x96 / Q96, tl, tu)
        f0, f1 = npm.functions.collect((tid, ch.addr, UINT128_MAX, UINT128_MAX)).call({'from': ch.addr}, block_identifier=block)
        e, u = pair.split((a0 + f0) / DEC, (a1 + f1) / DEC)
        return e + idle[0], u + idle[1]
    except Exception:
        return None


idle0 = list(dep['agent_idle'])
out_rows, archive_ok = [], True
for r in rows:
    adj_e = sum(e['eth'] for e in missing if e['block'] <= r['block'])
    adj_u = sum(e['usdc'] for e in missing if e['block'] <= r['block'])
    adj = pair.value_usd(adj_e, adj_u, r['pool_price'])
    agent_rec = r['agent_value'] + adj
    # idle tokens implied by on-chain events up to this block (all collects in, all increases out)
    ie = idle0[0] + sum(e['eth'] for e in events if e['event'] == 'Collect' and e['block'] <= r['block']) \
        - sum(e['eth'] for e in events if e['event'] == 'IncreaseLiquidity' and e['block'] <= r['block'] and e['tx'].lower() != dep['txs'][-1]['tx'].lower())
    iu = idle0[1] + sum(e['usdc'] for e in events if e['event'] == 'Collect' and e['block'] <= r['block']) \
        - sum(e['usdc'] for e in events if e['event'] == 'IncreaseLiquidity' and e['block'] <= r['block'] and e['tx'].lower() != dep['txs'][-1]['tx'].lower())
    cv = chain_value(r['block'], st['token_id'], (ie, iu)) if archive_ok else None
    if cv is None:
        archive_ok = False
    agent_chain = pair.value_usd(cv[0], cv[1], r['pool_price']) - r['model_cost_usd'] if cv else None
    out_rows.append({**{k: r[k] for k in ['step', 'replay_hour_utc', 'chain_time_utc', 'block', 'market_price', 'pool_price', 'action']},
                     'agent_value_recorded': round(r['agent_value'], 4), 'unrecorded_collect_usd': round(adj, 4),
                     'agent_value_reconciled': round(agent_rec, 4),
                     'agent_value_from_chain': round(agent_chain, 4) if agent_chain is not None else '',
                     'passive_value': round(r['passive_value'], 4),
                     'alpha_recorded': round(r['alpha_usd'], 4),
                     'alpha_reconciled': round(agent_rec - r['passive_value'], 4),
                     'agent_fees_usd': r['agent_fees_usd'], 'passive_fees_usd': r['passive_fees_usd'],
                     'liq_ratio': r['liq_ratio'], 'in_range': r['in_range'], 'model_cost_usd': r['model_cost_usd']})

with open(OUT / 'testnet_steps_reconciled.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)

alltx = [dict(step=0, **t) for t in dep['txs']] + [dict(step=r['step'], **t) for r in rows for t in r['txs']] + miss_tx
alltx.sort(key=lambda t: (t['block'], t['step']))
with open(OUT / 'testnet_transactions_reconciled.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['step', 'label', 'tx', 'block', 'time_utc', 'gas_used', 'eth_fee', 'url', 'status'])
    w.writeheader(); [w.writerow({k: t.get(k, 1) for k in w.fieldnames}) for t in alltx]

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
x = [r['step'] for r in out_rows]
ax[0].plot(x, [r['market_price'] for r in out_rows], color='#1F4E79'); ax[0].set_ylabel('ETH price (USD)')
ax[0].set_title('AIAMM DDQN agent on Uniswap v3 – Ethereum Sepolia testnet (reconciled)')
for r in out_rows:
    if r['action'] != 'Hold': ax[0].axvline(r['step'], color='#C00000', alpha=0.25)
ax[1].plot(x, [r['agent_value_reconciled'] for r in out_rows], label='AI agent (DDQN), reconciled', color='#C00000')
ax[1].plot(x, [r['passive_value'] for r in out_rows], label='Passive full-range', color='#7F7F7F')
ax[1].plot(x, [r['agent_value_recorded'] for r in out_rows], label='AI agent, as originally recorded', color='#C00000', ls=':', alpha=0.6)
ax[1].set_ylabel('Position value (USD)'); ax[1].set_xlabel('Step (one replayed hour)'); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT / 'testnet_chart_reconciled.png', dpi=200)

# --- wallet cross-check: tokens actually held vs tokens the run state believes are idle
erc = [ch.contract(a, 'AIAMMTestToken') for a in (dep['tETH'], dep['tUSDC'])]
bal = [c.functions.balanceOf(ch.addr).call() / DEC for c in erc]

last, fin = rows[-1], out_rows[-1]
gas_total = sum(t['gas_used'] for t in alltx); eth_total = sum(t['eth_fee'] for t in alltx)
n_dep = len(dep['txs']); n_mkt = sum(1 for t in alltx if t['label'].startswith('market')); n_ag = sum(1 for t in alltx if t['label'].startswith('AGENT'))
L = ['# AIAMM Sepolia run – reconciliation against the blockchain\n',
     f'Reconciled on {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} via {ch.rpc}. Read-only; no transactions sent.\n',
     '## Unrecorded on-chain collect transactions',
     '| Step | Block | Tx | tETH | tUSDC |', '|---|---|---|---|---|']
for e in missing:
    stp = next((r['step'] for r in rows if r['block'] >= e['block']), rows[-1]['step'])
    L.append(f"| {stp} | {e['block']} | {e['tx']} | {e['eth']:.6f} | {e['usdc']:.4f} |")
L += ['', '## All unrecorded transactions sent by the wallet', '| Step | Block | Label | Tx | Status |', '|---|---|---|---|---|']
for t in miss_tx:
    L.append(f"| {t['step']} | {t['block']} | {t['label']} | {t['tx']} | {'success' if t['status'] == 1 else 'reverted'} |")
L += ['', '## Final result (step 48)', '| Measure | As recorded | Reconciled |', '|---|---|---|',
      f"| Agent value (net of model gas ${last['model_cost_usd']:.2f}) | ${last['agent_value']:,.2f} | ${fin['agent_value_reconciled']:,.2f} |",
      f"| Passive value | ${last['passive_value']:,.2f} | ${last['passive_value']:,.2f} |",
      f"| Alpha | ${last['alpha_usd']:+,.2f} | ${fin['alpha_reconciled']:+,.2f} |",
      f"| Agent value recomputed independently from chain state | – | {('$' + format(fin['agent_value_from_chain'], ',.2f')) if fin['agent_value_from_chain'] != '' else 'RPC has no historical state'} |",
      f"| Transactions deployment / market / agent / total | 10 / 44 / 8 / 62 | {n_dep} / {n_mkt} / {n_ag} / {len(alltx)} |",
      f"| Gas used | 57,289,425 | {gas_total:,} ({eth_total:.5f} test ETH) |", '',
      '## Per-step alpha (recorded → reconciled)', '| Step | Action | Alpha recorded | Alpha reconciled | From chain |', '|---|---|---|---|---|']
for r in out_rows:
    fc = (f"{r['agent_value_from_chain'] - r['passive_value']:+.2f}" if r['agent_value_from_chain'] != '' else '–')
    L.append(f"| {r['step']} | {r['action']} | {r['alpha_recorded']:+.2f} | {r['alpha_reconciled']:+.2f} | {fc} |")
L += ['', '## Wallet cross-check', f'Wallet balances now: tETH {bal[0]:.6f}, tUSDC {bal[1]:.4f}.',
      f"Agent idle per run state: tETH {st['idle'][0]:.6f}, tUSDC {st['idle'][1]:.4f}; passive idle {dep['passive_idle']}.",
      '', '## All on-chain events of the agent position', '| Block | Event | Tx | tETH | tUSDC | In original log |', '|---|---|---|---|---|---|']
for e in events:
    L.append(f"| {e['block']} | {e['event']} | {e['tx']} | {e['eth']:.6f} | {e['usdc']:.4f} | {'yes' if e['recorded'] else 'NO'} |")
(OUT / 'testnet_reconciliation.md').write_text('\n'.join(L), encoding='utf-8')
print('\n'.join(L[:25]))
print('\nWritten: output/testnet_reconciliation.md, testnet_steps_reconciled.csv, testnet_transactions_reconciled.csv, testnet_chart_reconciled.png')
