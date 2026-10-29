"""Step 2 – run the trained AIAMM DDQN agent on-chain.

Each step replays the next real ETH/USDC hourly price: a "market" swap moves the Uniswap v3
pool to that price, the agent observes the on-chain state, chooses one of its five actions,
and the action is executed as real transactions. The run checkpoints after every step and can
be resumed by running the same command again.

  python 02_run_agent.py --steps 48              (Ethereum Sepolia)
  python 02_run_agent.py --network local --steps 48
"""
import argparse, math, random, time
import numpy as np
from tn_common import *

ap = argparse.ArgumentParser()
ap.add_argument('--network', default='sepolia')
ap.add_argument('--steps', type=int, default=48)
ap.add_argument('--test-actions', default='', help='(testing only) comma-separated forced actions')
ap.add_argument('--min-eth', type=float, default=0.004, help='stop safely if the wallet balance falls below this')
args = ap.parse_args()

dep = load_json(OUT / 'deployment.json')
if not dep:
    sys.exit('Run 01_deploy.py first.')
ch = Chain(args.network)
pair = Pair(dep['tETH'], dep['tUSDC'])
npm = ch.contract(ch.cfg['npm'], 'NonfungiblePositionManager')
router = ch.contract(ch.cfg['router'], 'SwapRouter02')
pool = ch.contract(dep['pool'], 'UniswapV3Pool')
v, agent = load_agent(HERE / 'AIAMM_data')
data = np.load(HERE / 'AIAMM_data' / 'binance_eth_usdc_hourly.npz')
C, O, H, Lw, VOL, TS = (data[k] for k in ('close', 'open', 'high', 'low', 'volume', 'open_times'))
P0, s0 = dep['P0'], dep['start_idx']
ACTIONS = ['Hold', 'Increase liquidity', 'Decrease liquidity', 'Shift range up 3%', 'Shift range down 3%']
GAS_MODEL = {1: 0.15, 2: 0.15, 3: 0.30, 4: 0.30}   # fixed USD-equivalent costs used in the V13 model

st = load_json(OUT / 'state.json') or {
    'step': 0, 'cooldown': 0, 'liq_ratio': 0.75, 'token_id': dep['agent_token_id'],
    'idle': dep['agent_idle'], 'center': dep['agent_center'], 'cum_cost': 0.0, 'fees_collected_usd': 0.0,
    'amm_hist': [1.0], 'mkt_hist': [1.0], 'sent_base': 0.0, 'prev_action': -1,
    'norm': {'n': agent.normalizer.n, 'mean': agent.normalizer.mean.tolist(), 'M2': agent.normalizer.M2.tolist()},
    'rng': 42, 'rows': []}
agent.normalizer.n = st['norm']['n']; agent.normalizer.mean = np.array(st['norm']['mean']); agent.normalizer.M2 = np.array(st['norm']['M2'])
agent.prev_action = st['prev_action']
np.random.seed(st['rng'] + st['step']); random.seed(st['rng'] + st['step'])
sa = v.SentimentAnalyzer(); sa.base_sentiment = st['sent_base']


def deadline():
    return ch.w3.eth.get_block('latest').timestamp + 1800


def move_market(target_usd):
    """Swap against the pool until its price equals the next replayed market price."""
    cur, _ = pool_usd_price(pool, pair)
    if abs(target_usd / cur - 1) < 1e-4:
        return []
    token_in, token_out = (pair.tusdc, pair.teth) if target_usd > cur else (pair.teth, pair.tusdc)
    limit = sqrt_x96_from_raw_price(pair.raw_from_usd(target_usd))
    try:
        r, _ = ch.send(router.functions.exactInputSingle((token_in, token_out, FEE, ch.addr, 10 ** 27, 0, limit)),
                       f'market move to ${target_usd:,.2f}')
        return [r]
    except RuntimeError as e:
        # Common failure mode on public RPCs: the first attempt actually succeeded on-chain
        # (pool price now at target) but the RPC returned "Block ... not found" on the receipt
        # query, so send() retried the same swap and Uniswap v3 rejected it with 'SPL'
        # (can't move price in the stated direction when we're already at the limit).
        if 'SPL' in str(e):
            cur_after, _ = pool_usd_price(pool, pair)
            if abs(target_usd / cur_after - 1) < 0.01:  # within 1% of target is close enough
                log(f'  SPL caught; pool price already at ${cur_after:,.2f} (target ${target_usd:,.2f}) — treating as success')
                return []
        raise


def unit_amounts(lo_usd, hi_usd, price_usd):
    """(eth, usdc) per unit of liquidity for a USD range, and the ticks."""
    tl, tu = pair.ticks_for_usd_range(lo_usd, hi_usd)
    _, sp = pool_usd_price(pool, pair)
    a0, a1 = amounts_for_liquidity(1.0, sp, tl, tu)
    e, u = pair.split(a0 / DEC, a1 / DEC)
    return e, u, tl, tu


def amounts_for_value(value_usd, lo_usd, hi_usd, price_usd, idle):
    e, u, tl, tu = unit_amounts(lo_usd, hi_usd, price_usd)
    per_l = e * price_usd + u
    L = value_usd / per_l if per_l > 0 else 0
    want_e, want_u = e * L, u * L
    scale = min(1.0, (idle[0] / want_e) if want_e > 0 else 1.0, (idle[1] / want_u) if want_u > 0 else 1.0)
    return want_e * scale * 0.999, want_u * scale * 0.999, tl, tu


def remove_and_collect(token_id, liquidity, label):
    recs = []
    if liquidity > 0:
        r, _ = ch.send(npm.functions.decreaseLiquidity((token_id, int(liquidity), 0, 0, deadline())), f'{label}: decrease liquidity')
        recs.append(r)
    r, rc = ch.send(npm.functions.collect((token_id, ch.addr, UINT128_MAX, UINT128_MAX)), f'{label}: collect')
    recs.append(r)
    ev = npm.events.Collect().process_receipt(rc, errors=DISCARD)
    e, u = pair.split(ev[0]['args']['amount0'] / DEC, ev[0]['args']['amount1'] / DEC) if ev else (0.0, 0.0)
    return recs, e, u


log(f'Agent run on {args.network}: steps {st["step"]} -> {args.steps}; wallet balance {ch.eth_balance():.4f} ETH')
while st['step'] < args.steps:
    if args.network != 'local' and ch.eth_balance() < args.min_eth:
        log('Wallet balance below safety limit – stopping. Top up from a faucet and run the same command to resume.')
        break
    t = st['step']; idx = s0 + t + 1
    gas0, eth0 = ch.gas_used_total, ch.eth_spent_total
    s = sa.get_sentiment(st['amm_hist'])                       # as in V13: sentiment from AMM price history
    txs = move_market(float(C[idx]))
    mkt, _ = pool_usd_price(pool, pair)                         # pool now at the replayed market price
    target = float(C[idx])
    st['amm_hist'].append(mkt / P0); st['mkt_hist'].append(target / P0)

    (ae, au), (afe, afu), aL = position_state(ch, npm, pool, pair, st['token_id'])
    (pe, pu), (pfe, pfu), _ = position_state(ch, npm, pool, pair, dep['passive_token_id'])
    agent_val = pair.value_usd(ae + afe + st['idle'][0], au + afu + st['idle'][1], mkt) - st['cum_cost']
    passive_val = pair.value_usd(pe + pfe + dep['passive_idle'][0], pu + pfu + dep['passive_idle'][1], mkt)
    alpha_pct = (agent_val - passive_val) / max(passive_val, 1.0)
    fees_now = st['fees_collected_usd'] + pair.value_usd(afe, afu, mkt)
    vol = v.rolling_vol(st['mkt_hist'], 20)
    disloc = (st['amm_hist'][-1] - st['mkt_hist'][-1]) / st['mkt_hist'][-1]
    mom = v.momentum_sign(st['mkt_hist'], 20)
    rlr = float(np.clip(math.log(st['center'] / mkt), -2, 2))
    hl = float((H[idx] - Lw[idx]) / C[idx]); oc = float((C[idx] - O[idx]) / O[idx])
    vr = float(VOL[idx] / max(VOL[max(0, idx - 24):idx + 1].mean(), 1e-9) - 1.0)
    state = agent.build_state(st['amm_hist'], st['mkt_hist'], s, st['liq_ratio'], fees_now / 2000.0, vol, disloc,
                              alpha_pct, mom, min(st['cooldown'] / 50.0, 1.0), rlr, t / max(args.steps - 1, 1), hl, oc, vr)
    action = agent.act(state)
    if args.test_actions:
        seq = [int(a) for a in args.test_actions.split(',')]
        action = seq[t % len(seq)]
    if st['cooldown'] < v.MIN_COOLDOWN and action != 0:
        action = 0
    note = ''
    if action == 0:
        st['cooldown'] += 1
    else:
        st['cooldown'] = 0
        st['cum_cost'] += GAS_MODEL[action]
        lo_usd, hi_usd = st['center'] * 0.60, st['center'] * 1.40
        if action == 1:                                       # add $200 (10% of capital) from idle tokens
            st['liq_ratio'] = min(st['liq_ratio'] + 0.1, 1.0)
            e, u, _, _ = amounts_for_value(200.0, lo_usd, hi_usd, mkt, st['idle'])
            if e + u > 0:
                a0, a1 = pair.join(int(e * DEC), int(u * DEC))
                r, rc = ch.send(npm.functions.increaseLiquidity((st['token_id'], a0, a1, 0, 0, deadline())), 'AGENT increase liquidity')
                txs.append(r); ev = npm.events.IncreaseLiquidity().process_receipt(rc, errors=DISCARD)[0]['args']
                ue, uu = pair.split(ev['amount0'] / DEC, ev['amount1'] / DEC)
                st['idle'] = [st['idle'][0] - ue, st['idle'][1] - uu]
            else:
                note = 'no idle tokens to add'
        elif action == 2:                                     # remove ~10% of capital
            frac = min(0.1 / max(st['liq_ratio'], 1e-9), 1.0)
            st['liq_ratio'] = max(st['liq_ratio'] - 0.1, 0.3)
            st['fees_collected_usd'] += pair.value_usd(afe, afu, mkt)
            recs, e, u = remove_and_collect(st['token_id'], aL * frac, 'AGENT')
            txs += recs; st['idle'] = [st['idle'][0] + e, st['idle'][1] + u]
        else:                                                 # shift range up / down by 3%
            st['center'] *= 1.03 if action == 3 else 0.97
            st['fees_collected_usd'] += pair.value_usd(afe, afu, mkt)
            recs, e, u = remove_and_collect(st['token_id'], aL, 'AGENT shift')
            txs += recs; st['idle'] = [st['idle'][0] + e, st['idle'][1] + u]
            total = pair.value_usd(st['idle'][0], st['idle'][1], mkt)
            e, u, tl, tu = amounts_for_value(st['liq_ratio'] * total, st['center'] * 0.60, st['center'] * 1.40, mkt, st['idle'])
            a0, a1 = pair.join(int(e * DEC), int(u * DEC))
            r, rc = ch.send(npm.functions.mint((pair.token0, pair.token1, FEE, tl, tu, a0, a1, 0, 0, ch.addr, deadline())),
                            'AGENT mint shifted range')
            txs.append(r); ev = npm.events.IncreaseLiquidity().process_receipt(rc, errors=DISCARD)[0]['args']
            st['token_id'] = ev['tokenId']
            ue, uu = pair.split(ev['amount0'] / DEC, ev['amount1'] / DEC)
            st['idle'] = [st['idle'][0] - ue, st['idle'][1] - uu]
    agent.prev_action = action

    # value after the action
    (ae, au), (afe, afu), _ = position_state(ch, npm, pool, pair, st['token_id'])
    agent_val = pair.value_usd(ae + afe + st['idle'][0], au + afu + st['idle'][1], mkt) - st['cum_cost']
    in_range = st['center'] * 0.60 <= mkt <= st['center'] * 1.40
    blk = ch.w3.eth.get_block('latest')
    row = {'step': t + 1, 'replay_hour_utc': time.strftime('%Y-%m-%d %H:%M', time.gmtime(int(TS[idx]) / 1000)),
           'block': blk.number, 'chain_time_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(blk.timestamp)),
           'market_price': round(target, 2), 'pool_price': round(mkt, 2), 'sentiment': round(float(s), 4),
           'action': ACTIONS[action], 'note': note, 'agent_value': round(agent_val, 4), 'passive_value': round(passive_val, 4),
           'alpha_usd': round(agent_val - passive_val, 4),
           'agent_fees_usd': round(st['fees_collected_usd'] + pair.value_usd(afe, afu, mkt), 6),
           'passive_fees_usd': round(pair.value_usd(pfe, pfu, mkt), 6), 'liq_ratio': round(st['liq_ratio'], 2),
           'agent_range_usd': [round(st['center'] * 0.6, 2), round(st['center'] * 1.4, 2)], 'in_range': in_range,
           'model_cost_usd': round(st['cum_cost'], 2), 'gas_used': ch.gas_used_total - gas0,
           'test_eth_fee': round(ch.eth_spent_total - eth0, 8), 'txs': txs}
    st['rows'].append(row)
    log(f"step {t + 1:3d}/{args.steps}  price ${target:,.2f}  action {ACTIONS[action]:<20s} "
        f"agent ${agent_val:,.2f}  passive ${passive_val:,.2f}  alpha {agent_val - passive_val:+.2f}")
    st['step'] = t + 1; st['sent_base'] = sa.base_sentiment; st['prev_action'] = action
    st['norm'] = {'n': agent.normalizer.n, 'mean': agent.normalizer.mean.tolist(), 'M2': agent.normalizer.M2.tolist()}
    save_json(OUT / 'state.json', st)

log(f'Finished at step {st["step"]}. Wallet balance {ch.eth_balance():.4f} ETH. Next: python 03_report.py')
