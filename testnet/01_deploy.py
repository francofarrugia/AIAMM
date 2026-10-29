"""Step 1 – deploy test tokens, create the Uniswap v3 pool and open the three positions.

  python 01_deploy.py              (Ethereum Sepolia)
  python 01_deploy.py --network local
"""
import argparse, time
import numpy as np
from tn_common import *

ap = argparse.ArgumentParser(); ap.add_argument('--network', default='sepolia')
ap.add_argument('--start-hours-before-end', type=int, default=200,
                help='replay window starts this many hours before the end of the dataset')
args = ap.parse_args()

ch = Chain(args.network)
log(f'Network {args.network} via {ch.rpc}; wallet {ch.addr}; balance {ch.eth_balance():.4f} ETH')
gp = ch.w3.eth.gas_price
est = (7_500_000 + 48 * 130_000 + 3_000_000) * gp / 1e18
log(f'Current gas price {gp / 1e9:.2f} gwei -> estimated cost for deployment + 48 steps: {est:.4f} test ETH')
if args.network != 'local' and ch.eth_balance() < max(0.005, est * 1.3):
    sys.exit(f'Balance too low ({ch.eth_balance():.4f} ETH). Request more Sepolia ETH from a faucet (about {max(0.01, est * 1.5):.3f} ETH needed).')
if load_json(OUT / 'deployment.json'):
    sys.exit('output/deployment.json already exists – deployment was done. Run 02_run_agent.py.')

data = np.load(HERE / 'AIAMM_data' / 'binance_eth_usdc_hourly.npz')
start_idx = len(data['close']) - args.start_hours_before_end
P0 = float(data['close'][start_idx])
log(f'Replay window starts at dataset index {start_idx} '
    f'({time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(data["open_times"][start_idx]) / 1000))} UTC), price ${P0:,.2f}')

dep = {'network': args.network, 'chain_id': ch.cfg['chain_id'], 'explorer': ch.cfg['explorer'], 'rpc': ch.rpc,
       'wallet': ch.addr, 'uniswap': {k: ch.cfg[k] for k in ('factory', 'npm', 'router')},
       'fee_tier': FEE, 'start_idx': start_idx, 'P0': P0, 'txs': []}
T = dep['txs']

# 1. test tokens (no monetary value)
r = ch.deploy('AIAMMTestToken', 'AIAMM Test ETH', 'tETH', 10 ** 9 * DEC, label='deploy tETH'); T.append(r); dep['tETH'] = r['address']
r = ch.deploy('AIAMMTestToken', 'AIAMM Test USDC', 'tUSDC', 10 ** 13 * DEC, label='deploy tUSDC'); T.append(r); dep['tUSDC'] = r['address']
pair = Pair(dep['tETH'], dep['tUSDC'])
dep['token0'], dep['token1'] = pair.token0, pair.token1

npm = ch.contract(ch.cfg['npm'], 'NonfungiblePositionManager')
factory = ch.contract(ch.cfg['factory'], 'UniswapV3Factory')

# 2. pool at the first replay price
r, _ = ch.send(npm.functions.createAndInitializePoolIfNecessary(pair.token0, pair.token1, FEE,
               sqrt_x96_from_raw_price(pair.raw_from_usd(P0))), 'create + initialise pool'); T.append(r)
dep['pool'] = factory.functions.getPool(pair.token0, pair.token1, FEE).call()
log(f'Pool {dep["pool"]}')

# 3. approvals
for tok in (dep['tETH'], dep['tUSDC']):
    t = ch.contract(tok, 'AIAMMTestToken')
    for spender, nm in ((ch.cfg['npm'], 'PositionManager'), (ch.cfg['router'], 'SwapRouter02')):
        r, _ = ch.send(t.functions.approve(spender, 2 ** 256 - 1), f'approve {nm}'); T.append(r)

def mint(eth_amt, usdc_amt, lo_tick, hi_tick, label):
    a0, a1 = pair.join(int(eth_amt * DEC), int(usdc_amt * DEC))
    deadline = ch.w3.eth.get_block('latest').timestamp + 1800
    r, rc = ch.send(npm.functions.mint((pair.token0, pair.token1, FEE, lo_tick, hi_tick, a0, a1, 0, 0, ch.addr, deadline)), label)
    ev = npm.events.IncreaseLiquidity().process_receipt(rc, errors=DISCARD)[0]['args']
    used_eth, used_usdc = pair.split(ev['amount0'] / DEC, ev['amount1'] / DEC)
    return r, ev['tokenId'], used_eth, used_usdc

# 4. background market liquidity (full range, $200,000) so that replayed prices can be traded
r, bg_id, _, _ = mint(100_000 / P0, 100_000, MIN_TICK, MAX_TICK, 'mint background liquidity (full range)'); T.append(r)
# 5. passive benchmark: $2,000 full range (50/50), never touched again
r, pas_id, pe, pu = mint(1000 / P0, 1000, MIN_TICK, MAX_TICK, 'mint PASSIVE benchmark position'); T.append(r)
# 6. AI agent: $2,000 budget; 75% deployed in a ±40% range around P0 (as in the V13 simulation)
lo, hi = pair.ticks_for_usd_range(P0 * 0.60, P0 * 1.40)
r, ai_id, ae, au = mint(0.75 * 1000 / P0, 0.75 * 1000, lo, hi, 'mint AI AGENT position (±40%)'); T.append(r)

dep.update({'background_token_id': bg_id, 'passive_token_id': pas_id, 'agent_token_id': ai_id,
            'passive_idle': [1000 / P0 - pe, 1000 - pu],
            'agent_idle': [1000 / P0 - ae, 1000 - au], 'agent_range_usd': [P0 * 0.60, P0 * 1.40],
            'agent_center': P0, 'initial_value_usd': 2000.0,
            'gas_used_deploy': ch.gas_used_total, 'eth_spent_deploy': ch.eth_spent_total})
save_json(OUT / 'deployment.json', dep)
log(f'Deployment complete: {len(T)} transactions, {ch.eth_spent_total:.5f} test ETH used. '
    f'Balance left {ch.eth_balance():.4f} ETH. Next: python 02_run_agent.py')
