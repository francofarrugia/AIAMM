"""Shared helpers for the AIAMM testnet demonstration (Ethereum Sepolia, Uniswap v3).

The kit deploys two test tokens with no monetary value, creates a Uniswap v3 pool,
and lets the trained AIAMM DDQN agent manage a concentrated-liquidity position next to
a passive full-range benchmark position, while real ETH/USDC hourly prices are replayed.
"""
import json, math, os, sys, time
from pathlib import Path

import numpy as np
from web3 import Web3
from web3.logs import DISCARD

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'
OUT.mkdir(exist_ok=True)

NETWORKS = {
    'sepolia': {
        'chain_id': 11155111,
        'rpc_urls': ['https://ethereum-sepolia-rpc.publicnode.com',
                     'https://sepolia.drpc.org',
                     'https://rpc.sepolia.org'],
        'explorer': 'https://sepolia.etherscan.io',
        # Official Uniswap v3 deployments on Ethereum Sepolia (developers.uniswap.org)
        'factory': '0x0227628f3F023bb0B980b67D528571c95c6DaC1c',
        'npm': '0x1238536071E1c677A632429e3655c799b22cDA52',
        'router': '0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E',
    },
    'local': {
        'chain_id': 31337,
        'rpc_urls': ['http://127.0.0.1:8545'],
        'explorer': '(local test chain – no explorer)',
    },
}

FEE = 3000            # 0.30% pool
TICK_SPACING = 60
MIN_TICK, MAX_TICK = -887220, 887220   # multiples of 60
Q96 = 2 ** 96
UINT128_MAX = 2 ** 128 - 1
DEC = 10 ** 18        # both test tokens use 18 decimals


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S') + '  ' + msg
    print(line, flush=True)
    with open(OUT / 'run.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def abi(name):
    return json.loads((HERE / 'abi' / f'{name}.json').read_text())['abi']


def load_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str))


class Chain:
    def __init__(self, network):
        if network not in NETWORKS:
            sys.exit(f'Unknown network {network}')
        self.network = network
        self.cfg = dict(NETWORKS[network])
        if network == 'local':
            self.cfg.update(load_json(HERE / 'local_addresses.json', {}))
            self.key = '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80'
        else:
            wallet = load_json(HERE / 'wallet.json')
            if not wallet:
                sys.exit('wallet.json not found – run  python 00_create_wallet.py  first.')
            self.key = wallet['private_key']
        self.w3 = None
        for url in self.cfg['rpc_urls']:
            try:
                w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 60}))
                if w3.is_connected() and w3.eth.chain_id == self.cfg['chain_id']:
                    self.w3, self.rpc = w3, url
                    break
            except Exception as e:  # try next RPC
                print(f'  RPC {url} unavailable: {e}')
        if self.w3 is None:
            sys.exit('Could not connect to any RPC endpoint for ' + network)
        self.acct = self.w3.eth.account.from_key(self.key)
        self.addr = self.acct.address
        self.gas_used_total = 0
        self.eth_spent_total = 0.0

    def explorer_tx(self, h):
        return f"{self.cfg['explorer']}/tx/{h}" if self.network != 'local' else h

    def contract(self, address, name):
        return self.w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi(name))

    def _fees(self):
        if self.network == 'local':
            return {'gasPrice': self.w3.eth.gas_price}
        base = self.w3.eth.get_block('latest')['baseFeePerGas']
        try:
            prio = self.w3.eth.max_priority_fee
        except Exception:
            prio = Web3.to_wei(1, 'gwei')
        prio = max(prio, Web3.to_wei(0.1, 'gwei'))
        return {'maxFeePerGas': int(base * 2 + prio), 'maxPriorityFeePerGas': int(prio)}

    def send(self, fn, label, value=0, retries=3):
        """Build, sign and send a contract call; wait for the receipt; return a record.

        Fixed 29 Oct 2026: a transaction is never re-sent once the RPC has accepted it. Public RPCs
        sometimes fail *after* broadcasting (e.g. "Block with id ... not found" while reading the
        receipt or block). The earlier version treated that as a failure and sent the call again,
        so a successful first transaction went unrecorded (and, for a collect, the tokens it
        withdrew were never credited to the agent). Now: only the broadcast is retried; after
        that we keep polling the receipt of the SAME hash, and a failed block lookup is non-fatal.
        """
        last_err, h = None, None
        for attempt in range(retries):
            try:
                tx = fn.build_transaction({'from': self.addr, 'nonce': self.w3.eth.get_transaction_count(self.addr, 'pending'),
                                           'chainId': self.cfg['chain_id'], 'value': value, **self._fees()})
                tx['gas'] = int(self.w3.eth.estimate_gas(tx) * 1.25)
                signed = self.acct.sign_transaction(tx)
                h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                break
            except Exception as e:
                last_err = e
                log(f'  retry {attempt + 1}/{retries} (broadcast) for {label}: {e}')
                time.sleep(10)
        if h is None:
            raise RuntimeError(f'{label} failed: {last_err}')
        hx = '0x' + h.hex().removeprefix('0x')
        rc = None
        for attempt in range(30):                      # up to ~10 min of polling the same hash
            try:
                rc = self.w3.eth.wait_for_transaction_receipt(h, timeout=20)
                break
            except Exception as e:
                last_err = e
                if attempt % 5 == 4:
                    log(f'  still waiting for receipt of {label} {hx}: {e}')
                time.sleep(5)
        if rc is None:
            raise RuntimeError(f'{label}: no receipt for {hx} (NOT re-sent; check the explorer): {last_err}')
        if rc.status != 1:
            raise RuntimeError(f'transaction reverted: {hx}')
        price = rc.get('effectiveGasPrice', tx.get('gasPrice', 0))
        eth = rc.gasUsed * price / 1e18
        self.gas_used_total += rc.gasUsed
        self.eth_spent_total += eth
        try:
            ts = self.w3.eth.get_block(rc.blockNumber).timestamp
        except Exception:
            ts = time.time()                           # block lookup failed: fall back to local clock
        rec = {'label': label, 'tx': hx, 'block': rc.blockNumber,
               'time_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts)),
               'gas_used': rc.gasUsed, 'eth_fee': eth, 'url': self.explorer_tx(hx)}
        log(f'  tx {label}: {hx}  gas {rc.gasUsed}')
        return rec, rc

    def deploy(self, name, *args, label=None):
        art = json.loads((HERE / 'abi' / f'{name}.json').read_text())
        c = self.w3.eth.contract(abi=art['abi'], bytecode=art['bytecode'])
        rec, rc = self.send(c.constructor(*args), label or f'deploy {name}')
        rec['address'] = rc.contractAddress
        return rec

    def eth_balance(self):
        return self.w3.eth.get_balance(self.addr) / 1e18


# ---------------- Uniswap v3 maths ----------------
def tick_to_sqrt(tick):
    return math.sqrt(1.0001 ** tick)

def price_to_tick(p):
    return math.floor(math.log(p) / math.log(1.0001))

def round_tick(t):
    return int(round(t / TICK_SPACING) * TICK_SPACING)

def sqrt_x96_from_raw_price(p_raw):
    return int(math.sqrt(p_raw) * Q96)

def amounts_for_liquidity(L, sqrt_p, tick_lower, tick_upper):
    sa, sb = tick_to_sqrt(tick_lower), tick_to_sqrt(tick_upper)
    if sqrt_p <= sa:
        return L * (sb - sa) / (sa * sb), 0.0
    if sqrt_p >= sb:
        return 0.0, L * (sb - sa)
    return L * (sb - sqrt_p) / (sqrt_p * sb), L * (sqrt_p - sa)


class Pair:
    """Price convention: USD price of tETH = tUSDC per tETH, whatever the token order."""
    def __init__(self, teth, tusdc):
        self.teth, self.tusdc = Web3.to_checksum_address(teth), Web3.to_checksum_address(tusdc)
        self.eth_is_token0 = int(self.teth, 16) < int(self.tusdc, 16)
        self.token0, self.token1 = (self.teth, self.tusdc) if self.eth_is_token0 else (self.tusdc, self.teth)

    def raw_from_usd(self, usd):        # raw price = token1 per token0
        return usd if self.eth_is_token0 else 1.0 / usd

    def usd_from_raw(self, raw):
        return raw if self.eth_is_token0 else 1.0 / raw

    def ticks_for_usd_range(self, lo_usd, hi_usd):
        a, b = price_to_tick(self.raw_from_usd(lo_usd)), price_to_tick(self.raw_from_usd(hi_usd))
        lo, hi = sorted((round_tick(a), round_tick(b)))
        return max(lo, MIN_TICK), min(hi, MAX_TICK)

    def split(self, amt0, amt1):
        """(token0, token1) amounts -> (eth, usdc)"""
        return (amt0, amt1) if self.eth_is_token0 else (amt1, amt0)

    def join(self, eth, usdc):
        """(eth, usdc) -> (token0, token1)"""
        return (eth, usdc) if self.eth_is_token0 else (usdc, eth)

    def value_usd(self, eth, usdc, usd_price):
        return eth * usd_price + usdc


def pool_usd_price(pool, pair):
    sqrt_x96 = pool.functions.slot0().call()[0]
    raw = (sqrt_x96 / Q96) ** 2
    return pair.usd_from_raw(raw), sqrt_x96 / Q96


def position_state(chain, npm, pool, pair, token_id):
    """Return (eth, usdc) held in the position, uncollected fees (eth, usdc) and liquidity."""
    if token_id is None:
        return (0.0, 0.0), (0.0, 0.0), 0
    p = npm.functions.positions(token_id).call()
    L, tl, tu = p[7], p[5], p[6]
    _, sqrt_p = pool_usd_price(pool, pair)
    a0, a1 = amounts_for_liquidity(L, sqrt_p, tl, tu)
    try:
        f0, f1 = npm.functions.collect((token_id, chain.addr, UINT128_MAX, UINT128_MAX)).call({'from': chain.addr})
    except Exception:
        f0, f1 = p[10], p[11]
    return pair.split(a0 / DEC, a1 / DEC), pair.split(f0 / DEC, f1 / DEC), L


def load_agent(v13dir=HERE):
    """Recreate the trained V13 DDQN agent from the exported weights."""
    sys.path.insert(0, str(v13dir))
    cwd = os.getcwd(); os.chdir(v13dir)
    import importlib
    v = importlib.import_module('AIAMM_V13')
    os.chdir(cwd)
    w = np.load(HERE / 'model' / 'ddqn_v13_seed42.npz', allow_pickle=False)
    agent = v.DQLAgent()
    for k in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        setattr(agent.online, k, w[k].copy())
    norm = json.loads(str(w['norm_attrs']))
    agent.normalizer.n = norm['n']
    agent.normalizer.mean = np.array(norm['mean'])
    agent.normalizer.M2 = np.array(norm['M2'])
    agent.epsilon = float(agent.epsilon_min)
    agent.eval_mode = True
    return v, agent
