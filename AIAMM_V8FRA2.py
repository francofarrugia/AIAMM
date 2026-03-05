# ============================================================
# AIAMM V8 — Deep RL for LP Management (Improvements over V7)
# ============================================================
#
# V7 PROBLEMS IDENTIFIED (from results):
#
# PROBLEM 1: Cumulative reward always deeply negative (→ -1300)
#   ROOT CAUSE: alpha_change is computed as a *fraction* of initial_value
#   (tiny, ~0.001), then divided by vol_scale (also tiny, ~1.5),
#   producing net rewards close to zero. But rebalance cost penalty
#   is NOT scaled the same way → net negative per step.
#   FIX V8: Unified reward scaling. Use direct dollar alpha change
#   (not ratio) and calibrate all reward components consistently.
#   Remove vol_scale division — it makes signals vanishingly small.
#
# PROBLEM 2: Hold dominates (663/1000 = 66%) despite being suboptimal
#   ROOT CAUSE: action-repeat penalty + cost penalty makes any action
#   worse than hold. Agent learns Hold as the safest local minimum.
#   FIX V8: Reduce action-repeat penalty. Add a small hold penalty
#   when the agent is out-of-range (active_fraction low) — passivity
#   costs fees, so it should be penalized when range is stale.
#
# PROBLEM 3: Win rate 47% — barely above random
#   ROOT CAUSE: Linear-in-time reward degradation prevents the agent
#   from learning long-horizon strategies. Gamma=0.95 means only
#   ~20 steps of meaningful credit assignment.
#   FIX V8: Double DQN (eliminates overestimation bias).
#   N-step returns (n=5) for longer credit assignment.
#   Increase gamma to 0.99.
#
# PROBLEM 4: ShallowNet with single-sample updates is noisy
#   ROOT CAUSE: Updating one sample at a time (inside minibatch loop)
#   produces high gradient variance. Also, standard SGD has no
#   momentum — slow convergence.
#   FIX V8: Adam optimizer (momentum + adaptive LR) implemented
#   in NumPy. Two hidden layers (32→64→32) for richer representation.
#   Minibatch update using vectorized forward pass.
#
# PROBLEM 5: State features are redundant/uninformative
#   ROOT CAUSE: (liq_ratio - 0.75)/0.25 is perfectly correlated with
#   liq_ratio already in state. alpha_norm and amm_ret encode similar
#   info.
#   FIX V8: Replace redundant feature with:
#   (a) range_center_log_ratio: log(range_center/market_price) — how
#       far off-center the LP is relative to market. Critical signal.
#   (b) fee_acceleration: change in fee rate (momentum of earnings).
#   (c) step_fraction: t/T — enables time-aware strategies.
#   Total: 12 features, all meaningful and low-redundancy.
#
# PROBLEM 6: Epsilon decays only after memory fills (8000 steps)
#   But memory fills partway through episode 8 at 1000 steps/ep.
#   After memory fills, epsilon decays at 0.998/step = very slow.
#   FIX V8: Decay epsilon per *episode* not per *batch update*.
#   Gives clean phase transitions: explore (eps=1→0.3) then exploit.
#
# ADDITIONAL: No reward shaping for being in-range
#   FIX V8: Add small bonus for active_fraction > 0.8 (concentrated
#   liquidity earns more fees). Encourages the agent to recenter.
#
# ADDITIONAL: N-step buffer for better credit assignment
#   FIX V8: Collect (s,a,r_sum,s_n,done) with n=5 step lookahead.
#   This propagates terminal outcomes back further in time.

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import random
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)

# ─────────────────────────────────────────────────────────────────────────────
# AMMEnvironment  (unchanged — microstructure is fine)
# ─────────────────────────────────────────────────────────────────────────────

class AMMEnvironment:
    def __init__(
        self,
        initial_token_a=10_000,
        initial_token_b=10_000,
        fee=0.003,
        price_feed=None,
        arbitrage_band_bps=12,
        max_trade_fraction=0.12,
        arb_gas_cost_b=0.25,
        rebalance_gas_cost_b=0.15,
    ):
        self.initial_token_a = initial_token_a
        self.initial_token_b = initial_token_b
        self.token_a = initial_token_a
        self.token_b = initial_token_b
        self.fee = fee
        self.max_trade_fraction = max_trade_fraction
        self.arbitrage_band_bps = arbitrage_band_bps
        self.arb_gas_cost_b = arb_gas_cost_b
        self.rebalance_gas_cost_b = rebalance_gas_cost_b
        self.market_price = self.get_price()
        self.price_history = [self.get_price()]
        self.market_price_history = [self.market_price]
        self.volume_history = []
        self.fee_history = []
        self.arb_volume_history = []
        self.dislocation_history = [0.0]
        self.step_count = 0
        self.price_feed = price_feed
        self.price_feed_index = 0
        self.market_volatility = 0.015
        self.market_drift = 0.0

    def get_price(self):
        return self.token_b / self.token_a if self.token_a > 0 else float("inf")

    def _next_market_price(self, sentiment=0.0):
        if self.price_feed:
            if self.price_feed_index < len(self.price_feed):
                p = self.price_feed[self.price_feed_index]
                self.price_feed_index += 1
                if p > 0:
                    self.market_price = float(p)
                    return self.market_price

        if self.step_count > 0 and self.step_count % 120 == 0:
            self.market_volatility = float(np.random.choice([0.008, 0.015, 0.03]))
            self.market_drift = np.random.normal(0, 0.0015)

        shock = np.random.normal(
            self.market_drift + sentiment * 0.003,
            self.market_volatility,
        )
        self.market_price = max(1e-6, self.market_price * np.exp(shock))
        return self.market_price

    def _simulate_order_flow(self, sentiment=0.0):
        base = self.token_b * 0.0025
        notional = base * np.random.lognormal(0.0, 0.65)
        buy_prob = np.clip(0.5 + 0.35 * np.tanh(sentiment), 0.05, 0.95)
        if np.random.rand() < buy_prob:
            _, vol_b, fee_b = self.swap_b_for_a(notional)
        else:
            amount_a = notional / max(self.market_price, 1e-9)
            _, vol_b, fee_b = self.swap_a_for_b(amount_a)
        return vol_b, fee_b

    def _run_arbitrage(self):
        amm = self.get_price()
        mkt = self.market_price
        if not np.isfinite(amm) or mkt <= 0:
            return 0.0, 0.0
        disloc = (amm - mkt) / mkt
        if abs(disloc) * 10_000 < self.arbitrage_band_bps:
            return 0.0, 0.0
        target = amm + 0.8 * (mkt - amm)
        k = self.token_a * self.token_b
        if target > amm:
            db = max(0.0, np.sqrt(k * target) - self.token_b)
            amt = db / max(1 - self.fee, 1e-9)
            if amt < self.arb_gas_cost_b * 8:
                return 0.0, 0.0
            _, vol_b, fee_b = self.swap_b_for_a(amt)
        else:
            da = max(0.0, np.sqrt(k / max(target, 1e-9)) - self.token_a)
            amt = da / max(1 - self.fee, 1e-9)
            if amt * max(self.market_price, 1e-9) < self.arb_gas_cost_b * 8:
                return 0.0, 0.0
            _, vol_b, fee_b = self.swap_a_for_b(amt)
        return vol_b, fee_b

    def simulate_market_activity(self, sentiment=0.0):
        self.step_count += 1
        self._next_market_price(sentiment)
        flow_vol, flow_fee = self._simulate_order_flow(sentiment)
        arb_vol, arb_fee = self._run_arbitrage()
        total_vol = flow_vol + arb_vol
        total_fee = flow_fee + arb_fee
        amm = self.get_price()
        disloc = (amm - self.market_price) / max(self.market_price, 1e-9)
        self.price_history.append(amm)
        self.market_price_history.append(self.market_price)
        self.volume_history.append(total_vol)
        self.fee_history.append(total_fee)
        self.arb_volume_history.append(arb_vol)
        self.dislocation_history.append(disloc)
        return {
            "market_price": self.market_price,
            "amm_price": amm,
            "price_dislocation": disloc,
            "volume_b": total_vol,
            "fees_collected_b": total_fee,
            "arb_volume_b": arb_vol,
        }

    def swap_a_for_b(self, amount_a):
        if amount_a <= 0:
            return 0.0, 0.0, 0.0
        amount_a = min(amount_a, self.token_a * self.max_trade_fraction)
        net = amount_a * (1 - self.fee)
        out_b = self.token_b * net / (self.token_a + net)
        fee_a = amount_a - net
        self.token_a += amount_a
        self.token_b -= out_b
        return out_b, amount_a * max(self.market_price, 1e-9), fee_a * max(self.market_price, 1e-9)

    def swap_b_for_a(self, amount_b):
        if amount_b <= 0:
            return 0.0, 0.0, 0.0
        amount_b = min(amount_b, self.token_b * self.max_trade_fraction)
        net = amount_b * (1 - self.fee)
        out_a = self.token_a * net / (self.token_b + net)
        fee_b = amount_b - net
        self.token_b += amount_b
        self.token_a -= out_a
        return out_a, amount_b, fee_b

    def reset(self):
        self.token_a = self.initial_token_a
        self.token_b = self.initial_token_b
        self.market_price = self.get_price()
        self.price_history = [self.get_price()]
        self.market_price_history = [self.market_price]
        self.volume_history = []
        self.fee_history = []
        self.arb_volume_history = []
        self.dislocation_history = [0.0]
        self.step_count = 0
        self.price_feed_index = 0


# ─────────────────────────────────────────────────────────────────────────────
# LiquidityProvider
# ─────────────────────────────────────────────────────────────────────────────

class LiquidityProvider:
    def __init__(self, deposit_a, deposit_b, entry_price):
        self.deposit_a = deposit_a
        self.deposit_b = deposit_b
        self.entry_price = entry_price
        self.initial_value = deposit_a * entry_price + deposit_b
        self.fees_earned = 0.0
        self.costs_paid = 0.0

    def calculate_impermanent_loss(self, current_price):
        if self.entry_price <= 0:
            return 0.0
        r = current_price / self.entry_price
        return 2 * np.sqrt(r) / (1 + r) - 1 if r > 0 else 0.0

    def hodl_value(self, current_price):
        return self.deposit_a * current_price + self.deposit_b

    def calculate_position_value(self, current_price, active_exposure=1.0, trading_costs=0.0):
        active_exposure = float(np.clip(active_exposure, 0.0, 1.0))
        il = self.calculate_impermanent_loss(current_price) * active_exposure
        hodl = self.hodl_value(current_price)
        return hodl * (1 + il) + self.fees_earned - trading_costs

    def add_fees(self, amount):
        self.fees_earned += max(0.0, amount)

    def add_cost(self, amount):
        self.costs_paid += max(0.0, amount)


# ─────────────────────────────────────────────────────────────────────────────
# SentimentAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_history = []
        self.base_sentiment = 0.0

    def get_sentiment(self, price_history, window=10):
        if len(price_history) < 2:
            s = np.random.normal(0, 0.1)
        else:
            recent = price_history[-min(window, len(price_history)):]
            rets = [(recent[i]-recent[i-1])/recent[i-1]
                    for i in range(1, len(recent)) if recent[i-1] != 0]
            if rets:
                mom = np.mean(rets)
                vol = np.std(rets) if len(rets) > 1 else 0.01
                s = np.tanh(mom / max(vol, 0.001) * 5) + np.random.normal(0, 0.2)
            else:
                s = np.random.normal(0, 0.1)
        self.base_sentiment = 0.7 * self.base_sentiment + 0.3 * s
        final = float(np.clip(self.base_sentiment, -1, 1))
        self.sentiment_history.append(final)
        return final


# ─────────────────────────────────────────────────────────────────────────────
# RunningNormalizer
# ─────────────────────────────────────────────────────────────────────────────

class RunningNormalizer:
    """Welford online z-score normaliser."""
    def __init__(self, size, clip=5.0):
        self.n = 0
        self.mean = np.zeros(size)
        self.M2 = np.ones(size)
        self.clip = clip

    def update_and_normalize(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
        if self.n < 2:
            return np.zeros_like(x)
        std = np.sqrt(self.M2 / (self.n - 1))
        std = np.where(std < 1e-8, 1.0, std)
        return np.clip((x - self.mean) / std, -self.clip, self.clip)


# ─────────────────────────────────────────────────────────────────────────────
# DeepNet — Two-hidden-layer Q-network with Adam optimizer
# FIX V8: Vectorized minibatch updates, adaptive LR, deeper capacity
# ─────────────────────────────────────────────────────────────────────────────

class DeepNet:
    """
    Architecture: input → ReLU → ReLU → output
    state_size → h1 → h2 → action_size
    Adam optimizer with vectorized minibatch updates.
    """
    def __init__(self, state_size, action_size, h1=64, h2=64, lr=3e-4):
        # He initialization for ReLU
        self.W1 = np.random.randn(state_size, h1) * np.sqrt(2.0 / state_size)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, action_size) * 0.01
        self.b3 = np.zeros(action_size)

        self.lr = lr
        self.beta1, self.beta2, self.eps_adam = 0.9, 0.999, 1e-8
        self.t = 0

        # Adam moments for each parameter
        params = ['W1','b1','W2','b2','W3','b3']
        self.m = {k: np.zeros_like(getattr(self, k)) for k in params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in params}

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        """x: (batch, state_size) or (state_size,)"""
        squeeze = x.ndim == 1
        if squeeze:
            x = x[np.newaxis, :]
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        out = a2 @ self.W3 + self.b3
        if squeeze:
            return out[0]
        return out  # (batch, action_size)

    def predict(self, x):
        return self.forward(x)

    def _adam_update(self, param_name, grad):
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * grad**2
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps_adam)

    def update_batch(self, states, actions, targets):
        """
        Vectorized minibatch update.
        states: (B, state_size), actions: (B,) int, targets: (B,) float
        """
        B = len(states)
        self.t += 1

        # ── Forward pass ──
        z1 = states @ self.W1 + self.b1           # (B, h1)
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2               # (B, h2)
        a2 = self._relu(z2)
        q_all = a2 @ self.W3 + self.b3            # (B, n_actions)

        # ── Loss: MSE only on the taken action ──
        q_taken = q_all[np.arange(B), actions]    # (B,)
        errors = targets - q_taken                 # (B,)

        # ── Backward pass ──
        # Output layer: gradient only on taken action
        dout = np.zeros_like(q_all)               # (B, n_actions)
        dout[np.arange(B), actions] = -2.0 * errors / B

        dW3 = a2.T @ dout                         # (h2, n_actions)
        db3 = dout.sum(axis=0)

        da2 = dout @ self.W3.T                    # (B, h2)
        dz2 = da2 * self._relu_grad(z2)           # (B, h2)

        dW2 = a1.T @ dz2                          # (h1, h2)
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T                     # (B, h1)
        dz1 = da1 * self._relu_grad(z1)           # (B, h1)

        dW1 = states.T @ dz1                      # (state, h1)
        db1 = dz1.sum(axis=0)

        # ── Adam updates ──
        for name, grad in [('W1', dW1), ('b1', db1), ('W2', dW2),
                            ('b2', db2), ('W3', dW3), ('b3', db3)]:
            update = self._adam_update(name, grad)
            setattr(self, name, getattr(self, name) - update)

        return float(np.mean(errors**2))

    def copy_weights_from(self, other):
        for attr in ['W1','b1','W2','b2','W3','b3']:
            setattr(self, attr, getattr(other, attr).copy())
        # Reset Adam moments after copy to avoid stale statistics
        for k in self.m:
            self.m[k] = np.zeros_like(self.m[k])
            self.v[k] = np.zeros_like(self.v[k])
        self.t = 0


# ─────────────────────────────────────────────────────────────────────────────
# NStepBuffer — collects transitions for n-step returns
# FIX V8: Better credit assignment across rebalancing decisions
# ─────────────────────────────────────────────────────────────────────────────

class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buf = deque()

    def add(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def flush(self, memory):
        """Convert buffered steps to n-step transitions and push to replay."""
        while len(self.buf) >= self.n or (self.buf and self.buf[-1][4]):  # last is done
            if not self.buf:
                break
            # Take the oldest transition
            s0, a0, _, _, _ = self.buf[0]

            # Compute discounted reward sum
            n_actual = min(self.n, len(self.buf))
            r_sum = 0.0
            for i, (_, _, ri, _, di) in enumerate(list(self.buf)[:n_actual]):
                r_sum += (self.gamma ** i) * ri
                if di:
                    # Terminal reached
                    s_n, done_n = self.buf[i][3], True
                    self.buf.popleft()
                    memory.append((s0, a0, r_sum, s_n, done_n))
                    return

            s_n = self.buf[n_actual - 1][3]
            done_n = self.buf[n_actual - 1][4]
            memory.append((s0, a0, r_sum, s_n, done_n))
            self.buf.popleft()

    def clear(self):
        self.buf.clear()


# ─────────────────────────────────────────────────────────────────────────────
# DQLAgent V8 — Double DQN, Adam, 2-hidden-layer net, n-step returns
# ─────────────────────────────────────────────────────────────────────────────

class DQLAgent:
    """
    V8 changes vs V7:
    - DeepNet (2 hidden ReLU layers) with Adam optimizer
    - Double DQN: online net selects action, target net evaluates value
    - N-step returns (n=5) for better credit assignment
    - Epsilon decays per episode, not per batch update
    - Reward no longer divided by vol_scale (fixes always-negative reward)
    - Range-center feature added to state (replaces redundant feature)
    - Hold penalty when out-of-range (encourages recentering)
    - Smaller action-repeat penalty (was over-penalising)
    - 12-feature state with step_fraction for time-awareness
    """

    ACTION_NAMES = [
        "Hold Position",
        "Increase Liquidity",
        "Decrease Liquidity",
        "Recenter Range Up",
        "Recenter Range Down",
    ]
    N_ACTIONS = 5

    MIN_LIQ = 0.5
    MAX_LIQ = 1.0

    def __init__(
        self,
        state_size=12,
        h1=64, h2=64,
        target_sync=200,
        memory_size=12_000,
        n_step=5,
        gamma=0.99,
    ):
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.97      # per-episode decay (not per step)
        self.batch_size = 128
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size

        self.online = DeepNet(state_size, self.N_ACTIONS, h1, h2, lr=3e-4)
        self.target = DeepNet(state_size, self.N_ACTIONS, h1, h2, lr=3e-4)
        self.target.copy_weights_from(self.online)

        self.target_sync = target_sync
        self.update_count = 0
        self.total_reward = 0.0
        self.reward_history = []
        self.loss_history = []
        self.normalizer = RunningNormalizer(state_size)

        self.prev_action = -1
        self.n_step_buf = NStepBuffer(n_step, gamma)

    def build_state(
        self,
        amm_hist, mkt_hist,
        sentiment,
        liq_ratio,
        fees_norm,
        volatility,
        dislocation,
        alpha_norm,
        momentum_sign_val,
        cooldown,
        range_center_log_ratio,   # V8: log(range_center / market_price)
        step_fraction,            # V8: t / T (time-awareness)
    ):
        amm_ret = (
            (amm_hist[-1] - amm_hist[-2]) / amm_hist[-2]
            if len(amm_hist) >= 2 and amm_hist[-2] != 0 else 0.0
        )
        mkt_ret = (
            (mkt_hist[-1] - mkt_hist[-2]) / mkt_hist[-2]
            if len(mkt_hist) >= 2 and mkt_hist[-2] != 0 else 0.0
        )
        raw = np.array([
            amm_ret,
            mkt_ret,
            sentiment,
            liq_ratio,
            fees_norm,
            volatility,
            dislocation,
            alpha_norm,
            momentum_sign_val,
            cooldown,
            range_center_log_ratio,   # V8 new feature
            step_fraction,            # V8 new feature
        ], dtype=float)
        return self.normalizer.update_and_normalize(raw)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.N_ACTIONS)
        return int(np.argmax(self.online.predict(state)))

    def remember(self, s, a, r, s2, done):
        """Store via n-step buffer."""
        self.n_step_buf.add(s, a, r, s2, done)
        self.n_step_buf.flush(self.memory)

    def end_episode(self):
        """Call at end of each episode to flush n-step buffer and decay epsilon."""
        self.n_step_buf.flush(self.memory)
        self.n_step_buf.clear()
        # Decay epsilon per episode (clean phase: explore → exploit)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states  = np.array([b[0] for b in batch], dtype=float)
        actions = np.array([b[1] for b in batch], dtype=int)
        rewards = np.array([b[2] for b in batch], dtype=float)
        nexts   = np.array([b[3] for b in batch], dtype=float)
        dones   = np.array([b[4] for b in batch], dtype=float)

        # Double DQN: online selects action, target evaluates value
        # FIX V8: reduces overestimation bias vs standard DQN
        online_q_next = self.online.forward(nexts)              # (B, A)
        best_actions  = np.argmax(online_q_next, axis=1)        # (B,)
        target_q_next = self.target.forward(nexts)              # (B, A)
        target_vals   = target_q_next[np.arange(len(batch)), best_actions]  # (B,)

        targets = rewards + (1.0 - dones) * (self.gamma ** 5) * target_vals
        # Note: gamma^n because rewards are already n-step discounted sums

        loss = self.online.update_batch(states, actions, targets)
        self.loss_history.append(loss)

        self.update_count += 1
        if self.update_count % self.target_sync == 0:
            self.target.copy_weights_from(self.online)

    def calculate_reward(
        self,
        alpha_change_dollars,   # V8: raw dollar change in (AI_value - static_value)
        fee_dollars,            # raw dollar fees this step
        cost_dollars,           # raw dollar cost this step
        action,
        active_fraction_val,    # how much capital is earning fees
        initial_value,          # to normalize
    ):
        """
        V8 reward: calibrated dollar-scale signals, no vol_scale division.

        Key insight: alpha_change_dollars is on the order of $0.1-$5 per step.
        Fee per step is ~$0.01-$0.5. Rebalance cost is $0.15-$0.30.
        All penalties/bonuses are on the same scale → coherent gradient.

        Scale to centidollars to keep gradient magnitudes reasonable.
        """
        scale = 0.01 * initial_value   # ~$20 for $2000 initial portfolio

        # Primary: beat static LP (can be positive or negative)
        reward = 3.0 * alpha_change_dollars / scale

        # Fee bonus: reward actual LP income proportionally
        reward += 1.0 * fee_dollars / scale

        # Cost penalty: gas/rebalancing cuts into profit
        reward -= 1.5 * cost_dollars / scale

        # In-range bonus: concentrated liquidity that's actually earning
        # FIX V8: encourages the agent to recenter rather than holding stale ranges
        if active_fraction_val > 0.8:
            reward += 0.05

        # Out-of-range + Hold penalty: passivity when out of range costs fees
        # FIX V8: this breaks the "always Hold" local minimum
        if action == 0 and active_fraction_val < 0.3:
            reward -= 0.10

        # Action-repeat penalty (smaller than V7 to avoid over-discouraging action)
        if action != 0 and action == self.prev_action:
            reward -= 0.10   # was 0.30 in V7

        self.prev_action = action
        self.total_reward += reward
        self.reward_history.append(reward)
        return reward


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pool_share(env, lp):
    return min(
        lp.deposit_a / max(env.initial_token_a, 1e-9),
        lp.deposit_b / max(env.initial_token_b, 1e-9),
    )

def rolling_vol(series, window=20):
    if len(series) < 2:
        return 0.01
    recent = series[-min(window, len(series)):]
    rets = [(recent[i]-recent[i-1])/recent[i-1]
            for i in range(1, len(recent)) if recent[i-1] > 0]
    return float(np.std(rets)) if rets else 0.01

def active_fraction(mkt_price, range_center, half_width, liq_ratio):
    """Fraction of capital earning fees. Hard cap at 1.0."""
    if mkt_price <= 0 or range_center <= 0:
        return 0.0
    lo = range_center * (1 - half_width)
    hi = range_center * (1 + half_width)
    if lo <= mkt_price <= hi:
        return float(np.clip(liq_ratio, 0.0, 1.0))
    dist = abs(np.log(max(mkt_price / range_center, 1e-9)))
    decay = np.exp(-8.0 * dist)
    return float(np.clip(liq_ratio * (0.1 + 0.4 * decay), 0.0, 1.0))

def momentum_sign(price_hist, window=20):
    if len(price_hist) < window + 1:
        return 0.0
    ret = (price_hist[-1] - price_hist[-window]) / max(price_hist[-window], 1e-9)
    return float(np.sign(ret))

def generate_price_path(num_steps, seed):
    """Synthetic market path — deterministic from seed."""
    set_seeds(seed)
    env = AMMEnvironment()
    sa = SentimentAnalyzer()
    path = []
    for _ in range(num_steps):
        s = sa.get_sentiment(env.price_history)
        env.simulate_market_activity(s)
        path.append(env.market_price)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Static strategy
# ─────────────────────────────────────────────────────────────────────────────

def run_static(price_feed, num_steps):
    set_seeds(0)
    env = AMMEnvironment(price_feed=list(price_feed))
    sa = SentimentAnalyzer()
    lp = LiquidityProvider(1000, 1000, env.get_price())
    share = pool_share(env, lp)
    values, il_list, disloc_list = [], [], []

    for _ in range(num_steps):
        s = sa.get_sentiment(env.price_history)
        info = env.simulate_market_activity(s)
        lp.add_fees(info["fees_collected_b"] * share)
        p = env.get_price()
        values.append(lp.calculate_position_value(p, active_exposure=1.0))
        il_list.append(lp.calculate_impermanent_loss(p))
        disloc_list.append(info["price_dislocation"])

    return {
        "values": values,
        "il": il_list,
        "fees": lp.fees_earned,
        "disloc": disloc_list,
        "price_hist": env.price_history,
        "mkt_hist": env.market_price_history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI strategy
# ─────────────────────────────────────────────────────────────────────────────

RANGE_HALF_WIDTH = 0.12

def run_ai(price_feed, num_steps, agent, static_values, static_il):
    set_seeds(1)
    env = AMMEnvironment(price_feed=list(price_feed))
    sa = SentimentAnalyzer()
    lp = LiquidityProvider(1000, 1000, env.get_price())
    share = pool_share(env, lp)

    liq_ratio = 0.75
    range_center = env.market_price
    cum_cost = 0.0
    cooldown = 0

    values, il_list, active_list, actions_list = [], [], [], []
    prev_ai_value = lp.initial_value
    prev_fees = 0.0

    for step in range(num_steps):
        s = sa.get_sentiment(env.price_history)
        info = env.simulate_market_activity(s)
        mkt = info["market_price"]
        p = env.get_price()

        il_raw = lp.calculate_impermanent_loss(p)
        vol = rolling_vol(env.market_price_history, 20)
        disloc = info["price_dislocation"]
        mom = momentum_sign(env.market_price_history, 20)

        active = active_fraction(mkt, range_center, RANGE_HALF_WIDTH, liq_ratio)
        il_now = il_raw * active

        static_val_now = static_values[step] if step < len(static_values) else static_values[-1]
        current_ai_val = lp.calculate_position_value(p, active, cum_cost)
        alpha_norm = (current_ai_val - static_val_now) / max(lp.initial_value, 1.0)

        # V8: new features
        range_log_ratio = float(np.clip(
            np.log(max(range_center, 1e-9) / max(mkt, 1e-9)),
            -2.0, 2.0
        ))
        step_frac = step / max(num_steps - 1, 1)

        state = agent.build_state(
            env.price_history, env.market_price_history,
            s, liq_ratio,
            lp.fees_earned / max(lp.initial_value, 1.0),
            vol, disloc, alpha_norm, mom,
            min(cooldown / 50.0, 1.0),
            range_log_ratio,
            step_frac,
        )

        action = agent.act(state)
        actions_list.append(action)

        # Apply action
        cost = 0.0
        if action == 0:
            cooldown += 1
        elif action == 1:
            liq_ratio = min(liq_ratio + 0.1, DQLAgent.MAX_LIQ)
            cost = env.rebalance_gas_cost_b
            cooldown = 0
        elif action == 2:
            liq_ratio = max(liq_ratio - 0.1, DQLAgent.MIN_LIQ)
            cost = env.rebalance_gas_cost_b
            cooldown = 0
        elif action == 3:
            range_center *= 1.02
            cost = env.rebalance_gas_cost_b * 2.0
            cooldown = 0
        elif action == 4:
            range_center *= 0.98
            cost = env.rebalance_gas_cost_b * 2.0
            cooldown = 0

        # Passive drift: center slowly follows market
        range_center = 0.97 * range_center + 0.03 * mkt
        cum_cost += cost

        # Recompute after action
        active = active_fraction(mkt, range_center, RANGE_HALF_WIDTH, liq_ratio)
        step_fee = info["fees_collected_b"] * share * active
        lp.add_fees(step_fee)

        new_ai_val = lp.calculate_position_value(p, active, cum_cost)

        # V8 Reward: raw dollar alpha change (not ratio)
        static_prev = static_values[step-1] if step > 0 else lp.initial_value
        alpha_change_dollars = (
            (new_ai_val - static_val_now) - (prev_ai_value - static_prev)
        )

        reward = agent.calculate_reward(
            alpha_change_dollars=alpha_change_dollars,
            fee_dollars=step_fee,
            cost_dollars=cost,
            action=action,
            active_fraction_val=active,
            initial_value=lp.initial_value,
        )

        # Build next state for replay
        range_log_ratio2 = float(np.clip(
            np.log(max(range_center, 1e-9) / max(mkt, 1e-9)),
            -2.0, 2.0
        ))
        next_state = agent.build_state(
            env.price_history, env.market_price_history,
            s, liq_ratio,
            lp.fees_earned / max(lp.initial_value, 1.0),
            vol, disloc,
            (new_ai_val - static_val_now) / max(lp.initial_value, 1.0),
            mom,
            min(cooldown / 50.0, 1.0),
            range_log_ratio2,
            (step + 1) / max(num_steps - 1, 1),
        )

        done = (step == num_steps - 1)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        prev_ai_value = new_ai_val
        prev_fees = lp.fees_earned
        values.append(new_ai_val)
        il_list.append(il_now)
        active_list.append(active)

    agent.end_episode()  # flush n-step buffer, decay epsilon

    return {
        "values": values,
        "il": il_list,
        "fees": lp.fees_earned,
        "active": active_list,
        "actions": actions_list,
        "disloc": env.dislocation_history[1:],
        "price_hist": env.price_history,
        "mkt_hist": env.market_price_history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulation orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AIAMMSimulation:
    def __init__(self, num_steps=1000, num_episodes=200, seed=42):
        self.num_steps = num_steps
        self.num_episodes = num_episodes
        self.base_seed = seed
        self.agent = DQLAgent()

    def run_comparison(self):
        print("=" * 72)
        print("  AIAMM V8 — Double DQN, Adam, n-step returns, calibrated reward")
        print("=" * 72)
        print(f"  Episodes: {self.num_episodes}  |  Steps/ep: {self.num_steps}  |  Seed: {self.base_seed}")
        print("-" * 72)

        s_vals_ep, ai_vals_ep = [], []
        s_il_ep, ai_il_ep = [], []
        s_fees_ep, ai_fees_ep = [], []
        alpha_ep = []

        last_static = last_ai = None

        for ep in range(self.num_episodes):
            ep_seed = self.base_seed + ep * 997

            path = generate_price_path(self.num_steps, ep_seed)
            static = run_static(path, self.num_steps)
            ai = run_ai(path, self.num_steps, self.agent, static["values"], static["il"])

            alpha = ai["values"][-1] - static["values"][-1]

            s_vals_ep.append(static["values"][-1])
            ai_vals_ep.append(ai["values"][-1])
            s_il_ep.append(np.mean(static["il"]) * 100)
            ai_il_ep.append(np.mean(ai["il"]) * 100)
            s_fees_ep.append(static["fees"])
            ai_fees_ep.append(ai["fees"])
            alpha_ep.append(alpha)

            last_static, last_ai = static, ai

            if (ep + 1) % 20 == 0:
                print(
                    f"  Ep {ep+1:>3}/{self.num_episodes}"
                    f"  α={np.mean(alpha_ep):+7.2f}"
                    f"  win={np.mean(np.array(alpha_ep)>0)*100:4.0f}%"
                    f"  ε={self.agent.epsilon:.3f}"
                    f"  mem={len(self.agent.memory)}"
                    f"  updates={self.agent.update_count}"
                )

        alpha_arr = np.array(alpha_ep)

        print()
        print("=" * 72)
        print("  RESULTS (means over all episodes, same price path per episode)")
        print("=" * 72)
        print(f"{'Metric':<40} {'Static':>10} {'AI':>10} {'Δ':>10}")
        print("-" * 75)

        sv = np.mean(s_vals_ep);  av = np.mean(ai_vals_ep)
        print(f"{'Avg Final Value ($)':<40} {sv:>10.2f} {av:>10.2f} {av-sv:>+10.2f}")

        si = np.mean(s_il_ep);   ai_i = np.mean(ai_il_ep)
        print(f"{'Avg IL (%)':<40} {si:>9.3f}% {ai_i:>9.3f}% {ai_i-si:>+9.3f}%")

        sf = np.mean(s_fees_ep); af = np.mean(ai_fees_ep)
        print(f"{'Avg Fees ($)':<40} {sf:>10.2f} {af:>10.2f} {af-sf:>+10.2f}")

        alpha_mean = np.mean(alpha_arr)
        alpha_std  = np.std(alpha_arr)
        sharpe = alpha_mean / (alpha_std + 1e-9)
        win_rate = np.mean(alpha_arr > 0) * 100
        print(f"{'Alpha mean ($)':<40} {'—':>10} {alpha_mean:>+10.2f}")
        print(f"{'Alpha std ($)':<40} {'—':>10} {alpha_std:>10.2f}")
        print(f"{'Alpha Sharpe':<40} {'—':>10} {sharpe:>+10.3f}")
        print(f"{'Win rate (%)':<40} {'—':>10} {win_rate:>9.1f}%")
        print()

        print("  Action distribution (last episode):")
        acts = last_ai["actions"]
        for i, name in enumerate(DQLAgent.ACTION_NAMES):
            c = acts.count(i)
            pct = c / len(acts) * 100
            bar = "█" * max(1, int(pct / 2))
            print(f"    {name:<26} {c:>5} ({pct:>5.1f}%) {bar}")

        print(f"\n  Final ε={self.agent.epsilon:.4f}  "
              f"total_reward={self.agent.total_reward:.2f}  "
              f"updates={self.agent.update_count}")

        self._plot(last_static, last_ai, alpha_arr, s_vals_ep, ai_vals_ep)

        return {
            "static_value": sv, "ai_value": av,
            "alpha_mean": alpha_mean, "alpha_sharpe": sharpe,
            "win_rate": win_rate,
            "static_il": si, "ai_il": ai_i,
            "static_fees": sf, "ai_fees": af,
        }

    def _plot(self, static, ai, alpha_arr, s_vals_ep, ai_vals_ep):
        fig, axes = plt.subplots(2, 3, figsize=(19, 11))
        fig.suptitle(
            "AIAMM V8: Double DQN + Adam — Controlled Same-Path Comparison",
            fontsize=15, fontweight="bold",
        )
        steps = range(self.num_steps)

        # 1) Portfolio value
        ax = axes[0, 0]
        ax.plot(steps, static["values"], label="Static LP", color="steelblue", alpha=0.85)
        ax.plot(steps, ai["values"],     label="AI Agent",  color="seagreen",  alpha=0.85)
        ax.axhline(static["values"][0], color="gray", linestyle=":", lw=0.8)
        ax.set_title("Portfolio Value (same price path)")
        ax.set_xlabel("Step"); ax.set_ylabel("$")
        ax.legend(); ax.grid(True, alpha=0.3)

        # 2) Per-step alpha
        ax = axes[0, 1]
        a_series = [ai["values"][i] - static["values"][i] for i in range(self.num_steps)]
        ax.plot(steps, a_series, color="darkorange", lw=0.9, alpha=0.9)
        ax.axhline(0, color="black", lw=0.8)
        ax.fill_between(steps, a_series, 0,
                         where=[v > 0 for v in a_series], color="green", alpha=0.25, label="AI ahead")
        ax.fill_between(steps, a_series, 0,
                         where=[v <= 0 for v in a_series], color="red", alpha=0.2, label="Static ahead")
        ax.set_title("Alpha: AI − Static (per step)")
        ax.set_xlabel("Step"); ax.set_ylabel("Alpha ($)")
        ax.legend(); ax.grid(True, alpha=0.3)

        # 3) IL comparison
        ax = axes[0, 2]
        ax.plot(steps, [v*100 for v in static["il"]], label="Static IL", color="tomato", alpha=0.8)
        ax.plot(steps, [v*100 for v in ai["il"]],     label="AI IL",     color="darkorange", alpha=0.8)
        ax.set_title("Impermanent Loss")
        ax.set_xlabel("Step"); ax.set_ylabel("IL (%)")
        ax.legend(); ax.grid(True, alpha=0.3)

        # 4) Action distribution
        ax = axes[1, 0]
        counts = [ai["actions"].count(i) for i in range(DQLAgent.N_ACTIONS)]
        colors = ["#888", "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]
        bars = ax.bar(range(DQLAgent.N_ACTIONS), counts, color=colors, alpha=0.8)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(c), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(DQLAgent.N_ACTIONS))
        ax.set_xticklabels(["Hold","Inc","Dec","Up","Down"], rotation=0)
        ax.set_title("Action Distribution (last episode)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3, axis="y")

        # 5) Cumulative reward + rolling mean loss
        ax = axes[1, 1]
        if self.agent.reward_history:
            cum_r = np.cumsum(self.agent.reward_history)
            ax.plot(cum_r, color="seagreen", alpha=0.85, lw=0.8, label="Cumulative Reward")
            ax.axhline(0, color="black", lw=0.8)
        ax2 = ax.twinx()
        if self.agent.loss_history:
            # Rolling average of loss
            loss_arr = np.array(self.agent.loss_history)
            window = max(1, len(loss_arr)//50)
            kernel = np.ones(window)/window
            smoothed = np.convolve(loss_arr, kernel, mode='valid')
            ax2.plot(smoothed, color="salmon", alpha=0.6, lw=0.7, label="MSE Loss (smoothed)")
            ax2.set_ylabel("Loss", color="salmon", fontsize=8)
            ax2.tick_params(axis='y', labelcolor='salmon', labelsize=7)
        ax.set_title("Agent Reward & Training Loss")
        ax.set_xlabel("Training step"); ax.set_ylabel("Σ reward")
        ax.grid(True, alpha=0.3)

        # 6) Alpha distribution across episodes + rolling win rate
        ax = axes[1, 2]
        alpha_arr_clean = np.array(alpha_arr)
        ax.hist(alpha_arr_clean, bins=30, color="darkorange", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", lw=1.2, ls="--", label="Break-even")
        ax.axvline(np.mean(alpha_arr_clean), color="seagreen", lw=1.5,
                   label=f"Mean α={np.mean(alpha_arr_clean):+.2f}")
        win = np.mean(alpha_arr_clean > 0) * 100
        ax.set_title(f"Alpha Distribution — Win Rate {win:.0f}%")
        ax.set_xlabel("Alpha ($)"); ax.set_ylabel("Episodes")
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("aiamm_v8_results.png", dpi=150, bbox_inches="tight")
        print("\nSaved: aiamm_v8_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔" + "═"*68 + "╗")
    print("║  AIAMM V8 — Double DQN + Adam + N-Step Returns                    ║")
    print("║  · Calibrated dollar-scale reward (fixes always-negative reward)  ║")
    print("║  · Two ReLU hidden layers (64→64) with Adam optimizer             ║")
    print("║  · Double DQN: reduces overestimation bias                        ║")
    print("║  · N-step returns (n=5): longer credit assignment (γ=0.99)        ║")
    print("║  · Per-episode epsilon decay: clean explore→exploit phases        ║")
    print("║  · Range-center log-ratio + step_fraction in state (12 features)  ║")
    print("║  · In-range bonus + out-of-range hold penalty                     ║")
    print("╚" + "═"*68 + "╝")
    print()
    sim = AIAMMSimulation(num_steps=1000, num_episodes=200, seed=42)
    results = sim.run_comparison()
    print("\nFinal:", results)


if __name__ == "__main__":
    main()
