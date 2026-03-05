import json, uuid

def cid():
    return str(uuid.uuid4())

MD_TITLE = """# 1AMMV7.5 Detailed Walkthrough

This notebook is a complete rewrite of `1AMMV5.5.ipynb` with ten structural
improvements to the RL training loop, market simulation, and evaluation metrics.

## Architecture (same 6-section layout as V5.5)

### Section 1 - AMMEnvironment
Constant-product AMM with external reference market, organic order flow, and
arbitrage alignment.
**V7.5 fix [10]:** `_simulate_order_flow` anchors `base_notional_b` to
`initial_token_b` (not live reserves), preventing order-size feedback loops.

### Section 2 - LiquidityProvider
LP position tracking with IL, fee, and cost accounting. No structural changes;
pool-share fix applied in `AIAMMSimulation._pool_share`.

### Section 3 - SentimentAnalyzer
EMA-smoothed momentum proxy for market sentiment. Unchanged from V5.5.

### Section 4 - MLPQNetwork + RunningNormalizer + DQLAgent
**V7.5 major upgrades:**
- **MLPQNetwork** - 2-layer MLP (state->64->32->actions) with He init,
  ReLU activations, and a fully vectorised NumPy batch backprop pass.
  Replaces the single linear layer from V5.5.
- **Target network** - a second MLPQNetwork whose weights are hard-copied
  from the online network every `target_update_freq` train steps.
- **RunningNormalizer** - EMA normalises each raw reward to zero-mean
  unit-variance before storage, keeping Q-value scales stable.
- **7-action space** - adds Widen Range and Narrow Range actions.
- **State vector grows from 8 to 9 features** - `range_half_width` added.
- **Replay buffer maxlen=50000** (was 2000).

### Section 5 - AIAMMSimulation
**V7.5 fixes:**
- Shared market path per episode for valid controlled comparison.
- Persistent DQLAgent created once, passed into every episode call.
- Geometric-mean pool share estimate.
- Dynamic `range_half_width` tracks volatility; agent can adjust via actions.
- Sharpe ratio and max drawdown added to summary table.

### Section 6 - main()
Entry-point unchanged in structure; banner updated for V7.5.
"""

MD_CHANGES = """## What Was Changed in V7.5 vs V5.5

| # | Area | V5.5 | V7.5 |
|---|------|-------|-------|
| 1 | Q-function | Linear: s @ W + b | 2-layer MLP (64-32) with ReLU |
| 2 | Target network | None | Hard-sync every 50 train steps |
| 3 | Episode comparison | Independent stochastic envs | Shared pre-generated price path |
| 4 | Reward scale | Raw mixed-scale reward | Z-score normalised via EMA |
| 5 | Agent lifetime | New agent per episode | Single agent across all 200 eps |
| 6 | Pool share | min(a/A, b/B) | Geometric mean sqrt(a/A * b/B) |
| 7 | Range width | Fixed 0.12, 5 actions | Dynamic vol-tracking + 7 actions |
| 8 | Metrics | Value, IL, fees, gap | + Sharpe ratio + max drawdown |
| 9 | Replay buffer | maxlen=2000 | maxlen=50000 |
| 10 | Order-flow base | self.token_b * 0.0025 | self.initial_token_b * 0.0025 |
"""

CODE_MAIN = """# %% [markdown]
# ## Section 0: Imports & Setup

# %%
import numpy as np
import random
from collections import deque
import matplotlib
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILE_LABEL = globals().get("__file__", "notebook")
print("RUNNING FILE", FILE_LABEL)
print("Section 0 done")

# %% [markdown]
# ## Section 1: AMM Environment

# %%


class AMMEnvironment:
    \"\"\"
    Constant-product AMM with external reference market, organic order flow,
    and arbitrage alignment.

    V7.5 fix [10]: _simulate_order_flow anchors base_notional_b to
    initial_token_b so order sizes do not self-amplify as reserves drift.
    \"\"\"

    def __init__(
        self,
        initial_token_a=10000,
        initial_token_b=10000,
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
        if self.token_a == 0:
            return float("inf")
        return self.token_b / self.token_a

    def _next_market_price(self, sentiment_signal=0.0):
        if self.price_feed:
            if self.price_feed_index < len(self.price_feed):
                next_price = self.price_feed[self.price_feed_index]
                self.price_feed_index += 1
                if next_price > 0:
                    self.market_price = float(next_price)
                    return self.market_price

        if self.step_count > 0 and self.step_count % 120 == 0:
            self.market_volatility = float(np.random.choice([0.008, 0.015, 0.03]))
            self.market_drift = np.random.normal(0, 0.0015)

        shock = np.random.normal(
            self.market_drift + sentiment_signal * 0.003,
            self.market_volatility,
        )
        self.market_price = max(1e-6, self.market_price * np.exp(shock))
        return self.market_price

    def _simulate_order_flow(self, sentiment_signal=0.0):
        # V7.5 fix [10]: anchor to initial pool depth to decouple order
        # size from reserve drift.
        base_notional_b = self.initial_token_b * 0.0025
        notional_b = base_notional_b * np.random.lognormal(mean=0.0, sigma=0.65)
        buy_prob = np.clip(0.5 + 0.35 * np.tanh(sentiment_signal), 0.05, 0.95)

        if np.random.rand() < buy_prob:
            _, volume_b, fee_b = self.swap_b_for_a(notional_b)
        else:
            amount_a = notional_b / max(self.market_price, 1e-9)
            _, volume_b, fee_b = self.swap_a_for_b(amount_a)
        return volume_b, fee_b

    def _run_arbitrage(self):
        amm_price = self.get_price()
        market_price = self.market_price
        if not np.isfinite(amm_price) or market_price <= 0:
            return 0.0, 0.0

        dislocation = (amm_price - market_price) / market_price
        if abs(dislocation) * 10000 < self.arbitrage_band_bps:
            return 0.0, 0.0

        target_price = amm_price + 0.8 * (market_price - amm_price)
        pool_k = self.token_a * self.token_b

        if target_price > amm_price:
            delta_b_effective = max(0.0, np.sqrt(pool_k * target_price) - self.token_b)
            amount_b_in = delta_b_effective / max(1 - self.fee, 1e-9)
            if amount_b_in < self.arb_gas_cost_b * 8:
                return 0.0, 0.0
            _, volume_b, fee_b = self.swap_b_for_a(amount_b_in)
        else:
            delta_a_effective = max(
                0.0,
                np.sqrt(pool_k / max(target_price, 1e-9)) - self.token_a,
            )
            amount_a_in = delta_a_effective / max(1 - self.fee, 1e-9)
            if amount_a_in * max(self.market_price, 1e-9) < self.arb_gas_cost_b * 8:
                return 0.0, 0.0
            _, volume_b, fee_b = self.swap_a_for_b(amount_a_in)

        return volume_b, fee_b

    def simulate_market_activity(self, sentiment_signal=0.0):
        self.step_count += 1
        self._next_market_price(sentiment_signal)

        flow_volume_b, flow_fees_b = self._simulate_order_flow(sentiment_signal)
        arb_volume_b, arb_fees_b = self._run_arbitrage()

        step_volume_b = flow_volume_b + arb_volume_b
        step_fees_b = flow_fees_b + arb_fees_b
        amm_price = self.get_price()
        dislocation = (amm_price - self.market_price) / max(self.market_price, 1e-9)

        self.price_history.append(amm_price)
        self.market_price_history.append(self.market_price)
        self.volume_history.append(step_volume_b)
        self.fee_history.append(step_fees_b)
        self.arb_volume_history.append(arb_volume_b)
        self.dislocation_history.append(dislocation)

        return {
            "market_price": self.market_price,
            "amm_price": amm_price,
            "price_dislocation": dislocation,
            "volume_b": step_volume_b,
            "fees_collected_b": step_fees_b,
            "arb_volume_b": arb_volume_b,
        }

    def swap_a_for_b(self, amount_a):
        if amount_a <= 0:
            return 0.0, 0.0, 0.0
        amount_a = min(amount_a, self.token_a * self.max_trade_fraction)
        amount_a_with_fee = amount_a * (1 - self.fee)
        amount_b_out = self.token_b * amount_a_with_fee / (
            self.token_a + amount_a_with_fee
        )
        fee_a = amount_a - amount_a_with_fee
        self.token_a += amount_a
        self.token_b -= amount_b_out
        traded_notional_b = amount_a * max(self.market_price, 1e-9)
        fee_value_b = fee_a * max(self.market_price, 1e-9)
        return amount_b_out, traded_notional_b, fee_value_b

    def swap_b_for_a(self, amount_b):
        if amount_b <= 0:
            return 0.0, 0.0, 0.0
        amount_b = min(amount_b, self.token_b * self.max_trade_fraction)
        amount_b_with_fee = amount_b * (1 - self.fee)
        amount_a_out = self.token_a * amount_b_with_fee / (
            self.token_b + amount_b_with_fee
        )
        fee_b = amount_b - amount_b_with_fee
        self.token_b += amount_b
        self.token_a -= amount_a_out
        traded_notional_b = amount_b
        fee_value_b = fee_b
        return amount_a_out, traded_notional_b, fee_value_b

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


print("Section 1 done")

# %% [markdown]
# ## Section 2: Liquidity Provider

# %%


class LiquidityProvider:
    \"\"\"
    LP position with IL, fee, and execution-cost-aware valuation.
    No structural changes in V7.5; pool-share fix is in AIAMMSimulation.
    \"\"\"

    def __init__(self, deposit_a, deposit_b, entry_price):
        self.deposit_a = deposit_a
        self.deposit_b = deposit_b
        self.entry_price = entry_price
        self.initial_value = deposit_a * entry_price + deposit_b

        self.fees_earned = 0.0
        self.costs_paid = 0.0
        self.total_deposited_value = self.initial_value

    def calculate_impermanent_loss(self, current_price):
        if self.entry_price == 0:
            return 0
        price_ratio = current_price / self.entry_price
        if price_ratio <= 0:
            return 0
        return 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1

    def calculate_position_value(
        self, current_price, active_exposure=1.0, trading_costs=0.0
    ):
        active_exposure = float(np.clip(active_exposure, 0.0, 1.5))
        il = self.calculate_impermanent_loss(current_price) * active_exposure
        hodl_value = self.deposit_a * current_price + self.deposit_b
        return hodl_value * (1 + il) + self.fees_earned - trading_costs

    def calculate_current_value(self, current_price, pool_share=1.0):
        return self.calculate_position_value(current_price)

    def add_fees(self, fee_amount):
        self.fees_earned += fee_amount

    def add_cost(self, cost_amount):
        self.costs_paid += cost_amount


print("Section 2 done")

# %% [markdown]
# ## Section 3: Sentiment Analysis

# %%


class SentimentAnalyzer:
    \"\"\"
    EMA-smoothed momentum proxy for market sentiment. Unchanged from V5.5.
    \"\"\"

    def __init__(self, sentiment_feed=None):
        self.sentiment_history = []
        self.base_sentiment = 0.0
        self.sentiment_feed = sentiment_feed
        self.sentiment_feed_index = 0

    def get_sentiment(self, price_history, window=10):
        if self.sentiment_feed:
            if self.sentiment_feed_index < len(self.sentiment_feed):
                sentiment = self.sentiment_feed[self.sentiment_feed_index]
                self.sentiment_feed_index += 1
            else:
                sentiment = 0.0
        elif len(price_history) < 2:
            sentiment = np.random.normal(0, 0.1)
        else:
            recent = price_history[-min(window, len(price_history)):]
            returns = [
                (recent[i] - recent[i - 1]) / recent[i - 1]
                for i in range(1, len(recent))
                if recent[i - 1] != 0
            ]
            if returns:
                momentum = np.mean(returns)
                volatility = np.std(returns) if len(returns) > 1 else 0.01
                nlp_noise = np.random.normal(0, 0.2)
                sentiment = (
                    np.tanh(momentum / max(volatility, 0.001) * 5) + nlp_noise
                )
            else:
                sentiment = np.random.normal(0, 0.1)

        self.base_sentiment = 0.7 * self.base_sentiment + 0.3 * sentiment
        final_sentiment = np.clip(self.base_sentiment, -1, 1)
        self.sentiment_history.append(final_sentiment)
        return final_sentiment


print("Section 3 done")

# %% [markdown]
# ## Section 4: MLP Q-Network + DQL Agent
#
# V7.5 major upgrades: MLPQNetwork [fix 1], target network [fix 2],
# RunningNormalizer [fix 4], 7-action space [fix 7], state_size=9,
# replay buffer 50k [fix 9].

# %%


class MLPQNetwork:
    \"\"\"
    2-layer MLP Q-value approximator in NumPy.
    Architecture: input_size -> 64 -> 32 -> action_size.
    Uses He initialisation, ReLU activations, and vectorised batch backprop.
    \"\"\"

    def __init__(self, input_size, action_size, hidden1=64, hidden2=32, lr=0.001):
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, action_size) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(action_size)

    def forward(self, x):
        \"\"\"Forward pass. x may be (input_size,) or (batch, input_size).\"\"\"
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]
        Z1 = x @ self.W1 + self.b1
        A1 = np.maximum(0.0, Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = np.maximum(0.0, Z2)
        Q = A2 @ self.W3 + self.b3
        if single:
            return Q[0], (x[0], Z1[0], A1[0], Z2[0], A2[0])
        return Q, (x, Z1, A1, Z2, A2)

    def predict(self, x):
        \"\"\"Return Q-values (single vector or batch matrix).\"\"\"
        q, _ = self.forward(x)
        return q

    def update_batch(self, states, actions, targets):
        \"\"\"
        Vectorised TD update for a mini-batch.
        states  : (batch, input_size)
        actions : (batch,) int
        targets : (batch,) float
        \"\"\"
        batch_size = states.shape[0]

        # Forward pass
        Z1 = states @ self.W1 + self.b1          # (B, h1)
        A1 = np.maximum(0.0, Z1)
        Z2 = A1 @ self.W2 + self.b2              # (B, h2)
        A2 = np.maximum(0.0, Z2)
        Q = A2 @ self.W3 + self.b3               # (B, out)

        # TD error on taken actions only, normalised by batch size
        dQ = np.zeros_like(Q)
        for i, a in enumerate(actions):
            dQ[i, a] = (Q[i, a] - targets[i]) / batch_size

        # Backprop layer 3
        dW3 = A2.T @ dQ                          # (h2, out)
        db3 = dQ.sum(axis=0)
        dA2 = dQ @ self.W3.T                     # (B, h2)

        # ReLU gate 2
        dZ2 = dA2 * (Z2 > 0)
        dW2 = A1.T @ dZ2                         # (h1, h2)
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T                    # (B, h1)

        # ReLU gate 1
        dZ1 = dA1 * (Z1 > 0)
        dW1 = states.T @ dZ1                     # (in, h1)
        db1 = dZ1.sum(axis=0)

        # SGD weight update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def copy_weights_from(self, source):
        \"\"\"Hard-copy weights from another MLPQNetwork (target-net sync).\"\"\"
        self.W1 = source.W1.copy()
        self.b1 = source.b1.copy()
        self.W2 = source.W2.copy()
        self.b2 = source.b2.copy()
        self.W3 = source.W3.copy()
        self.b3 = source.b3.copy()


class RunningNormalizer:
    \"\"\"
    Online reward normaliser using exponential moving mean/variance.
    Keeps Q-value scales stable across long multi-episode runs.
    \"\"\"

    def __init__(self, alpha=0.005):
        self.mean = 0.0
        self.var = 1.0
        self.alpha = alpha
        self.count = 0

    def normalize(self, x):
        self.count += 1
        if self.count < 50:
            return float(x)
        delta = x - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * self.var + self.alpha * delta ** 2
        std = max(np.sqrt(self.var), 1e-6)
        return float(np.clip((x - self.mean) / std, -10.0, 10.0))


class DQLAgent:
    \"\"\"
    Deep Q-Learning agent for dynamic AMM liquidity management.

    V7.5 upgrades:
    [1] MLPQNetwork replaces linear approximator
    [2] Separate target network, synced every target_update_freq steps
    [4] Reward normalisation via RunningNormalizer
    [5] Persistent across episodes (created once in run_comparison)
    [7] 7-action space: adds Widen Range and Narrow Range
        state_size=9 (range_half_width added as observable feature)
    [9] Replay buffer maxlen=50000
    \"\"\"

    def __init__(self, state_size=9, action_size=7, target_update_freq=50):
        self.state_size = state_size
        self.action_size = action_size
        self.target_update_freq = target_update_freq
        self.train_step = 0

        self.action_names = [
            "Hold Position",
            "Increase Liquidity",
            "Decrease Liquidity",
            "Recenter Range Up",
            "Recenter Range Down",
            "Widen Range",
            "Narrow Range",
        ]

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 32

        # [9] Larger buffer
        self.memory = deque(maxlen=50000)

        # [1] + [2] Online network + target network
        self.q_network = MLPQNetwork(state_size, action_size)
        self.target_network = MLPQNetwork(state_size, action_size)
        self.target_network.copy_weights_from(self.q_network)

        # [4] Reward normaliser
        self.reward_normalizer = RunningNormalizer()

        self.total_reward = 0.0
        self.reward_history = []

    def get_state(
        self,
        amm_price_history,
        market_price_history,
        sentiment,
        liquidity_ratio,
        il_current,
        fees_earned,
        volatility,
        range_half_width,
    ):
        \"\"\"9-feature state vector (adds range_half_width vs V5.5's 8).\"\"\"
        if len(amm_price_history) < 2:
            amm_return = 0.0
        else:
            prev = amm_price_history[-2]
            amm_return = (amm_price_history[-1] - prev) / prev if prev != 0 else 0.0

        if len(market_price_history) < 2:
            market_return = 0.0
        else:
            prev = market_price_history[-2]
            market_return = (
                (market_price_history[-1] - prev) / prev if prev != 0 else 0.0
            )

        current_market = market_price_history[-1] if market_price_history else 1.0
        current_amm = amm_price_history[-1] if amm_price_history else current_market
        dislocation = (
            (current_amm - current_market) / max(current_market, 1e-9)
            if current_market > 0
            else 0.0
        )

        return np.array(
            [
                amm_return,
                market_return,
                sentiment,
                liquidity_ratio,
                il_current,
                fees_earned,
                volatility,
                dislocation,
                range_half_width,   # NEW in V7.5
            ],
            dtype=float,
        )

    def act(self, state):
        \"\"\"Epsilon-greedy action selection.\"\"\"
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(state)
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        \"\"\"Experience-replay mini-batch update with target network.\"\"\"
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch], dtype=float)

        # [2] TD target uses the target network, not the online network
        next_q = self.target_network.predict(next_states)    # (B, actions)
        targets = rewards + self.gamma * np.max(next_q, axis=1) * (1.0 - dones)

        self.q_network.update_batch(states, actions, targets)

        # Periodic hard-sync of target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(
        self,
        value_change,
        fees_gained,
        il_change,
        dislocation,
        rebalance_cost_ratio,
    ):
        \"\"\"Cost-aware reward with EMA normalisation [fix 4].\"\"\"
        raw = (
            4.0 * value_change
            + 2.5 * fees_gained
            - 4.0 * il_change
            - 1.5 * abs(dislocation)
            - 2.0 * rebalance_cost_ratio
        )
        normalised = self.reward_normalizer.normalize(raw)
        self.total_reward += normalised
        self.reward_history.append(normalised)
        return normalised


print("Section 4 done")

# %% [markdown]
# ## Section 5: Simulation Engine

# %%


def fetch_coingecko_ohlc(coin_id="ethereum", vs_currency="usd", days=1):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs_currency, "days": days}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


class AIAMMSimulation:
    \"\"\"
    Orchestrates the static vs AI LP comparison.

    V7.5 fixes applied here:
    [3]  Shared pre-generated market path per episode
    [5]  Persistent DQLAgent passed in each episode run
    [6]  Geometric-mean pool share
    [7]  Dynamic range_half_width + 7-action support
    [8]  Sharpe ratio + max drawdown in summary table
    \"\"\"

    def __init__(
        self,
        num_steps=50,
        num_episodes=2,
        price_feed=None,
        sentiment_feed=None,
        seed=None,
    ):
        self.num_steps = num_steps
        self.num_episodes = num_episodes
        self.price_feed = price_feed
        self.sentiment_feed = sentiment_feed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.results = {
            "static": {"values": [], "il": [], "fees": [], "gap": []},
            "ai":     {"values": [], "il": [], "fees": [], "gap": []},
        }

    @staticmethod
    def _pool_share(env, lp):
        # [6] Geometric mean is more balanced than min() for unequal ratios.
        share_a = lp.deposit_a / max(env.initial_token_a, 1e-9)
        share_b = lp.deposit_b / max(env.initial_token_b, 1e-9)
        return float(np.sqrt(share_a * share_b))

    @staticmethod
    def _recent_volatility(series, window=12):
        if len(series) < 2:
            return 0.01
        recent = series[-min(window, len(series)):]
        returns = []
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            if prev > 0:
                returns.append((recent[i] - prev) / prev)
        return float(np.std(returns)) if returns else 0.01

    @staticmethod
    def _active_fraction(market_price, range_center, range_half_width, liquidity_ratio):
        if market_price <= 0 or range_center <= 0:
            return 0.0
        lower = range_center * (1 - range_half_width)
        upper = range_center * (1 + range_half_width)
        if lower <= market_price <= upper:
            return float(np.clip(liquidity_ratio, 0.0, 1.5))
        distance = abs(np.log(market_price / range_center))
        decay = np.exp(-8.0 * distance)
        return float(np.clip(liquidity_ratio * (0.1 + 0.4 * decay), 0.0, 1.5))

    @staticmethod
    def _generate_market_path(num_steps, initial_price=1.0):
        \"\"\"
        Pre-generate a stochastic external-market price path for one episode.
        Both static and AI runs receive this identical sequence, giving a
        valid controlled comparison [fix 3].
        \"\"\"
        prices = []
        price = initial_price
        vol = 0.015
        drift = 0.0
        for i in range(num_steps):
            if i > 0 and i % 120 == 0:
                vol = float(np.random.choice([0.008, 0.015, 0.03]))
                drift = np.random.normal(0, 0.0015)
            shock = np.random.normal(drift, vol)
            price = max(1e-6, price * np.exp(shock))
            prices.append(price)
        return prices

    @staticmethod
    def _compute_sharpe(values):
        \"\"\"Per-step Sharpe ratio (mean/std of returns, scaled by sqrt N).\"\"\"
        if len(values) < 2:
            return 0.0
        returns = np.diff(values) / np.maximum(np.array(values[:-1]), 1e-9)
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        if std_r < 1e-9:
            return 0.0
        return float(mean_r / std_r * np.sqrt(len(returns)))

    @staticmethod
    def _compute_max_drawdown(values):
        \"\"\"Maximum drawdown as a fraction of the running peak.\"\"\"
        if len(values) < 2:
            return 0.0
        arr = np.array(values, dtype=float)
        peak = np.maximum.accumulate(arr)
        drawdowns = (arr - peak) / np.maximum(peak, 1e-9)
        return float(np.min(drawdowns))

    def run_static_strategy(self, price_feed=None):
        \"\"\"Passive LP baseline on a given price path.\"\"\"
        env = AMMEnvironment(price_feed=price_feed)
        sentiment_analyzer = SentimentAnalyzer(sentiment_feed=self.sentiment_feed)
        lp = LiquidityProvider(
            deposit_a=1000,
            deposit_b=1000,
            entry_price=env.get_price(),
        )
        pool_share = self._pool_share(env, lp)

        values, il_history, dislocation_history = [], [], []

        for _ in range(self.num_steps):
            sentiment = sentiment_analyzer.get_sentiment(env.price_history)
            step_info = env.simulate_market_activity(sentiment)
            lp.add_fees(step_info["fees_collected_b"] * pool_share)

            current_price = env.get_price()
            values.append(lp.calculate_position_value(current_price, active_exposure=1.0))
            il_history.append(lp.calculate_impermanent_loss(current_price))
            dislocation_history.append(step_info["price_dislocation"])

        return (
            values,
            il_history,
            lp.fees_earned,
            env.price_history,
            env.market_price_history,
            dislocation_history,
        )

    def run_ai_strategy(self, agent, price_feed=None):
        \"\"\"
        AI-driven DQL LP management.
        [5]  agent is passed in (persistent, created once in run_comparison).
        [7]  7-action space + dynamic range_half_width.
        \"\"\"
        env = AMMEnvironment(price_feed=price_feed)
        sentiment_analyzer = SentimentAnalyzer(sentiment_feed=self.sentiment_feed)
        lp = LiquidityProvider(
            deposit_a=1000,
            deposit_b=1000,
            entry_price=env.get_price(),
        )
        pool_share = self._pool_share(env, lp)

        liquidity_ratio = 1.0
        range_center = env.market_price
        # [7] range width starts at 0.12 but softly tracks volatility
        range_half_width = 0.12
        cumulative_rebalance_cost = 0.0

        values, il_history, active_history, actions_taken = [], [], [], []
        prev_il = 0.0
        prev_value = lp.initial_value

        for step in range(self.num_steps):
            sentiment = sentiment_analyzer.get_sentiment(env.price_history)
            step_info = env.simulate_market_activity(sentiment)

            current_price = env.get_price()
            market_price = step_info["market_price"]
            il_raw = lp.calculate_impermanent_loss(current_price)
            vol = self._recent_volatility(env.market_price_history, window=12)

            # [7] Soft drift toward vol-implied range width
            vol_target_width = float(np.clip(2.5 * vol, 0.05, 0.35))
            range_half_width = 0.95 * range_half_width + 0.05 * vol_target_width

            active_fraction = self._active_fraction(
                market_price, range_center, range_half_width, liquidity_ratio
            )
            il = il_raw * active_fraction

            state = agent.get_state(
                env.price_history,
                env.market_price_history,
                sentiment,
                liquidity_ratio,
                il,
                lp.fees_earned / max(lp.initial_value, 1),
                vol,
                range_half_width,
            )

            action = agent.act(state)
            actions_taken.append(action)

            # Apply action effects
            action_cost = 0.0
            if action == 1:
                liquidity_ratio = min(liquidity_ratio + 0.1, 1.5)
                action_cost += env.rebalance_gas_cost_b
            elif action == 2:
                liquidity_ratio = max(liquidity_ratio - 0.1, 0.2)
                action_cost += env.rebalance_gas_cost_b
            elif action == 3:
                range_center *= 1.02
                action_cost += env.rebalance_gas_cost_b
            elif action == 4:
                range_center *= 0.98
                action_cost += env.rebalance_gas_cost_b
            elif action == 5:
                # Widen: captures more volume, less capital-efficient
                range_half_width = min(range_half_width + 0.02, 0.40)
                action_cost += env.rebalance_gas_cost_b * 0.5
            elif action == 6:
                # Narrow: higher fee density if in-range, higher IL risk
                range_half_width = max(range_half_width - 0.02, 0.03)
                action_cost += env.rebalance_gas_cost_b * 0.5

            # Passive drift: center slowly follows market
            range_center = 0.98 * range_center + 0.02 * market_price
            cumulative_rebalance_cost += action_cost

            # Recompute active fraction after action
            active_fraction = self._active_fraction(
                market_price, range_center, range_half_width, liquidity_ratio
            )
            il = il_raw * active_fraction

            step_fee = step_info["fees_collected_b"] * pool_share * active_fraction
            lp.add_fees(step_fee)

            current_value = lp.calculate_position_value(
                current_price,
                active_exposure=active_fraction,
                trading_costs=cumulative_rebalance_cost,
            )

            il_change = il - prev_il
            value_change = (current_value - prev_value) / max(prev_value, 1)
            fee_yield = step_fee / max(lp.initial_value, 1)
            reward = agent.calculate_reward(
                value_change=value_change,
                fees_gained=fee_yield,
                il_change=il_change,
                dislocation=step_info["price_dislocation"],
                rebalance_cost_ratio=action_cost / max(lp.initial_value, 1),
            )

            next_state = agent.get_state(
                env.price_history,
                env.market_price_history,
                sentiment,
                liquidity_ratio,
                il,
                lp.fees_earned / max(lp.initial_value, 1),
                vol,
                range_half_width,
            )

            done = step == self.num_steps - 1
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            prev_il = il
            prev_value = current_value
            values.append(current_value)
            il_history.append(il)
            active_history.append(active_fraction)

        return (
            values,
            il_history,
            lp.fees_earned,
            env.price_history,
            env.market_price_history,
            actions_taken,
            agent,
            active_history,
            env.dislocation_history,
        )

    def run_comparison(self):
        \"\"\"
        Full static vs AI comparison.
        [3]  Shared price path per episode.
        [5]  Agent created once, passed into every episode.
        [8]  Sharpe ratio + max drawdown added to results.
        \"\"\"
        print("=" * 70)
        print("  AIAMM V7.5: AI Agents with AMMs for Smarter Liquidity")
        print("=" * 70)
        print()

        # [5] Create agent ONCE; it accumulates experience across ALL episodes.
        agent = DQLAgent()

        all_static_final, all_ai_final     = [], []
        all_static_il,    all_ai_il        = [], []
        all_static_fees,  all_ai_fees      = [], []
        all_static_gap,   all_ai_gap       = [], []
        all_static_sharpe, all_ai_sharpe   = [], []
        all_static_mdd,    all_ai_mdd      = [], []

        print(f"Running {self.num_episodes} simulation episodes...")
        print(f"Each episode: {self.num_steps} time steps")
        print("-" * 70)

        for episode in range(self.num_episodes):
            # [3] Pre-generate one price path shared by both runs this episode.
            ep_price_feed = (
                self.price_feed
                if self.price_feed is not None
                else self._generate_market_path(self.num_steps)
            )

            s_values, s_il, s_fees, _, _, s_dislocation = self.run_static_strategy(
                price_feed=ep_price_feed
            )
            (
                a_values, a_il, a_fees, _, _, _, _, _, a_dislocation,
            ) = self.run_ai_strategy(agent=agent, price_feed=ep_price_feed)

            all_static_final.append(s_values[-1])
            all_static_il.append(np.mean(s_il))
            all_static_fees.append(s_fees)
            all_static_gap.append(np.mean(np.abs(s_dislocation)) * 100)
            all_static_sharpe.append(self._compute_sharpe(s_values))
            all_static_mdd.append(self._compute_max_drawdown(s_values))

            all_ai_final.append(a_values[-1])
            all_ai_il.append(np.mean(a_il))
            all_ai_fees.append(a_fees)
            all_ai_gap.append(np.mean(np.abs(a_dislocation[1:])) * 100)
            all_ai_sharpe.append(self._compute_sharpe(a_values))
            all_ai_mdd.append(self._compute_max_drawdown(a_values))

            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{self.num_episodes} completed")

        print("-" * 70)
        print()
        print("=" * 70)
        print("  RESULTS SUMMARY")
        print("=" * 70)
        print()
        print(
            f"{'Metric':<35} {'Static LP':>15} {'AI Agent':>15} {'Improvement':>15}"
        )
        print("-" * 80)

        static_val = np.mean(all_static_final)
        ai_val     = np.mean(all_ai_final)
        imp_val    = (ai_val - static_val) / static_val * 100
        print(
            f"{'Avg Final Portfolio Value ($)':<35} {static_val:>15.2f} "
            f"{ai_val:>15.2f} {imp_val:>14.1f}%"
        )

        static_il = np.mean(all_static_il) * 100
        ai_il     = np.mean(all_ai_il) * 100
        il_red    = (static_il - ai_il) / abs(static_il) * 100 if static_il != 0 else 0
        print(
            f"{'Avg Impermanent Loss (%)':<35} {static_il:>14.3f}% "
            f"{ai_il:>14.3f}% {il_red:>14.1f}%"
        )

        static_fees = np.mean(all_static_fees)
        ai_fees     = np.mean(all_ai_fees)
        fee_imp     = (ai_fees - static_fees) / static_fees * 100 if static_fees != 0 else 0
        print(
            f"{'Avg Fees Earned ($)':<35} {static_fees:>15.2f} "
            f"{ai_fees:>15.2f} {fee_imp:>14.1f}%"
        )

        static_gap = np.mean(all_static_gap)
        ai_gap     = np.mean(all_ai_gap)
        gap_red    = (
            (static_gap - ai_gap) / max(static_gap, 1e-9) * 100
            if static_gap > 0 else 0
        )
        print(
            f"{'Avg |AMM-Market Gap| (%)':<35} {static_gap:>14.3f}% "
            f"{ai_gap:>14.3f}% {gap_red:>14.1f}%"
        )

        # [8] Risk-adjusted metrics
        static_sharpe = np.mean(all_static_sharpe)
        ai_sharpe     = np.mean(all_ai_sharpe)
        sh_imp        = (
            (ai_sharpe - static_sharpe) / abs(static_sharpe) * 100
            if static_sharpe != 0 else 0
        )
        print(
            f"{'Avg Sharpe Ratio':<35} {static_sharpe:>15.3f} "
            f"{ai_sharpe:>15.3f} {sh_imp:>14.1f}%"
        )

        static_mdd = np.mean(all_static_mdd) * 100
        ai_mdd     = np.mean(all_ai_mdd) * 100
        mdd_imp    = (
            (static_mdd - ai_mdd) / abs(static_mdd) * 100
            if static_mdd != 0 else 0
        )
        print(
            f"{'Avg Max Drawdown (%)':<35} {static_mdd:>14.3f}% "
            f"{ai_mdd:>14.3f}% {mdd_imp:>14.1f}%"
        )

        print()
        print(
            f"{'Std Dev Portfolio Value ($)':<35} "
            f"{np.std(all_static_final):>15.2f} {np.std(all_ai_final):>15.2f}"
        )
        print(
            f"{'Std Dev Impermanent Loss (%)':<35} "
            f"{np.std(all_static_il)*100:>14.3f}% "
            f"{np.std(all_ai_il)*100:>14.3f}%"
        )
        print()

        # Single-episode diagnostic trace (uses same trained agent)
        print("=" * 70)
        print("  DETAILED SINGLE EPISODE ANALYSIS")
        print("=" * 70)
        print()

        diag_feed = self._generate_market_path(self.num_steps)
        (
            s_values, s_il, s_fees, s_price_hist, s_market_hist, _,
        ) = self.run_static_strategy(price_feed=diag_feed)
        (
            a_values, a_il, a_fees, a_price_hist, _, actions, agent,
            active_history, a_dislocation,
        ) = self.run_ai_strategy(agent=agent, price_feed=diag_feed)

        print("AI Agent Action Distribution:")
        print("-" * 40)
        for i in range(agent.action_size):
            count = actions.count(i)
            pct   = count / len(actions) * 100
            bar   = "#" * int(pct / 2)
            print(f"  {agent.action_names[i]:<25} {count:>5} ({pct:>5.1f}%) {bar}")
        print()

        print("AI Agent Learning Metrics:")
        print("-" * 40)
        print(f"  Final Exploration Rate (epsilon):  {agent.epsilon:.4f}")
        print(f"  Total Cumulative Reward:           {agent.total_reward:.2f}")
        print(f"  Memory Buffer Size:                {len(agent.memory)}")
        print(
            f"  Target Network Sync Count:         "
            f"{agent.train_step // agent.target_update_freq}"
        )
        print()

        self.generate_plots(
            s_values=s_values, a_values=a_values,
            s_il=s_il, a_il=a_il,
            s_price_hist=s_price_hist, a_price_hist=a_price_hist,
            s_market_hist=s_market_hist,
            actions=actions, agent=agent,
            active_history=active_history,
            a_dislocation=a_dislocation,
            s_fees=s_fees, a_fees=a_fees,
        )

        return {
            "static_value":   static_val,  "ai_value":    ai_val,
            "improvement":    imp_val,
            "static_il":      static_il,   "ai_il":       ai_il,
            "il_reduction":   il_red,
            "static_fees":    static_fees, "ai_fees":     ai_fees,
            "static_gap":     static_gap,  "ai_gap":      ai_gap,
            "gap_reduction":  gap_red,
            "static_sharpe":  static_sharpe,"ai_sharpe":  ai_sharpe,
            "static_mdd":     static_mdd,  "ai_mdd":      ai_mdd,
        }

    def generate_plots(
        self,
        s_values, a_values,
        s_il, a_il,
        s_price_hist, a_price_hist, s_market_hist,
        actions, agent, active_history, a_dislocation,
        s_fees, a_fees,
    ):
        \"\"\"2x3 diagnostic plot panel.\"\"\"
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "AIAMM V7.5: AI-Driven vs Static Liquidity Management",
            fontsize=16, fontweight="bold",
        )

        ax1 = axes[0, 0]
        ax1.plot(s_values, label="Static LP",   alpha=0.8, color="blue")
        ax1.plot(a_values, label="AI Agent LP", alpha=0.8, color="green")
        ax1.set_title("Portfolio Value Over Time")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Value ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot([x * 100 for x in s_il], label="Static LP IL", alpha=0.8, color="red")
        ax2.plot([x * 100 for x in a_il], label="AI Agent IL",  alpha=0.8, color="orange")
        ax2.set_title("Impermanent Loss Over Time")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Impermanent Loss (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[0, 2]
        ax3.plot(s_price_hist,  color="tab:blue",  alpha=0.7, label="Static AMM")
        ax3.plot(a_price_hist,  color="tab:green", alpha=0.7, label="AI AMM")
        ax3.plot(
            s_market_hist, color="tab:gray", alpha=0.8,
            linestyle="--", label="Reference Market",
        )
        ax3.set_title("AMM vs Reference Price")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Price (Token B / Token A)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 0]
        action_counts = [actions.count(i) for i in range(agent.action_size)]
        colors = ["gray", "green", "red", "blue", "orange", "purple", "brown"]
        ax4.bar(
            range(agent.action_size), action_counts,
            color=colors[: agent.action_size], alpha=0.7,
        )
        ax4.set_title("AI Agent Action Distribution")
        ax4.set_xlabel("Action")
        ax4.set_ylabel("Count")
        ax4.set_xticks(range(agent.action_size))
        ax4.set_xticklabels(
            ["Hold", "Inc Liq", "Dec Liq", "Up", "Down", "Widen", "Narrow"],
            rotation=45, ha="right",
        )
        ax4.grid(True, alpha=0.3)

        ax5 = axes[1, 1]
        cum_rewards = np.cumsum(agent.reward_history)
        ax5.plot(cum_rewards, color="green", alpha=0.8)
        ax5.set_title("AI Agent Cumulative Reward (normalised)")
        ax5.set_xlabel("Time Step")
        ax5.set_ylabel("Cumulative Reward")
        ax5.grid(True, alpha=0.3)

        ax6 = axes[1, 2]
        gap_series = (
            np.abs(np.array(a_dislocation[1: len(active_history) + 1])) * 100
        )
        ax6.plot(
            np.array(active_history) * 100,
            label="Active Liquidity (%)", color="teal", alpha=0.8,
        )
        ax6.plot(
            gap_series,
            label="|AMM-Market Gap| (%)", color="tab:brown", alpha=0.8,
        )
        ax6.set_title("Execution Frictions and Range Activity")
        ax6.set_xlabel("Time Step")
        ax6.set_ylabel("Percent")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("aiamm_results.png", dpi=150, bbox_inches="tight")
        print("Visualization saved to 'aiamm_results.png'")
        print(f"Static fees: {s_fees:.2f} | AI fees: {a_fees:.2f}")
        print()


print("Section 5 done")

# %% [markdown]
# ## Section 6: Main Execution

# %%


def main():
    print()
    print("+" + "-" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|  AIAMM V7.5: AI Agents with Automated Market Makers              |")
    print("|             for Smarter Liquidity Management                     |")
    print("|" + " " * 68 + "|")
    print("|  Author: Franco Farrugia                                         |")
    print("|  Research Area: Future Digital Technologies                      |")
    print("|" + " " * 68 + "|")
    print("|  V7.5 Upgrades vs V5.5:                                          |")
    print("|  [1] MLP Q-Network (64->32) replaces linear approximator        |")
    print("|  [2] Target network with periodic hard-sync                     |")
    print("|  [3] Shared episode market path for valid comparison             |")
    print("|  [4] Reward EMA normalisation for stable Q-updates              |")
    print("|  [5] Persistent agent accumulates learning across episodes       |")
    print("|  [6] Geometric-mean pool share estimate                         |")
    print("|  [7] Dynamic range width + 7-action space (Widen/Narrow)        |")
    print("|  [8] Sharpe ratio + max drawdown in results table               |")
    print("|  [9] Replay buffer 50k (was 2k)                                 |")
    print("|  [10] Order-flow anchored to initial pool depth                 |")
    print("|" + " " * 68 + "|")
    print("+" + "-" * 68 + "+")
    print()

    use_real_data = False
    price_feed = None
    if use_real_data:
        try:
            ohlc_data = fetch_coingecko_ohlc("ethereum", "usd", days=1)
            price_feed = [row[4] for row in ohlc_data]
            print(f"Loaded {len(price_feed)} real price points.")
        except Exception as exc:
            print(f"Real data fetch failed, using simulation: {exc}")
            price_feed = None

    simulation = AIAMMSimulation(
        num_steps=5000,
        num_episodes=200,
        price_feed=price_feed,
        seed=42,
    )
    results = simulation.run_comparison()
    print("Simulation complete.")
    print(results)


if __name__ == "__main__":
    main()
"""

# ---------------------------------------------------------------------------
# Build notebook JSON
# ---------------------------------------------------------------------------

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "cells": [
        {
            "cell_type": "markdown",
            "id": cid(),
            "metadata": {},
            "source": MD_TITLE,
        },
        {
            "cell_type": "markdown",
            "id": cid(),
            "metadata": {},
            "source": MD_CHANGES,
        },
        {
            "cell_type": "code",
            "id": cid(),
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": CODE_MAIN,
        },
    ],
}

out_path = "C:/Users/Franco Farrugia/OneDrive/Desktop/Version 1/1AMMV7.5.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Written:", out_path)
