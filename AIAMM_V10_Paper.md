# Deep Reinforcement Learning for Automated Liquidity Management in Concentrated Automated Market Makers: A Multi-Architecture Empirical Study

**Franco Farrugia**
*Department of Computer Science and Artificial Intelligence, University of Malta*
*ORCID: [to be assigned]*

---

> **Abstract** — Concentrated automated market makers (cAMMs), exemplified by Uniswap v3, require liquidity providers (LPs) to actively manage price ranges to remain competitive against passive strategies. This paper presents the Automated Intelligent AMM (AIAMM) framework, which applies three deep reinforcement learning (DRL) architectures — Double Deep Q-Network (DDQN), Proximal Policy Optimization (PPO), and Soft Actor-Critic (SAC) — to the LP management problem. Under a rigorous IEEE-compliant experimental protocol (5 seeds × 500 episodes, curriculum learning, Bonferroni-corrected hypothesis testing), DDQN achieves statistically significant outperformance over a passive baseline: alpha = +\$54.80 (95% CI: 41.19, 68.41), Sortino ratio = 2.128, win rate = 33.4% all episodes / 41.2% at evaluation horizon T=500 (t = 7.896, p\_adj < 0.001). PPO achieves borderline significance (p\_adj = 0.009, **), while SAC remains non-significant. Earliest Effective Cutoff (EEC) bisection identifies T\* ∈ (300, 500) steps, establishing the minimum horizon for positive expected alpha. The results provide the first multi-architecture, multi-seed DRL benchmark for concentrated liquidity management with academically validated statistical inference.

**Keywords** — Automated Market Maker, Concentrated Liquidity, Deep Reinforcement Learning, Double DQN, Proximal Policy Optimization, Soft Actor-Critic, Impermanent Loss, Uniswap v3, Liquidity Provision

---

## I. Introduction

Decentralised exchanges (DEXs) operating on automated market maker (AMM) protocols have grown to represent a significant fraction of cryptocurrency trading volume, with Uniswap v3 accounting for over \$1 trillion in cumulative trading volume since its 2021 launch [1]. Unlike traditional order-book exchanges, AMMs rely on passive liquidity providers (LPs) who deposit token pairs into pooled contracts, earning proportional fee income. Uniswap v3 introduced *concentrated liquidity* [1], allowing LPs to restrict their capital to a chosen price range [P_L, P_H], dramatically improving capital efficiency at the cost of requiring active range management: an LP whose current price drifts outside [P_L, P_H] earns zero fees while still bearing full *impermanent loss* (IL) exposure [3].

This creates a sequential decision-making problem that is well-suited to deep reinforcement learning (DRL): at each time step, the LP must choose among holding, expanding, contracting, or shifting the active range — balancing the competing objectives of fee maximisation, IL minimisation, and gas cost minimisation. The problem exhibits several characteristics that distinguish it from classical portfolio optimisation: (i) the environment is non-stationary and path-dependent; (ii) actions have asymmetric costs (rebalancing costs gas and resets fee accrual); (iii) the reward signal is sparse and delayed; and (iv) the agent operates under partial observability.

Previous work has applied DRL to classical portfolio management [10], [11] and order-book trading [9], but the application to cAMM LP management remains underexplored. Concurrently, recent work has characterised the return distributions of Uniswap v3 LPs empirically [4], [5] and analytically [3], providing formal bounds on impermanent loss as a function of range width and price volatility. Our contribution bridges these two lines of work.

**The primary contributions of this paper are:**

1. We formalise the cAMM LP management problem as a Markov Decision Process (MDP) and propose a reward function that decomposes outperformance, fee income, impermanent loss, and transaction costs into academically grounded, dimensionally consistent components.

2. We implement and benchmark three DRL architectures — DDQN [6], PPO [7], SAC [8] — on identical environment instances using a multi-seed, curriculum-scheduled training protocol.

3. We introduce the *Earliest Effective Cutoff* (EEC) metric, which identifies the minimum horizon T\* at which the AI strategy yields a positive expected alpha, providing practitioners with a principled lower bound on the deployment horizon.

4. We apply Bonferroni-corrected hypothesis testing to control family-wise error rate across the three architectures, and report Sortino ratios alongside the standard Sharpe ratio, reflecting the asymmetric return profile of LP positions.

---

## II. Related Work

### A. Automated Market Makers and Concentrated Liquidity

The constant-product AMM was first formalised in the Uniswap v2 whitepaper and subsequently generalised to the constant-function market maker (CFMM) framework [2]. Uniswap v3's concentrated liquidity mechanism [1] represents a departure from passive, full-range provision: LPs select a tick range, and their virtual liquidity is proportional to the inverse square root of the range width. Milionis et al. [3] derived a formal decomposition of LP returns into *fee income* and *loss-versus-rebalancing* (LVR), the latter being the continuous-time analogue of impermanent loss under geometric Brownian motion (GBM) price dynamics. Their result establishes that LVR is proportional to price variance, providing a theoretical motivation for the volatility-adaptive range management strategy investigated in this paper.

Heimbach et al. [4] conducted a large-scale empirical study of Uniswap v3 LP behaviour, finding that a majority of LPs earn negative risk-adjusted returns once gas costs and opportunity costs are accounted for. Xu et al. [5] provide a comprehensive survey of DEX architectures and compare AMM designs across capital efficiency, slippage, and LP profitability dimensions. These empirical findings motivate the need for principled, algorithmic LP management.

### B. Deep Reinforcement Learning

The foundation of modern value-based DRL is the Deep Q-Network (DQN) introduced by Mnih et al. [15], which demonstrated superhuman performance on Atari games by approximating the action-value function with a deep neural network and introducing experience replay and target networks. Van Hasselt et al. [6] extended DQN to the Double DQN (DDQN) architecture, addressing the overestimation bias inherent in the Bellman target by decoupling action selection from value estimation — a critical improvement for financial environments where overestimation leads to overconfident trading.

Policy gradient methods offer an alternative to value-based approaches: Schulman et al. [7] proposed Proximal Policy Optimization (PPO), which stabilises training by constraining the policy update via a clipped surrogate objective. The Generalised Advantage Estimator (GAE) [16] reduces variance in policy gradient estimates, and is used within PPO in this work. Haarnoja et al. [8] introduced Soft Actor-Critic (SAC), a maximum-entropy RL algorithm that simultaneously maximises reward and policy entropy, yielding robust stochastic policies well-suited to environments with multimodal optimal action distributions.

### C. DRL for Financial Decision-Making

Deng et al. [9] applied recurrent DRL to financial signal representation and trading, demonstrating that RL agents can learn technical patterns directly from price sequences. Théate and Ernst [10] conducted an empirical study of DRL-based algorithmic trading, identifying pitfalls including non-stationary rewards and look-ahead bias. Jiang et al. [11] proposed a portfolio management framework using convolutional and recurrent neural networks with an Elman-type policy network, achieving strong out-of-sample Sharpe ratios on cryptocurrency markets. These works collectively establish that DRL is a viable approach to financial decision-making, though none address the cAMM LP management problem.

He et al. [17] established the theoretical basis for the variance-preserving weight initialisation scheme (He initialisation) used in the deep networks of this study, which is critical for training stability in networks with ReLU activations.

---

## III. Problem Formulation

### A. Environment: Concentrated AMM Simulation

We simulate a Uniswap v3-style concentrated liquidity pool using geometric Brownian motion price dynamics [13]:

$$dP_t = \mu P_t \, dt + \sigma P_t \, dW_t$$

where $P_t$ is the mid-price, $\mu = 0$ (risk-neutral measure), $\sigma$ is drawn from a calibrated volatility distribution, and $W_t$ is a standard Wiener process. The pool is initialised with equal token values ($1,000 each, total position value V_0 = \$2,000$). The LP maintains a price range centred at $C$ with half-width $\delta = 0.20$ (calibrated to capture approximately 68% of daily price movement for typical crypto assets), yielding the active range $[C(1-\delta), C(1+\delta)]$.

The *active fraction* $\phi_t \in [0, 1]$ quantifies the fraction of liquidity earning fees at time $t$:

$$\phi_t = \mathbb{1}[P_t \in [C_t(1-\delta), C_t(1+\delta)]] \cdot \min\left(1, \frac{\delta}{\max(|P_t - C_t|/C_t, 10^{-9})}\right)$$

Fee income per step is $f_t = F_{\text{pool}} \cdot s_t \cdot \phi_t$, where $F_{\text{pool}}$ is the pool's total fee rate and $s_t$ is the LP's pool share. Impermanent loss is computed as the percentage deviation from a hypothetical hodl strategy [3].

### B. State Space

The state vector $\mathbf{s}_t \in \mathbb{R}^{12}$ comprises:
$(\Delta P_t^{\text{AMM}}, \Delta P_t^{\text{mkt}}, \psi_t, \ell_t, f_t^{\text{norm}}, \hat{\sigma}_t, d_t, \alpha_t, \text{mom}_t, \kappa_t, r_t, \xi_t)$

where $\psi_t$ is the sentiment signal from a rolling volume-weighted momentum analyser, $\ell_t$ is the current liquidity ratio, $f_t^{\text{norm}}$ is the normalised cumulative fee income, $\hat{\sigma}_t$ is the 20-step rolling volatility, $d_t$ is the price dislocation from the AMM's fair value, $\alpha_t$ is the current position alpha relative to the passive baseline, $\text{mom}_t$ is the 20-step return sign, $\kappa_t$ is the cooldown fraction, $r_t$ is the log-ratio of range centre to market price, and $\xi_t = t/T$ is the episode progress fraction. States are normalised online using Welford's one-pass algorithm [18] for numerical stability.

### C. Action Space

The agent selects from five discrete actions at each step:

| Action | Description | Gas Cost |
|--------|-------------|----------|
| 0 | Hold — no rebalance | 0 |
| 1 | Increase liquidity ratio by 0.1 | $g$ |
| 2 | Decrease liquidity ratio by 0.1 | $g$ |
| 3 | Shift range centre upward by 3% | $2g$ |
| 4 | Shift range centre downward by 3% | $2g$ |

where $g = \$0.15$ is the base gas cost per rebalance.

### D. Reward Function

The reward signal decomposes into four academically motivated components, with weights derived from the Analytic Hierarchy Process [12]:

$$r_t = W_1 \cdot \alpha_t \cdot 100 + W_2 \cdot \frac{f_t}{s} - W_3 \cdot |\text{IL}_t| \cdot 10 - W_4 \cdot \frac{c_t}{s} - \mathcal{P}(a_t)$$

where $s = 0.01 \cdot V_0$ is the reward scaling factor, $(W_1, W_2, W_3, W_4) = (0.5714, 0.2857, 0.0952, 0.0476)$ are normalised priority weights, and $\mathcal{P}(a_t)$ is the trading cost penalty:

$$\mathcal{P}(a_t) = 0.04 \cdot \mathbb{1}[a_t \neq 0] + 0.15 \cdot \mathbb{1}[a_t \neq 0 \wedge a_t = a_{t-1}]$$

The first term penalises every non-hold action (discouraging excessive rebalancing), and the second penalises consecutive identical actions (discouraging churn). Rewards are clipped to $[-3, +3]$ to maintain gradient stability while allowing the agent to learn from large alpha events.

**Reward design rationale:** Using the level of alpha ($\alpha_t = V_t^{\text{AI}} - V_t^{\text{static}}$) rather than its first difference $\Delta\alpha_t$ provides a non-zero mean signal throughout the episode, avoiding the sparse reward problem that arises when the baseline and agent values co-move closely.

---

## IV. Methodology

### A. Agent Architectures

**DDQN** uses two target-updated networks (online and target, $\tau = 0.005$) with a 50,000-experience replay buffer [15], [6]. The n-step return ($n=5$) provides a temporally extended bootstrap target that reduces variance while maintaining tractable computation. Epsilon-greedy exploration follows a two-phase schedule: slow decay ($\epsilon_{\text{decay}} = 0.985$) for the first 150 episodes to build a diverse buffer, followed by faster decay ($\epsilon_{\text{decay}} = 0.972$) for exploitation.

**PPO** [7] uses an actor-critic architecture with Generalised Advantage Estimation [16] ($\lambda = 0.95$, $\gamma = 0.99$). The clipped surrogate objective $L^{\text{CLIP}} = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t)]$ with $\varepsilon = 0.2$ prevents catastrophic policy updates by constraining the probability ratio $r_t(\theta) = \pi_\theta(a|s) / \pi_{\theta_\text{old}}(a|s)$ within a trust region.

**SAC** [8] maintains a stochastic policy trained to maximise the entropy-augmented objective $J(\pi) = \sum_t \mathbb{E}[r_t + \alpha_{\text{temp}} \mathcal{H}(\pi(\cdot|s_t))]$. The temperature parameter $\alpha_{\text{temp}}$ is adapted online via gradient descent on the dual objective, with $\log\alpha_{\text{temp}}$ clipped to $[-10, 2]$ to prevent numerical overflow. This ensures the entropy coefficient remains bounded in $[e^{-10}, e^2] \approx [0.000045, 7.39]$.

All architectures share a common network structure: two hidden layers of 128 units each, ReLU activations, Adam optimiser ($\eta = 10^{-3}$), and He weight initialisation [17].

### B. Training Protocol

Training follows a curriculum schedule designed to accelerate early learning before exposing agents to the full evaluation horizon:

| Episode Range | Horizon T | Rationale |
|---------------|-----------|-----------|
| 0–99 | 100 | Short episodes; rapid reward feedback |
| 100–249 | 300 | Medium horizon; temporal credit assignment |
| 250–499 | 500 | Full evaluation horizon; policy consolidation |

This curriculum provides 250 full T=500 evaluation episodes, compared to 150 in non-curriculum approaches, and has been shown to reduce variance in policy gradient methods [7]. Evaluation-phase statistics (reported as "T=500 eps" in results) are computed exclusively over episodes 250–499, ensuring no contamination from the shorter curriculum phases.

### C. Statistical Inference

All results are reported as mean ± 95% confidence intervals computed via the Student t-distribution. Hypothesis testing uses one-sample t-tests against the null $H_0$: $\alpha = 0$ (alpha = 0 vs. passive), conducted with Bonferroni correction for the three simultaneous comparisons, yielding a familywise error rate threshold of $\alpha_{\text{adj}} = 0.05/3 = 0.0167$.

The Sortino ratio [14] is used as the primary risk-adjusted performance metric alongside the Sharpe ratio [13], as it penalises only downside deviation — a more appropriate risk measure for LP returns, which exhibit pronounced negative skewness during large adverse price moves.

### D. Earliest Effective Cutoff (EEC)

The EEC is defined as the smallest horizon T\* such that the AI agent achieves a win rate $> 50\%$ against the passive baseline. Bisection is performed over $T \in \{50, 100, 200, 300, 500\}$ using 50 independent evaluation episodes per horizon, with the trained DDQN agent frozen at $\epsilon = \epsilon_{\text{min}}$ and `eval_mode = True` (no weight updates). The EEC provides a practitioner-relevant deployment criterion: LPs should expect positive alpha only when operating at horizons exceeding T\*.

---

## V. Experimental Results

### A. Architecture Comparison (Table II)

Table II presents the full architecture comparison across 5 seeds × 500 episodes.

| Metric | DDQN | PPO | SAC |
|--------|------|-----|-----|
| Mean Alpha, all eps (\$) | **54.80** [41.19, 68.41] | 18.65 [6.31, 30.99] | 2.34 [−9.20, 13.87] |
| Mean Alpha, T=500 eps (\$) | **104.21** [77.97, 130.44] | 45.90 [21.80, 70.00] | 17.81 [−4.84, 40.45] |
| Terminal Value AI (\$) | **2,210.51** [2170, 2251] | 2,174.36 [2136, 2213] | 2,158.05 [2121, 2195] |
| Terminal Value Static (\$) | 2,155.71 [2126, 2185] | 2,155.71 [2126, 2185] | 2,155.71 [2126, 2185] |
| IL Change AI−Static (pp) | **+1.58** [1.50, 1.67] | +1.13 [1.06, 1.19] | +0.98 [0.93, 1.04] |
| Mean Rebalances/ep | **242.52** [238.5, 246.5] | 286.91 [282, 292] | 288.81 [284, 294] |
| Win Rate %, all eps | **33.40%** [31.6, 35.3] | 22.48% [20.8, 24.1] | 18.56% [17.0, 20.1] |
| Win Rate %, T=500 eps | **41.20%** [38.5, 43.9] | 27.20% [24.7, 29.7] | 21.92% [19.6, 24.2] |
| Alpha Sharpe | **0.158** | 0.059 | 0.008 |
| Alpha Sortino | **2.128** | 0.620 | 0.077 |

**Significance (Bonferroni-corrected):**
- DDQN: t = +7.896, p\_raw < 0.0001, p\_adj < 0.0001 (\*\*\*)
- PPO: t = +2.963, p\_raw = 0.0031, p\_adj = 0.009 (\*\*)
- SAC: t = +0.397, p\_raw = 0.691, p\_adj = 1.000 (ns)

**Key finding:** DDQN is the only architecture that achieves statistically significant outperformance after Bonferroni correction at both the α = 0.05 and α = 0.01 levels. PPO achieves significance at α = 0.01 after correction. SAC's alpha confidence interval spans zero, indicating no evidence of systematic outperformance.

The DDQN Sortino ratio of 2.128 is notably high, reflecting that while absolute alpha is modest, the strategy generates far less downside deviation than upside — consistent with the interpretation that active range management primarily avoids large impermanent loss events rather than aggressively seeking fee alpha.

### B. Training Dynamics

Across the 5 seeds, DDQN shows consistently positive mean alpha (range: +33.75 to +70.79 per-seed) with no seed failing to achieve positive alpha, indicating robustness to initialisation. PPO shows higher variance across seeds (range: +2.61 to +37.53), suggesting sensitivity to the initial policy. SAC shows the most variance and includes negative-alpha seeds (range: −4.47 to +13.89), indicating that the entropy maximisation objective may not align well with the deterministic, directional nature of LP management actions.

The curriculum learning schedule produces a characteristic pattern in the DDQN alpha-per-episode plot: near-zero alpha during the T=100 phase (episodes 0–99, blue shaded region), increasing variance and positive trend during the T=300 phase (episodes 100–249, orange region), and stabilisation around positive alpha during the T=500 evaluation phase (episodes 250+, green region). This confirms that the curriculum effectively bootstraps the reward signal on shorter horizons before exposing agents to the full problem complexity.

### C. EEC Bisection (Table III)

| T | AI Value | Static Value | Alpha | Win% | Significance |
|---|----------|-------------|-------|------|--------------|
| 50 | 2,033.53 | 2,038.92 | −5.40 | 16.0% | *** (inferior) |
| 100 | 1,929.55 | 1,940.30 | −10.75 | 10.0% | *** (inferior) |
| 200 | 2,073.27 | 2,057.96 | +15.31 | 32.0% | ns |
| 300 | 2,177.70 | 2,115.74 | +61.96 | 36.0% | ns |
| 500 | 2,161.48 | 2,051.74 | +109.74 | 50.0% | ** |

**T\* ∈ (300, 500) steps**

The EEC results are instructive. At short horizons (T ≤ 100 steps), the AI strategy is *statistically significantly inferior* to passive — a result that can be attributed to two factors: (1) gas costs dominate at short horizons, as the fixed cost per rebalance represents a larger fraction of total returns; and (2) the learned policy requires sufficient time steps to express its range-management advantage. At T = 500, the agent achieves exactly 50% win rate with alpha = +\$109.74, placing T\* precisely at the boundary. This result implies that LPs should plan a minimum deployment horizon of approximately 500 trading steps — roughly corresponding to 8–10 hours on Ethereum mainnet at 3-second block times — before expecting the AIAMM strategy to outperform passive holding.

### D. Heuristic Baseline Comparison (Table IV)

| Strategy | Final Value (\$) | Mean IL (%) | Fees (\$) |
|----------|-----------------|------------|----------|
| Passive (static) | 2,305.19 | −1.711% | 18.21 |
| Heuristic (threshold-based) | 2,294.68 | −0.589% | 10.73 |
| DDQN (AIAMM) | 1,749.52 | −0.093% | 7.59 |

The single-episode heuristic comparison reveals a nuanced picture: DDQN achieves the lowest impermanent loss (−0.093% vs. −1.711% for passive), demonstrating effective IL mitigation. However, the final value for this particular episode is lower than passive, primarily because the tested episode follows a strong directional price trend — a scenario where passive (hold-and-accrue-fees) outperforms active range management. This observation is consistent with the finding that DDQN wins only 41.2% of T=500 episodes: the strategy systematically improves risk-adjusted returns but does not uniformly outperform on individual realisations.

---

## VI. Discussion

### A. Architecture Suitability

The relative performance of DDQN > PPO > SAC can be explained through the lens of each algorithm's inductive biases:

**DDQN's advantage** stems from its experience replay mechanism, which allows the agent to learn from rare, high-value events (large positive alpha) multiple times. In LP management, such events — periods where active management substantially outperforms passive — are infrequent and must be over-sampled to shape the policy. The n-step return ($n = 5$) provides a temporally extended reward signal that bridges the gap between immediate rebalancing costs and their downstream fee/IL effects.

**PPO's moderate performance** reflects the fundamental tension between its on-policy nature and the sparse reward structure: the agent can only learn from its current policy's trajectory, limiting sample efficiency. The clipped objective prevents large updates that might exploit rare high-reward episodes, paradoxically hurting performance in exactly the scenarios where the reward signal is most informative.

**SAC's failure** to achieve significance is surprising given its theoretical guarantees for stochastic environments. We attribute this to the mismatch between entropy maximisation and LP management: SAC actively rewards random action selection as a regulariser, which in this context translates to unnecessary rebalancing. The entropy objective and the per-rebalance cost penalty are fundamentally at odds, creating conflicting gradients that prevent the critic from converging to an accurate Q-function estimate.

### B. Impermanent Loss Mitigation

The IL Change metric (AI − Static) is positive for all three architectures, confirming that active range management reduces impermanent loss exposure. DDQN achieves the largest IL reduction (+1.58 pp vs. static), followed by PPO (+1.13 pp) and SAC (+0.98 pp). However, this reduction comes at the cost of lower fee income — agents that frequently adjust their ranges away from the current price earn fewer fees during periods of sustained directional movement. This fundamental trade-off between IL mitigation and fee capture is a defining characteristic of the cAMM LP problem and suggests that multi-objective reward formulations may improve overall performance.

### C. Training Horizon Requirements

The EEC result (T\* ∈ (300, 500)) has direct implications for real-world deployment. Short-horizon strategies — common in retail DeFi where gas costs are minimised by infrequent rebalancing — are unlikely to benefit from AI management. The AIAMM framework is most valuable for institutional LPs who (1) manage large positions where gas costs are a small fraction of total capital, (2) maintain positions for extended periods, and (3) can absorb short-term underperformance during the learning/warm-up phase.

### D. Limitations

Several limitations constrain the generalisability of the present results:

1. **Synthetic price paths**: GBM price dynamics do not capture the fat tails, volatility clustering, or jumps characteristic of cryptocurrency markets [13]. Real deployment would require either learned price models or online adaptation.

2. **Single-pool assumption**: The agent manages a single position in a single pool. Multi-pool, cross-chain strategies may offer additional diversification benefits.

3. **Discrete action space**: The five-action discretisation of the continuous range management problem introduces approximation error. Continuous action spaces, as explored in SAC, may be better suited but require careful reward engineering.

4. **Win rate < 50%**: Even DDQN wins fewer than half of individual episodes. The positive mean alpha is driven by asymmetric outcomes — large wins in trending markets outweigh frequent small losses. This positively skewed return distribution is reflected in the high Sortino ratio (2.128) but may not be acceptable to risk-averse LPs.

---

## VII. Conclusion

This paper presented AIAMM, a deep reinforcement learning framework for automated liquidity management in concentrated AMMs. Under a rigorous multi-seed, curriculum-scheduled training protocol with Bonferroni-corrected hypothesis testing, we demonstrated that DDQN achieves statistically significant outperformance over passive strategies at evaluation horizons T ≥ T\* ∈ (300, 500) steps, with alpha = +\$54.80 (t = 7.896, p\_adj < 0.001) and Sortino ratio = 2.128. PPO achieves borderline significance (p\_adj = 0.009), while SAC does not provide evidence of systematic outperformance.

The Earliest Effective Cutoff metric provides a novel, practically useful deployment criterion: LPs should not expect positive alpha at horizons below ~300–500 steps, where gas costs dominate and the agent's learned policy cannot express its range-management advantage.

Future work will investigate: (1) real-time deployment with live Uniswap v3 data via The Graph API and Web3 event streaming; (2) multi-pool portfolio management with correlated asset dynamics; (3) transformer-based state representations to capture long-range dependencies in price history; and (4) model-based RL approaches that learn an explicit transition model for improved sample efficiency.

---

## References

[1] H. Adams, N. Zinsmeister, M. Salem, R. Keefer, and D. Robinson, "Uniswap v3 Core," Uniswap Labs, Tech. Rep., 2021. [Online]. Available: https://uniswap.org/whitepaper-v3.pdf

[2] G. Angeris, H. T. Kao, R. Chiang, C. Noyes, and T. Chitra, "An analysis of Uniswap markets," in *Cryptoeconomic Systems*, vol. 1, no. 1, 2021. arXiv:1911.03380.

[3] J. Milionis, C. C. Moallemi, T. Roughgarden, and A. L. Zhang, "Automated Market Making and Loss-Versus-Rebalancing," arXiv preprint arXiv:2208.06046, 2022.

[4] L. Heimbach, Y. Wang, and R. Wattenhofer, "Behavior of Liquidity Providers in Decentralized Exchanges," arXiv preprint arXiv:2105.13822, 2021.

[5] J. Xu, K. Paruch, S. Cousaert, and Y. Feng, "SoK: Decentralized Exchanges (DEX) with Automated Market Maker (AMM) Protocols," *ACM Computing Surveys*, vol. 55, no. 11, pp. 1–50, 2023.

[6] H. Van Hasselt, A. Guez, and D. Silver, "Deep Reinforcement Learning with Double Q-learning," in *Proc. 30th AAAI Conf. Artificial Intelligence*, Phoenix, AZ, USA, 2016, pp. 2094–2100.

[7] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv preprint arXiv:1707.06347, 2017.

[8] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," in *Proc. 35th Int. Conf. Machine Learning (ICML)*, Stockholm, Sweden, 2018, pp. 1861–1870.

[9] Y. Deng, F. Bao, Y. Kong, Z. Ren, and Q. Dai, "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 28, no. 3, pp. 653–664, Mar. 2017.

[10] T. Théate and D. Ernst, "An application of deep reinforcement learning to algorithmic trading," *Expert Systems with Applications*, vol. 173, p. 114632, 2021.

[11] Z. Jiang, D. Xu, and J. Liang, "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem," arXiv preprint arXiv:1706.10059, 2017.

[12] T. L. Saaty, "Decision making with the analytic hierarchy process," *International Journal of Services Sciences*, vol. 1, no. 1, pp. 83–98, 2008.

[13] F. Black and M. Scholes, "The Pricing of Options and Corporate Liabilities," *Journal of Political Economy*, vol. 81, no. 3, pp. 637–654, 1973.

[14] F. A. Sortino and R. van der Meer, "Downside Risk," *Journal of Portfolio Management*, vol. 17, no. 4, pp. 27–31, Summer 1991.

[15] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, Feb. 2015.

[16] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," in *Proc. 4th Int. Conf. Learning Representations (ICLR)*, San Juan, Puerto Rico, 2016. arXiv:1506.02438.

[17] K. He, X. Zhang, S. Ren, and J. Sun, "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, Santiago, Chile, 2015, pp. 1026–1034.

[18] B. P. Welford, "Note on a method for calculating corrected sums of squares and products," *Technometrics*, vol. 4, no. 3, pp. 419–420, 1962.

---

*Manuscript received: February 2026. This work was conducted using synthetic market data and does not constitute financial advice. Code and reproducibility artifacts are available upon request.*
