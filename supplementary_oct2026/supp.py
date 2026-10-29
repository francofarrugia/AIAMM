"""Supplementary analysis on the unchanged AIAMM V13 code (DDQN, real ETH/USDC data).
mode 'base'   : standard V13 DDQN, with extra metrics logged (capital efficiency, ROI,
                fees, VaR, return variance, market regime) + explainability on the trained agent.
mode 'nosent' : identical, but the sentiment feature in the agent's state is set to 0
                (ablation for Objective 2c). The simulated market itself is unchanged.
"""
import sys, json, os, time
import numpy as np
sys.path.insert(0, os.environ['V13DIR'])
os.chdir(os.environ['V13DIR'])
import importlib
v = importlib.import_module('13_AIAMM_V13')

mode = sys.argv[1]
N_SEEDS = int(os.environ.get('N_SEEDS', 5)); N_EP = int(os.environ.get('N_EP', 1000))
EVAL_START = 500

if mode == 'nosent':
    _orig = v.RLAgentBase.build_state
    def build_state_nosent(self, amm_hist, mkt_hist, sentiment, *a, **k):
        return _orig(self, amm_hist, mkt_hist, 0.0, *a, **k)
    v.RLAgentBase.build_state = build_state_nosent

def episode_std(vals):
    a = np.array(vals); r = np.diff(a) / a[:-1]
    return float(r.std())

out = {'mode': mode, 'seeds': []}
t0 = time.time()
for s_idx in range(N_SEEDS):
    seed = 42 + s_idx * 10_000
    v.set_seeds(seed)
    agent = v.DQLAgent()
    rec = []
    for ep in range(N_EP):
        T = v.curriculum_steps(ep, N_EP)
        ep_seed = seed + ep * 997
        path, start_idx = v.generate_price_path(T, ep_seed)
        st = v.run_static(path, T, seed=ep_seed)
        ai = v.run_ai(path, T, agent, st['values'], seed=ep_seed + 1, start_idx=start_idx)
        lr = np.diff(np.log(np.array(path)))
        rec.append(dict(ep=ep, T=T, start=int(start_idx),
            alpha=ai['values'][-1] - st['values'][-1],
            ai_val=ai['values'][-1], st_val=st['values'][-1],
            ai_fees=ai['fees'], st_fees=st['fees'],
            util=float(np.mean(ai['active'])),
            il_ai=float(np.mean(ai['il']) * 100), il_st=float(np.mean(st['il']) * 100),
            reb=sum(1 for a in ai['actions'] if a != 0),
            drift=float(path[-1] / path[0] - 1), rv=float(lr.std()),
            sd_ai=episode_std(ai['values']), sd_st=episode_std(st['values'])))
    seed_out = {'seed': seed, 'episodes': rec}

    if mode == 'base':
        # ---- explainability on the trained, frozen agent (greedy) ----
        agent.eval_mode = True
        agent.epsilon = 0.0
        states, acts = [], []
        orig_act = agent.act
        def rec_act(state, _o=orig_act):
            a = _o(state); states.append(np.array(state)); acts.append(a); return a
        agent.act = rec_act
        for k in range(30):
            es = 777_777 + seed + k * 131
            path, start_idx = v.generate_price_path(500, es)
            st = v.run_static(path, 500, seed=es)
            v.run_ai(path, 500, agent, st['values'], seed=es + 1, start_idx=start_idx)
        agent.act = orig_act
        S = np.array(states)
        Q = agent.online.predict(S); base_a = Q.argmax(1)
        rng = np.random.default_rng(seed)
        imp = []
        for j in range(S.shape[1]):
            changed = []
            for rep in range(5):
                Sp = S.copy(); Sp[:, j] = rng.permutation(Sp[:, j])
                changed.append(float(np.mean(agent.online.predict(Sp).argmax(1) != base_a)))
            imp.append(float(np.mean(changed)))
        seed_out['xai'] = {'perm_importance': imp,
                           'greedy_action_dist': np.bincount(base_a, minlength=5).tolist(),
                           'n_states': int(len(S))}
    out['seeds'].append(seed_out)
    print(f'[{mode}] seed {s_idx+1}/{N_SEEDS} done, {time.time()-t0:.0f}s, '
          f'mean alpha {np.mean([r["alpha"] for r in rec]):+.2f}', flush=True)

json.dump(out, open(f'{os.environ["OUTDIR"]}/supp_{mode}.json', 'w'))
print('FINISHED', mode, f'{time.time()-t0:.0f}s')
