# Train the V13 DDQN (seed 42, 1,000 curriculum episodes on real ETH/USDC data) and export weights
import sys, os, json, numpy as np
V = sys.argv[1]; sys.path.insert(0, V); os.chdir(V)
import importlib; v = importlib.import_module('13_AIAMM_V13')
seed = 42; v.set_seeds(seed); agent = v.DQLAgent(); alphas = []
for ep in range(1000):
    T = v.curriculum_steps(ep, 1000); es = seed + ep * 997
    path, st_idx = v.generate_price_path(T, es)
    st = v.run_static(path, T, seed=es)
    ai = v.run_ai(path, T, agent, st['values'], seed=es + 1, start_idx=st_idx)
    alphas.append(ai['values'][-1] - st['values'][-1])
    if ep % 100 == 99: print(ep + 1, np.mean(alphas[-100:]), flush=True)
n = agent.online; nz = agent.normalizer
out = sys.argv[2]
np.savez(out, W1=n.W1, b1=n.b1, W2=n.W2, b2=n.b2, W3=n.W3, b3=n.b3,
         norm_attrs=json.dumps({k: (val.tolist() if isinstance(val, np.ndarray) else val) for k, val in vars(nz).items()}),
         epsilon=agent.epsilon, train_mean_alpha=np.mean(alphas), eval_mean_alpha=np.mean(alphas[500:]))
print('saved', out, 'mean alpha all', np.mean(alphas), 'eval', np.mean(alphas[500:]))
