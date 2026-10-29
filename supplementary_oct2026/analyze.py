import json, numpy as np
from scipy import stats
D = '/tmp/claude-0/-home-claude/8d1a4307-7dd1-5fd0-8063-4ce88d897096/scratchpad/supp/'
FEAT = ['AMM return', 'Market return', 'Sentiment indicator', 'Liquidity ratio', 'Fees earned',
        'Volatility', 'Price dislocation', 'Alpha signal', 'Momentum sign', 'Cooldown',
        'Range position', 'Episode progress', 'High–low range', 'Open–close return', 'Volume ratio']
ACT = ['Hold', 'Increase liquidity', 'Decrease liquidity', 'Shift range up', 'Shift range down']

def load(m):
    d = json.load(open(D + f'supp_{m}.json'))
    eps = [e for s in d['seeds'] for e in s['episodes']]
    ev = [e for s in d['seeds'] for e in s['episodes'] if e['ep'] >= 500]
    return d, eps, ev

def ci(x):
    x = np.array(x); m = x.mean(); h = stats.sem(x) * stats.t.ppf(0.975, len(x) - 1)
    return m, m - h, m + h

R = {}
for m in ['base', 'nosent']:
    d, eps, ev = load(m)
    a_all = np.array([e['alpha'] for e in eps]); a = np.array([e['alpha'] for e in ev])
    r = {}
    r['n_all'], r['n_eval'] = len(eps), len(ev)
    r['alpha_all'] = ci(a_all); r['win_all'] = float((a_all > 0).mean() * 100)
    r['alpha_eval'] = ci(a); r['win_eval'] = float((a > 0).mean() * 100)
    if m == 'base':
        r['roi_ai'] = ci([(e['ai_val'] / 2000 - 1) * 100 for e in ev])
        r['roi_st'] = ci([(e['st_val'] / 2000 - 1) * 100 for e in ev])
        r['fees_ai'] = ci([e['ai_fees'] for e in ev]); r['fees_st'] = ci([e['st_fees'] for e in ev])
        r['util'] = ci([e['util'] * 100 for e in ev])
        r['fee_per_active'] = float(np.mean([e['ai_fees'] / max(e['util'], 1e-6) for e in ev]))
        r['var95'] = float(np.percentile(a, 5)); r['cvar95'] = float(a[a <= np.percentile(a, 5)].mean())
        r['sd_ai'] = float(np.mean([e['sd_ai'] for e in ev]) * 100); r['sd_st'] = float(np.mean([e['sd_st'] for e in ev]) * 100)
        r['roi_sd_ai'] = float(np.std([(e['ai_val'] / 2000 - 1) * 100 for e in ev]))
        r['roi_sd_st'] = float(np.std([(e['st_val'] / 2000 - 1) * 100 for e in ev]))
        r['il_diff'] = ci([e['il_ai'] - e['il_st'] for e in ev])
        r['reb'] = float(np.mean([e['reb'] for e in ev]))
        reg = {}
        for name, f in [('Bull (price +5% or more)', lambda e: e['drift'] >= 0.05),
                        ('Bear (price −5% or more)', lambda e: e['drift'] <= -0.05),
                        ('Sideways (within ±5%)', lambda e: -0.05 < e['drift'] < 0.05)]:
            x = np.array([e['alpha'] for e in ev if f(e)])
            reg[name] = (len(x), *ci(x), float((x > 0).mean() * 100)) if len(x) > 1 else (len(x),)
        rvs = np.array([e['rv'] for e in ev]); q = np.percentile(rvs, [33.3, 66.7])
        for name, f in [('Lowest-volatility third', lambda e: e['rv'] <= q[0]),
                        ('Middle third', lambda e: q[0] < e['rv'] <= q[1]),
                        ('Highest-volatility third', lambda e: e['rv'] > q[1])]:
            x = np.array([e['alpha'] for e in ev if f(e)])
            reg[name] = (len(x), *ci(x), float((x > 0).mean() * 100))
        r['regimes'] = reg
        r['rv_q'] = q.tolist()
        imp = np.mean([s['xai']['perm_importance'] for s in d['seeds']], axis=0) * 100
        r['xai'] = sorted(zip(FEAT, imp.tolist()), key=lambda t: -t[1])
        ad = np.sum([s['xai']['greedy_action_dist'] for s in d['seeds']], axis=0)
        r['actdist'] = list(zip(ACT, (ad / ad.sum() * 100).tolist()))
    R[m] = r
ab = np.array([e['alpha'] for e in load('base')[2]]); an = np.array([e['alpha'] for e in load('nosent')[2]])
t, p = stats.ttest_ind(ab, an, equal_var=False)
R['ablation_t'] = (float(t), float(p)); R['ablation_diff'] = float(ab.mean() - an.mean())
json.dump(R, open(D + 'summary.json', 'w'), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
