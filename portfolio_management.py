"""
Portfolio Management for Alpha Generation System
=================================================
Reads the alpha strategies CSV and applies:
  1. Strategy scoring & filtering
  2. Mean-Variance Optimization (Efficient Frontier)
  3. CVaR Optimization (tail-risk aware allocation)
  4. Portfolio weight output to CSV
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')   # headless — swap to 'TkAgg' if you want interactive plots
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. LOAD & INSPECT
# ─────────────────────────────────────────────

CSV_PATH = r'C:\Users\vaibh\OneDrive\Desktop\ \MultiStrategyGenerator\results\strategy_results_60m_top500_lookback_commission_standard_new_data_btcusdt_updated_validated.csv'   # <-- change to your actual file path

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("RAW DATA")
print("=" * 60)
print(df.to_string(index=False))
print(f"\nShape : {df.shape[0]} strategies x {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")

# ─────────────────────────────────────────────
# 1. FILTER — KEEP STRATEGIES WITH POSITIVE
#    TEST RETURNS AND ACCEPTABLE DRAWDOWN
# ─────────────────────────────────────────────

filtered = df[
    (df['test_return']   > 0)   &   # must be profitable on test set
    (df['test_drawdown'] > -20) &   # max drawdown not worse than -20 %
    (df['test_trades']   >= 5)      # at least 5 trades (not a fluke)
].copy()

print("\n" + "=" * 60)
print(f"AFTER FILTERING : {len(filtered)} / {len(df)} strategies pass")
print("=" * 60)
print(filtered[['id','direction','test_return','test_sharpe',
                 'test_drawdown','test_trades','test_winrate','test_score']].to_string(index=False))

if len(filtered) < 2:
    print("\n[WARNING] Fewer than 2 strategies pass filters — "
          "loosening criteria (keeping top 5 by test_score).")
    filtered = df.nlargest(5, 'test_score').copy()

# ─────────────────────────────────────────────
# 2. BUILD RETURNS MATRIX
#    rows = strategies, columns = metrics used
#    as "scenario" proxies for optimization
# ─────────────────────────────────────────────

# We treat each strategy's (train_return, test_return) as two
# "scenario" observations — a simple but principled approach
# when you have no time-series of daily P&L.

returns_matrix = filtered[['train_return', 'test_return']].values.T  # (2 scenarios x n_strats)
strategy_ids   = filtered['id'].tolist()
n              = len(strategy_ids)

print(f"\nReturns matrix shape : {returns_matrix.shape}  "
      f"(2 scenarios x {n} strategies)")

# ─────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def portfolio_performance(weights, returns_matrix):
    """Expected return and risk (std dev) of a weighted portfolio.
    returns_matrix : (n_scenarios x n_strategies)
    weights        : (n_strategies,)
    result         : scalar ret, scalar risk
    """
    scenario_returns = returns_matrix @ weights          # (n_scenarios,)
    ret  = np.mean(scenario_returns)
    risk = np.std(scenario_returns, ddof=1) if len(scenario_returns) > 1 else 0.0
    return ret, risk


def neg_sharpe(weights, returns_matrix, rf=0.0):
    ret, risk = portfolio_performance(weights, returns_matrix)
    if risk == 0:
        return 0.0
    return -(ret - rf) / risk


def calculate_cvar(weights, returns_matrix, confidence=0.95):
    """Returns *positive* CVaR (loss magnitude) for minimization."""
    scenario_returns = returns_matrix @ weights
    var_threshold    = np.percentile(scenario_returns, (1 - confidence) * 100)
    tail             = scenario_returns[scenario_returns <= var_threshold]
    cvar             = tail.mean() if len(tail) > 0 else var_threshold
    return -cvar     # negate so minimizing = minimizing tail loss


def portfolio_return_scalar(weights, returns_matrix):
    return np.mean(returns_matrix @ weights)


# Shared optimization setup
constraints  = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds       = tuple((0, 1) for _ in range(n))
init_weights = np.array([1 / n] * n)

# ─────────────────────────────────────────────
# 4. MEAN-VARIANCE OPTIMIZATION
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("MEAN-VARIANCE OPTIMIZATION")
print("=" * 60)

opt_sharpe   = minimize(neg_sharpe,   init_weights,
                        args=(returns_matrix,),
                        method='SLSQP', bounds=bounds, constraints=constraints)

opt_variance = minimize(lambda w: portfolio_performance(w, returns_matrix)[1],
                        init_weights,
                        method='SLSQP', bounds=bounds, constraints=constraints)


def print_mv_result(label, weights, returns_matrix):
    ret, risk = portfolio_performance(weights, returns_matrix)
    sharpe    = (ret / risk) if risk != 0 else 0
    print(f"\n  [{label}]")
    for sid, w in zip(strategy_ids, weights):
        if w > 0.001:
            print(f"    {sid[:8]}  {w * 100:6.2f}%")
    print(f"    Expected Return : {ret:.4f}%")
    print(f"    Risk (Std Dev)  : {risk:.4f}%")
    print(f"    Sharpe Ratio    : {sharpe:.4f}")
    return ret, risk, sharpe, weights


mv_sharpe_ret,   mv_sharpe_risk,   mv_sharpe_sr,   mv_sharpe_w   = print_mv_result("Max Sharpe",   opt_sharpe.x,   returns_matrix)
mv_variance_ret, mv_variance_risk, mv_variance_sr, mv_variance_w = print_mv_result("Min Variance", opt_variance.x, returns_matrix)

# Efficient Frontier
frontier_returns, frontier_risks = [], []
mean_rets = np.mean(returns_matrix, axis=0)   # per-strategy mean, shape (n,)

for target in np.linspace(mean_rets.min(), mean_rets.max(), 80):
    cons_ef = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, t=target: portfolio_performance(w, returns_matrix)[0] - t}
    ]
    res = minimize(lambda w: portfolio_performance(w, returns_matrix)[1],
                   init_weights, method='SLSQP', bounds=bounds, constraints=cons_ef)
    if res.success:
        r, s = portfolio_performance(res.x, returns_matrix)
        frontier_returns.append(r)
        frontier_risks.append(s)

# ─────────────────────────────────────────────
# 5. CVaR OPTIMIZATION
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("CVaR OPTIMIZATION (95% confidence)")
print("=" * 60)

opt_cvar = minimize(calculate_cvar, init_weights,
                    args=(returns_matrix, 0.95),
                    method='SLSQP', bounds=bounds, constraints=constraints)

cvar_limit = 5.0   # max acceptable CVaR loss %

opt_return_cvar = minimize(
    lambda w: -portfolio_return_scalar(w, returns_matrix),
    init_weights,
    args=(),
    method='SLSQP',
    bounds=bounds,
    constraints=[
        {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: cvar_limit - calculate_cvar(w, returns_matrix, 0.95)}
    ]
)


def print_cvar_result(label, weights, returns_matrix, confidence=0.95):
    sc_rets = returns_matrix @ weights
    var     = np.percentile(sc_rets, (1 - confidence) * 100)
    tail    = sc_rets[sc_rets <= var]
    cvar    = tail.mean() if len(tail) > 0 else var
    ret     = portfolio_return_scalar(weights, returns_matrix)
    print(f"\n  [{label}]")
    for sid, w in zip(strategy_ids, weights):
        if w > 0.001:
            print(f"    {sid[:8]}  {w * 100:6.2f}%")
    print(f"    Expected Return : {ret:.4f}%")
    print(f"    VaR  (95%)      : -{abs(var):.4f}%")
    print(f"    CVaR (95%)      : -{abs(cvar):.4f}%")
    return ret, abs(var), abs(cvar), weights


cvar_min_ret, cvar_min_var, cvar_min_cvar, cvar_min_w   = print_cvar_result("Min CVaR",             opt_cvar.x,         returns_matrix)
cvar_ret_ret, cvar_ret_var, cvar_ret_cvar, cvar_ret_w   = print_cvar_result("Max Return (CVaR cap)", opt_return_cvar.x,  returns_matrix)

# CVaR across confidence levels
print("\n  [CVaR sensitivity across confidence levels]")
cvar_rows = []
for conf in [0.90, 0.95, 0.99]:
    opt = minimize(calculate_cvar, init_weights, args=(returns_matrix, conf),
                   method='SLSQP', bounds=bounds, constraints=constraints)
    sc_rets = returns_matrix @ opt.x
    var  = np.percentile(sc_rets, (1 - conf) * 100)
    tail = sc_rets[sc_rets <= var]
    cvar = abs(tail.mean()) if len(tail) > 0 else abs(var)
    ret  = portfolio_return_scalar(opt.x, returns_matrix)
    row  = {'Confidence': f'{int(conf*100)}%', 'Return_%': round(ret, 4),
            'VaR_%': round(abs(var), 4), 'CVaR_%': round(cvar, 4)}
    for sid, w in zip(strategy_ids, opt.x):
        row[f'w_{sid[:8]}'] = round(w * 100, 2)
    cvar_rows.append(row)

cvar_conf_df = pd.DataFrame(cvar_rows)
print(cvar_conf_df.to_string(index=False))

# ─────────────────────────────────────────────
# 6. SCORE-WEIGHTED ALLOCATION
#    Simple benchmark: weight by test_score
# ─────────────────────────────────────────────

scores       = filtered['test_score'].values
score_weights = np.maximum(scores, 0)          # clip negatives to 0
if score_weights.sum() > 0:
    score_weights = score_weights / score_weights.sum()
else:
    score_weights = init_weights

sw_ret, sw_risk = portfolio_performance(score_weights, returns_matrix)
print("\n" + "=" * 60)
print("SCORE-WEIGHTED ALLOCATION (benchmark)")
print("=" * 60)
for sid, w in zip(strategy_ids, score_weights):
    print(f"  {sid[:8]}  {w * 100:6.2f}%")
print(f"  Expected Return : {sw_ret:.4f}%")
print(f"  Risk (Std Dev)  : {sw_risk:.4f}%")

# ─────────────────────────────────────────────
# 7. EXPORT RESULTS TO CSV
# ─────────────────────────────────────────────

output_rows = []
allocations = {
    'MaxSharpe_MV'      : mv_sharpe_w,
    'MinVariance_MV'    : mv_variance_w,
    'MinCVaR'           : cvar_min_w,
    'MaxReturn_CVaRCap' : cvar_ret_w,
    'ScoreWeighted'     : score_weights,
    'EqualWeight'       : init_weights,
}

for sid, row in filtered.iterrows():
    base = {
        'strategy_id'    : row['id'],
        'direction'      : row['direction'],
        'signals'        : row['signals'],
        'n_signals'      : row['n_signals'],
        'tp'             : round(row['tp'], 6),
        'sl'             : round(row['sl'], 6),
        'train_return'   : row['train_return'],
        'train_sharpe'   : row['train_sharpe'],
        'train_drawdown' : row['train_drawdown'],
        'train_trades'   : row['train_trades'],
        'train_winrate'  : row['train_winrate'],
        'test_return'    : row['test_return'],
        'test_sharpe'    : row['test_sharpe'],
        'test_drawdown'  : row['test_drawdown'],
        'test_trades'    : row['test_trades'],
        'test_winrate'   : row['test_winrate'],
        'test_score'     : row['test_score'],
    }
    idx = strategy_ids.index(row['id'])
    for alloc_name, weights in allocations.items():
        base[f'weight_{alloc_name}_%'] = round(weights[idx] * 100, 4)
    output_rows.append(base)

output_df = pd.DataFrame(output_rows)
output_path = r'C:\Users\vaibh\OneDrive\Desktop\ \MultiStrategyGenerator\results\portfolio_allocations.csv'
output_df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print(f"EXPORTED → {output_path}")
print("=" * 60)
print(output_df[['strategy_id', 'direction',
                  'weight_MaxSharpe_MV_%', 'weight_MinCVaR_%',
                  'weight_ScoreWeighted_%', 'weight_EqualWeight_%']].to_string(index=False))

# ─────────────────────────────────────────────
# 8. SUMMARY COMPARISON TABLE
# ─────────────────────────────────────────────

summary_rows = [
    {'Method': 'Max Sharpe (MV)',       'Return_%': round(mv_sharpe_ret,   4), 'Risk_%': round(mv_sharpe_risk,   4), 'Sharpe': round(mv_sharpe_sr,   4)},
    {'Method': 'Min Variance (MV)',     'Return_%': round(mv_variance_ret, 4), 'Risk_%': round(mv_variance_risk, 4), 'Sharpe': round(mv_variance_sr, 4)},
    {'Method': 'Min CVaR',              'Return_%': round(cvar_min_ret,    4), 'Risk_%': round(cvar_min_var,     4), 'Sharpe': '-'},
    {'Method': 'Max Return (CVaR cap)', 'Return_%': round(cvar_ret_ret,    4), 'Risk_%': round(cvar_ret_var,     4), 'Sharpe': '-'},
    {'Method': 'Score-Weighted',        'Return_%': round(sw_ret,          4), 'Risk_%': round(sw_risk,          4), 'Sharpe': round(sw_ret / sw_risk, 4) if sw_risk > 0 else '-'},
    {'Method': 'Equal Weight',          'Return_%': round(portfolio_performance(init_weights, returns_matrix)[0], 4),
                                         'Risk_%': round(portfolio_performance(init_weights, returns_matrix)[1], 4),
                                         'Sharpe': '-'},
]
summary_df = pd.DataFrame(summary_rows)
summary_path = r'C:\Users\vaibh\OneDrive\Desktop\ \MultiStrategyGenerator\results\portfolio_summary.csv'
summary_df.to_csv(summary_path, index=False)

print(f"\nEXPORTED → {summary_path}")
print(summary_df.to_string(index=False))

# ─────────────────────────────────────────────
# 9. PLOTS  (saved as PNGs)
# ─────────────────────────────────────────────

# --- Plot 1: Efficient Frontier ---
fig, ax = plt.subplots(figsize=(10, 6))
if frontier_risks:
    ax.plot(frontier_risks, frontier_returns, 'b-', linewidth=2, label='Efficient Frontier')

ax.scatter(mv_sharpe_risk,   mv_sharpe_ret,   color='red',   s=120, zorder=5,
           label=f'Max Sharpe (ret={mv_sharpe_ret:.2f}%)')
ax.scatter(mv_variance_risk, mv_variance_ret, color='green', s=120, zorder=5,
           label=f'Min Variance (ret={mv_variance_ret:.2f}%)')
ax.scatter(sw_risk,          sw_ret,          color='purple', s=120, zorder=5, marker='D',
           label=f'Score-Weighted (ret={sw_ret:.2f}%)')

# Individual strategies
strat_rets  = np.mean(returns_matrix, axis=0)   # mean across scenarios per strategy
strat_risks = np.std(returns_matrix,  axis=0, ddof=1)
for i, sid in enumerate(strategy_ids):
    ax.scatter(strat_risks[i], strat_rets[i], s=60, zorder=4, alpha=0.6)
    ax.annotate(sid[:8], (strat_risks[i], strat_rets[i]),
                textcoords='offset points', xytext=(5, 4), fontsize=7)

ax.set_xlabel('Risk (Std Dev %)')
ax.set_ylabel('Return %')
ax.set_title('Efficient Frontier — Alpha Strategy Allocation')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\vaibh\OneDrive\Desktop\ \MultiStrategyGenerator\results\efficient_frontier.png', dpi=150)
plt.close()
print("\nEXPORTED → efficient_frontier.png")

# --- Plot 2: Portfolio Weights Comparison ---
alloc_labels  = list(allocations.keys())
weight_matrix = np.array([allocations[k] for k in alloc_labels]) * 100   # (methods x strategies)

fig, ax = plt.subplots(figsize=(12, 5))
x       = np.arange(len(alloc_labels))
width   = 0.8 / n
colors  = plt.cm.tab10(np.linspace(0, 1, n))

for i, sid in enumerate(strategy_ids):
    ax.bar(x + i * width - (n - 1) * width / 2,
           weight_matrix[:, i], width, label=sid[:8], color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(alloc_labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Weight %')
ax.set_title('Strategy Weights Across Allocation Methods')
ax.legend(fontsize=8, ncol=min(n, 5))
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\vaibh\OneDrive\Desktop\ \MultiStrategyGenerator\results\portfolio_weights.png', dpi=150)
plt.close()
print("EXPORTED → portfolio_weights.png")

# --- Plot 3: Risk-Return scatter of strategies ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sc = axes[0].scatter(filtered['test_drawdown'].abs(), filtered['test_return'],
                     c=filtered['test_score'], cmap='RdYlGn', s=80, zorder=3)
for _, row in filtered.iterrows():
    axes[0].annotate(row['id'][:8],
                     (abs(row['test_drawdown']), row['test_return']),
                     textcoords='offset points', xytext=(4, 3), fontsize=7)
plt.colorbar(sc, ax=axes[0], label='test_score')
axes[0].set_xlabel('Max Drawdown % (abs)')
axes[0].set_ylabel('Test Return %')
axes[0].set_title('Drawdown vs Return (color = score)')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(filtered['test_trades'], filtered['test_winrate'],
                c=filtered['test_return'], cmap='RdYlGn', s=80, zorder=3)
for _, row in filtered.iterrows():
    axes[1].annotate(row['id'][:8],
                     (row['test_trades'], row['test_winrate']),
                     textcoords='offset points', xytext=(4, 3), fontsize=7)
axes[1].set_xlabel('Test Trades')
axes[1].set_ylabel('Win Rate %')
axes[1].set_title('Trades vs Win Rate (color = return)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\vaibh\OneDrive\Desktop\ \MultiStrategyGenerator\results\strategy_scatter.png', dpi=150)
plt.close()
print("EXPORTED → strategy_scatter.png")

print("\n" + "=" * 60)
print("ALL DONE")
print(f"  portfolio_allocations.csv  — per-strategy weights")
print(f"  portfolio_summary.csv      — method comparison")
print(f"  efficient_frontier.png")
print(f"  portfolio_weights.png")
print(f"  strategy_scatter.png")
print("=" * 60)