"""
Step 5: Performance Comparison & Final Charts.

Compares: Spot / M1 / M3 / M6 / UNG
Outputs: annual returns, term structure, summary table
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR, FIG_DIR, TABLE_DIR, setup_matplotlib, save_fig
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def run():
    print('=' * 60)
    print('Step 5: Performance Comparison (UNG / NG)')
    print('=' * 60)

    setup_matplotlib()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    nav_df = pd.read_csv(os.path.join(DATA_DIR, 'nav_strategies.csv'),
                         index_col=0, parse_dates=True)

    cols = [c for c in ['Spot', 'M1', 'M3', 'M6', 'UNG'] if c in nav_df.columns]
    nav_df = nav_df[cols]

    daily_ret = nav_df.pct_change().dropna()
    n_years = len(nav_df) / 252

    # ═══════════════════════════════
    # Annual returns
    # ═══════════════════════════════
    print('\n  [1] Annual returns...')

    annual_ret = daily_ret.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    # Add TE columns
    for col in ['M1', 'M3', 'M6', 'UNG']:
        if col in annual_ret.columns:
            annual_ret[f'{col}_vs_Spot'] = annual_ret[col] - annual_ret['Spot']

    annual_ret.to_csv(os.path.join(TABLE_DIR, 'annual_returns.csv'))
    print(annual_ret[cols].round(4).to_string())

    # ── Chart: Annual returns bar ──
    labels_map = {'Spot': 'Spot', 'M1': 'M1', 'M3': 'M3', 'M6': 'M6', 'UNG': 'UNG'}
    colors_map = {'Spot': '#E91E63', 'M1': '#4CAF50', 'M3': '#2196F3',
                  'M6': '#9C27B0', 'UNG': '#FF9800'}

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(annual_ret))
    width = 0.15
    n_cols = len(cols)
    for i, col in enumerate(cols):
        offset = (i - n_cols/2 + 0.5) * width
        ax.bar(x + offset, annual_ret[col] * 100, width,
               label=labels_map.get(col, col), color=colors_map.get(col, 'gray'), alpha=0.8)

    years = [str(d.year) for d in annual_ret.index]
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha='right')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_title('Annual Returns: Spot vs Roll Strategies vs UNG ETF',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Annual Return (%)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_fig(fig, '02_annual_returns.png')

    # ── Chart: Annual TE ──
    te_cols = [f'{c}_vs_Spot' for c in ['M1', 'M6', 'UNG'] if f'{c}_vs_Spot' in annual_ret.columns]
    if te_cols:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(annual_ret))
        te_colors = {'M1_vs_Spot': '#4CAF50', 'M6_vs_Spot': '#9C27B0', 'UNG_vs_Spot': '#FF9800'}
        te_labels = {'M1_vs_Spot': 'M1 vs Spot', 'M6_vs_Spot': 'M6 vs Spot', 'UNG_vs_Spot': 'UNG vs Spot'}
        width = 0.25

        for i, col in enumerate(te_cols):
            offset = (i - len(te_cols)/2 + 0.5) * width
            ax.bar(x + offset, annual_ret[col] * 100, width,
                   label=te_labels.get(col, col), color=te_colors.get(col, 'gray'), alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, ha='right')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title('Annual Tracking Error vs Spot\nNegative = Underperforming Spot',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('TE (%)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_fig(fig, '02b_tracking_deficit.png')

    # ═══════════════════════════════
    # Term Structure Snapshot
    # ═══════════════════════════════
    print('\n  [2] Term structure snapshot...')

    ts_path = os.path.join(DATA_DIR, 'term_structure.csv')
    if os.path.exists(ts_path):
        ts = pd.read_csv(ts_path)
        ts = ts.sort_values(['year', 'month_order'])

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(range(len(ts)), ts['last_close'], 'bo-', markersize=8, linewidth=2)

        for i in range(len(ts) - 1):
            color = 'red' if ts.iloc[i+1]['last_close'] > ts.iloc[i]['last_close'] else 'green'
            ax.annotate('', xy=(i+1, ts.iloc[i+1]['last_close']),
                        xytext=(i, ts.iloc[i]['last_close']),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        labels = ts['month_name'] + "'" + ts['year'].astype(str).str[-2:]
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels(labels, rotation=45, ha='right')

        front = ts.iloc[0]['last_close']
        back = ts.iloc[-1]['last_close']
        state = 'Contango' if back > front else 'Backwardation'

        ax.set_title(f'NG Futures Forward Curve (Term Structure)\n'
                     f'Market state: {state} | Front={front:.3f} vs Back={back:.3f}',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Price (USD/MMBtu)')
        ax.grid(True, alpha=0.3)
        ax.axhline(front, color='gray', linestyle=':', alpha=0.5)
        plt.tight_layout()
        save_fig(fig, '08_term_structure.png')

    # ═══════════════════════════════
    # Summary Performance Table
    # ═══════════════════════════════
    print('\n  [3] Performance summary...')

    results = []
    for col in cols:
        total = nav_df[col].iloc[-1] / nav_df[col].iloc[0] - 1
        ann = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
        vol = daily_ret[col].std() * np.sqrt(252)
        maxdd = (nav_df[col] / nav_df[col].cummax() - 1).min()
        sharpe = ann / vol if vol > 0 else 0
        corr = daily_ret[col].corr(daily_ret['Spot']) if col != 'Spot' else 1.0
        te_ann = (total - (nav_df['Spot'].iloc[-1]/nav_df['Spot'].iloc[0]-1)) / n_years if col != 'Spot' else 0

        results.append({
            'Strategy': col,
            'Total_Return': total,
            'Ann_Return': ann,
            'Ann_Volatility': vol,
            'Sharpe': sharpe,
            'Max_Drawdown': maxdd,
            'Corr_vs_Spot': corr,
            'Ann_TE_vs_Spot': te_ann,
        })

    summary = pd.DataFrame(results).set_index('Strategy')
    summary.to_csv(os.path.join(TABLE_DIR, 'performance_summary.csv'))

    name_map = {'Spot': 'Henry Hub Spot', 'M1': 'M1 (front-month)',
                'M3': 'M3 (3rd-month)', 'M6': 'M6 (6th-month)', 'UNG': 'UNG ETF'}

    print(f'\n  {"Strategy":20s} {"Total":>8s} {"Ann":>7s} {"Vol":>7s} {"Sharpe":>7s} {"MaxDD":>8s} {"rho(S)":>7s} {"TE/yr":>7s}')
    print('  ' + '-' * 78)
    for _, r in summary.iterrows():
        n = name_map.get(r.name, r.name)
        print(f'  {n:20s} {r["Total_Return"]:+7.2%} {r["Ann_Return"]:+6.2%} '
              f'{r["Ann_Volatility"]:6.2%} {r["Sharpe"]:+6.3f} {r["Max_Drawdown"]:+7.2%} '
              f'{r["Corr_vs_Spot"]:6.4f} {r["Ann_TE_vs_Spot"]:+6.2%}')

    # ── Conclusions ──
    print('\n  [Conclusions]')
    print('  1. TE is driven by contango roll cost, NOT hedge ratio design')
    print('  2. h* = 0.43 but R2 = 3.3% -> futures barely explain spot moves')
    print('  3. UNG uses h=1 by design (directional exposure, not min-variance)')
    print('  4. M6 strategy reduces TE vs M1 (less contango drag at longer tenors)')
    print('  5. OLS regression: Roll_cost and Basis explain 42.5% of monthly TE')

    print('\nStep 5 complete.')


if __name__ == '__main__':
    run()
