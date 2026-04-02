"""
Step 3: Front-Month Replication Strategy Simulation.

Strategy: hold front-month natural gas futures (NG=F) with 1:1 hedge ratio,
50% idle cash in SHY, daily fee deduction (1.11% / 250 per day).

Compared against:
- Real spot: Henry Hub daily spot (FRED DHHNGSP, USD/MMBtu)
- Actual ETF: UNG (United States Natural Gas Fund)

All series are daily — no monthly resampling needed since DHHNGSP is daily.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (DATA_DIR, FIG_DIR, TABLE_DIR, setup_matplotlib,
                       simulate_nav, save_fig, performance_summary)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HEDGE_RATIO  = 1.0
BOND_WEIGHT  = 0.5
ANNUAL_FEE   = 0.0111    # UNG expense ratio: 1.11%
ROLL_COST    = 0.0003    # per roll event (~0.03%, estimated)
INITIAL_NAV  = 100.0


def run():
    print('=' * 60)
    print('Step 3: Front-Month Replication Simulation (UNG / NG)')
    print('=' * 60)

    setup_matplotlib()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    # ── Load daily data ──
    daily_prices  = pd.read_csv(os.path.join(DATA_DIR, 'prices_daily.csv'),
                                index_col=0, parse_dates=True)
    daily_returns = pd.read_csv(os.path.join(DATA_DIR, 'returns_daily.csv'),
                                index_col=0, parse_dates=True)

    # ── Simulate front-month NAV (daily) ──
    print('\n  Simulating front-month replication (daily)...')
    daily_fee = ANNUAL_FEE / 250
    nav_values = [INITIAL_NAV]
    for i in range(len(daily_returns)):
        r_f = daily_returns['NG_F_ret'].iloc[i]
        r_b = daily_returns['SHY_ret'].iloc[i]
        daily_r = HEDGE_RATIO * r_f + BOND_WEIGHT * r_b - daily_fee
        nav_values.append(nav_values[-1] * (1 + daily_r))

    nav_sim = pd.Series(nav_values[1:], index=daily_returns.index, name='Front_Month_Sim')

    # ── Normalize UNG and Spot to same starting NAV ──
    ung_nav  = daily_prices['UNG']  / daily_prices['UNG'].iloc[0]  * INITIAL_NAV
    spot_nav = daily_prices['Spot'] / daily_prices['Spot'].iloc[0] * INITIAL_NAV

    # ── Build unified daily NAV DataFrame ──
    nav_df = pd.DataFrame({
        'Spot':            spot_nav,
        'Front_Month_Sim': nav_sim,
        'UNG_ETF':         ung_nav
    }).dropna()

    # ── Save ──
    nav_path = os.path.join(DATA_DIR, 'nav_series.csv')
    nav_df.to_csv(nav_path)
    print(f'  Saved daily NAV: {nav_path} ({len(nav_df)} trading days)')

    # ── Correlations ──
    daily_ret = nav_df.pct_change().dropna()
    print(f'\n  Daily return correlations:')
    print(f'    Front-month sim vs Spot:  {daily_ret["Front_Month_Sim"].corr(daily_ret["Spot"]):.4f}')
    print(f'    UNG ETF vs Spot:          {daily_ret["UNG_ETF"].corr(daily_ret["Spot"]):.4f}')
    print(f'    Front-month sim vs UNG:   {daily_ret["Front_Month_Sim"].corr(daily_ret["UNG_ETF"]):.4f}')

    # ── Performance ──
    print(f'\n  Performance (start={nav_df.index[0].date()}, end={nav_df.index[-1].date()}):')
    for col in nav_df.columns:
        p = performance_summary(nav_df[col], col)
        total = nav_df[col].iloc[-1] / nav_df[col].iloc[0] - 1
        ann   = p['annualized_return']
        vol   = p['annualized_volatility']
        mdd   = p['max_drawdown']
        print(f'    {col:20s}: total={total:+.2%}, ann={ann:+.2%}, vol={vol:.2%}, maxDD={mdd:+.2%}')

    # ── Annual returns table ──
    annual_ret = daily_ret.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    annual_ret['Sim_vs_Spot'] = annual_ret['Front_Month_Sim'] - annual_ret['Spot']
    annual_ret['UNG_vs_Spot'] = annual_ret['UNG_ETF']         - annual_ret['Spot']

    ann_path = os.path.join(TABLE_DIR, 'annual_returns.csv')
    annual_ret.to_csv(ann_path)
    print(f'\n  Annual returns saved: {ann_path}')
    print(annual_ret[['Spot', 'Front_Month_Sim', 'UNG_ETF',
                       'Sim_vs_Spot', 'UNG_vs_Spot']].round(4).to_string())

    # ── Plot 1: NAV comparison (log scale) ──
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(nav_df.index, nav_df['Spot'],
            label='天然氣現貨 (Henry Hub, DHHNGSP)', color='#E91E63', linewidth=2)
    ax.plot(nav_df.index, nav_df['Front_Month_Sim'],
            label='近月期貨複製策略 (模擬)', color='#4CAF50', linewidth=1.5)
    ax.plot(nav_df.index, nav_df['UNG_ETF'],
            label='UNG ETF (實際)', color='#FF9800', linewidth=1.5)

    ax.axhline(100, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_yscale('log')
    ax.set_title('UNG ETF 複製策略績效比較圖\n'
                 '現貨 = Henry Hub 天然氣現貨 (EIA/FRED DHHNGSP, USD/MMBtu)\n'
                 f'起始淨值=100，對數刻度',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('淨值 (對數刻度)', fontsize=12)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    for col, color in [('Spot', '#E91E63'), ('Front_Month_Sim', '#4CAF50'), ('UNG_ETF', '#FF9800')]:
        v = nav_df[col].iloc[-1]
        ax.annotate(f'{v:.1f}', xy=(nav_df.index[-1], v),
                    fontsize=10, color=color, fontweight='bold',
                    xytext=(10, 0), textcoords='offset points')

    save_fig(fig, '01_nav_comparison.png')

    # ── Plot 1b: Linear scale (zoomed last 5 years) ──
    recent = nav_df[nav_df.index >= nav_df.index[-1] - pd.DateOffset(years=5)]
    # Re-normalize to 100 at the start of this window
    recent_norm = recent.divide(recent.iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(recent_norm.index, recent_norm['Spot'],
            label='天然氣現貨 (Henry Hub)', color='#E91E63', linewidth=2)
    ax.plot(recent_norm.index, recent_norm['Front_Month_Sim'],
            label='近月期貨複製策略', color='#4CAF50', linewidth=1.5)
    ax.plot(recent_norm.index, recent_norm['UNG_ETF'],
            label='UNG ETF (實際)', color='#FF9800', linewidth=1.5)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title('近5年績效比較 (線性刻度)', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('淨值')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    save_fig(fig, '01b_nav_recent5yr.png')

    print('\nFront-month simulation complete.')


if __name__ == '__main__':
    run()
