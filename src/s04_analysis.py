"""
Step 4: Tracking Error Analysis.

Part A: Hedge ratio (h*) — h* = rho * sigma_S / sigma_F
Part B: Basis analysis (daily Spot vs Futures)
Part C: Tracking error decomposition (roll yield + fees + residual)
Part D: OLS regression (what explains monthly TE?)
Part E: Seasonal analysis (which months have worst contango?)
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (DATA_DIR, FIG_DIR, TABLE_DIR, setup_matplotlib, save_fig,
                       minimum_variance_hedge_ratio, rolling_hedge_ratio)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def run():
    print('=' * 60)
    print('Step 4: Tracking Error Analysis (UNG / NG)')
    print('=' * 60)

    setup_matplotlib()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    daily_prices  = pd.read_csv(os.path.join(DATA_DIR, 'prices_daily.csv'),
                                index_col=0, parse_dates=True)
    daily_returns = pd.read_csv(os.path.join(DATA_DIR, 'returns_daily.csv'),
                                index_col=0, parse_dates=True)
    nav_df        = pd.read_csv(os.path.join(DATA_DIR, 'nav_strategies.csv'),
                                index_col=0, parse_dates=True)
    roll_costs    = pd.read_csv(os.path.join(DATA_DIR, 'roll_costs.csv'),
                                parse_dates=['roll_date'])

    # ═══════════════════════════════════════════════
    # Part A: Hedge Ratio h*
    # ═══════════════════════════════════════════════
    print('\n  [A] Hedge Ratio Analysis')

    spot_ret = daily_returns['Spot_ret']
    fut_ret  = daily_returns['NG_F_ret']

    # Method 1: formula
    rho = spot_ret.corr(fut_ret)
    sigma_s = spot_ret.std()
    sigma_f = fut_ret.std()
    h_star = rho * (sigma_s / sigma_f)
    r_squared = rho ** 2

    # Method 2: OLS regression (verification)
    slope, intercept, r_val, p_val, std_err = sp_stats.linregress(fut_ret, spot_ret)

    print(f'  Method 1 (formula): h*={h_star:.4f}, rho={rho:.4f}, R2={r_squared:.4f}')
    print(f'  Method 2 (OLS):     h*={slope:.4f}, R2={r_val**2:.4f}, p={p_val:.2e}')
    print(f'  Methods match: {abs(h_star - slope) < 0.001}')
    print(f'  sigma_S (ann): {sigma_s * np.sqrt(252):.1%}')
    print(f'  sigma_F (ann): {sigma_f * np.sqrt(252):.1%}')
    print(f'  -> R2={r_squared:.1%}: futures explain only {r_squared:.1%} of spot daily variance')

    # Rolling h*
    h_60d  = rolling_hedge_ratio(spot_ret, fut_ret, window=60)
    h_252d = rolling_hedge_ratio(spot_ret, fut_ret, window=252)

    # ── Plot: Rolling h* + R2 ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.plot(h_60d.index, h_60d.values, 'b-', alpha=0.4, linewidth=0.8, label='60-day rolling h*')
    ax.plot(h_252d.index, h_252d.values, 'r-', linewidth=1.2, label='252-day rolling h*')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=0.8, label='h=1.0 (UNG uses this)')
    ax.axhline(h_star, color='green', linestyle=':', linewidth=1,
               label=f'Full-period h*={h_star:.3f}')
    ax.set_title(f'Minimum Variance Hedge Ratio h* (Spot vs NG=F, daily)\n'
                 f'Full period: h*={h_star:.4f}, rho={rho:.4f}, R2={r_squared:.4f}',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('h*')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    ax = axes[1]
    rolling_r2 = spot_ret.rolling(252).corr(fut_ret) ** 2
    ax.fill_between(rolling_r2.index, rolling_r2.values, alpha=0.3, color='purple')
    ax.plot(rolling_r2.index, rolling_r2.values, 'purple', linewidth=1)
    ax.axhline(r_squared, color='green', linestyle=':', label=f'Full-period R2={r_squared:.4f}')
    ax.set_title('Rolling Hedge Effectiveness R2 (252-day)', fontsize=13, fontweight='bold')
    ax.set_ylabel('R2')
    ax.set_xlabel('Date')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.3, rolling_r2.max() * 1.1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    save_fig(fig, '05_hedge_ratio.png')

    # ═══════════════════════════════════════════════
    # Part B: Basis Analysis
    # ═══════════════════════════════════════════════
    print('\n  [B] Basis Analysis (NG=F vs Henry Hub Spot)')

    basis     = daily_prices['NG_F'] - daily_prices['Spot']
    basis_pct = basis / daily_prices['Spot'] * 100

    n_contango = (basis > 0).sum()
    print(f'  Avg basis: {basis.mean():.3f} USD ({basis_pct.mean():.1f}%)')
    print(f'  Contango: {n_contango}/{len(basis)} days ({n_contango/len(basis):.1%})')

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    ax.plot(daily_prices.index, daily_prices['Spot'], label='Spot (Henry Hub)', color='#E91E63', linewidth=1)
    ax.plot(daily_prices.index, daily_prices['NG_F'], label='Front-month Futures (NG=F)', color='#2196F3', linewidth=1)
    ax.set_title('Natural Gas: Spot vs Front-Month Futures', fontsize=14)
    ax.set_ylabel('Price (USD/MMBtu)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    ax = axes[1]
    basis_mo = basis_pct.resample('MS').mean()
    colors_b = ['red' if b > 0 else 'green' for b in basis_mo.values]
    ax.bar(basis_mo.index, basis_mo.values, width=25, color=colors_b, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(basis_pct.mean(), color='blue', linestyle='--',
               label=f'Average = {basis_pct.mean():.1f}%')
    ax.set_title('Monthly Average Basis (Futures - Spot) / Spot (%)\n'
                 'Red = Contango (roll loss) | Green = Backwardation (roll gain)',
                 fontsize=13)
    ax.set_ylabel('Basis (%)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, '03_basis_analysis.png')

    # ═══════════════════════════════════════════════
    # Part C: Tracking Error Decomposition
    # ═══════════════════════════════════════════════
    print('\n  [C] Tracking Error Decomposition')

    n_years = len(nav_df) / 252

    for col in ['M1', 'M6', 'UNG']:
        if col not in nav_df.columns:
            continue
        total = nav_df[col].iloc[-1] / nav_df[col].iloc[0] - 1
        spot_total = nav_df['Spot'].iloc[-1] / nav_df['Spot'].iloc[0] - 1
        te = total - spot_total
        print(f'  {col:4s}: total={total:+.2%}, spot={spot_total:+.2%}, TE={te:+.2%} ({te/n_years:+.2%}/yr)')

    # Roll yield contribution
    if len(roll_costs) > 0:
        total_roll = roll_costs['roll_cost_pct'].sum()
        fee_total = 0.0111 * n_years
        print(f'\n  TE Decomposition (M1 strategy):')
        print(f'    Roll yield (sum of all rolls):  {total_roll:+.2%}')
        print(f'    Management fee (1.11% x {n_years:.1f}yr): -{fee_total:.2%}')
        print(f'    -> Roll cost is the dominant factor')

    # Cumulative TE chart
    nav_ret = nav_df.pct_change().dropna()
    fig, ax = plt.subplots(figsize=(14, 6))

    for col, color, label in [('M1', '#4CAF50', 'M1 vs Spot'),
                               ('M6', '#9C27B0', 'M6 vs Spot'),
                               ('UNG', '#FF9800', 'UNG vs Spot')]:
        if col not in nav_ret.columns:
            continue
        cum_te = (nav_ret[col] - nav_ret['Spot']).cumsum() * 100
        ax.plot(cum_te.index, cum_te.values, color=color, linewidth=1.3, label=label)

    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_title('Cumulative Tracking Error vs Spot (daily return gaps)\n'
                 'Negative = underperforming spot', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative TE (percentage points)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    save_fig(fig, '05b_cumulative_tracking_error.png')

    # ═══════════════════════════════════════════════
    # Part D: OLS Regression — What explains monthly TE?
    # ═══════════════════════════════════════════════
    print('\n  [D] OLS Regression: Monthly TE Explanatory Variables')

    # Build monthly dataset
    monthly_nav = nav_df.resample('ME').last()
    monthly_ret = monthly_nav.pct_change().dropna()

    if 'M1' in monthly_ret.columns:
        monthly_te = monthly_ret['M1'] - monthly_ret['Spot']
        monthly_te.name = 'TE'

        # Monthly basis (average)
        monthly_basis = basis_pct.resample('ME').mean()
        monthly_basis.name = 'Basis_pct'

        # Monthly spot volatility
        monthly_vol = daily_returns['Spot_ret'].resample('ME').std() * np.sqrt(21)
        monthly_vol.name = 'Spot_vol'

        # Monthly roll cost (sum of rolls in that month)
        if len(roll_costs) > 0:
            rc_monthly = roll_costs.set_index('roll_date')['roll_cost_pct'].resample('ME').sum()
            rc_monthly.name = 'Roll_cost'
        else:
            rc_monthly = pd.Series(0, index=monthly_te.index, name='Roll_cost')

        # Contango dummy
        contango_dummy = (monthly_basis > 0).astype(int)
        contango_dummy.name = 'Contango'

        # Merge
        reg_df = pd.DataFrame({
            'TE': monthly_te,
            'Roll_cost': rc_monthly,
            'Basis_pct': monthly_basis,
            'Spot_vol': monthly_vol,
            'Contango': contango_dummy,
        }).dropna()

        if len(reg_df) >= 20:
            # OLS
            X = reg_df[['Roll_cost', 'Basis_pct', 'Spot_vol', 'Contango']]
            X = X.assign(const=1)
            y = reg_df['TE']

            # Manual OLS
            XtX_inv = np.linalg.inv(X.values.T @ X.values)
            beta = XtX_inv @ X.values.T @ y.values
            y_hat = X.values @ beta
            residuals = y.values - y_hat
            sse = residuals @ residuals
            sst = ((y - y.mean()) ** 2).sum()
            r2 = 1 - sse / sst
            n_obs = len(y)
            k = X.shape[1]
            se = np.sqrt(np.diag(sse / (n_obs - k) * XtX_inv))
            t_stats = beta / se

            print(f'  OLS Results (n={n_obs}, R2={r2:.4f}):')
            print(f'  {"Variable":12s} {"Coeff":>10s} {"Std Err":>10s} {"t-stat":>8s}')
            print('  ' + '-' * 45)
            for name, b, s, t in zip(X.columns, beta, se, t_stats):
                sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.64 else ''
                print(f'  {name:12s} {b:+10.4f} {s:10.4f} {t:+7.2f} {sig}')

            # Save regression table
            reg_result = pd.DataFrame({
                'Variable': X.columns,
                'Coefficient': beta,
                'Std_Error': se,
                't_stat': t_stats,
            })
            reg_result.to_csv(os.path.join(TABLE_DIR, 'ols_te_regression.csv'), index=False)

            # ── Scatter: TE vs Roll Cost ──
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            ax = axes[0]
            ax.scatter(reg_df['Roll_cost']*100, reg_df['TE']*100,
                       alpha=0.5, s=30, color='#2196F3')
            # Regression line
            x_range = np.linspace(reg_df['Roll_cost'].min(), reg_df['Roll_cost'].max(), 50)
            slope_rc, int_rc, _, _, _ = sp_stats.linregress(reg_df['Roll_cost'], reg_df['TE'])
            ax.plot(x_range*100, (int_rc + slope_rc*x_range)*100, 'r--', linewidth=1.5)
            ax.set_title(f'Monthly TE vs Roll Cost\nSlope={slope_rc:.2f}, Simple R2={(reg_df["Roll_cost"].corr(reg_df["TE"]))**2:.3f}',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Monthly Roll Cost (%)')
            ax.set_ylabel('Monthly Tracking Error (%)')
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.scatter(reg_df['Basis_pct'], reg_df['TE']*100,
                       alpha=0.5, s=30, color='#FF9800')
            slope_b, int_b, _, _, _ = sp_stats.linregress(reg_df['Basis_pct'], reg_df['TE'])
            x_range2 = np.linspace(reg_df['Basis_pct'].min(), reg_df['Basis_pct'].max(), 50)
            ax.plot(x_range2, (int_b + slope_b*x_range2)*100, 'r--', linewidth=1.5)
            ax.set_title(f'Monthly TE vs Basis Level\nSlope={slope_b:.4f}',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Average Monthly Basis (%)')
            ax.set_ylabel('Monthly Tracking Error (%)')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_fig(fig, '07_te_regression.png')
        else:
            print(f'  Not enough data for regression ({len(reg_df)} months)')

    # ═══════════════════════════════════════════════
    # Part E: Seasonal Analysis
    # ═══════════════════════════════════════════════
    print('\n  [E] Seasonal Analysis')

    if len(roll_costs) > 0:
        seasonal_rc = roll_costs.groupby('roll_month')['roll_cost_pct'].agg(['mean', 'std', 'count'])
        month_labels = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                        7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        seasonal_rc.index = seasonal_rc.index.map(lambda m: month_labels.get(m, str(m)))

        print('  Roll cost by month:')
        for idx, row in seasonal_rc.iterrows():
            print(f'    {idx}: mean={row["mean"]*100:+.2f}%, n={row["count"]:.0f}')

        fig, ax = plt.subplots(figsize=(11, 6))
        colors_s = ['red' if m > 0 else 'green' for m in seasonal_rc['mean']]
        ax.bar(seasonal_rc.index, seasonal_rc['mean']*100,
               color=colors_s, alpha=0.75, edgecolor='black', linewidth=0.5)
        ax.errorbar(seasonal_rc.index, seasonal_rc['mean']*100,
                    yerr=seasonal_rc['std']*100, fmt='none', color='black',
                    capsize=4, linewidth=0.8)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title('Average Roll Cost by Calendar Month\n'
                     'Red = Contango (loss) | Green = Backwardation (gain)',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Roll Cost (%)')
        ax.set_xlabel('Roll Month')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_fig(fig, '04_seasonal_roll_cost.png')

        seasonal_rc.to_csv(os.path.join(TABLE_DIR, 'seasonal_roll_cost.csv'))

    # Seasonal basis
    basis_by_mo = basis_pct.groupby(basis_pct.index.month).agg(['mean', 'std'])
    basis_by_mo.index = basis_by_mo.index.map(lambda m: month_labels.get(m, str(m)))

    fig, ax = plt.subplots(figsize=(11, 6))
    colors_bm = ['red' if m > 0 else 'green' for m in basis_by_mo['mean']]
    ax.bar(basis_by_mo.index, basis_by_mo['mean'], color=colors_bm, alpha=0.75,
           edgecolor='black', linewidth=0.5)
    ax.errorbar(basis_by_mo.index, basis_by_mo['mean'], yerr=basis_by_mo['std'],
                fmt='none', color='black', capsize=4, linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_title('Average Basis by Calendar Month (Futures - Spot) / Spot %',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Basis (%)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_fig(fig, '04b_seasonal_basis.png')

    print('\nStep 4 complete.')


if __name__ == '__main__':
    run()
