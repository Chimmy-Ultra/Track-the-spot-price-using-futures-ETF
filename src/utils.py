"""
Shared utility functions for UNG Natural Gas ETF replication project.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── paths ──
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLE_DIR = os.path.join(OUTPUT_DIR, 'tables')

# ── matplotlib CJK font setup ──
def setup_matplotlib():
    """Configure matplotlib for Traditional Chinese display."""
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    # bbox_inches='tight' is passed directly in save_fig instead

# ── data loading ──
def load_csv(filename):
    """Load a CSV from the data directory."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def align_dataframes(*dfs, method='inner'):
    """Align multiple dataframes by date index."""
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how=method)
    return result.dropna()

# ── return calculations ──
def daily_returns(prices):
    """Calculate daily simple returns."""
    return prices.pct_change().dropna()

def cumulative_returns(daily_ret):
    """Calculate cumulative returns from daily returns."""
    return (1 + daily_ret).cumprod() - 1

def annualized_return(daily_ret, trading_days=252):
    """Calculate annualized return."""
    total = (1 + daily_ret).prod()
    n_years = len(daily_ret) / trading_days
    if n_years <= 0:
        return 0.0
    return total ** (1 / n_years) - 1

def annualized_volatility(daily_ret, trading_days=252):
    """Calculate annualized volatility."""
    return daily_ret.std() * np.sqrt(trading_days)

# ── hedge ratio ──
def minimum_variance_hedge_ratio(spot_returns, futures_returns):
    """
    h* = rho * (sigma_S / sigma_F)
    Returns h*, rho, R-squared.
    """
    rho = spot_returns.corr(futures_returns)
    sigma_s = spot_returns.std()
    sigma_f = futures_returns.std()
    h_star = rho * (sigma_s / sigma_f)
    r_squared = rho ** 2
    return h_star, rho, r_squared

def rolling_hedge_ratio(spot_returns, futures_returns, window=60):
    """Calculate rolling minimum-variance hedge ratio."""
    h_stars = []
    dates = []
    for i in range(window, len(spot_returns)):
        s = spot_returns.iloc[i-window:i]
        f = futures_returns.iloc[i-window:i]
        h, _, _ = minimum_variance_hedge_ratio(s, f)
        h_stars.append(h)
        dates.append(spot_returns.index[i])
    return pd.Series(h_stars, index=dates, name=f'h*_{window}d')

# ── NAV simulation ──
def simulate_nav(futures_returns, shy_returns, hedge_ratio=1.0,
                 bond_weight=0.5, annual_fee=0.0111, roll_cost_per_event=0.0005,
                 roll_dates=None, initial_nav=100.0):
    """
    Simulate ETF NAV from futures returns.

    NAV(t) = NAV(t-1) * (1 + h*R_f(t) + w_bond*R_bond(t) - daily_fee - roll_cost_if_roll)

    Default annual_fee = 1.11% (UNG expense ratio)
    """
    daily_fee = annual_fee / 250
    nav = [initial_nav]

    if roll_dates is None:
        roll_dates = set()
    else:
        roll_dates = set(roll_dates)

    for i in range(len(futures_returns)):
        date = futures_returns.index[i]
        r_f = futures_returns.iloc[i]
        r_b = shy_returns.iloc[i] if i < len(shy_returns) else 0.0

        roll_cost = roll_cost_per_event if date in roll_dates else 0.0

        daily_r = hedge_ratio * r_f + bond_weight * r_b - daily_fee - roll_cost
        nav.append(nav[-1] * (1 + daily_r))

    return pd.Series(nav[1:], index=futures_returns.index, name='NAV')

# ── roll detection ──
def detect_roll_dates(futures_prices, threshold_pct=0.02):
    """
    Detect roll dates from continuous futures contract.
    Roll dates show: large price gap near contract expiry.

    Natural gas (NG) contract months: all 12 months
    F(Jan), G(Feb), H(Mar), J(Apr), K(May), M(Jun),
    N(Jul), Q(Aug), U(Sep), V(Oct), X(Nov), Z(Dec)
    Expiry is around the 3rd-5th business day before the first calendar day of the contract month.
    In practice, front-month rolls around the 25th-28th of the prior month.
    """
    returns = futures_prices.pct_change()

    # NG rolls every month — look for large returns around day 20-28 of each month
    roll_dates = []
    roll_yields = []

    for i in range(1, len(returns)):
        date = returns.index[i]
        day = date.day

        # Rolls typically happen in the last week of the month (day 22-28)
        near_expiry = (22 <= day <= 28)

        if near_expiry:
            r = returns.iloc[i]
            if abs(r) > threshold_pct:
                roll_dates.append(date)
                roll_yields.append(r)

    return roll_dates, roll_yields

# ── performance metrics ──
def performance_summary(nav_series, name='Strategy'):
    """Calculate comprehensive performance metrics."""
    ret = nav_series.pct_change().dropna()
    total_ret = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    n_years = len(ret) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (nav_series / nav_series.cummax() - 1).min()

    return {
        'name': name,
        'total_return': total_ret,
        'annualized_return': ann_ret,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'n_years': n_years
    }

def annual_returns_table(nav_series, name='Strategy'):
    """Calculate annual returns from NAV series."""
    daily_ret = nav_series.pct_change()
    annual = (1 + daily_ret).resample('YE').prod() - 1
    annual.name = name
    return annual

# ── plotting helpers ──
def save_fig(fig, filename):
    """Save figure to output/figures/."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
