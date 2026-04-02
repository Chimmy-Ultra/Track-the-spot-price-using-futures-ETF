"""
Step 2: Clean and merge all downloaded data.

Inputs:
- Daily: UNG ETF, NG=F continuous futures, SHY
- Daily: Henry Hub spot price (FRED DHHNGSP, EIA schedule ~Mon-Fri)

Outputs:
- prices_daily.csv  : aligned daily prices  (UNG, NG_F, SHY, Spot)
- returns_daily.csv : aligned daily returns (UNG_ret, NG_F_ret, SHY_ret, Spot_ret)

Note: Spot (Henry Hub) is published on EIA business days only, not all
exchange trading days. We forward-fill up to 3 days so weekends/holidays
in the spot series align with futures trading days.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR


def load_and_extract_close(filename, col_name):
    """Load CSV and extract Close price column."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if 'Close' in df.columns:
        return df[['Close']].rename(columns={'Close': col_name})
    elif 'Adj Close' in df.columns:
        return df[['Adj Close']].rename(columns={'Adj Close': col_name})
    else:
        raise ValueError(f'No Close column in {filename}: {df.columns.tolist()}')


def run():
    print('=' * 60)
    print('Step 2: Data Cleaning & Merging')
    print('=' * 60)

    # ── Load daily market data ──
    ung_etf  = load_and_extract_close('ung_etf.csv', 'UNG')
    futures  = load_and_extract_close('ng_futures_continuous.csv', 'NG_F')
    shy      = load_and_extract_close('shy.csv', 'SHY')

    print(f'  UNG ETF: {ung_etf.index[0].date()} to {ung_etf.index[-1].date()}, {len(ung_etf)} rows')
    print(f'  NG=F:    {futures.index[0].date()} to {futures.index[-1].date()}, {len(futures)} rows')
    print(f'  SHY:     {shy.index[0].date()} to {shy.index[-1].date()}, {len(shy)} rows')

    # ── Load Henry Hub daily spot ──
    print('\n  Loading Henry Hub spot price (daily, FRED/EIA)...')
    spot_path = os.path.join(DATA_DIR, 'ng_spot_daily.csv')
    spot = pd.read_csv(spot_path, index_col=0, parse_dates=True)
    spot = spot.rename(columns={'NG_Spot_USD_MMBtu': 'Spot'})
    print(f'  Spot:    {spot.index[0].date()} to {spot.index[-1].date()}, {len(spot)} rows')

    # ── Align all four series ──
    # Strategy: outer join first, then forward-fill Spot (EIA gaps), then inner align
    # This handles EIA holidays & weekends while preserving trading day coverage

    # Step 1: merge UNG/NG_F/SHY on common trading days
    mkt = ung_etf.join(futures, how='inner').join(shy, how='inner')
    mkt = mkt.ffill(limit=3).dropna()
    print(f'\n  Market data (UNG+NG_F+SHY) common days: {len(mkt)} rows')

    # Step 2: forward-fill Spot to all calendar days, then align to market days
    spot_filled = spot.reindex(
        pd.date_range(spot.index.min(), spot.index.max(), freq='D')
    ).ffill(limit=5)
    spot_aligned = spot_filled.reindex(mkt.index).dropna()
    print(f'  Spot after ffill + reindex to mkt days: {len(spot_aligned)} rows')

    # Step 3: inner join
    daily_merged = mkt.join(spot_aligned, how='inner')
    daily_merged = daily_merged.dropna()
    print(f'  Final merged: {daily_merged.index[0].date()} to {daily_merged.index[-1].date()}, {len(daily_merged)} rows')

    # ── Compute daily returns ──
    daily_returns = daily_merged.pct_change().dropna()
    daily_returns.columns = ['UNG_ret', 'NG_F_ret', 'SHY_ret', 'Spot_ret']

    # ── Save ──
    os.makedirs(DATA_DIR, exist_ok=True)
    daily_merged.to_csv(os.path.join(DATA_DIR, 'prices_daily.csv'))
    daily_returns.to_csv(os.path.join(DATA_DIR, 'returns_daily.csv'))
    print(f'\n  Saved: prices_daily.csv ({len(daily_merged)} rows)')
    print(f'  Saved: returns_daily.csv ({len(daily_returns)} rows)')

    # ── Correlation report ──
    print('\n  Daily return correlations:')
    print(daily_returns.corr().round(4).to_string())

    rho_ung_spot = daily_returns['UNG_ret'].corr(daily_returns['Spot_ret'])
    rho_ngf_spot = daily_returns['NG_F_ret'].corr(daily_returns['Spot_ret'])
    print(f'\n  Key correlations:')
    print(f'    UNG vs Spot (Henry Hub):  {rho_ung_spot:.4f}')
    print(f'    NG=F vs Spot (Henry Hub): {rho_ngf_spot:.4f}')
    print(f'    UNG vs NG=F:              {daily_returns["UNG_ret"].corr(daily_returns["NG_F_ret"]):.4f}')

    # ── Basis (futures - spot) ──
    basis = daily_merged['NG_F'] - daily_merged['Spot']
    basis_pct = basis / daily_merged['Spot'] * 100
    print(f'\n  Basis (NG=F - Henry Hub Spot):')
    print(f'    Mean: {basis.mean():.3f} USD/MMBtu ({basis_pct.mean():.1f}%)')
    print(f'    Contango (futures > spot): {(basis > 0).sum()}/{len(basis)} days ({(basis > 0).mean():.1%})')

    print('\nData cleaning complete.')
    return daily_merged, daily_returns


if __name__ == '__main__':
    run()
