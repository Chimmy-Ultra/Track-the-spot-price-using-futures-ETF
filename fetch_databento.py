"""
Databento NG Futures Data Fetcher
===================================
Downloads ALL individual natural gas (Henry Hub) futures contracts from CME Globex.
Each row = one contract on one trading day (OHLCV).

Usage:
    Set DATABENTO_API_KEY environment variable, then:
        python fetch_databento.py

Or paste your API key directly in the API_KEY variable below.

Output:
    data/ng_futures_databento.csv    -- raw per-contract daily bars
    data/ng_roll_schedule.csv        -- derived roll dates + roll yields
"""
import os
import sys
import pandas as pd
import databento as db

# ── API Key ──────────────────────────────────────────────
# Option 1: set env var  DATABENTO_API_KEY=db-xxxx
# Option 2: paste key here directly
API_KEY = os.environ.get('DATABENTO_API_KEY', 'YOUR_API_KEY_HERE')
# ─────────────────────────────────────────────────────────

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
START_DATE = '2007-04-01'
END_DATE   = None   # None = latest available

# NG contract month codes (all 12)
NG_MONTH_CODES = {
    'F':'Jan', 'G':'Feb', 'H':'Mar', 'J':'Apr',
    'K':'May', 'M':'Jun', 'N':'Jul', 'Q':'Aug',
    'U':'Sep', 'V':'Oct', 'X':'Nov', 'Z':'Dec'
}


def fetch_ng_futures():
    if API_KEY == 'YOUR_API_KEY_HERE':
        print('ERROR: Please set your Databento API key.')
        print('  Edit fetch_databento.py and replace YOUR_API_KEY_HERE')
        print('  or set the DATABENTO_API_KEY environment variable.')
        sys.exit(1)

    print('Connecting to Databento...')
    client = db.Historical(API_KEY)

    # ── Cost estimate before downloading ──
    print('Estimating cost...')
    try:
        cost = client.metadata.get_cost(
            dataset='GLBX.MDP3',
            symbols=['NG.FUT'],
            schema='ohlcv-1d',
            start=START_DATE,
            end=END_DATE,
            stype_in='parent',
        )
        print(f'  Estimated cost: ${cost:.4f} USD')
        confirm = input('  Proceed? (y/n): ').strip().lower()
        if confirm != 'y':
            print('Aborted.')
            sys.exit(0)
    except Exception as e:
        print(f'  Could not estimate cost: {e}')
        confirm = input('  Proceed anyway? (y/n): ').strip().lower()
        if confirm != 'y':
            sys.exit(0)

    # ── Download ──
    print(f'\nDownloading NG futures daily bars from {START_DATE}...')
    data = client.timeseries.get_range(
        dataset='GLBX.MDP3',
        symbols=['NG.FUT'],
        schema='ohlcv-1d',
        start=START_DATE,
        end=END_DATE,
        stype_in='parent',
    )

    df = data.to_df()
    print(f'  Raw rows: {len(df)}')
    print(f'  Columns:  {df.columns.tolist()}')

    # ── Save raw ──
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, 'ng_futures_databento.csv')
    df.to_csv(raw_path)
    print(f'\n  Saved raw: {raw_path}')

    return df


def build_roll_schedule(df):
    """
    From per-contract daily bars, compute:
    - Which contract was the front month on each trading day
    - Roll dates (when front month switches)
    - Roll yield = (new front price - old front price) / old front price

    Assumes the front month is the nearest-expiry contract
    with sufficient volume on each day.
    """
    print('\nBuilding roll schedule from individual contracts...')

    # Parse contract expiry from instrument_id or symbol
    # Databento OHLCV schema includes: ts_event, open, high, low, close, volume, instrument_id
    # We need to join with the instrument definitions to get the contract month

    # The symbol column (if present) looks like: NGH4, NGJ4, etc.
    if 'symbol' in df.columns:
        df['contract'] = df['symbol']
    elif 'instrument_id' in df.columns:
        # Would need instrument definitions - skip for now
        print('  Note: instrument_id only; symbol lookup would require definitions endpoint.')
        return None

    # Identify front month = highest volume contract on each day
    df['date'] = pd.to_datetime(df.index).normalize() if df.index.name == 'ts_event' else pd.to_datetime(df['ts_event']).dt.normalize()

    daily_front = df.sort_values('volume', ascending=False).groupby('date').first()

    # Roll dates = when front-month contract changes
    daily_front['prev_contract'] = daily_front['contract'].shift(1)
    roll_dates = daily_front[daily_front['contract'] != daily_front['prev_contract']].copy()

    if len(roll_dates) > 0:
        roll_dates['roll_yield'] = (daily_front.loc[roll_dates.index, 'close'] -
                                    daily_front.loc[roll_dates.index, 'open']) / daily_front.loc[roll_dates.index, 'open']
        roll_path = os.path.join(DATA_DIR, 'ng_roll_schedule.csv')
        roll_dates[['contract', 'prev_contract', 'close', 'volume', 'roll_yield']].to_csv(roll_path)
        print(f'  Roll events found: {len(roll_dates)}')
        print(f'  Saved: {roll_path}')
        print(roll_dates[['contract', 'prev_contract', 'roll_yield']].head(10).to_string())

    return daily_front


if __name__ == '__main__':
    df = fetch_ng_futures()
    build_roll_schedule(df)
    print('\nDone.')
