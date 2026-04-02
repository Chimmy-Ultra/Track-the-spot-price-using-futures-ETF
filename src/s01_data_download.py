"""
Step 1: Download all required data for UNG Natural Gas ETF replication.

Downloads:
- UNG ETF prices (Yahoo Finance, inception 2007-04-18)
- NG=F continuous front-month natural gas futures (Yahoo Finance)
- SHY short-term Treasury ETF (Yahoo Finance)
- Henry Hub natural gas spot price from FRED (daily, EIA)
- Individual NG futures contracts (Databento GLBX.MDP3) -- skipped if file exists
- Currently active NG individual contracts (term structure snapshot)

Databento API Key:
  Set env var DATABENTO_API_KEY=db-xxxx  or paste key in DATABENTO_API_KEY below.
  If key is blank or file already exists, the Databento step is skipped.
  Individual contract data is used for accurate roll cost calculation.
"""
import os
import sys
import yfinance as yf
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR

START_DATE = '2007-04-18'   # UNG ETF inception date
END_DATE = None             # latest available

# Set your Databento API key as environment variable: DATABENTO_API_KEY=db-xxxx
# Or create a .env file (never commit it to git)
DATABENTO_API_KEY = os.environ.get('DATABENTO_API_KEY', '')


def download_single(ticker, filename, start=START_DATE, end=END_DATE):
    """Download a single ticker and save to CSV."""
    print(f'  Downloading {ticker}...')
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        print(f'  WARNING: No data for {ticker}')
        return None

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path)
    print(f'  Saved {len(df)} rows -> {path}')
    return df


def download_ng_spot():
    """
    Download Henry Hub Natural Gas Spot Price from FRED (daily).

    Series: DHHNGSP = Henry Hub Natural Gas Spot Price (Dollars per Million Btu)
    Source: EIA (U.S. Energy Information Administration)
    Frequency: Daily (Mon-Fri, EIA publication schedule)
    This is the actual cash/spot price, independent from futures markets.
    """
    print('\n  Downloading Henry Hub natural gas spot price from FRED (EIA)...')
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP&cosd=2007-01-01'
    df = pd.read_csv(url, parse_dates=[0])
    df.columns = ['Date', 'NG_Spot_USD_MMBtu']
    df = df.set_index('Date')

    # FRED encodes missing values as '.' — replace with NaN then drop
    df['NG_Spot_USD_MMBtu'] = pd.to_numeric(df['NG_Spot_USD_MMBtu'], errors='coerce')
    df = df.dropna()

    path = os.path.join(DATA_DIR, 'ng_spot_daily.csv')
    df.to_csv(path)
    print(f'  Saved {len(df)} daily observations -> {path}')
    print(f'  Range: {df.index[0].date()} to {df.index[-1].date()}')
    print(f'  Unit: USD/MMBtu (Henry Hub, Louisiana)')
    return df


def download_term_structure():
    """
    Download currently active NG individual contracts for term structure snapshot.

    Natural gas contract months (all 12):
    F(Jan), G(Feb), H(Mar), J(Apr), K(May), M(Jun),
    N(Jul), Q(Aug), U(Sep), V(Oct), X(Nov), Z(Dec)
    Yahoo ticker format: NG{code}{YY}.NYM
    """
    print('\n  Downloading active natural gas futures contracts (term structure)...')

    # All 12 natural gas contract month codes in order
    month_codes = [
        ('F', 'Jan'), ('G', 'Feb'), ('H', 'Mar'), ('J', 'Apr'),
        ('K', 'May'), ('M', 'Jun'), ('N', 'Jul'), ('Q', 'Aug'),
        ('U', 'Sep'), ('V', 'Oct'), ('X', 'Nov'), ('Z', 'Dec')
    ]
    month_order = {code: i for i, (code, _) in enumerate(month_codes)}

    contracts = {}
    # Try next 2-3 years of contracts
    for year in range(25, 28):
        for code, month_name in month_codes:
            ticker = f'NG{code}{year}.NYM'
            try:
                df = yf.download(ticker, period='5d', progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    last_close = float(df['Close'].iloc[-1])
                    last_date = df.index[-1]
                    contracts[ticker] = {
                        'ticker': ticker,
                        'month_code': code,
                        'month_name': month_name,
                        'month_order': month_order[code],
                        'year': 2000 + year,
                        'last_close': last_close,
                        'last_date': last_date
                    }
                    print(f'    {ticker}: {last_close:.3f}')
            except Exception:
                pass

    if contracts:
        ts_df = pd.DataFrame(contracts.values())
        # Sort by year then month order for proper forward curve display
        ts_df = ts_df.sort_values(['year', 'month_order']).reset_index(drop=True)
        path = os.path.join(DATA_DIR, 'term_structure.csv')
        ts_df.to_csv(path, index=False)
        print(f'  Saved {len(ts_df)} contracts -> {path}')
        return ts_df
    else:
        print('  WARNING: No active contracts found. Term structure chart will be skipped.')
        return None


def download_databento_ng():
    """
    Download all individual NG futures contracts from Databento (CME Globex).

    Dataset : GLBX.MDP3 (CME Globex)
    Symbol  : NG.FUT   (all NG futures, parent symbol)
    Schema  : ohlcv-1d (daily OHLCV per contract)

    Skipped if:
    - No API key is configured
    - data/ng_futures_databento.csv already exists (avoid re-downloading / re-billing)

    To force re-download: delete ng_futures_databento.csv first.
    """
    out_path = os.path.join(DATA_DIR, 'ng_futures_databento.csv')

    # ── Skip if file already exists (check both locations) ──
    databento_dir = os.path.join(DATA_DIR, 'databento')
    existing = None
    if os.path.exists(out_path):
        existing = out_path
    elif os.path.isdir(databento_dir):
        for f in os.listdir(databento_dir):
            if 'OUTRIGHTS' in f and f.endswith('.csv'):
                existing = os.path.join(databento_dir, f)
                break

    if existing:
        df = pd.read_csv(existing, nrows=5)
        total = sum(1 for _ in open(existing)) - 1
        print(f'  Databento data exists ({total:,} rows) -> skipping download')
        print(f'  File: {existing}')
        return None

    # ── Skip if no API key ──
    if not DATABENTO_API_KEY:
        print('  No DATABENTO_API_KEY set -> skipping individual contract download')
        print('  Set DATABENTO_API_KEY in s01_data_download.py or as env var to enable.')
        return None

    try:
        import databento as db
    except ImportError:
        print('  databento package not installed -> pip install databento')
        return None

    # GLBX.MDP3 dataset available from 2010-06-06
    db_start = '2010-06-06'

    print(f'\n  Databento: NG individual contracts (GLBX.MDP3, ohlcv-1d)')
    print(f'  Start: {db_start}')

    client = db.Historical(DATABENTO_API_KEY)

    # ── Step 1: estimate cost FIRST (no charge) ──
    params = dict(
        dataset='GLBX.MDP3',
        symbols=['NG.FUT'],
        schema='ohlcv-1d',
        start=db_start,
        end=END_DATE,
        stype_in='parent',
    )

    try:
        cost = client.metadata.get_cost(**params)
        print(f'  Estimated cost: ${cost:.4f} USD')
    except Exception as e:
        print(f'  Could not estimate cost ({e}), proceeding...')
        cost = None

    # ── Step 2: confirm before spending money ──
    print()
    print('  ' + '=' * 50)
    print('  Databento Download Confirmation')
    print('  ' + '=' * 50)
    print(f'  Dataset:    GLBX.MDP3 (CME Globex)')
    print(f'  Symbol:     NG.FUT (all individual NG futures)')
    print(f'  Schema:     ohlcv-1d (daily OHLCV per contract)')
    print(f'  Date range: {db_start} ~ present')
    if cost is not None:
        print(f'  Est. cost:  ${cost:.4f} USD')
    else:
        print(f'  Est. cost:  unknown (API query failed)')
    print(f'  Save to:    {out_path}')
    print('  ' + '-' * 50)
    print('  NOTE: This will charge your Databento account.')
    print('        File will be cached locally after download.')
    print('        Re-running will NOT re-download if file exists.')
    print('  ' + '=' * 50)

    ans = input('  Type "yes" to confirm download: ').strip().lower()
    if ans != 'yes':
        print('  Skipped Databento download (user did not confirm).')
        return None

    # ── Step 3: actual download ──
    print('  Downloading...')
    data = client.timeseries.get_range(**params)

    df = data.to_df()
    df.to_csv(out_path)
    print(f'  Saved {len(df)} rows -> {out_path}')
    print(f'  Columns: {df.columns.tolist()}')
    return df


def run():
    print('=' * 60)
    print('Step 1: Data Download (UNG Natural Gas ETF)')
    print('=' * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Main daily datasets ──
    download_single('UNG', 'ung_etf.csv')
    download_single('NG=F', 'ng_futures_continuous.csv')
    download_single('SHY', 'shy.csv')

    # ── Henry Hub daily spot (FRED/EIA) ──
    download_ng_spot()

    # ── Individual NG contracts (Databento) ──
    download_databento_ng()

    # ── Term structure snapshot ──
    download_term_structure()

    print('\nData download complete.')


if __name__ == '__main__':
    run()
