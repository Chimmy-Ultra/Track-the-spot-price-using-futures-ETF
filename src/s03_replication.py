"""
Step 3: Build Roll Schedule & Simulate M1/M3/M6 Strategies.

Uses Databento individual contract data (OUTRIGHTS.csv).

KEY INSIGHT: Single-digit year codes recycle every 10 years (NGN0 = Jul 2010
AND Jul 2020). We use `instrument_id` (unique per contract) to distinguish them,
and derive expiry from each contract's last trading date in the data.

Output files:
  data/roll_schedule.csv     - daily M1~M6 assignments + prices
  data/roll_costs.csv        - each roll event with cost
  data/nav_strategies.csv    - daily NAV for all strategies + spot + UNG
"""
import os
import sys
import re
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR, FIG_DIR, TABLE_DIR, setup_matplotlib, save_fig
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ANNUAL_FEE = 0.0111
DAILY_FEE  = ANNUAL_FEE / 250

# ── Transaction cost parameters ──
# Based on NYMEX NG contract specs:
#   Contract size: 10,000 MMBtu
#   Min tick: $0.001/MMBtu = $10/tick
#   Front-month bid-ask spread: ~1 tick ($0.001)
#   2nd-month bid-ask spread:   ~2 ticks ($0.002)
#   CME exchange fee: ~$1.50/side
#   Broker commission: ~$0.50/side (institutional)
# At $3.50/MMBtu, contract value = $35,000
#
# Per roll: sell M1 (half-spread) + buy M2 (half-spread)
#   = 0.0143% + 0.0286% + exchange + broker
#   = ~0.054% per roll = ~5.4 bps
#   Annualized: ~0.65%/year (12 rolls)
ROLL_TRANSACTION_COST = 0.00054   # 5.4 bps per roll event
DAILY_REBAL_COST      = 0.000005  # ~0.5 bp daily trading friction

# UNG-specific market impact (large fund ~$500M AUM)
# sqrt(0.89% participation) * spread ~ 2.7 bps additional
UNG_MARKET_IMPACT     = 0.00027   # per roll, added on top for UNG-style sim

MONTH_MAP = {
    'F': 1,  'G': 2,  'H': 3,  'J': 4,  'K': 5,  'M': 6,
    'N': 7,  'Q': 8,  'U': 9,  'V': 10, 'X': 11, 'Z': 12
}


def load_outrights():
    """Load Databento outright contracts, keyed by instrument_id."""
    print('  Loading Databento outright contracts...')
    path = None
    for f in os.listdir(os.path.join(DATA_DIR, 'databento')):
        if 'OUTRIGHTS' in f and f.endswith('.csv'):
            path = os.path.join(DATA_DIR, 'databento', f)
            break
    if path is None:
        raise FileNotFoundError('No OUTRIGHTS csv in data/databento/')

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['ts_event']).dt.normalize().dt.tz_localize(None)
    df = df[['date', 'instrument_id', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()

    print(f'  Loaded {len(df):,} rows')
    print(f'  Unique symbols: {df["symbol"].nunique()}, Unique instrument_ids: {df["instrument_id"].nunique()}')
    print(f'  Date range: {df["date"].min().date()} ~ {df["date"].max().date()}')
    return df


def smart_expiry(symbol, first_date, last_date):
    """
    Compute contract expiry using symbol + trading dates to disambiguate decade.

    2-digit year (NGJ26): unambiguous -> 2026
    1-digit year (NGN0):  try 2010/2020/2030, pick decade where expiry
                          is closest to last_date and >= first_date - 30d
    """
    m = re.match(r'^NG([FGHJKMNQUVXZ])(\d{1,2})$', symbol)
    if not m:
        return pd.NaT
    month = MONTH_MAP[m.group(1)]
    yr_raw = int(m.group(2))

    if yr_raw >= 10:
        year = 2000 + yr_raw
    else:
        best_year, best_diff = None, float('inf')
        for base in [2010, 2020, 2030]:
            y = base + yr_raw
            try:
                exp = pd.Timestamp(y, month, 1) - pd.tseries.offsets.BusinessDay(4)
            except Exception:
                continue
            if exp < first_date - pd.Timedelta(days=30):
                continue
            diff = abs((exp - last_date).days)
            if diff < best_diff:
                best_diff = diff
                best_year = y
        year = best_year if best_year else 2020 + yr_raw

    return pd.Timestamp(year, month, 1) - pd.tseries.offsets.BusinessDay(4)


def build_contract_calendar(outrights):
    """
    Build contract calendar using instrument_id as unique key.
    ALL contracts use smart_expiry() for correct decade disambiguation.
    """
    print('\n  Building contract calendar (by instrument_id)...')

    cal = (outrights.groupby('instrument_id')
           .agg(symbol=('symbol', 'first'),
                first_date=('date', 'min'),
                last_date=('date', 'max'),
                total_volume=('volume', 'sum'))
           .reset_index())

    cal['expiry'] = cal.apply(
        lambda r: smart_expiry(r['symbol'], r['first_date'], r['last_date']), axis=1
    )
    cal = cal.dropna(subset=['expiry'])
    cal = cal.sort_values('expiry').reset_index(drop=True)

    n = len(cal)
    years_span = (cal['expiry'].iloc[-1] - cal['expiry'].iloc[0]).days / 365.25
    per_year = n / years_span if years_span > 0 else 0

    print(f'  Contracts: {n}')
    print(f'  Expiry range: {cal["expiry"].iloc[0].date()} ~ {cal["expiry"].iloc[-1].date()}')
    print(f'  Contracts/year: {per_year:.1f} (expect ~12)')

    for _, r in cal.head(5).iterrows():
        print(f'    id={r["instrument_id"]:>7d}  sym={r["symbol"]:6s}  '
              f'traded={r["first_date"].date()}~{r["last_date"].date()}  expiry={r["expiry"].date()}')

    return cal


def build_roll_schedule(outrights, cal):
    """
    Build daily roll schedule.

    For each trading day, assign M1/M2/.../M6 by sorting all non-expired
    contracts by expiry ascending. Uses instrument_id for uniqueness.

    Price lookup: pivot table (date x instrument_id), forward-filled.
    """
    print('\n  Building daily roll schedule...')

    trading_days = sorted(outrights['date'].unique())

    # Pivot: rows=date, cols=instrument_id, values=close
    # Forward-fill up to 5 days for contracts that don't trade every day
    price_pivot = outrights.pivot_table(
        index='date', columns='instrument_id', values='close', aggfunc='last'
    ).ffill(limit=5)

    # Contract arrays sorted by expiry (already sorted in cal)
    cids = cal['instrument_id'].values
    syms = cal['symbol'].values
    expiries = cal['expiry'].values

    # Max reasonable expiry for M1: within 60 days of trade date
    # (NG contracts expire monthly, so M1 should expire within ~35 days)
    # M6 can be up to ~210 days out
    MAX_M1_DAYS = 60
    MAX_M6_DAYS = 400

    records = []
    for date in trading_days:
        date_np = np.datetime64(date)
        active_mask = expiries >= date_np
        active_cids = cids[active_mask]
        active_syms = syms[active_mask]
        active_expiries = expiries[active_mask]

        # Filter: only keep contracts with reasonable expiry for each slot
        # M1 should expire within 60 days; M6 within 400 days
        reasonable_mask = (active_expiries - date_np) <= np.timedelta64(MAX_M6_DAYS, 'D')
        active_cids = active_cids[reasonable_mask]
        active_syms = active_syms[reasonable_mask]
        active_expiries = active_expiries[reasonable_mask]

        row = {'date': date}
        for i, label in enumerate(['M1', 'M2', 'M3', 'M4', 'M5', 'M6']):
            if i < len(active_cids):
                cid = active_cids[i]
                sym = active_syms[i]
                close = price_pivot.at[date, cid] if (date in price_pivot.index and cid in price_pivot.columns) else np.nan
                row[f'{label}_symbol'] = sym
                row[f'{label}_cid'] = cid
                row[f'{label}_close'] = close
            else:
                row[f'{label}_symbol'] = None
                row[f'{label}_cid'] = None
                row[f'{label}_close'] = np.nan

        records.append(row)

    schedule = pd.DataFrame(records)
    schedule['date'] = pd.to_datetime(schedule['date'])
    schedule = schedule.sort_values('date').reset_index(drop=True)

    # Detect rolls (M1 contract_id changes)
    schedule['M1_prev_cid'] = schedule['M1_cid'].shift(1)
    schedule['is_roll'] = schedule['M1_cid'] != schedule['M1_prev_cid']
    schedule.loc[schedule.index[0], 'is_roll'] = False

    n_rolls = schedule['is_roll'].sum()
    n_years = len(schedule) / 252
    print(f'  Schedule: {len(schedule)} days ({n_years:.1f} years)')
    print(f'  Roll events: {n_rolls} ({n_rolls/n_years:.1f}/year, expect ~12)')

    rolls_preview = schedule[schedule['is_roll']].head(5)
    print(f'  First 5 rolls:')
    for _, r in rolls_preview.iterrows():
        print(f'    {r["date"].date()}: {schedule.loc[r.name-1, "M1_symbol"]} -> {r["M1_symbol"]}')

    return schedule


def calculate_roll_costs(schedule):
    """
    Roll cost = (M2_close - M1_close) / M1_close on the day BEFORE roll.

    This is the cost you pay to switch from near-month to next-month.
    Positive = contango (loss), negative = backwardation (gain).
    """
    print('\n  Calculating roll costs...')

    rolls = []
    for idx in schedule.index[schedule['is_roll']]:
        if idx == 0:
            continue
        prev = schedule.loc[idx - 1]
        curr = schedule.loc[idx]

        m1 = prev['M1_close']
        m2 = prev['M2_close']

        if pd.isna(m1) or pd.isna(m2) or m1 == 0:
            continue

        spread = m2 - m1
        cost_pct = spread / m1

        rolls.append({
            'roll_date': curr['date'],
            'old_contract': prev['M1_symbol'],
            'new_contract': curr['M1_symbol'],
            'M1_close': m1,
            'M2_close': m2,
            'spread_usd': spread,
            'roll_cost_pct': cost_pct,
            'market_state': 'contango' if spread > 0 else 'backwardation',
        })

    roll_df = pd.DataFrame(rolls)
    roll_df['roll_date'] = pd.to_datetime(roll_df['roll_date'])
    roll_df['roll_month'] = roll_df['roll_date'].dt.month

    n = len(roll_df)
    if n == 0:
        print('  WARNING: No roll costs calculated.')
        return roll_df

    n_c = (roll_df['roll_cost_pct'] > 0).sum()
    avg = roll_df['roll_cost_pct'].mean()
    total = roll_df['roll_cost_pct'].sum()

    print(f'  Total rolls: {n}')
    print(f'  Contango: {n_c} ({n_c/n:.1%}) | Backwardation: {n-n_c} ({(n-n_c)/n:.1%})')
    print(f'  Avg roll cost: {avg*100:+.2f}%/roll')
    print(f'  Cumulative: {total*100:+.1f}%')
    print(f'  Annualized: {avg*12*100:+.1f}%/year (avg x 12 rolls)')

    return roll_df


def simulate_strategy_nav(schedule, month_label='M1', strategy_name='M1',
                          include_txcost=True):
    """
    Simulate NAV by holding the Nth-month contract daily.
    Uses CLOSE-to-CLOSE returns (same as ETF NAV calculation).

    Price convention:
      - All returns are computed from close_yesterday to close_today.
      - This matches how ETF NAVs are marked: using end-of-day settlement.

    On same-contract days:
      return = close_today / close_yesterday - 1 - daily_fee
    On roll days:
      return = 0 (NAV carries forward, minus fee and transaction cost).
      Why? When you roll, you sell X units of old contract and buy Y units
      of new contract at market prices. Your TOTAL VALUE doesn't change.
      The roll cost is already reflected in the new contract's subsequent
      returns (it starts from a different price level).
    """
    close_col = f'{month_label}_close'
    cid_col = f'{month_label}_cid'

    closes = schedule[close_col].values
    cids = schedule[cid_col].values
    n = len(closes)

    nav = np.full(n, np.nan)
    nav[0] = 100.0
    n_missing = 0
    n_roll_skip = 0

    for i in range(1, n):
        if np.isnan(closes[i]) or np.isnan(closes[i-1]) or closes[i-1] == 0:
            nav[i] = nav[i-1]
            n_missing += 1
        elif cids[i] != cids[i-1]:
            # Roll day: hold flat, deduct fee + transaction cost
            cost = DAILY_FEE
            if include_txcost:
                cost += ROLL_TRANSACTION_COST
            nav[i] = nav[i-1] * (1 - cost)
            n_roll_skip += 1
        else:
            # Same contract: close-to-close return
            daily_cost = DAILY_FEE
            if include_txcost:
                daily_cost += DAILY_REBAL_COST
            ret = closes[i] / closes[i-1] - 1 - daily_cost
            nav[i] = nav[i-1] * (1 + ret)

    if n_missing > 0:
        print(f'    {strategy_name}: {n_missing}/{n} days missing ({n_missing/n:.1%}), '
              f'{n_roll_skip} roll-day skips')

    return pd.Series(nav, index=schedule['date'], name=strategy_name)


def simulate_ung_prospectus_nav(schedule, cal, include_txcost=True):
    """
    Simulate UNG-style NAV using the SEC prospectus rolling rule.

    UNG Roll Mechanics (from SEC prospectus / USCF documentation):
    ──────────────────────────────────────────────────────────────
    "During the four business days from two weeks (= 10 business days)
    before the near month natural gas futures contract expires, the fund
    rolls its position into the next-month contract. Approximately 25%
    of the position is traded on each of these four days."

    Weight schedule (START-of-day weights used for today's return):
      bd_remain >= 10:  (1.0, 0.0)   normal holding of near-month
      bd_remain == 9:   (0.75, 0.25) first 25% already rolled at prev close
      bd_remain == 8:   (0.50, 0.50)
      bd_remain == 7:   (0.25, 0.75)
      bd_remain <= 6:   (0.0, 1.0)   fully in next-month

    The 4 rebalance trades happen at the close of days with
    bd_remain = 10, 9, 8, 7. The weight change is observed on the
    NEXT day (bd_remain = 9, 8, 7, 6 respectively).

    Return calculation:
      - On normal days (no contract transition):
          r = w_m1 * r_m1_same_contract + w_m2 * r_m2_same_contract
      - On schedule contract-transition days (M1_cid changes because
        old M1 expired):
          The contract we held (yesterday's schedule M2) is now today's
          schedule M1. Return is computed as:
            r = schedule.M1_close[i] / schedule.M2_close[i-1] - 1

    Transaction costs (only if include_txcost=True):
      - DAILY_FEE applied every day (management fee)
      - DAILY_REBAL_COST applied every day (daily trading friction)
      - On roll days (weight changed from yesterday), apply:
          w_diff * (ROLL_TRANSACTION_COST + UNG_MARKET_IMPACT)
        where w_diff = |w_m1[i] - w_m1[i-1]| (= 0.25 on each of 4 roll days)
    """
    # ── Step 0: Build expiry lookup (instrument_id -> expiry as datetime64[D]) ──
    cal_expiry = {}
    for _, row in cal.iterrows():
        cid = row['instrument_id']
        expiry_ts = pd.Timestamp(row['expiry'])
        if pd.isna(expiry_ts):
            continue
        cal_expiry[cid] = np.datetime64(expiry_ts.date(), 'D')

    dates_np = pd.to_datetime(schedule['date'].values).to_numpy().astype('datetime64[D]')
    m1_close = schedule['M1_close'].values.astype(float)
    m2_close = schedule['M2_close'].values.astype(float)
    m1_cid   = schedule['M1_cid'].values
    n = len(dates_np)

    # ── Step 1: Compute bd_remain and start-of-day weights for each day ──
    # bd_remain = business days from today (inclusive) to expiry (exclusive)
    bd_remain = np.full(n, 999, dtype=int)
    w_m1 = np.ones(n)
    w_m2 = np.zeros(n)

    for i in range(n):
        cid = m1_cid[i]
        if cid is None or (isinstance(cid, float) and np.isnan(cid)):
            continue
        if cid not in cal_expiry:
            continue
        expiry_np = cal_expiry[cid]
        bd = int(np.busday_count(dates_np[i], expiry_np))
        bd_remain[i] = bd

        if bd >= 10:
            w_m1[i], w_m2[i] = 1.0, 0.0
        elif bd == 9:
            w_m1[i], w_m2[i] = 0.75, 0.25
        elif bd == 8:
            w_m1[i], w_m2[i] = 0.50, 0.50
        elif bd == 7:
            w_m1[i], w_m2[i] = 0.25, 0.75
        else:  # bd <= 6
            w_m1[i], w_m2[i] = 0.0, 1.0

    # ── Step 2: Simulate NAV ──
    nav = np.full(n, np.nan)
    nav[0] = 100.0

    n_expiry_transitions = 0
    n_roll_trade_days    = 0
    n_incomplete_roll    = 0

    for i in range(1, n):
        contract_transition = (m1_cid[i] != m1_cid[i-1])

        if contract_transition:
            # Old near-month expired. The contract we held (yesterday's M2)
            # is now today's M1. Compute its return across the transition.
            if (not np.isnan(m1_close[i]) and not np.isnan(m2_close[i-1])
                    and m2_close[i-1] != 0):
                r = m1_close[i] / m2_close[i-1] - 1
            else:
                r = 0.0

            if w_m1[i-1] > 0.001:
                n_incomplete_roll += 1

            cost = DAILY_FEE
            if include_txcost:
                cost += DAILY_REBAL_COST
            nav[i] = nav[i-1] * (1 + r - cost)
            n_expiry_transitions += 1

        else:
            # Same contracts as yesterday — compute blended return
            if (not np.isnan(m1_close[i]) and not np.isnan(m1_close[i-1])
                    and m1_close[i-1] != 0):
                r_m1 = m1_close[i] / m1_close[i-1] - 1
            else:
                r_m1 = 0.0

            if (not np.isnan(m2_close[i]) and not np.isnan(m2_close[i-1])
                    and m2_close[i-1] != 0):
                r_m2 = m2_close[i] / m2_close[i-1] - 1
            else:
                r_m2 = r_m1

            r = w_m1[i] * r_m1 + w_m2[i] * r_m2

            # Cost: if weights changed from yesterday, a trade occurred at prev close
            w_diff = abs(w_m1[i] - w_m1[i-1])
            cost = DAILY_FEE
            if include_txcost:
                cost += DAILY_REBAL_COST
                if w_diff > 0.001:
                    cost += w_diff * (ROLL_TRANSACTION_COST + UNG_MARKET_IMPACT)
                    n_roll_trade_days += 1

            nav[i] = nav[i-1] * (1 + r - cost)

    print(f'    UNG (prospectus rule):')
    print(f'      {n_expiry_transitions} contract-expiry transitions')
    print(f'      {n_roll_trade_days} roll-trade days '
          f'(expect ~4 per expiry = {4 * n_expiry_transitions})')
    if n_incomplete_roll > 0:
        print(f'      WARNING: {n_incomplete_roll} expiries with incomplete roll')

    return pd.Series(nav, index=schedule['date'], name='UNG_sim')


def run():
    print('=' * 60)
    print('Step 3: Roll Schedule & Strategy Simulation (Databento)')
    print('=' * 60)

    setup_matplotlib()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    # ── Load & build calendar ──
    outrights = load_outrights()
    cal = build_contract_calendar(outrights)

    # ── Roll schedule ──
    schedule = build_roll_schedule(outrights, cal)
    schedule.to_csv(os.path.join(DATA_DIR, 'roll_schedule.csv'), index=False)

    # ── Roll costs ──
    roll_costs = calculate_roll_costs(schedule)
    roll_costs.to_csv(os.path.join(DATA_DIR, 'roll_costs.csv'), index=False)

    # ── Simulate NAVs ──
    print('\n  Simulating strategy NAVs...')
    print('  (all using close-to-close returns, with transaction costs)')
    m1_nav = simulate_strategy_nav(schedule, 'M1', 'M1')
    m3_nav = simulate_strategy_nav(schedule, 'M3', 'M3')
    m6_nav = simulate_strategy_nav(schedule, 'M6', 'M6')
    ung_sim = simulate_ung_prospectus_nav(schedule, cal)

    for name, s in [('M1', m1_nav), ('M3', m3_nav), ('M6', m6_nav),
                     ('UNG_sim', ung_sim)]:
        total = s.iloc[-1] / s.iloc[0] - 1
        print(f'    {name}: end={s.iloc[-1]:.2f}, total={total:+.2%}')

    # ── Load Spot + UNG, align to Databento range ──
    prices = pd.read_csv(os.path.join(DATA_DIR, 'prices_daily.csv'),
                         index_col=0, parse_dates=True)
    start = schedule['date'].iloc[0]
    end = schedule['date'].iloc[-1]

    spot = prices.loc[start:end, 'Spot'].dropna()
    ung = prices.loc[start:end, 'UNG'].dropna()
    spot_nav = spot / spot.iloc[0] * 100
    ung_nav = ung / ung.iloc[0] * 100

    nav_all = pd.DataFrame({
        'Spot': spot_nav, 'M1': m1_nav, 'M3': m3_nav, 'M6': m6_nav,
        'UNG_sim': ung_sim, 'UNG': ung_nav,
    }).dropna()

    nav_all.to_csv(os.path.join(DATA_DIR, 'nav_strategies.csv'))
    print(f'\n  Saved nav_strategies.csv ({len(nav_all)} days)')

    # ── Performance table ──
    n_years = len(nav_all) / 252
    rets = nav_all.pct_change().dropna()

    print(f'\n  Performance ({nav_all.index[0].date()} ~ {nav_all.index[-1].date()}, {n_years:.1f} yrs):')
    print(f'  {"":8s} {"Total":>9s} {"Ann":>8s} {"Vol":>8s} {"MaxDD":>8s} {"rho(S)":>7s}')
    print('  ' + '-' * 50)
    for col in nav_all.columns:
        t = nav_all[col].iloc[-1] / nav_all[col].iloc[0] - 1
        a = (1+t)**(1/n_years)-1 if n_years > 0 else 0
        v = rets[col].std() * np.sqrt(252)
        m = (nav_all[col] / nav_all[col].cummax() - 1).min()
        r = rets[col].corr(rets['Spot']) if col != 'Spot' else 1.0
        print(f'  {col:8s} {t:+8.2%} {a:+7.2%} {v:7.2%} {m:+7.2%} {r:6.4f}')

    # ── Validation: M1 vs NG=F ──
    print('\n  Validation: M1 close vs NG=F (Yahoo)...')
    ngf = prices.loc[start:end, 'NG_F'].dropna()
    m1p = schedule.set_index('date')['M1_close']
    common = pd.DataFrame({'M1': m1p, 'NGF': ngf}).dropna()
    print(f'  Price corr: {common["M1"].corr(common["NGF"]):.4f}')
    print(f'  Mean |diff|: ${(common["M1"]-common["NGF"]).abs().mean():.3f}')

    # ── Plot 1: NAV comparison ──
    colors = {'Spot':'#E91E63','M1':'#4CAF50','M3':'#2196F3','M6':'#9C27B0',
              'UNG_sim':'#FFEB3B','UNG':'#FF9800'}
    labels = {'Spot':'Henry Hub Spot','M1':'M1 (1-day roll)','M3':'M3 (3rd-month)',
              'M6':'M6 (6th-month)','UNG_sim':'UNG sim (4-day roll)','UNG':'UNG ETF (actual)'}

    fig, ax = plt.subplots(figsize=(14, 8))
    for col in ['Spot','M1','M3','M6','UNG_sim','UNG']:
        ax.plot(nav_all.index, nav_all[col], color=colors[col],
                linewidth=2 if col=='Spot' else 1.3, label=labels[col])
    ax.set_yscale('log')
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title(f'NAV Comparison: Spot vs Roll Strategies vs UNG\n'
                 f'Start=100, Log Scale | {nav_all.index[0].date()} ~ {nav_all.index[-1].date()}',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('NAV (log)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    for col in nav_all.columns:
        v = nav_all[col].iloc[-1]
        ax.annotate(f'{v:.1f}', xy=(nav_all.index[-1], v),
                    fontsize=9, color=colors[col], fontweight='bold',
                    xytext=(10, 0), textcoords='offset points')
    save_fig(fig, '01_nav_comparison.png')

    # ── Plot 2: Roll costs ──
    if len(roll_costs) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(14, 9))

        ax = axes[0]
        bc = ['red' if c > 0 else 'green' for c in roll_costs['roll_cost_pct']]
        ax.bar(roll_costs['roll_date'], roll_costs['roll_cost_pct']*100,
               width=20, color=bc, alpha=0.7)
        ax.axhline(0, color='k', linewidth=0.5)
        avg = roll_costs['roll_cost_pct'].mean()*100
        ax.axhline(avg, color='blue', linestyle='--', label=f'Avg = {avg:+.2f}%')
        ax.set_title('Monthly Roll Cost (M2-M1 spread at transition)\n'
                     'Red = Contango (loss) | Green = Backwardation (gain)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Roll Cost (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

        ax = axes[1]
        cum = roll_costs['roll_cost_pct'].cumsum()*100
        ax.plot(roll_costs['roll_date'], cum, 'r-', linewidth=1.5)
        ax.fill_between(roll_costs['roll_date'], cum, 0,
                        where=(cum>0), alpha=0.15, color='red')
        ax.fill_between(roll_costs['roll_date'], cum, 0,
                        where=(cum<=0), alpha=0.15, color='green')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title('Cumulative Roll Cost', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative (%)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

        plt.tight_layout()
        save_fig(fig, '06_roll_costs.png')

    print('\nStep 3 complete.')


if __name__ == '__main__':
    run()
