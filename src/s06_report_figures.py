"""Polished figures for the homework report."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

ROOT = Path(r"C:\Users\kevin\Desktop\期貨追蹤現貨")
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Microsoft JhengHei",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.unicode_minus": False,
})

C_SPOT = "#E63946"
C_FUT  = "#457B9D"
C_UNG  = "#F4A261"
C_POS  = "#2A9D8F"
C_NEG  = "#E76F51"
C_DARK = "#1D3557"
C_GREY = "#6C757D"

START = pd.Timestamp("2010-06-07")


# ================== Fig 1: Cumulative performance ==================
nav = pd.read_csv(ROOT / "data" / "nav_strategies.csv", index_col=0, parse_dates=True)

fig, ax = plt.subplots(figsize=(12, 5.8))
ax.plot(nav.index, nav["Spot"],    color=C_SPOT, lw=1.9, label="Spot (Henry Hub)")
ax.plot(nav.index, nav["UNG_sim"], color=C_FUT,  lw=1.8, label="UNG_sim (our replication, prospectus rule)")
ax.plot(nav.index, nav["UNG"],     color=C_UNG,  lw=1.8, label="UNG ETF (actual)", alpha=0.9, linestyle="--")
ax.set_yscale("log")
ax.set_title("Cumulative Performance — UNG Replication vs UNG ETF vs Spot   (2010-06 = 100, log scale)", pad=14)
ax.set_ylabel("NAV")

last_date = nav.index[-1]
last = nav.iloc[-1]
ax.set_xlim(nav.index[0], last_date + pd.Timedelta(days=520))

for name, col, color in [("Spot", "Spot", C_SPOT), ("UNG_sim", "UNG_sim", C_FUT), ("UNG", "UNG", C_UNG)]:
    ax.annotate(f"{name}: {last[col]:.1f}",
                xy=(last_date, last[col]),
                xytext=(8, 0), textcoords="offset points",
                color=color, fontweight="bold", va="center", fontsize=10)

# Event annotations – placed on the spot curve with short arrows
events = [("2021-02-15", "Winter Storm Uri", (-70, 35)),
          ("2024-01-12", "2024 cold snap",  (-20, 55)),
          ("2026-01-23", "2026 spike",      (-90, -40))]
for date, label, offset in events:
    d = pd.to_datetime(date)
    idx = nav.index.get_indexer([d], method="nearest")[0]
    y = nav["Spot"].iloc[idx]
    ax.annotate(label, xy=(nav.index[idx], y),
                xytext=offset, textcoords="offset points",
                fontsize=9, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.7, alpha=0.9))

ax.legend(loc="lower left", ncol=3, fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.savefig(OUT / "report_01_cumulative.png")
plt.close()
print("wrote", OUT / "report_01_cumulative.png")


# ================== Fig 2: Basis time series ==================
prices = pd.read_csv(ROOT / "data" / "prices_daily.csv", parse_dates=["Date"], index_col="Date")
prices = prices[prices.index >= START]
basis_pct = (prices["NG_F"] - prices["Spot"]) / prices["Spot"] * 100
monthly = basis_pct.resample("ME").mean().dropna()

fig, ax = plt.subplots(figsize=(12, 4.8))
colors = [C_NEG if v > 0 else C_POS for v in monthly.values]
ax.bar(monthly.index, monthly.values, width=26, color=colors, alpha=0.88, edgecolor="none")
ax.axhline(0, color="k", lw=0.6)
ax.axhline(monthly.mean(), color=C_DARK, ls="--", lw=1, alpha=0.7)

ax.set_title("Basis = (Futures - Spot) / Spot   —   Monthly Average", pad=12)
ax.set_ylabel("Basis (%)")

n_c = (monthly > 0).sum()
n_t = len(monthly)
info = (f"Contango months: {n_c} / {n_t}   ({n_c/n_t:.1%})\n"
        f"Mean basis: {monthly.mean():+.2f}%")
ax.text(0.015, 0.96, info, transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor=C_GREY, alpha=0.9, boxstyle="round,pad=0.45"))

p1 = mpatches.Patch(color=C_NEG, label="Contango  (F > S  ->  roll loss)")
p2 = mpatches.Patch(color=C_POS, label="Backwardation  (F < S  ->  roll gain)")
ax.legend(handles=[p1, p2], loc="upper right", fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.savefig(OUT / "report_02_basis.png")
plt.close()
print("wrote", OUT / "report_02_basis.png")


# ================== Fig 3: TE attribution ==================
components = [
    ("Management\nFee",      -1.11, C_GREY),
    ("Trading\nCost",        -0.97, C_GREY),
    ("Roll Yield\nDrag",    -20.00, C_NEG),
]
total = sum(v for _, v, _ in components)

fig, ax = plt.subplots(figsize=(9.5, 5.8))
names  = [n for n, _, _ in components] + ["Total TE\n(annualized)"]
values = [v for _, v, _ in components] + [total]
cols   = [c for _, _, c in components] + [C_DARK]

bars = ax.bar(names, values, color=cols, width=0.62,
              edgecolor="white", linewidth=1.6)
for bar, v in zip(bars, values):
    inside = v < -3
    offset = -16 if inside else 10
    color  = "white" if inside else C_DARK
    ax.annotate(f"{v:+.2f}%",
                xy=(bar.get_x() + bar.get_width()/2, v),
                xytext=(0, offset), textcoords="offset points",
                ha="center", color=color, fontweight="bold", fontsize=11.5)

ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("Annualized contribution (%)")
ax.set_title("Tracking Error Attribution — Why UNG lags Spot by ~22% / year", pad=12)

# Callout for roll yield dominance — placed to the right of the bar
share = abs(-20.00) / abs(total)
ax.annotate(f"Roll drag is ~{share:.0%}\nof total TE",
            xy=(2, -10), xytext=(2.55, -13),
            ha="left", va="center", fontsize=10.5,
            color=C_NEG, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.2,
                            connectionstyle="arc3,rad=0.1"))

ax.set_ylim(-25, 3)
ax.grid(axis="x", visible=False)
plt.savefig(OUT / "report_03_attribution.png")
plt.close()
print("wrote", OUT / "report_03_attribution.png")


# ================== Fig 4: Daily scatter (futures vs spot) ==================
rets = pd.read_csv(ROOT / "data" / "returns_daily.csv", parse_dates=["Date"], index_col="Date")
rets = rets.dropna(subset=["Spot_ret", "NG_F_ret"])
rets = rets[rets.index >= START]

x_full = rets["Spot_ret"].values * 100
y_full = rets["NG_F_ret"].values * 100

slope = np.cov(x_full, y_full, ddof=0)[0, 1] / np.var(x_full, ddof=0)
intercept = y_full.mean() - slope * x_full.mean()
rho = np.corrcoef(x_full, y_full)[0, 1]

LIM = 25  # main window in %
fig, ax = plt.subplots(figsize=(8.5, 7.6))

inside = (np.abs(x_full) <= LIM) & (np.abs(y_full) <= LIM)
ax.scatter(x_full[inside], y_full[inside],
           s=12, alpha=0.35, color=C_FUT, edgecolor="none", rasterized=True)

ax.plot([-LIM, LIM], [-LIM, LIM], color=C_GREY, ls="--", lw=1, label="1:1 line")
xs = np.array([-LIM, LIM])
ax.plot(xs, slope*xs + intercept, color=C_NEG, lw=2.2,
        label=f"OLS fit  (β = {slope:.3f})")

ax.text(0.03, 0.97,
        f"ρ = {rho:.3f}\nR² = {rho**2:.3f}\nn = {len(x_full):,}  days",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(facecolor="white", edgecolor=C_GREY, alpha=0.92, boxstyle="round,pad=0.5"))

# Flag extreme spot spikes as clipped markers on the right edge
highlights = ["2021-02-17", "2024-01-12", "2026-01-23"]
edge_positions = [(LIM*0.96, 3), (LIM*0.96, -5), (LIM*0.96, -11)]
for (date, (ex, ey)) in zip(highlights, edge_positions):
    d = pd.to_datetime(date)
    if d in rets.index:
        sx = rets.loc[d, "Spot_ret"] * 100
        sy = rets.loc[d, "NG_F_ret"] * 100
        ax.scatter([ex], [ey], s=110, color=C_NEG, marker=">",
                   edgecolor="white", linewidth=1.2, zorder=5, clip_on=False)
        ax.annotate(f"{date}\nspot +{sx:.0f}%, fut {sy:+.1f}%",
                    xy=(ex, ey), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5,
                    color=C_NEG, fontweight="bold")

ax.axhline(0, color="k", lw=0.5, alpha=0.35)
ax.axvline(0, color="k", lw=0.5, alpha=0.35)
ax.set_xlabel("Spot daily return (%)")
ax.set_ylabel("Futures daily return (%)")
ax.set_title("Daily Returns: Futures vs Spot\nρ = 0.20 → hedge ratio cannot capture extreme spot spikes", pad=10)
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=10)

note = (f"{(~inside).sum()} extreme day(s) outside ±{LIM}% window\n"
        f"(max spot: +{x_full.max():.0f}% on {rets.index[np.argmax(x_full)].date()})")
ax.text(0.03, 0.03, note, transform=ax.transAxes, va="bottom", fontsize=8.5,
        color=C_GREY, style="italic")

plt.savefig(OUT / "report_04_scatter.png")
plt.close()
print("wrote", OUT / "report_04_scatter.png")
