# =============================================================================
# alpha_lab | Umut Öztürk | ECON484 Extended Research
# portfolio_sim.py — Dynamic Position Sizing Backtest
# Walk-forward: lookahead bias yok, sadece t anındaki bilgi
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR    = Path(__file__).resolve().parent.parent.parent
ALPHA_DATA  = BASE_DIR / "original_data" / "alpha_data"
ALPHA_PLOTS = BASE_DIR / "plots" / "alpha_charts"
ALPHA_DATA.mkdir(exist_ok=True)
ALPHA_PLOTS.mkdir(exist_ok=True)

MARKETS         = ["sp500", "bist100", "dax"]
INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = {
    "sp500":   0.0010,
    "bist100": 0.0015,
    "dax":     0.0010,
}

MIN_HOLD_DAYS = {
    "sp500":   5,
    "bist100": 1,
    "dax":     3,
}

# ---------------------------------------------------------------------------
# Position sizing: regime + stress zone kombinasyonu
# Cheating yok — sadece t anındaki bilgi
# ---------------------------------------------------------------------------
POSITION_SIZING = {
    "sp500": {
        ("Calm/Bull",   "low"):  1.00,
        ("Calm/Bull",   "mid"):  0.70,
        ("Calm/Bull",   "high"): 0.40,
        ("Transition",  "low"):  0.80,
        ("Transition",  "mid"):  0.50,
        ("Transition",  "high"): 0.10,
        ("Stress/Bear", "low"):  0.20,   # ← 0.60'tan düşür, ama sıfır değil
        ("Stress/Bear", "mid"):  0.10,   # ← 0.20'den düşür
        ("Stress/Bear", "high"): 0.00,   # aynı
    },
    "bist100": {
    ("Calm/Bull",   "low"):  1.00,
    ("Calm/Bull",   "mid"):  1.00,
    ("Calm/Bull",   "high"): 0.80,
    ("Transition",  "low"):  1.00,
    ("Transition",  "mid"):  0.80,
    ("Transition",  "high"): 0.60,
    ("Stress/Bear", "low"):  0.60,
    ("Stress/Bear", "mid"):  0.30,
    ("Stress/Bear", "high"): 0.00,
},
    "dax": {
        ("Calm/Bull",   "low"):  1.00,
        ("Calm/Bull",   "mid"):  0.70,
        ("Calm/Bull",   "high"): 0.40,
        ("Transition",  "low"):  0.00,   # veri bunu söyledi, aynı kalıyor
        ("Transition",  "mid"):  0.00,
        ("Transition",  "high"): 0.00,
        ("Stress/Bear", "low"):  0.20,   # ← 0.40'tan düşür ama sıfır değil (Yol B)
        ("Stress/Bear", "mid"):  0.10,   # ← küçük taban pozisyon
        ("Stress/Bear", "high"): 0.00,   # aynı
    },
}

def get_stress_zone(score: float) -> str:
    if score < 0.33:
        return "low"
    elif score < 0.66:
        return "mid"
    else:
        return "high"

# ---------------------------------------------------------------------------
# CORE: dinamik pozisyon backtest
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, market: str):
    df      = df.copy().reset_index(drop=True)
    cost    = TRANSACTION_COST[market]
    mhp     = MIN_HOLD_DAYS[market]
    sizing  = POSITION_SIZING[market]

    capital      = INITIAL_CAPITAL
    position     = 0.0
    hold_counter = 0
    prev_target  = 0.0

    equity    = []
    trade_log = []

    for i, row in df.iterrows():
        date   = row["Date"]
        close  = row["Close"]
        regime = row["Regime_Name"]
        score  = row["Stress_Score"] if not pd.isna(row["Stress_Score"]) else 0.0

        if pd.isna(close) or close <= 0:
            equity.append({"Date": date, "Equity": capital,
                           "Position": position, "Trade": None,
                           "Close": close, "Regime": regime, "Target_Pct": 0})
            continue

        zone       = get_stress_zone(score)
        target_pct = sizing.get((regime, zone), 0.0)
        trade_action = None

        if hold_counter > 0:
            hold_counter -= 1

        if hold_counter == 0 and abs(target_pct - prev_target) >= 0.10:
            total_value   = capital + position * close
            target_value  = total_value * target_pct
            current_value = position * close
            delta         = target_value - current_value

            if delta > 0:
                buy_value    = min(delta, capital)
                shares_buy   = (buy_value * (1 - cost)) / close
                position    += shares_buy
                capital     -= buy_value
                trade_action = "BUY"
                trade_log.append({"Date": date, "Action": "BUY",
                                   "Price": close, "Target_Pct": target_pct,
                                   "Regime": regime, "Score": score})
            elif delta < 0:
                shares_sell  = min(abs(delta) / close, position)
                proceeds     = shares_sell * close * (1 - cost)
                position    -= shares_sell
                capital     += proceeds
                trade_action = "SELL"
                trade_log.append({"Date": date, "Action": "SELL",
                                   "Price": close, "Target_Pct": target_pct,
                                   "Regime": regime, "Score": score})

            prev_target  = target_pct
            hold_counter = mhp

        current_equity = capital + position * close
        equity.append({
            "Date":       date,
            "Equity":     current_equity,
            "Position":   position,
            "Trade":      trade_action,
            "Close":      close,
            "Signal":     row["Signal"],
            "Regime":     regime,
            "Target_Pct": target_pct,
        })

    return pd.DataFrame(equity), pd.DataFrame(trade_log)

# ---------------------------------------------------------------------------
# Buy & Hold benchmark
# ---------------------------------------------------------------------------
def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    df    = df.dropna(subset=["Close"]).reset_index(drop=True)
    start = df["Close"].iloc[0]
    return (df["Close"] / start) * INITIAL_CAPITAL

# ---------------------------------------------------------------------------
# Performance metrikleri
# ---------------------------------------------------------------------------
def performance_metrics(eq_df: pd.DataFrame, trade_df: pd.DataFrame,
                         market: str) -> dict:
    eq         = eq_df["Equity"].values
    daily_ret  = pd.Series(eq).pct_change().dropna()
    total_ret  = (eq[-1] / eq[0] - 1) * 100
    annual_ret = ((eq[-1] / eq[0]) ** (252 / len(eq)) - 1) * 100
    sharpe     = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0)
    peak   = pd.Series(eq).cummax()
    max_dd = ((pd.Series(eq) - peak) / peak).min() * 100

    win_rate = None
    if len(trade_df) >= 2:
        buys  = trade_df[trade_df["Action"] == "BUY"]["Price"].values
        sells = trade_df[trade_df["Action"] == "SELL"]["Price"].values
        n     = min(len(buys), len(sells))
        if n > 0:
            win_rate = round((sells[:n] > buys[:n]).sum() / n * 100, 1)

    return {
        "market":        market,
        "total_return":  round(total_ret, 2),
        "annual_return": round(annual_ret, 2),
        "sharpe":        round(sharpe, 3),
        "max_drawdown":  round(max_dd, 2),
        "win_rate":      win_rate,
        "n_trades":      len(trade_df[trade_df["Action"] == "BUY"]) if len(trade_df) else 0,
        "final_equity":  round(eq[-1], 2),
    }

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_equity_curve(eq_df: pd.DataFrame, bh_series: pd.Series,
                      trade_df: pd.DataFrame, market: str,
                      metrics: dict) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(
        f"alpha_lab | {market.upper()} — Portfolio Simulation (Dynamic Sizing)\n"
        f"Return: {metrics['total_return']:+.1f}%  "
        f"Annual: {metrics['annual_return']:+.1f}%  "
        f"Sharpe: {metrics['sharpe']:.2f}  "
        f"MaxDD: {metrics['max_drawdown']:.1f}%  "
        f"WinRate: {metrics['win_rate']}%\n"
        "Umut Öztürk | ECON484 Extended Research",
        fontsize=10
    )

    # Panel 1: Equity curves
    ax1.plot(eq_df["Date"], eq_df["Equity"],
             color="#2980b9", linewidth=1.5, label="Dynamic Strategy")
    ax1.plot(eq_df["Date"], bh_series.values[:len(eq_df)],
             color="#95a5a6", linewidth=1.2, linestyle="--", label="Buy & Hold")

    if len(trade_df):
        buys  = trade_df[trade_df["Action"] == "BUY"]
        sells = trade_df[trade_df["Action"] == "SELL"]
        buy_eq  = eq_df[eq_df["Date"].isin(buys["Date"])]["Equity"]
        sell_eq = eq_df[eq_df["Date"].isin(sells["Date"])]["Equity"]
        ax1.scatter(buys["Date"],  buy_eq,  marker="^", color="#27ae60", s=40, zorder=5, label="BUY")
        ax1.scatter(sells["Date"], sell_eq, marker="v", color="#c0392b", s=40, zorder=5, label="SELL")

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(alpha=0.3)

    # Panel 2: Drawdown
    peak = eq_df["Equity"].cummax()
    dd   = (eq_df["Equity"] - peak) / peak * 100
    ax2.fill_between(eq_df["Date"], dd, 0, color="#e74c3c", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(dd.min() * 1.2, 5)
    ax2.grid(alpha=0.3)

    # Panel 3: Position sizing
    ax3.fill_between(eq_df["Date"], eq_df["Target_Pct"] * 100,
                     alpha=0.6, color="#2980b9")
    ax3.set_ylabel("Position %")
    ax3.set_ylim(0, 110)
    ax3.grid(alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=30)
    plt.tight_layout()

    out = ALPHA_PLOTS / f"portfolio_sim_{market}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] plots/alpha_charts/portfolio_sim_{market}.png")


def plot_comparison_summary(all_metrics: list) -> None:
    df = pd.DataFrame(all_metrics)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "alpha_lab | Portfolio Simulation Summary — Dynamic Position Sizing\n"
        "Umut Öztürk | ECON484 Extended Research", fontsize=11
    )
    metrics_cfg = [
        ("total_return",  "Total Return (%)",  "#2980b9"),
        ("annual_return", "Annual Return (%)", "#27ae60"),
        ("sharpe",        "Sharpe Ratio",      "#8e44ad"),
        ("win_rate",      "Win Rate (%)",      "#e67e22"),
    ]
    for ax, (col, title, color) in zip(axes, metrics_cfg):
        vals = df[col].fillna(0).values
        bars = ax.bar(df["market"].str.upper(), vals,
                      color=color, alpha=0.85, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(bar.get_height()) * 0.02,
                    f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = ALPHA_PLOTS / "portfolio_sim_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] plots/alpha_charts/portfolio_sim_summary.png")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("alpha_lab | Umut Öztürk | ECON484 Extended Research")
    print("Portfolio Simulation — Dynamic Position Sizing")
    print("=" * 60)

    all_metrics = []

    for market in MARKETS:
        print(f"\n[{market.upper()}]")
        try:
            df = pd.read_csv(
                ALPHA_DATA / f"alpha_signal_{market}.csv",
                parse_dates=["Date"]
            )

            eq_df, trade_df = run_backtest(df, market)
            bh_series       = buy_and_hold(df)
            metrics         = performance_metrics(eq_df, trade_df, market)

            bh_return = (bh_series.iloc[-1] / INITIAL_CAPITAL - 1) * 100

            print(f"  Total Return    : {metrics['total_return']:+.2f}%")
            print(f"  Annual Return   : {metrics['annual_return']:+.2f}%")
            print(f"  Sharpe Ratio    : {metrics['sharpe']:.3f}")
            print(f"  Max Drawdown    : {metrics['max_drawdown']:.2f}%")
            print(f"  Win Rate        : {metrics['win_rate']}%")
            print(f"  # Trades (BUY)  : {metrics['n_trades']}")
            print(f"  Final Equity    : ${metrics['final_equity']:,.2f}")
            print(f"  B&H Return      : {bh_return:+.2f}%")
            print(f"  Alpha vs B&H    : {metrics['total_return'] - bh_return:+.2f}%")

            plot_equity_curve(eq_df, bh_series, trade_df, market, metrics)
            eq_df.to_csv(ALPHA_DATA / f"portfolio_sim_{market}.csv", index=False)
            all_metrics.append(metrics)

        except Exception:
            import traceback
            traceback.print_exc()

    plot_comparison_summary(all_metrics)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(pd.DataFrame(all_metrics)[
        ["market", "total_return", "annual_return",
         "sharpe", "max_drawdown", "win_rate", "n_trades"]
    ].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()