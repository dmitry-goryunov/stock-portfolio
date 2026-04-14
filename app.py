import streamlit as st
import yfinance as yf
import pandas as pd
import pyomo.environ as pyo
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations
import pathlib, contextlib, os, logging

logging.getLogger("pyomo.core").setLevel(logging.ERROR)

st.set_page_config(page_title="Portfolio Optimiser", layout="wide")
st.title("Portfolio Optimisation")
st.caption("Markowitz mean-variance optimisation · Pyomo + IPOPT")

# ── IPOPT path ─────────────────────────────────────────────────────────────
try:
    import idaes
    ipopt_exe = os.path.join(idaes.bin_directory, "ipopt.exe")
except Exception:
    ipopt_exe = "ipopt"

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    ALL_TICKERS = [
        "SHEL.L", "BP.L", "HSBA.L", "BARC.L", "AZN.L", "GSK.L",
        "ULVR.L", "DGE.L", "RIO.L", "BA.L", "NG.L", "VOD.L",
        "ISF.L", "VMID.L", "HUKX.L",
    ]
    selected_tickers = st.multiselect("Assets", options=ALL_TICKERS, default=ALL_TICKERS)

    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start", value=pd.Timestamp("2019-10-01"))
    end_date   = c2.date_input("End",   value=pd.Timestamp("2024-10-31"))
    start_str  = start_date.strftime("%Y-%m-%d")
    end_str    = end_date.strftime("%Y-%m-%d")

    reward_floor      = st.slider("Reward floor (monthly)", 0.001, 0.015, 0.004, 0.001, format="%.3f")
    risk_free_rate    = st.slider("Risk-free rate (annual)", 0.01, 0.08, 0.045, 0.005, format="%.3f")
    include_dividends = st.toggle("Include dividends", value=True)
    run_3asset        = st.toggle("Find best 3-asset portfolio", value=False,
                                  help="Slow — checks all C(n,3) combinations")

    go_btn = st.button("Run Optimisation", type="primary", use_container_width=True)


# ── Helpers ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading price data...")
def load_returns(tickers, start, end, dividends):
    for db in pathlib.Path.home().glob("AppData/Local/py-yfinance/*.db"):
        with contextlib.suppress(Exception):
            db.unlink()

    if dividends:
        data = yf.download(list(tickers), start=start, end=end,
                           interval="1mo", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            l0 = set(data.columns.get_level_values(0))
            l1 = set(data.columns.get_level_values(1))
            if {"Close", "Adj Close"} & l0:
                col = "Adj Close" if "Adj Close" in l0 else "Close"
                df = data[col].copy()
            else:
                col = "Adj Close" if "Adj Close" in l1 else "Close"
                df = data.xs(col, level=1, axis=1).copy()
        else:
            df = data[["Close"]].copy()
    else:
        frames = {}
        for t in tickers:
            try:
                h = yf.Ticker(t).history(start=start, end=end, interval="1mo", auto_adjust=False)
                if not h.empty and "Close" in h.columns:
                    frames[t] = h["Close"]
            except Exception:
                pass
        df = pd.DataFrame(frames)

    df.index = pd.to_datetime(df.index).strftime("%b-%Y")
    df = df.dropna(axis=1, how="all")
    df = df.pct_change().iloc[1:]
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="any")
    return df


@st.cache_data(show_spinner="Computing dividend-adjusted returns...")
def load_div_mean(tickers, start, end):
    frames = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=False)
            if not h.empty and len(h) >= 5:
                c = h["Close"]
                r = (c.diff() + h["Dividends"]) / c.shift(1)
                r = r.iloc[1:].dropna()
                if r.index.tz is not None:
                    r.index = pd.DatetimeIndex([ts.replace(tzinfo=None) for ts in r.index])
                frames[t] = (1 + r).resample("ME").prod() - 1
        except Exception:
            pass
    if not frames:
        return {}
    df = pd.DataFrame(frames)
    df.index = df.index.strftime("%b-%Y")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    return df.mean().to_dict()


def solve_portfolio(stocks_list, mean_ret, cov_dict, floor, exe):
    m = pyo.ConcreteModel()
    m.stocks = pyo.Set(initialize=stocks_list)
    m.p = pyo.Var(m.stocks, domain=pyo.NonNegativeReals)
    m.obj = pyo.Objective(
        expr=sum(cov_dict[i, j] * m.p[i] * m.p[j] for i in m.stocks for j in m.stocks),
        sense=pyo.minimize,
    )
    m.c1 = pyo.Constraint(expr=sum(m.p[i] * mean_ret[i] for i in m.stocks) >= floor)
    m.c2 = pyo.Constraint(expr=sum(m.p[i] for i in m.stocks) == 1)
    solver = pyo.SolverFactory("ipopt", executable=exe)
    res = solver.solve(m, tee=False)
    if res.solver.termination_condition == pyo.TerminationCondition.optimal:
        return {i: pyo.value(m.p[i]) for i in m.stocks}, pyo.value(m.obj)
    return None, None


def compute_frontier(mean_ret, cov_dict, stocks_list, exe, n=40):
    min_f = min(mean_ret.values()) + 1e-4
    max_f = max(mean_ret.values()) * 0.95
    vars_, rets_ = [], []
    for rf in np.linspace(min_f, max_f, n):
        w, v = solve_portfolio(stocks_list, mean_ret, cov_dict, rf, exe)
        if w is not None:
            vars_.append(v)
            rets_.append(rf)
    return vars_, rets_


# ── Main ───────────────────────────────────────────────────────────────────

if not go_btn:
    st.info("Configure parameters in the sidebar and click **Run Optimisation**.")
    st.stop()

if len(selected_tickers) < 2:
    st.error("Select at least 2 tickers.")
    st.stop()

# 1. Returns
df = load_returns(tuple(selected_tickers), start_str, end_str, include_dividends)
if df.shape[1] == 0:
    st.error("No valid data returned. Check tickers and date range.")
    st.stop()

stocks_list = df.columns.tolist()
cov_vals    = df.cov().values
cov_dict    = {(stocks_list[i], stocks_list[j]): cov_vals[i, j]
               for i in range(len(stocks_list)) for j in range(len(stocks_list))}
mean_ret    = df.mean().to_dict()

# 2. Optimise
with st.spinner("Solving mean-variance optimisation..."):
    weights, port_var = solve_portfolio(stocks_list, mean_ret, cov_dict, reward_floor, ipopt_exe)

if weights is None:
    st.error("Solver failed — try lowering the reward floor or adding more assets.")
    st.stop()

# 3. Dividend yields
with st.spinner("Fetching dividend yields..."):
    div_yields, ticker_names = {}, {}
    for t in stocks_list:
        try:
            info = yf.Ticker(t).info
            raw = info.get("dividendYield") or 0.0
            div_yields[t]   = raw if raw > 1 else raw * 100
            ticker_names[t] = info.get("longName") or info.get("shortName") or t
        except Exception:
            div_yields[t]   = 0.0
            ticker_names[t] = t

# 4. Returns with/without dividends
mean_ret_div = load_div_mean(tuple(stocks_list), start_str, end_str)

exp_ret_div = (1 + sum(weights[i] * mean_ret_div.get(i, 0) for i in stocks_list)) ** 12 - 1
exp_ret_po  = (1 + sum(weights[i] * mean_ret.get(i, 0)     for i in stocks_list)) ** 12 - 1
ann_std     = (port_var ** 0.5) * (12 ** 0.5)
sharpe_div  = (exp_ret_div - risk_free_rate) / ann_std
sharpe_po   = (exp_ret_po  - risk_free_rate) / ann_std

weights_df = (
    pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
    .sort_values("Weight", ascending=False)
)
weights_df["Weight %"]        = (weights_df["Weight"] * 100).round(2)
weights_df["Div Yield %"]     = pd.Series(div_yields).round(2)
weights_df["Wtd Div Yield %"] = (weights_df["Weight"] * weights_df["Div Yield %"]).round(3)
port_div = weights_df["Wtd Div Yield %"].sum()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Optimal Portfolio", "Efficient Frontier", "Correlation", "Best 3-Asset", "Custom Portfolio"
])

# ── Tab 1: Optimal Portfolio ───────────────────────────────────────────────
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Return (with div)",    f"{exp_ret_div * 100:.2f}%")
    m2.metric("Return (price only)",  f"{exp_ret_po  * 100:.2f}%")
    m3.metric("Sharpe (with div)",    f"{sharpe_div:.2f}")
    m4.metric("Wtd Div Yield",        f"{port_div:.2f}%")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Weights")
        disp = weights_df[weights_df["Weight %"] > 0.01][
            ["Weight %", "Div Yield %", "Wtd Div Yield %"]
        ].copy()
        disp.index = [ticker_names.get(t, t) for t in disp.index]
        st.dataframe(disp, use_container_width=True)

    with col_b:
        st.subheader("Allocation")
        active = weights_df[weights_df["Weight %"] > 0.01]
        fig = go.Figure(go.Bar(
            x=active.index,
            y=active["Weight %"],
            marker_color="steelblue",
            text=(active["Weight %"].round(1).astype(str) + "%"),
            textposition="outside",
        ))
        fig.update_layout(xaxis_title="", yaxis_title="Weight (%)",
                          showlegend=False, margin=dict(t=20, b=40))
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Efficient Frontier ──────────────────────────────────────────────
with tab2:
    with st.spinner("Computing efficient frontier (this takes ~1 min)..."):
        var_div_f, ret_div_f = compute_frontier(mean_ret_div, cov_dict, stocks_list, ipopt_exe)
        var_nd_f,  ret_nd_f  = compute_frontier(mean_ret,     cov_dict, stocks_list, ipopt_exe)

    ann_ret_div_f = [(1 + r) ** 12 - 1 for r in ret_div_f]
    ann_std_div_f = [v ** 0.5 * 12 ** 0.5 for v in var_div_f]
    ann_ret_nd_f  = [(1 + r) ** 12 - 1 for r in ret_nd_f]
    ann_std_nd_f  = [v ** 0.5 * 12 ** 0.5 for v in var_nd_f]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[x * 100 for x in ann_std_div_f], y=[y * 100 for y in ann_ret_div_f],
        mode="lines+markers", name="With dividends", line=dict(color="steelblue")
    ))
    fig.add_trace(go.Scatter(
        x=[x * 100 for x in ann_std_nd_f], y=[y * 100 for y in ann_ret_nd_f],
        mode="lines+markers", name="Without dividends",
        line=dict(color="tomato", dash="dash")
    ))
    fig.update_layout(
        xaxis_title="Annual Std Dev (%)", yaxis_title="Expected Annual Return (%)",
        legend=dict(x=0.02, y=0.98), margin=dict(t=30)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Correlation Heatmap ─────────────────────────────────────────────
with tab3:
    corr = df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Best 3-Asset ────────────────────────────────────────────────────
with tab4:
    if not run_3asset:
        st.info("Enable **Find best 3-asset portfolio** in the sidebar to run this search.")
    else:
        combos = list(combinations(stocks_list, 3))
        prog   = st.progress(0, text=f"Checking {len(combos)} combinations...")
        best_var3, best_combo, best_w3, feasible = float("inf"), None, None, 0

        for idx, combo in enumerate(combos):
            w3, v3 = solve_portfolio(list(combo), mean_ret, cov_dict, reward_floor, ipopt_exe)
            if w3 is not None:
                feasible += 1
                if v3 < best_var3:
                    best_var3, best_combo, best_w3 = v3, combo, w3
            prog.progress((idx + 1) / len(combos),
                          text=f"{idx + 1}/{len(combos)} combinations checked")

        prog.empty()

        if best_combo:
            exp3    = (1 + sum(best_w3[i] * mean_ret_div.get(i, 0) for i in best_combo)) ** 12 - 1
            std3    = (best_var3 ** 0.5) * (12 ** 0.5)
            sharpe3 = (exp3 - risk_free_rate) / std3

            ca, cb, cc = st.columns(3)
            ca.metric("Feasible combos",    f"{feasible}/{len(combos)}")
            cb.metric("Return (with div)",  f"{exp3 * 100:.2f}%")
            cc.metric("Sharpe",             f"{sharpe3:.2f}")

            w3_df = pd.DataFrame.from_dict(best_w3, orient="index", columns=["Weight"])
            w3_df["Weight %"] = (w3_df["Weight"] * 100).round(2)
            w3_df.index = [ticker_names.get(t, t) for t in w3_df.index]
            st.dataframe(w3_df[["Weight %"]], use_container_width=True)

# ── Tab 5: Custom Portfolio ────────────────────────────────────────────────
with tab5:
    st.subheader("Custom Portfolio Analyser")

    n_assets = st.number_input("Number of positions", min_value=1, max_value=10, value=4)
    all_options = stocks_list + ["CASH"]
    cp_input = {}

    cols = st.columns(2)
    for k in range(int(n_assets)):
        with cols[k % 2]:
            default_idx = min(k, len(all_options) - 1)
            t = st.selectbox(f"Asset {k+1}", options=all_options,
                             index=default_idx, key=f"cp_t{k}")
            w = st.number_input(f"Weight {k+1}", min_value=0.0, max_value=1.0,
                                value=round(1 / int(n_assets), 2), key=f"cp_w{k}")
            cp_input[t] = cp_input.get(t, 0) + w

    if st.button("Analyse", type="secondary"):
        total_w = sum(cp_input.values())
        if total_w <= 0:
            st.error("All weights are zero.")
        else:
            cp = {t: w / total_w for t, w in cp_input.items() if w > 0}
            CASH_KEY   = "CASH"
            rf_monthly = (1 + risk_free_rate) ** (1 / 12) - 1
            risky      = [t for t in cp if t != CASH_KEY]

            cp_mean_div = {t: mean_ret_div.get(t, 0) for t in risky}
            cp_mean_div[CASH_KEY] = rf_monthly
            cp_mean_po  = {t: mean_ret.get(t, 0) for t in risky}
            cp_mean_po[CASH_KEY]  = rf_monthly

            risky_in_df = [t for t in risky if t in df.columns]
            if risky_in_df:
                cp_cov      = df[risky_in_df].cov()
                port_var_cp = sum(
                    cp.get(i, 0) * cp.get(j, 0) * cp_cov.loc[i, j]
                    for i in risky_in_df for j in risky_in_df
                )
            else:
                port_var_cp = 0.0

            exp_div_cp = (1 + sum(cp.get(t, 0) * cp_mean_div.get(t, 0) for t in cp)) ** 12 - 1
            exp_po_cp  = (1 + sum(cp.get(t, 0) * cp_mean_po.get(t, 0)  for t in cp)) ** 12 - 1
            ann_std_cp = (port_var_cp ** 0.5) * (12 ** 0.5) if port_var_cp > 0 else 0.0
            sharpe_cp  = (exp_div_cp - risk_free_rate) / ann_std_cp if ann_std_cp > 0 else float("nan")

            ca, cb, cc, cd = st.columns(4)
            ca.metric("Return (with div)",   f"{exp_div_cp * 100:.2f}%")
            cb.metric("Return (price only)", f"{exp_po_cp  * 100:.2f}%")
            cc.metric("Annual Std Dev",      f"{ann_std_cp * 100:.2f}%")
            cd.metric("Sharpe",              f"{sharpe_cp:.2f}" if not np.isnan(sharpe_cp) else "N/A")
