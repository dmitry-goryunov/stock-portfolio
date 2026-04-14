import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from itertools import combinations
import pathlib, contextlib  # noqa: F401

st.set_page_config(page_title="Portfolio Optimiser", layout="wide")
st.title("Portfolio Optimisation")
st.caption("Markowitz mean-variance optimisation")

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
    frames = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start=start, end=end,
                                     interval="1mo", auto_adjust=dividends)
            if not h.empty and "Close" in h.columns:
                frames[t] = h["Close"]
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).tz_localize(None).strftime("%b-%Y")
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
                if h.index.tz is not None:
                    h.index = h.index.tz_localize(None)
                c = h["Close"]
                r = (c.diff() + h["Dividends"]) / c.shift(1)
                r = r.iloc[1:].dropna()
                frames[t] = (1 + r).resample("ME").prod() - 1
        except Exception:
            pass
    if not frames:
        return {}
    df = pd.DataFrame(frames)
    df.index = df.index.tz_localize(None).strftime("%b-%Y")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    return df.mean().to_dict()


def solve_portfolio(stocks_list, mean_ret_arr, cov_matrix, floor):
    n = len(stocks_list)
    w0 = np.ones(n) / n

    def variance(w):
        return w @ cov_matrix @ w

    constraints = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: np.dot(w, mean_ret_arr) - floor},
    ]
    bounds = [(0, 1)] * n

    res = minimize(variance, w0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"ftol": 1e-9, "maxiter": 1000})
    if res.success:
        return dict(zip(stocks_list, res.x)), float(res.fun)
    return None, None


def compute_frontier(mean_ret_arr, cov_matrix, stocks_list, n=40):
    min_f = float(np.min(mean_ret_arr)) + 1e-4
    max_f = float(np.max(mean_ret_arr)) * 0.95
    vars_, rets_ = [], []
    for floor in np.linspace(min_f, max_f, n):
        w, v = solve_portfolio(stocks_list, mean_ret_arr, cov_matrix, floor)
        if w is not None:
            vars_.append(v)
            rets_.append(floor)
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

stocks_list   = df.columns.tolist()
cov_matrix    = df.cov().values
mean_ret_arr  = df.mean().values
mean_ret_dict = df.mean().to_dict()

# 2. Optimise
with st.spinner("Solving mean-variance optimisation..."):
    weights, port_var = solve_portfolio(stocks_list, mean_ret_arr, cov_matrix, reward_floor)

if weights is None:
    st.error("Solver failed — try lowering the reward floor or adding more assets.")
    st.stop()

# 3. Dividend yields & names
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
exp_ret_po  = (1 + sum(weights[i] * mean_ret_dict.get(i, 0) for i in stocks_list)) ** 12 - 1
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
    m1.metric("Return (with div)",   f"{exp_ret_div * 100:.2f}%")
    m2.metric("Return (price only)", f"{exp_ret_po  * 100:.2f}%")
    m3.metric("Sharpe (with div)",   f"{sharpe_div:.2f}")
    m4.metric("Wtd Div Yield",       f"{port_div:.2f}%")

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
    with st.spinner("Computing efficient frontier..."):
        mean_div_arr = np.array([mean_ret_div.get(t, 0) for t in stocks_list])
        var_div_f, ret_div_f = compute_frontier(mean_div_arr, cov_matrix, stocks_list)
        var_nd_f,  ret_nd_f  = compute_frontier(mean_ret_arr, cov_matrix, stocks_list)

    ann = lambda rets, vars_: (
        [(1 + r) ** 12 - 1 for r in rets],
        [v ** 0.5 * 12 ** 0.5 for v in vars_]
    )
    ret_d, std_d = ann(ret_div_f, var_div_f)
    ret_n, std_n = ann(ret_nd_f,  var_nd_f)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x * 100 for x in std_d], y=[y * 100 for y in ret_d],
                             mode="lines+markers", name="With dividends",
                             line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=[x * 100 for x in std_n], y=[y * 100 for y in ret_n],
                             mode="lines+markers", name="Without dividends",
                             line=dict(color="tomato", dash="dash")))
    fig.update_layout(xaxis_title="Annual Std Dev (%)", yaxis_title="Expected Annual Return (%)",
                      legend=dict(x=0.02, y=0.98), margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Correlation Heatmap ─────────────────────────────────────────────
with tab3:
    fig = px.imshow(df.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
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
            idx_list = [stocks_list.index(t) for t in combo]
            sub_cov  = cov_matrix[np.ix_(idx_list, idx_list)]
            sub_mean = mean_ret_arr[idx_list]
            w3, v3   = solve_portfolio(list(combo), sub_mean, sub_cov, reward_floor)
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
            ca.metric("Feasible combos",   f"{feasible}/{len(combos)}")
            cb.metric("Return (with div)", f"{exp3 * 100:.2f}%")
            cc.metric("Sharpe",            f"{sharpe3:.2f}")

            w3_df = pd.DataFrame.from_dict(best_w3, orient="index", columns=["Weight"])
            w3_df["Weight %"] = (w3_df["Weight"] * 100).round(2)
            w3_df.index = [ticker_names.get(t, t) for t in w3_df.index]
            st.dataframe(w3_df[["Weight %"]], use_container_width=True)

# ── Tab 5: Custom Portfolio Analyser (Section 10) ─────────────────────────
with tab5:
    st.subheader("Custom Portfolio Analyser")
    st.caption("Pick assets and weights — results update automatically. Set weight to 0 to exclude.")

    all_options = stocks_list + ["CASH"]
    defaults    = ["ULVR.L", "AZN.L", "HSBA.L", "CASH", all_options[0], all_options[0]]
    def_weights = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0]

    cp_raw = {}
    cols_a, cols_b = st.columns(2)
    for k in range(6):
        default_t = defaults[k]
        default_i = all_options.index(default_t) if default_t in all_options else 0
        col = cols_a if k % 2 == 0 else cols_b
        with col:
            t = st.selectbox(f"Asset {k+1}", options=all_options,
                             index=default_i, key=f"cp_t{k}")
            w = st.number_input(f"Weight {k+1}", min_value=0.0, max_value=1.0,
                                value=def_weights[k], step=0.05, key=f"cp_w{k}")
            if w > 0:
                cp_raw[t] = cp_raw.get(t, 0) + w

    st.divider()

    if not cp_raw:
        st.info("Set at least one weight above zero to see results.")
    else:
        total_w = sum(cp_raw.values())
        if abs(total_w - 1.0) > 1e-6:
            st.caption(f"Weights summed to {total_w:.2f} — normalised to 1.0.")
        cp = {t: w / total_w for t, w in cp_raw.items()}

        rf_monthly = (1 + risk_free_rate) ** (1 / 12) - 1
        risky      = [t for t in cp if t != "CASH"]

        cp_mean_div = {t: mean_ret_div.get(t, 0) for t in risky}
        cp_mean_div["CASH"] = rf_monthly
        cp_mean_po  = {t: mean_ret_dict.get(t, 0) for t in risky}
        cp_mean_po["CASH"]  = rf_monthly

        risky_in_df = [t for t in risky if t in df.columns]
        if risky_in_df:
            cp_cov      = df[risky_in_df].cov().values
            cp_w_arr    = np.array([cp.get(t, 0) for t in risky_in_df])
            port_var_cp = float(cp_w_arr @ cp_cov @ cp_w_arr)
        else:
            port_var_cp = 0.0

        exp_div_cp = (1 + sum(cp.get(t, 0) * cp_mean_div.get(t, 0) for t in cp)) ** 12 - 1
        exp_po_cp  = (1 + sum(cp.get(t, 0) * cp_mean_po.get(t, 0)  for t in cp)) ** 12 - 1
        ann_std_cp = (port_var_cp ** 0.5) * (12 ** 0.5) if port_var_cp > 0 else 0.0
        sharpe_cp  = (exp_div_cp - risk_free_rate) / ann_std_cp if ann_std_cp > 0 else float("nan")

        cp_names, cp_divs = {}, {}
        for t in cp:
            if t == "CASH":
                cp_names[t] = f"Risk-Free Bond ({risk_free_rate*100:.1f}%)"
                cp_divs[t]  = 0.0
            else:
                cp_names[t] = ticker_names.get(t, t)
                cp_divs[t]  = div_yields.get(t, 0.0)

        res_df = pd.DataFrame({
            "Company":     {t: cp_names[t] for t in cp},
            "Weight %":    {t: round(cp[t] * 100, 2) for t in cp},
            "Div Yield %": {t: round(cp_divs[t], 2) for t in cp},
        })
        res_df["Wtd Div %"] = (res_df["Weight %"] / 100 * res_df["Div Yield %"]).round(3)
        port_wtd_div = res_df["Wtd Div %"].sum()

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Return (with div)",   f"{exp_div_cp * 100:.2f}%")
        cb.metric("Return (price only)", f"{exp_po_cp  * 100:.2f}%")
        cc.metric("Annual Std Dev",      f"{ann_std_cp * 100:.2f}%")
        cd.metric("Sharpe (with div)",   f"{sharpe_cp:.2f}" if not np.isnan(sharpe_cp) else "N/A")

        col_t, col_p = st.columns(2)
        with col_t:
            st.subheader("Holdings")
            st.dataframe(res_df, use_container_width=True)
            st.metric("Portfolio Wtd Div Yield", f"{port_wtd_div:.2f}%")
        with col_p:
            st.subheader("Allocation")
            fig_pie = go.Figure(go.Pie(
                labels=[cp_names[t] for t in cp],
                values=[cp[t] * 100 for t in cp],
                hole=0.35, textinfo="label+percent",
            ))
            fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
