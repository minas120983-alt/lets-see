import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenPort · ESG Portfolio Optimiser",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a2e1a;
}

.stApp {
    background: #f5f2ec;
    color: #1a2e1a;
}

.block-container {
    padding-top: 3.2rem;
    color: #1a2e1a;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #163116 0%, #102910 100%) !important;
    border-right: 1px solid #284528;
}
[data-testid="stSidebar"] * { color: #d6e5cb !important; }
[data-testid="stSidebar"] hr { border-color: #335333 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #d6e5cb !important; }
[data-testid="stSidebar"] label {
    font-size: 0.84rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: #d4a020 !important;
    border: 2px solid #fff7df !important;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input {
    background: #f7f5ef !important;
    color: #1a2e1a !important;
    border: 1px solid #86a173 !important;
    border-radius: 8px !important;
}

h1, h2, h3, h4, h5, h6, p, div, label, span { color: #1a2e1a; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem;
    color: #1a2e1a;
    line-height: 1.06;
    margin-top: 0.9rem;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.02rem;
    color: #5f7a5d;
    margin-bottom: 2.2rem;
    font-weight: 400;
}
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: #1a2e1a;
    border-bottom: 2px solid #c8dab8;
    padding-bottom: 0.4rem;
    margin-top: 0.25rem;
    margin-bottom: 1rem;
}
.info-box {
    background: #e8f5e0;
    border-left: 4px solid #4a8a3a;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.9rem;
    color: #2a4a2a;
}
.warn-box {
    background: #fff8e8;
    border-left: 4px solid #d4a020;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.9rem;
    color: #5a4010;
}
.error-box {
    background: #fdeceb;
    border-left: 4px solid #c0392b;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.9rem;
    color: #7a1f17;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #d4e0c8;
    border-radius: 12px;
    padding: 1.15rem 1.35rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(26, 46, 26, 0.06);
}
.metric-label {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #6e8e62;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #1a2e1a;
    line-height: 1;
}
.metric-unit { font-size: 0.85rem; color: #7d9b72; margin-left: 2px; }

div.stButton > button {
    background: #2d6a2d;
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.8rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 0.98rem;
    letter-spacing: 0.03em;
    width: 100%;
}
div.stButton > button:hover { background: #215221; color: #ffffff !important; }

.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: #ffffff !important;
    color: #1a2e1a !important;
    border: 1px solid #c8d8b8 !important;
    border-radius: 8px !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #1a2e1a !important;
    border: 1px solid #c8d8b8 !important;
}
.stRadio label { color: #1a2e1a !important; }
.stRadio div[role="radiogroup"] label { font-size: 0.96rem !important; font-weight: 600 !important; }
.stCheckbox div[data-testid="stMarkdownContainer"] p { color: #1a2e1a !important; }
.stDataFrame, [data-testid="stDataEditor"] { border-radius: 10px; overflow: hidden; }
[data-testid="stDataEditor"] * { color: #1a2e1a !important; }
[data-testid="stTable"] * { color: #1a2e1a !important; }
[data-testid="stExpander"] {
    border: 1px solid #d4e0c8 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * { color: #1a2e1a !important; font-weight: 600; }
table { color: #1a2e1a !important; }
thead tr th { color: #1a2e1a !important; }
tbody tr td { color: #1a2e1a !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_stats(weights, mu, cov, esg_scores, rf):
    w = np.array(weights)
    ep = float(w @ mu)
    vp = float(w @ cov @ w)
    sp = max(vp, 1e-12) ** 0.5
    sharpe = (ep - rf) / sp if sp > 1e-9 else 0.0
    esg = float(w @ esg_scores)
    return ep, sp, sharpe, esg


def utility(weights, mu, cov, esg_scores, rf, gamma, lam):
    ep, sp, _, esg = portfolio_stats(weights, mu, cov, esg_scores, rf)
    return -(ep - gamma / 2 * sp**2 + lam * esg)


def build_frontier(mu, cov, esg_scores, rf, n_points=160):
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    esg_min = float(np.min(esg_scores))
    esg_max = float(np.max(esg_scores))
    esg_targets = np.linspace(esg_min, esg_max, n_points)

    frontier_sharpe, frontier_ret, frontier_std, frontier_esg, frontier_weights = [], [], [], [], []
    w0 = np.ones(n) / n

    for esg_t in esg_targets:
        esg_constraint = {"type": "ineq", "fun": lambda w, t=esg_t: w @ esg_scores - t}
        def neg_sharpe(w):
            ep, sp, sr, _ = portfolio_stats(w, mu, cov, esg_scores, rf)
            return -sr
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                       constraints=[sum_constraint, esg_constraint],
                       options={"ftol": 1e-9, "maxiter": 500})
        if res.success:
            ep, sp, sr, eg = portfolio_stats(res.x, mu, cov, esg_scores, rf)
            frontier_sharpe.append(sr); frontier_ret.append(ep)
            frontier_std.append(sp); frontier_esg.append(eg)
            frontier_weights.append(res.x)
        else:
            frontier_sharpe.append(np.nan); frontier_ret.append(np.nan)
            frontier_std.append(np.nan); frontier_esg.append(esg_t)
            frontier_weights.append(None)

    return (np.array(frontier_esg), np.array(frontier_sharpe),
            np.array(frontier_ret), np.array(frontier_std), frontier_weights)


def find_optimal(mu, cov, esg_scores, rf, gamma, lam):
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0 = np.ones(n) / n
    res = minimize(utility, w0, args=(mu, cov, esg_scores, rf, gamma, lam),
                   method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    return res.x if res.success else w0


@st.cache_data(show_spinner=False)
def fetch_market_data(tickers, period="3y"):
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=False, group_by="ticker", threads=False)
    close = None
    if isinstance(raw.columns, pd.MultiIndex):
        close_frames = []
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                try:
                    s = raw[t]["Close"].rename(t)
                    close_frames.append(s)
                except Exception:
                    pass
        if close_frames:
            close = pd.concat(close_frames, axis=1)
    else:
        if "Close" in raw.columns:
            close = raw[["Close"]].copy()
            if len(tickers) == 1:
                close.columns = [tickers[0]]

    if close is None or close.empty:
        raise ValueError("No price data could be downloaded for the selected tickers.")

    close = close.dropna(axis=1, how="all").dropna(how="all")
    returns = close.pct_change().dropna(how="all")

    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Not enough valid return data.")

    mu = returns.mean() * 252
    vols = returns.std() * np.sqrt(252)
    cov = returns.cov() * 252
    corr = returns.corr()
    return close, returns, mu, vols, cov, corr


def nearest_psd(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < 1e-8] = 1e-8
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def yahoo_total_esg_to_app_score(total_esg):
    """
    Sustainalytics/Yahoo totalEsg: lower = better (risk score, typically 0–50).
    We convert to a 0–10 higher-is-better scale by mapping 0→10, 50→0.
    """
    if total_esg is None or pd.isna(total_esg):
        return None
    score = 10.0 - (float(total_esg) / 5.0)   # 50 → 0, 0 → 10
    return float(np.clip(score, 0.0, 10.0))


def _try_get_sustainability(ticker_str):
    """
    Robustly attempt to pull sustainability data from yfinance,
    handling both the old .sustainability attribute and new get_sustainability().
    Returns a pd.Series or None.
    """
    tk = yf.Ticker(ticker_str)

    # Attempt 1: new API
    try:
        sus = tk.get_sustainability()
        if sus is not None and not (hasattr(sus, "empty") and sus.empty):
            if isinstance(sus, pd.DataFrame) and not sus.empty:
                return sus.iloc[:, 0]
            if isinstance(sus, pd.Series) and not sus.empty:
                return sus
    except Exception:
        pass

    # Attempt 2: legacy attribute
    try:
        sus = tk.sustainability
        if sus is not None and not (hasattr(sus, "empty") and sus.empty):
            if isinstance(sus, pd.DataFrame) and not sus.empty:
                return sus.iloc[:, 0]
            if isinstance(sus, pd.Series) and not sus.empty:
                return sus
    except Exception:
        pass

    return None


def _series_get(series, *keys):
    """Case-insensitive key lookup in a pd.Series."""
    normalised = {str(k).lower().replace("_", "").replace(" ", ""): k for k in series.index}
    for key in keys:
        target = key.lower().replace("_", "").replace(" ", "")
        if target in normalised:
            val = series.loc[normalised[target]]
            if pd.notna(val):
                return val
    return None


@st.cache_data(show_spinner=False)
def fetch_esg_for_ticker(ticker):
    """
    Primary: yfinance sustainability.
    Fallback: yahooquery esg_scores (if installed).
    Returns a dict with has_esg, app_esg (0-10 scale), and raw sub-scores.
    """
    base = dict(ticker=ticker, yahoo_total_esg=None, environment_score=None,
                social_score=None, governance_score=None, rating_year=None,
                rating_month=None, app_esg=None, has_esg=False, source=None, error=None)

    # ── yfinance ──────────────────────────────────────────────────────────────
    try:
        s = _try_get_sustainability(ticker)
        if s is not None and not s.empty:
            total = _series_get(s, "totalEsg", "totalESG")
            env   = _series_get(s, "environmentScore", "environmentalScore", "eScore")
            soc   = _series_get(s, "socialScore", "sScore")
            gov   = _series_get(s, "governanceScore", "gScore")
            ry    = _series_get(s, "ratingYear")
            rm    = _series_get(s, "ratingMonth")

            if total is not None:
                base.update(
                    yahoo_total_esg=float(total),
                    environment_score=float(env) if env is not None else None,
                    social_score=float(soc) if soc is not None else None,
                    governance_score=float(gov) if gov is not None else None,
                    rating_year=int(ry) if ry is not None else None,
                    rating_month=int(rm) if rm is not None else None,
                    app_esg=yahoo_total_esg_to_app_score(float(total)),
                    has_esg=True,
                    source="yfinance",
                )
                return base
    except Exception as e:
        base["error"] = f"yfinance: {e}"

    # ── yahooquery fallback ────────────────────────────────────────────────────
    try:
        from yahooquery import Ticker as YQTicker
        data = YQTicker(ticker).esg_scores
        payload = (data or {}).get(ticker, {})
        if isinstance(payload, dict) and payload:
            total = payload.get("totalEsg")
            if total is not None and pd.notna(total):
                env = payload.get("environmentScore") or payload.get("environomentScore")
                soc = payload.get("socialScore")
                gov = payload.get("governanceScore")
                base.update(
                    yahoo_total_esg=float(total),
                    environment_score=float(env) if env is not None else None,
                    social_score=float(soc) if soc is not None else None,
                    governance_score=float(gov) if gov is not None else None,
                    rating_year=int(payload["ratingYear"]) if payload.get("ratingYear") else None,
                    rating_month=int(payload["ratingMonth"]) if payload.get("ratingMonth") else None,
                    app_esg=yahoo_total_esg_to_app_score(float(total)),
                    has_esg=True,
                    source="yahooquery",
                )
                return base
    except ImportError:
        yq_err = "yahooquery not installed"
    except Exception as e:
        yq_err = str(e)
    else:
        yq_err = "no data"

    prev_err = base.get("error") or ""
    base["error"] = f"{prev_err} | yahooquery: {yq_err}".strip(" |")
    return base


def fetch_esg_for_tickers(tickers):
    return pd.DataFrame([fetch_esg_for_ticker(t) for t in tickers])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## GreenPort")
    st.markdown("---")
    st.markdown("### Investor Preferences")
    gamma = st.slider("Risk Aversion (γ)", 0.5, 10.0, 3.0, 0.5,
                      help="Higher γ = more risk-averse.")
    lam = st.slider("ESG Preference (λ)", 0.0, 5.0, 1.0, 0.1,
                    help="Higher λ = more weight on ESG quality.")
    rf = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 4.0, 0.1, format="%.1f") / 100

    st.markdown("---")
    st.markdown("### ESG Filter")
    use_exclusion = st.checkbox("Apply exclusion screen", value=False,
                                help="Exclude assets below a minimum ESG score.")
    min_esg = 0.0
    if use_exclusion:
        min_esg = st.slider("Minimum ESG score", 0.0, 10.0, 4.0, 0.5)

    st.markdown("---")
    st.markdown(
        "<small style='color:#d6e5cb'>ECN316 · Sustainable Finance<br>Group Assignment 2026</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">GreenPort</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ESG-aware portfolio optimiser for retail investors</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT MODE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Asset Universe</div>', unsafe_allow_html=True)

mode_col1, mode_col2 = st.columns([1.5, 3])
with mode_col1:
    input_mode = st.radio("Input method", ["Manual input", "Ticker-based input"], horizontal=False)

default_names   = ["Tech ETF","Green Bond","Energy Stock","Healthcare","Consumer ETF",
                   "Infra Fund","EM Equity","Gov Bond","Real Estate","Commodity"]
default_ret     = [9.0, 4.5, 7.0, 7.5, 6.5, 5.5, 10.0, 3.0, 6.0, 5.0]
default_vol     = [18.0, 5.0, 22.0, 15.0, 14.0, 10.0, 25.0, 4.0, 13.0, 20.0]
default_esg     = [6.5, 8.5, 2.0, 7.0, 5.5, 7.5, 4.0, 6.0, 5.0, 3.5]
default_tickers = ["AAPL","MSFT","XOM","JNJ","SPY","TLT","NVDA","VWO","GLD","META"]

asset_data      = []
ticker_rows     = []
corr_df         = None
lookback_period = "3y"

# ══════════════════════════════════════════════════════════════════════════════
# MANUAL MODE
# ══════════════════════════════════════════════════════════════════════════════

if input_mode == "Manual input":
    col_left, col_right = st.columns([2, 1])
    with col_right:
        n_assets = st.number_input("Number of assets", min_value=2, max_value=10, value=3, step=1)
        st.markdown('<div class="info-box">Enter expected return, volatility and ESG score for each asset.</div>',
                    unsafe_allow_html=True)
    with col_left:
        cols = st.columns([2, 1.2, 1.2, 1.2])
        cols[0].markdown("**Asset name**")
        cols[1].markdown("**E[R] (%)**")
        cols[2].markdown("**σ (%)**")
        cols[3].markdown("**ESG (0–10)**")
        for i in range(int(n_assets)):
            c0, c1, c2, c3 = st.columns([2, 1.2, 1.2, 1.2])
            name = c0.text_input("", value=default_names[i], key=f"name_{i}", label_visibility="collapsed")
            ret  = c1.number_input("", value=default_ret[i], key=f"ret_{i}",  label_visibility="collapsed", format="%.1f")
            vol  = c2.number_input("", value=default_vol[i], key=f"vol_{i}",  label_visibility="collapsed", format="%.1f", min_value=0.1)
            esg  = c3.number_input("", value=default_esg[i], key=f"esg_{i}",  label_visibility="collapsed", format="%.1f", min_value=0.0, max_value=10.0)
            asset_data.append({"name": name, "ret": ret / 100, "vol": vol / 100, "esg": esg})

    st.markdown("**Correlation Matrix**")
    st.markdown('<div class="info-box">Enter pairwise correlations (−1 to 1). Diagonal is fixed at 1.</div>',
                unsafe_allow_html=True)
    n = int(n_assets)
    corr_init = pd.DataFrame(np.eye(n),
                             columns=[asset_data[i]["name"] for i in range(n)],
                             index=[asset_data[i]["name"] for i in range(n)])
    for r in range(n):
        for c in range(n):
            if r != c:
                corr_init.iloc[r, c] = 0.25
    corr_df = st.data_editor(corr_init, use_container_width=True, key="corr_matrix")


# ══════════════════════════════════════════════════════════════════════════════
# TICKER MODE  ── two-phase: collect tickers → fetch ESG → show fallback only
#                              where needed
# ══════════════════════════════════════════════════════════════════════════════

else:
    col_left, col_right = st.columns([2, 1])
    with col_right:
        n_assets = st.number_input("Number of assets", min_value=2, max_value=10, value=3,
                                   step=1, key="n_ticker_assets")
        lookback_period = st.selectbox("History window", ["1y", "3y", "5y", "10y"], index=1)

    # ── Phase 1: collect tickers & names only ─────────────────────────────────
    with col_left:
        cols = st.columns([1.1, 1.8])
        cols[0].markdown("**Ticker**")
        cols[1].markdown("**Display name**")
        for i in range(int(n_assets)):
            c1, c2 = st.columns([1.1, 1.8])
            ticker = c1.text_input("", value=default_tickers[i], key=f"ticker_{i}",
                                   label_visibility="collapsed").upper().strip()
            name   = c2.text_input("", value=default_names[i],   key=f"ticker_name_{i}",
                                   label_visibility="collapsed")
            ticker_rows.append({"ticker": ticker, "name": name or ticker, "manual_esg": None})

    # ── Phase 2: fetch ESG immediately (cached) & show fallback inputs only
    #            for tickers that failed ────────────────────────────────────────
    valid_tickers = [r["ticker"] for r in ticker_rows if r["ticker"]]
    if valid_tickers:
        with st.spinner("Checking ESG data availability…"):
            esg_preview = {r["ticker"]: fetch_esg_for_ticker(r["ticker"]) for r in ticker_rows if r["ticker"]}

        missing_esg = [t for t, res in esg_preview.items() if not res["has_esg"]]

        if missing_esg:
            st.markdown(
                f'<div class="warn-box"><strong>ESG data not found automatically</strong> for: '
                f'<strong>{", ".join(missing_esg)}</strong>. '
                f'Please enter a manual ESG score (0–10) for each below.</div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Manual ESG scores for tickers without automatic data:**")
            fallback_cols = st.columns(min(len(missing_esg), 5))
            manual_overrides = {}
            for idx, t in enumerate(missing_esg):
                col = fallback_cols[idx % len(fallback_cols)]
                default_idx = default_tickers.index(t) if t in default_tickers else 0
                manual_overrides[t] = col.number_input(
                    f"{t} ESG",
                    value=float(default_esg[default_idx]),
                    min_value=0.0,
                    max_value=10.0,
                    step=0.1,
                    format="%.1f",
                    key=f"manual_esg_{t}",
                )
        else:
            manual_overrides = {}
            st.markdown(
                '<div class="info-box">✓ ESG data found automatically for all tickers.</div>',
                unsafe_allow_html=True,
            )

        # store manual_esg back into ticker_rows
        for row in ticker_rows:
            t = row["ticker"]
            if t in manual_overrides:
                row["manual_esg"] = manual_overrides[t]
            else:
                row["manual_esg"] = None  # will use auto fetch


# ══════════════════════════════════════════════════════════════════════════════
# RUN BUTTON
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
run_col, _ = st.columns([1, 3])
with run_col:
    run = st.button("Optimise Portfolio")


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if run:
    ticker_data_display = None

    # ── Manual mode ───────────────────────────────────────────────────────────
    if input_mode == "Manual input":
        names      = [d["name"] for d in asset_data]
        mu         = np.array([d["ret"] for d in asset_data], dtype=float)
        vols       = np.array([d["vol"] for d in asset_data], dtype=float)
        esg_scores = np.array([d["esg"] for d in asset_data], dtype=float)
        n          = len(names)

        try:
            corr_np = corr_df.values.astype(float)
        except Exception:
            st.error("Please make sure all correlation values are numeric.")
            st.stop()

        corr_np = (corr_np + corr_np.T) / 2
        np.fill_diagonal(corr_np, 1.0)
        corr_np = np.clip(corr_np, -0.999, 0.999)
        cov = np.outer(vols, vols) * corr_np

    # ── Ticker mode ────────────────────────────────────────────────────────────
    else:
        tickers = [row["ticker"] for row in ticker_rows if row["ticker"]]
        if len(tickers) < 2:
            st.error("Please enter at least two valid ticker symbols.")
            st.stop()

        try:
            prices, returns, mu_series, vols_series, cov_df, corr_df_market = \
                fetch_market_data(tickers, period=lookback_period)
        except Exception as e:
            st.error(f"Failed to fetch ticker data: {e}")
            st.stop()

        available     = [t for t in tickers if t in mu_series.index]
        filtered_rows = [row for row in ticker_rows if row["ticker"] in available]

        if len(available) < 2:
            st.error("Not enough valid tickers returned. Please check the symbols.")
            st.stop()

        # Re-use already-fetched ESG (cached) ──────────────────────────────────
        esg_map = {t: fetch_esg_for_ticker(t) for t in available}

        resolved_rows   = []
        used_manual_for = []

        for row in filtered_rows:
            t    = row["ticker"]
            meta = esg_map.get(t, {})

            if meta.get("has_esg") and meta.get("app_esg") is not None:
                final_esg  = float(meta["app_esg"])
                esg_source = meta.get("source", "Automatic")
            else:
                # Use the manual override entered by the user in Phase 2
                manual_val = row.get("manual_esg")
                if manual_val is None:
                    # Safety fallback if somehow manual not entered
                    manual_val = 5.0
                final_esg  = float(manual_val)
                esg_source = "Manual fallback"
                used_manual_for.append(t)

            resolved_rows.append({
                "ticker":          t,
                "name":            row["name"],
                "final_esg":       final_esg,
                "esg_source":      esg_source,
                "yahoo_total_esg": meta.get("yahoo_total_esg"),
                "environment_score": meta.get("environment_score"),
                "social_score":    meta.get("social_score"),
                "governance_score":meta.get("governance_score"),
                "rating_year":     meta.get("rating_year"),
                "rating_month":    meta.get("rating_month"),
                "error":           meta.get("error"),
            })

        if used_manual_for:
            st.markdown(
                f'<div class="error-box"><strong>Manual ESG scores used for:</strong> '
                f'{", ".join(used_manual_for)}. These values were taken from your manual inputs above.</div>',
                unsafe_allow_html=True,
            )

        names      = [row["name"]      for row in resolved_rows]
        esg_scores = np.array([row["final_esg"] for row in resolved_rows], dtype=float)
        mu         = mu_series.loc[available].values.astype(float)
        vols       = vols_series.loc[available].values.astype(float)
        cov        = cov_df.loc[available, available].values.astype(float)
        corr_np    = corr_df_market.loc[available, available].values.astype(float)
        n          = len(available)

        ticker_data_display = pd.DataFrame({
            "Ticker":               available,
            "Display Name":         names,
            "Expected Return E[R] (%)": (mu_series.loc[available].values * 100).round(2),
            "Volatility σ (%)":     (vols_series.loc[available].values * 100).round(2),
            "Yahoo totalEsg":       [esg_map[t].get("yahoo_total_esg") for t in available],
            "Yahoo Environment":    [esg_map[t].get("environment_score") for t in available],
            "Yahoo Social":         [esg_map[t].get("social_score") for t in available],
            "Yahoo Governance":     [esg_map[t].get("governance_score") for t in available],
            "Yahoo Rating Year":    [esg_map[t].get("rating_year") for t in available],
            "Yahoo Rating Month":   [esg_map[t].get("rating_month") for t in available],
            "Final ESG Used (0-10)":[row["final_esg"] for row in resolved_rows],
            "ESG Source":           [row["esg_source"] for row in resolved_rows],
            "Fetch Error":          [row["error"] for row in resolved_rows],
        })

        st.markdown(
            f'<div class="info-box">Market data loaded for: {", ".join(available)}. '
            f'Returns annualised from daily data over {lookback_period}.</div>',
            unsafe_allow_html=True,
        )

    # ── PSD check ─────────────────────────────────────────────────────────────
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < -1e-8):
        st.markdown(
            '<div class="warn-box">Covariance matrix is not PSD. A numerical adjustment has been applied.</div>',
            unsafe_allow_html=True,
        )
        cov = nearest_psd(cov)

    # ── ESG exclusion screen ──────────────────────────────────────────────────
    active_mask = np.ones(n, dtype=bool)
    if use_exclusion:
        active_mask = esg_scores >= min_esg
        excluded = [names[i] for i in range(n) if not active_mask[i]]
        if excluded:
            st.markdown(f'<div class="warn-box">Excluded by ESG screen: {", ".join(excluded)}</div>',
                        unsafe_allow_html=True)

    active_idx = np.where(active_mask)[0]
    if len(active_idx) < 2:
        st.error("At least two assets must pass the ESG screen. Relax the filter.")
        st.stop()

    mu_a    = mu[active_idx]
    cov_a   = cov[np.ix_(active_idx, active_idx)]
    esg_a   = esg_scores[active_idx]
    names_a = [names[i] for i in active_idx]
    vols_a  = vols[active_idx]

    w_opt_a   = find_optimal(mu_a, cov_a, esg_a, rf, gamma, lam)
    w_opt     = np.zeros(n)
    for idx, wi in zip(active_idx, w_opt_a):
        w_opt[idx] = wi

    ep, sp, sr, esg_bar = portfolio_stats(w_opt_a, mu_a, cov_a, esg_a, rf)

    with st.spinner("Building ESG-efficient frontier…"):
        f_esg, f_sharpe, f_ret, f_std, f_weights = build_frontier(mu_a, cov_a, esg_a, rf, n_points=180)

    valid = ~np.isnan(f_sharpe)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Optimal Portfolio</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, unit in [
        (m1, "Expected Return", f"{ep*100:.2f}", "%"),
        (m2, "Volatility (σ)",  f"{sp*100:.2f}", "%"),
        (m3, "Sharpe Ratio",    f"{sr:.3f}",     ""),
        (m4, "ESG Score",       f"{esg_bar:.2f}", "/ 10"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{val}<span class="metric-unit">{unit}</span></div></div>',
                unsafe_allow_html=True,
            )

    u_val = ep - gamma / 2 * sp**2 + lam * esg_bar
    st.markdown(
        f'<div class="info-box">U = E[Rp] − (γ/2)σ² + λs̄ = <strong>{u_val:.4f}</strong>'
        f' &nbsp;|&nbsp; γ = {gamma}, λ = {lam}, r_f = {rf*100:.1f}%</div>',
        unsafe_allow_html=True,
    )

    # ── Weights table ─────────────────────────────────────────────────────────
    st.markdown("#### Portfolio Weights")
    weight_df = pd.DataFrame({
        "Asset":      names,
        "Weight (%)": [f"{w*100:.2f}" for w in w_opt],
        "E[R] (%)":   [f"{r*100:.2f}" for r in mu],
        "σ (%)":      [f"{v*100:.2f}" for v in vols],
        "ESG Score":  [f"{s:.2f}" for s in esg_scores],
        "Included":   ["Yes" if m else "No" for m in active_mask],
    })
    st.dataframe(weight_df, use_container_width=True, hide_index=True)

    if input_mode == "Ticker-based input":
        st.markdown("#### Ticker Data Used")
        st.dataframe(ticker_data_display, use_container_width=True, hide_index=True)
        st.markdown("#### Estimated Correlation Matrix")
        st.dataframe(pd.DataFrame(corr_np, index=names, columns=names).round(3),
                     use_container_width=True)
        st.markdown("#### Historical Adjusted Close Prices")
        display_prices = prices[available].copy()
        display_prices.columns = names
        st.dataframe(display_prices.tail(30), use_container_width=True)
        st.markdown("#### Historical Daily Returns")
        display_returns = returns[available].copy()
        display_returns.columns = names
        st.dataframe((display_returns.tail(30) * 100).round(3), use_container_width=True)

    # ── Frontier charts ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">ESG-Efficient Frontier</div>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor('#f5f2ec'); ax.set_facecolor('#f5f2ec')
        if valid.sum() > 1:
            ax.plot(f_esg[valid], f_sharpe[valid], color='#2d6a2d', lw=2.5, label='ESG-efficient frontier')
            ax.fill_between(f_esg[valid], f_sharpe[valid], alpha=0.12, color='#6aa35d')
        for i in range(len(mu_a)):
            sr_i = (mu_a[i] - rf) / vols_a[i]
            ax.scatter(esg_a[i], sr_i, color='#88b179', zorder=5, s=60, edgecolors='#2d6a2d', lw=1)
            ax.annotate(names_a[i], (esg_a[i], sr_i), textcoords="offset points",
                        xytext=(5, 4), fontsize=7.5, color='#2d4a2d')
        ax.scatter(esg_bar, sr, color='#c76b2f', zorder=10, s=120, edgecolors='white', lw=2, label='Optimal')
        ax.set_xlabel("Average ESG Score", fontsize=9, color='#2d4a2d')
        ax.set_ylabel("Sharpe Ratio", fontsize=9, color='#2d4a2d')
        ax.set_title("ESG Score vs Sharpe Ratio", fontsize=11, fontweight='bold', color='#1a2e1a', pad=10)
        ax.tick_params(colors='#5a7a5a', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('#c8d8b8')
        ax.legend(fontsize=8, framealpha=0.85, facecolor='#f5f2ec', edgecolor='#c8d8b8')
        ax.grid(True, alpha=0.3, color='#c8d8b8', linestyle='--')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        fig2.patch.set_facecolor('#f5f2ec'); ax2.set_facecolor('#f5f2ec')
        if valid.sum() > 1:
            sc = ax2.scatter(f_std[valid]*100, f_ret[valid]*100, c=f_esg[valid],
                             cmap='YlGn', s=14, zorder=3,
                             vmin=float(np.min(esg_a)), vmax=float(np.max(esg_a)))
            cb = fig2.colorbar(sc, ax=ax2, pad=0.02)
            cb.set_label("ESG Score", fontsize=8, color='#2d4a2d')
            cb.ax.tick_params(labelsize=7, colors='#5a7a5a')
        for i in range(len(mu_a)):
            ax2.scatter(vols_a[i]*100, mu_a[i]*100, color='#2d6a2d', zorder=5,
                        s=60, edgecolors='white', lw=1.2)
            ax2.annotate(names_a[i], (vols_a[i]*100, mu_a[i]*100),
                         textcoords="offset points", xytext=(5, 3), fontsize=7.5, color='#2d4a2d')
        ax2.scatter(sp*100, ep*100, color='#c76b2f', zorder=10, s=140,
                    edgecolors='white', lw=2, label='Optimal', marker='*')
        ax2.set_xlabel("Volatility σ (%)", fontsize=9, color='#2d4a2d')
        ax2.set_ylabel("Expected Return E[R] (%)", fontsize=9, color='#2d4a2d')
        ax2.set_title("Mean-Variance Space", fontsize=11, fontweight='bold', color='#1a2e1a', pad=10)
        ax2.tick_params(colors='#5a7a5a', labelsize=8)
        for spine in ax2.spines.values(): spine.set_color('#c8d8b8')
        ax2.legend(fontsize=8, framealpha=0.85, facecolor='#f5f2ec', edgecolor='#c8d8b8')
        ax2.grid(True, alpha=0.3, color='#c8d8b8', linestyle='--')
        fig2.tight_layout(); st.pyplot(fig2); plt.close()

    # ── Allocation charts ─────────────────────────────────────────────────────
    st.markdown("#### Portfolio Allocation")
    pie_col, bar_col = st.columns(2)
    nonzero = [(names[i], w_opt[i]) for i in range(n) if w_opt[i] > 0.005]
    if nonzero:
        pie_labels, pie_vals = zip(*nonzero)
        greens = ['#1a4a1a','#2d6a2d','#4a8a3a','#6aaa5a','#8aba7a',
                  '#a8cc98','#c4deb8','#d4e8c8','#e4f0d8','#f0f8ec']
        with pie_col:
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            fig3.patch.set_facecolor('#f5f2ec'); ax3.set_facecolor('#f5f2ec')
            ax3.pie(pie_vals, labels=pie_labels, autopct='%1.1f%%',
                    colors=greens[:len(pie_vals)], startangle=140,
                    textprops={'fontsize': 8, 'color': '#1a2e1a'},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
            ax3.set_title("Weight Allocation", fontsize=11, fontweight='bold', color='#1a2e1a', pad=10)
            fig3.tight_layout(); st.pyplot(fig3); plt.close()

        with bar_col:
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            fig4.patch.set_facecolor('#f5f2ec'); ax4.set_facecolor('#f5f2ec')
            bar_names   = [names[i]        for i in range(n) if w_opt[i] > 0.005]
            bar_weights = [w_opt[i]*100    for i in range(n) if w_opt[i] > 0.005]
            bar_esg     = [esg_scores[i]   for i in range(n) if w_opt[i] > 0.005]
            bar_colors  = [plt.cm.YlGn(s / 10) for s in bar_esg]
            bars = ax4.barh(bar_names, bar_weights, color=bar_colors, edgecolor='white', height=0.6)
            for bar, ev in zip(bars, bar_esg):
                ax4.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                         f'ESG {ev:.1f}', va='center', fontsize=7.5, color='#2d4a2d')
            ax4.set_xlabel("Weight (%)", fontsize=9, color='#2d4a2d')
            ax4.set_title("Weights with ESG Scores", fontsize=11, fontweight='bold', color='#1a2e1a', pad=10)
            ax4.tick_params(colors='#5a7a5a', labelsize=8)
            for spine in ax4.spines.values(): spine.set_color('#c8d8b8')
            ax4.grid(True, alpha=0.3, color='#c8d8b8', axis='x', linestyle='--')
            fig4.tight_layout(); st.pyplot(fig4); plt.close()

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    with st.expander("Sensitivity Analysis — ESG Preference (λ)"):
        st.markdown("How the optimal portfolio changes as λ varies from 0 to 5.")
        lam_vals  = np.linspace(0, 5, 20)
        sens_rows = []
        for lv in lam_vals:
            ww = find_optimal(mu_a, cov_a, esg_a, rf, gamma, lv)
            ep2, sp2, sr2, esg2 = portfolio_stats(ww, mu_a, cov_a, esg_a, rf)
            sens_rows.append({"λ": round(float(lv),2), "E[R] (%)": round(ep2*100,2),
                               "σ (%)": round(sp2*100,2), "Sharpe": round(sr2,3),
                               "ESG": round(esg2,2)})
        sens_df = pd.DataFrame(sens_rows)
        fig5, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        fig5.patch.set_facecolor('#f5f2ec')
        for ax_ in axes:
            ax_.set_facecolor('#f5f2ec'); ax_.tick_params(colors='#5a7a5a', labelsize=8)
            for sp_ in ax_.spines.values(): sp_.set_color('#c8d8b8')
            ax_.grid(True, alpha=0.3, color='#c8d8b8', linestyle='--')
        axes[0].plot(sens_df["λ"], sens_df["Sharpe"], color='#2d6a2d', lw=2)
        axes[0].set_title("Sharpe vs λ", fontsize=10, color='#1a2e1a')
        axes[0].set_xlabel("λ", fontsize=9); axes[0].set_ylabel("Sharpe Ratio", fontsize=9)
        axes[1].plot(sens_df["λ"], sens_df["ESG"], color='#4a8a3a', lw=2)
        axes[1].set_title("ESG Score vs λ", fontsize=10, color='#1a2e1a')
        axes[1].set_xlabel("λ", fontsize=9); axes[1].set_ylabel("ESG Score", fontsize=9)
        axes[2].plot(sens_df["λ"], sens_df["E[R] (%)"], color='#6aaa5a', lw=2, label='E[R]')
        axes[2].plot(sens_df["λ"], sens_df["σ (%)"], color='#c76b2f', lw=2, linestyle='--', label='σ')
        axes[2].set_title("Return and Risk vs λ", fontsize=10, color='#1a2e1a')
        axes[2].set_xlabel("λ", fontsize=9); axes[2].set_ylabel("%", fontsize=9)
        axes[2].legend(fontsize=8, facecolor='#f5f2ec', edgecolor='#c8d8b8')
        fig5.tight_layout(); st.pyplot(fig5); plt.close()
        st.dataframe(sens_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        '<div class="info-box"><strong>Methodology:</strong> Utility U = E[Rp] − (γ/2)σ²p + λs̄. '
        'ESG-efficient frontier maximises Sharpe ratio subject to a minimum ESG constraint at each point. '
        'Optimisation uses SLSQP with no short-selling. Sustainalytics totalEsg (lower = better) is '
        'converted to a 0–10 higher-is-better scale via score = 10 − totalEsg/5.</div>',
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        '<div class="warn-box">Configure the asset universe and preferences, then click '
        '<strong>Optimise Portfolio</strong> to generate results.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("How does the model work?"):
        st.markdown("""
**Utility Function**

$$U = E[R_p] - \\frac{\\gamma}{2}\\sigma_p^2 + \\lambda \\bar{s}$$

| Symbol | Meaning |
|--------|---------|
| $E[R_p]$ | Expected portfolio return |
| $\\sigma_p^2$ | Portfolio variance |
| $\\gamma$ | Risk aversion parameter |
| $\\lambda$ | ESG preference intensity |
| $\\bar{s}$ | Weighted average ESG score |

**ESG-Efficient Frontier**

For each ESG level, the model finds the portfolio with the highest Sharpe ratio subject to meeting that ESG target, producing a frontier in ESG–Sharpe space.

**ESG Score Conversion**

Yahoo Finance / Sustainalytics provide a `totalEsg` *risk* score (lower = better, roughly 0–50). GreenPort converts this to a 0–10 higher-is-better scale:

$$\\text{app\\_esg} = 10 - \\frac{\\text{totalEsg}}{5}$$

**Input Modes**

- **Manual input** — enter expected returns, volatilities and correlations directly.
- **Ticker-based input** — estimates return and risk from Yahoo Finance history. ESG is fetched automatically; manual inputs appear **only** for tickers where automatic data is unavailable.

**Optional fallback**

```bash
pip install yahooquery
```
""")
