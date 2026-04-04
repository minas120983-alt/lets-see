import warnings
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenPort · ESG Portfolio Optimiser",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #1a2e1a; }
.stApp { background: #f5f2ec; color: #1a2e1a; }
.block-container { padding-top: 3.2rem; color: #1a2e1a; }
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
    font-size: 0.84rem !important; font-weight: 700 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: #d4a020 !important; border: 2px solid #fff7df !important;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input {
    background: #f7f5ef !important; color: #1a2e1a !important;
    border: 1px solid #86a173 !important; border-radius: 8px !important;
}
h1, h2, h3, h4, h5, h6, p, div, label, span { color: #1a2e1a; }
.hero-title {
    font-family: 'DM Serif Display', serif; font-size: 3.4rem;
    color: #1a2e1a; line-height: 1.06; margin-top: 0.9rem; margin-bottom: 0.2rem;
}
.hero-sub { font-size: 1.02rem; color: #5f7a5d; margin-bottom: 2.2rem; font-weight: 400; }
.section-header {
    font-family: 'DM Serif Display', serif; font-size: 1.45rem; color: #1a2e1a;
    border-bottom: 2px solid #c8dab8; padding-bottom: 0.4rem;
    margin-top: 0.25rem; margin-bottom: 1rem;
}
.info-box {
    background: #e8f5e0; border-left: 4px solid #4a8a3a; border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem; margin: 0.6rem 0; font-size: 0.9rem; color: #2a4a2a;
}
.warn-box {
    background: #fff8e8; border-left: 4px solid #d4a020; border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem; margin: 0.6rem 0; font-size: 0.9rem; color: #5a4010;
}
.error-box {
    background: #fdeceb; border-left: 4px solid #c0392b; border-radius: 0 8px 8px 0;
    padding: 0.85rem 1rem; margin: 0.6rem 0; font-size: 0.9rem; color: #7a1f17;
}
.metric-card {
    background: #ffffff; border: 1px solid #d4e0c8; border-radius: 12px;
    padding: 1.15rem 1.35rem; margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(26,46,26,0.06);
}
.metric-label {
    font-size: 0.74rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #6e8e62; margin-bottom: 0.2rem;
}
.metric-value { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #1a2e1a; line-height: 1; }
.metric-unit { font-size: 0.85rem; color: #7d9b72; margin-left: 2px; }
div.stButton > button {
    background: #2d6a2d; color: #ffffff !important; border: none; border-radius: 8px;
    padding: 0.7rem 1.8rem; font-family: 'DM Sans', sans-serif;
    font-weight: 700; font-size: 0.98rem; letter-spacing: 0.03em; width: 100%;
}
div.stButton > button:hover { background: #215221; color: #ffffff !important; }
.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: #ffffff !important; color: #1a2e1a !important;
    border: 1px solid #c8d8b8 !important; border-radius: 8px !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff !important; color: #1a2e1a !important; border: 1px solid #c8d8b8 !important;
}
.stRadio label { color: #1a2e1a !important; }
.stRadio div[role="radiogroup"] label { font-size: 0.96rem !important; font-weight: 600 !important; }
.stCheckbox div[data-testid="stMarkdownContainer"] p { color: #1a2e1a !important; }
.stDataFrame, [data-testid="stDataEditor"] { border-radius: 10px; overflow: hidden; }
[data-testid="stDataEditor"] * { color: #1a2e1a !important; }
[data-testid="stTable"] * { color: #1a2e1a !important; }
[data-testid="stExpander"] {
    border: 1px solid #d4e0c8 !important; border-radius: 10px !important; background: #ffffff !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * { color: #1a2e1a !important; font-weight: 600; }
table, thead tr th, tbody tr td { color: #1a2e1a !important; }

/* ── Chatbot ── */
.chat-wrap {
    border-radius: 14px; overflow: hidden;
    border: 1px solid #c8dab8;
    box-shadow: 0 4px 20px rgba(26,46,26,0.10);
    margin-top: 0.5rem;
}
.chat-header {
    background: linear-gradient(135deg, #163116 0%, #2d6a2d 100%);
    padding: 1rem 1.4rem 0.85rem;
    display: flex; align-items: center; gap: 0.75rem;
}
.chat-header-icon { font-size: 1.7rem; line-height: 1; }
.chat-header-title { color: #f0f8ec !important; margin: 0 0 0.15rem; font-size: 1.05rem; font-weight: 700; }
.chat-header-sub   { color: #a8c898 !important; margin: 0; font-size: 0.8rem; }
.chat-body { background: #ffffff; padding: 1rem 1.2rem 0.8rem; }
.chat-msg-user {
    background: #2d6a2d; color: #ffffff !important;
    border-radius: 14px 14px 4px 14px;
    padding: 0.55rem 0.85rem; margin: 0.35rem 0 0.35rem 18%;
    font-size: 0.875rem; display: block; text-align: right;
}
.chat-msg-assistant {
    background: #f0f7eb; color: #1a2e1a !important;
    border-radius: 14px 14px 14px 4px;
    padding: 0.65rem 0.9rem; margin: 0.35rem 18% 0.35rem 0;
    font-size: 0.875rem; display: block; line-height: 1.6;
}
.chat-pills { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0 0.7rem; }
.chat-pill {
    background: #e8f5e0; border: 1px solid #6aaa5a;
    border-radius: 20px; padding: 0.28rem 0.72rem;
    font-size: 0.76rem; color: #2a4a2a !important; cursor: pointer;
    transition: background 0.15s;
}
.chat-pill:hover { background: #c8e8b0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ESG DATABASE — loaded from the uploaded LSEG CSV
# valuescore: 0–1 scale, higher = better (LSEG/Refinitiv ESGCombinedScore).
# We take the most recent year per ticker and scale to 0–10 for display.
# ══════════════════════════════════════════════════════════════════════════════

# Raw GitHub URL for the ESG CSV (filename has a space — encoded as %20)
_ESG_CSV_URL = (
    "https://raw.githubusercontent.com/minas120983-alt/lets-see/main/ESG%20data%202026.csv"
)
# Local fallback path (works when running locally or on Streamlit Community Cloud)
_ESG_CSV_LOCAL = "/mnt/user-data/uploads/ESG data 2026.csv"


def _parse_esg_df(df: pd.DataFrame) -> dict:
    """Convert a raw ESG DataFrame into the app's ticker→dict lookup."""
    df = df[df["fieldname"] == "ESGCombinedScore"].copy()
    df["valuescore"] = pd.to_numeric(df["valuescore"], errors="coerce")
    df = df.dropna(subset=["valuescore", "ticker"])
    df["ticker"] = df["ticker"].str.upper().str.strip()
    latest = df.sort_values("year").groupby("ticker").last().reset_index()
    return {
        row["ticker"]: {
            "app_esg": round(float(row["valuescore"]) * 10, 3),
            "letter":  str(row["value"]),
            "year":    int(row["year"]),
            "source":  f"LSEG ESGCombinedScore ({int(row['year'])})",
            "has_esg": True,
        }
        for _, row in latest.iterrows()
    }


@st.cache_data(show_spinner=False)
def load_esg_db() -> dict:
    """
    Load ESG data with two-source fallback:
      1. Raw GitHub URL (primary — works on any deployment)
      2. Local upload path (fallback for Streamlit Cloud / local runs)
    Returns a ticker→dict lookup or empty dict on complete failure.
    """
    # Source 1: GitHub raw URL
    try:
        resp = requests.get(_ESG_CSV_URL, timeout=15)
        resp.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(resp.text))
        result = _parse_esg_df(df)
        if result:
            return result
    except Exception:
        pass

    # Source 2: local file
    try:
        df = pd.read_csv(_ESG_CSV_LOCAL)
        result = _parse_esg_df(df)
        if result:
            return result
    except Exception:
        pass

    return {}


_ESG_DB: dict = load_esg_db()


def lookup_esg(ticker: str) -> dict:
    t = ticker.upper().strip()
    if t in _ESG_DB:
        return {"ticker": t, **_ESG_DB[t], "error": None}
    return {"ticker": t, "app_esg": None, "letter": None, "year": None,
            "source": None, "has_esg": False,
            "error": f"'{t}' not found in ESG CSV."}


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO MATH
# ══════════════════════════════════════════════════════════════════════════════

def port_ret(w, mu):    return float(np.asarray(w) @ np.asarray(mu))
def port_var(w, cov):   return float(np.asarray(w) @ np.asarray(cov) @ np.asarray(w))
def port_sd(w, cov):    return float(max(port_var(w, cov), 1e-14) ** 0.5)
def port_sr(w, mu, cov, rf): ep = port_ret(w,mu); sp = port_sd(w,cov); return (ep-rf)/sp if sp>1e-9 else 0.

def port_stats(w, mu, cov, esg, rf):
    w = np.asarray(w)
    ep = port_ret(w, mu); sp = port_sd(w, cov)
    return ep, sp, (ep-rf)/sp if sp>1e-9 else 0., float(w @ esg)

def _minimise_sd(mu, cov, extra_constraints=(), bounds=None, n_pts=1):
    n = len(mu)
    b = bounds or [(0., 1.)] * n
    res = minimize(lambda w: port_sd(w, cov), np.ones(n)/n, method="SLSQP",
                   bounds=b,
                   constraints=[{"type":"eq","fun": lambda w: np.sum(w)-1}, *extra_constraints],
                   options={"ftol":1e-10,"maxiter":800})
    return res.x if res.success else np.ones(n)/n

def find_tangency(mu, cov, rf, bounds=None):
    n = len(mu)
    b = bounds or [(0.,1.)]*n
    res = minimize(lambda w: -port_sr(w,mu,cov,rf), np.ones(n)/n, method="SLSQP",
                   bounds=b,
                   constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
                   options={"ftol":1e-10,"maxiter":800})
    wt = res.x if res.success else np.ones(n)/n
    return wt, port_ret(wt,mu), port_sd(wt,cov), port_sr(wt,mu,cov,rf)

def find_optimal(mu, cov, esg, rf, gamma, lam):
    n = len(mu)
    res = minimize(
        lambda w: -(port_ret(w,mu) - gamma/2*port_var(w,cov) + lam*float(np.asarray(w)@esg)),
        np.ones(n)/n, method="SLSQP",
        bounds=[(0.,1.)]*n,
        constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
        options={"ftol":1e-10,"maxiter":1000})
    return res.x if res.success else np.ones(n)/n

def build_mv_frontier(mu, cov, bounds=None, n_points=100):
    """
    True mean-variance frontier by minimising σ for each target return.
    Returns (std_arr_pct, ret_arr_pct).
    """
    n = len(mu)
    b = bounds or [(0.,1.)]*n
    w_mv = _minimise_sd(mu, cov, bounds=b)
    ret_min = port_ret(w_mv, mu)
    # Upper bound: max return achievable with these bounds
    ret_max = float(np.max([port_ret(np.eye(n)[i], mu) for i in range(n)
                             if b[i][1] > 0]))
    targets = np.linspace(ret_min, ret_max, n_points)
    stds, rets = [], []
    for rt in targets:
        c_ret = {"type":"eq","fun": lambda w, r=rt: port_ret(w,mu)-r}
        res = minimize(lambda w: port_sd(w,cov), np.ones(n)/n, method="SLSQP",
                       bounds=b,
                       constraints=[{"type":"eq","fun":lambda w:np.sum(w)-1}, c_ret],
                       options={"ftol":1e-10,"maxiter":500})
        if res.success:
            stds.append(port_sd(res.x, cov)*100)
            rets.append(port_ret(res.x, mu)*100)
    return np.array(stds), np.array(rets)


def nearest_psd(matrix):
    ev, evec = np.linalg.eigh(matrix)
    ev[ev < 1e-8] = 1e-8
    return evec @ np.diag(ev) @ evec.T


# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT — portfolio explainer
# ══════════════════════════════════════════════════════════════════════════════

def build_portfolio_context(names, mu, vols, esg_scores, w_opt,
                             ep, sp, sr, esg_bar, gamma, lam, rf,
                             ep_tan_all, sp_tan_all, sr_tan_all,
                             ep_tan_esg, sp_tan_esg, sr_tan_esg,
                             active_mask, esg_thresh):
    """Build a plain-text summary of the computed portfolio for the AI system prompt."""
    lines = [
        "=== GreenPort Portfolio Summary ===",
        f"Investor parameters: gamma (risk aversion)={gamma}, lambda (ESG preference)={lam}, "
        f"risk-free rate={rf*100:.2f}%",
        "",
        "--- Assets ---",
    ]
    for i, name in enumerate(names):
        flag = "yes" if active_mask[i] else f"no (ESG {esg_scores[i]:.2f} < threshold {esg_thresh:.1f})"
        lines.append(
            f"  {name}: E[R]={mu[i]*100:.2f}%, sigma={vols[i]*100:.2f}%, "
            f"ESG={esg_scores[i]:.2f}/10, weight={w_opt[i]*100:.2f}%, in ESG frontier: {flag}"
        )
    lines += [
        "",
        "--- ESG-Aware Optimal Portfolio ---",
        f"  Expected return:  {ep*100:.2f}%",
        f"  Volatility:       {sp*100:.2f}%",
        f"  Sharpe ratio:     {sr:.4f}",
        f"  ESG score:        {esg_bar:.3f} / 10",
        f"  Utility U:        {ep - gamma/2*sp**2 + lam*esg_bar:.5f}",
        "",
        "--- Tangency Portfolio (all assets, unconstrained) ---",
        f"  E[R]={ep_tan_all*100:.2f}%, sigma={sp_tan_all*100:.2f}%, Sharpe={sr_tan_all:.4f}",
        "",
        "--- Tangency Portfolio (ESG-screened assets only) ---",
        f"  E[R]={ep_tan_esg*100:.2f}%, sigma={sp_tan_esg*100:.2f}%, Sharpe={sr_tan_esg:.4f}",
        "",
        "--- Model ---",
        "Utility: U = E[Rp] - (gamma/2)*sigma^2 + lambda*ESG_bar",
        "Blue frontier = unconstrained MV frontier (all assets).",
        "Green frontier = MV frontier for ESG-screened assets only.",
        "Green lies RIGHT of blue: same return requires more risk when ESG-constrained.",
        "ESG data: LSEG ESGCombinedScore scaled 0-10 (higher = better).",
    ]
    return "\n".join(lines)


def call_claude(system_prompt, messages):
    """Call the Anthropic claude-sonnet-4-20250514 messages API.
    API key is read from st.secrets['ANTHROPIC_API_KEY'].
    On Streamlit Community Cloud add this in Settings > Secrets:
        ANTHROPIC_API_KEY = "sk-ant-..."
    """
    # Resolve API key — secrets first, then environment variable fallback
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    if not api_key:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return (
            "No API key found. Please add your Anthropic API key to Streamlit secrets: "
            "open the app in Streamlit Cloud → Settings → Secrets, and add: "
            "ANTHROPIC_API_KEY = \"sk-ant-...\""
        )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": messages,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
    except Exception as exc:
        return f"Sorry, I could not reach the AI service right now. Error: {exc}"


SUGGESTED_QUESTIONS = [
    "Why does my portfolio have these weights?",
    "What is the Sharpe ratio and is mine good?",
    "Why does the green frontier sit to the right of blue?",
    "What does the ESG preference (λ) actually do?",
    "How does increasing risk aversion (γ) change things?",
    "Which asset drags down my ESG score the most?",
    "Explain the Capital Market Line in my chart.",
    "What is the cost of the ESG constraint here?",
]


# ══════════════════════════════════════════════════════════════════════════════
# MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def fetch_market_data(tickers, period="3y"):
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=False, group_by="ticker", threads=False)
    close = None
    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                try: frames.append(raw[t]["Close"].rename(t))
                except Exception: pass
        if frames: close = pd.concat(frames, axis=1)
    else:
        if "Close" in raw.columns:
            close = raw[["Close"]].copy()
            if len(tickers) == 1: close.columns = [tickers[0]]
    if close is None or close.empty:
        raise ValueError("No price data downloaded.")
    close = close.dropna(axis=1, how="all").dropna(how="all")
    ret   = close.pct_change().dropna(how="all")
    if ret.empty or ret.shape[1] < 2:
        raise ValueError("Not enough return data.")
    return close, ret, ret.mean()*252, ret.std()*np.sqrt(252), ret.cov()*252, ret.corr()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## GreenPort")
    st.markdown("---")
    st.markdown("### Investor Preferences")
    gamma = st.slider("Risk Aversion (γ)", 0.5, 10.0, 3.0, 0.5)
    lam   = st.slider("ESG Preference (λ)", 0.0, 5.0, 1.0, 0.1)
    rf    = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 4.0, 0.1, format="%.1f") / 100
    st.markdown("---")
    st.markdown("### ESG Screen")
    use_exclusion  = st.checkbox("Apply ESG exclusion screen", value=False)
    min_esg_filter = 0.0
    if use_exclusion:
        min_esg_filter = st.slider("Min ESG score (0–10)", 0.0, 10.0, 4.0, 0.5)
    st.markdown("---")
    st.markdown("<small style='color:#d6e5cb'>ECN316 · Sustainable Finance · 2026</small>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">GreenPort</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ESG-aware portfolio optimiser · ECN316 Sustainable Finance</div>',
            unsafe_allow_html=True)

if _ESG_DB:
    st.markdown(
        f'<div class="info-box"> ESG database loaded: <strong>{len(_ESG_DB):,} tickers</strong> '
        f'from LSEG ESGCombinedScore CSV — most recent year per ticker, scaled 0–10.</div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="error-box">Warning: Could not load ESG data. '
        f'Tried GitHub: <code>{_ESG_CSV_URL}</code> and local fallback. '
        f'Check your internet connection or place the CSV at <code>{_ESG_CSV_LOCAL}</code>.</div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT MODE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Asset Universe</div>', unsafe_allow_html=True)

with st.columns([1.5, 3])[0]:
    input_mode = st.radio("Input method", ["Manual input", "Ticker-based input"], horizontal=False)

default_names   = ["Tech ETF","Green Bond","Energy Stock","Healthcare","Consumer ETF",
                   "Infra Fund","EM Equity","Gov Bond","Real Estate","Commodity"]
default_ret     = [9.0, 4.5, 7.0, 7.5, 6.5, 5.5, 10.0, 3.0, 6.0, 5.0]
default_vol     = [18.0, 5.0, 22.0, 15.0, 14.0, 10.0, 25.0, 4.0, 13.0, 20.0]
default_esg     = [6.5, 8.5, 2.0, 7.0, 5.5, 7.5, 4.0, 6.0, 5.0, 3.5]
default_tickers = ["AAPL","MSFT","XOM","JNJ","SPY","TLT","NVDA","VWO","GLD","META"]

asset_data = []; ticker_rows = []; corr_df = None; lookback_period = "3y"

# ── Manual ───────────────────────────────────────────────────────────────────
if input_mode == "Manual input":
    cl, cr = st.columns([2, 1])
    with cr:
        n_assets = st.number_input("Number of assets", 2, 10, 3, 1)
        st.markdown('<div class="info-box">Enter expected return, volatility and ESG score (0–10).</div>',
                    unsafe_allow_html=True)
    with cl:
        h = st.columns([2,1.2,1.2,1.2])
        h[0].markdown("**Asset name**"); h[1].markdown("**E[R] (%)**")
        h[2].markdown("**σ (%)**");      h[3].markdown("**ESG (0–10)**")
        for i in range(int(n_assets)):
            c0,c1,c2,c3 = st.columns([2,1.2,1.2,1.2])
            name = c0.text_input("",value=default_names[i],key=f"name_{i}",label_visibility="collapsed")
            ret  = c1.number_input("",value=default_ret[i], key=f"ret_{i}", label_visibility="collapsed",format="%.1f")
            vol  = c2.number_input("",value=default_vol[i], key=f"vol_{i}", label_visibility="collapsed",format="%.1f",min_value=0.1)
            esg  = c3.number_input("",value=default_esg[i], key=f"esg_{i}", label_visibility="collapsed",format="%.1f",min_value=0.0,max_value=10.0)
            asset_data.append({"name":name,"ret":ret/100,"vol":vol/100,"esg":esg})

    st.markdown("**Correlation Matrix**")
    st.markdown('<div class="info-box">Enter pairwise correlations (−1 to 1). Diagonal fixed at 1.</div>',
                unsafe_allow_html=True)
    n = int(n_assets)
    ci = pd.DataFrame(np.eye(n),columns=[asset_data[i]["name"] for i in range(n)],
                      index=[asset_data[i]["name"] for i in range(n)])
    for r in range(n):
        for c in range(n):
            if r != c: ci.iloc[r,c] = 0.25
    corr_df = st.data_editor(ci, use_container_width=True, key="corr_matrix")

# ── Ticker ────────────────────────────────────────────────────────────────────
else:
    cl, cr = st.columns([2,1])
    with cr:
        n_assets = st.number_input("Number of assets",2,10,3,1,key="n_ticker_assets")
        lookback_period = st.selectbox("History window",["1y","3y","5y","10y"],index=1)
    with cl:
        h = st.columns([1.1,1.8])
        h[0].markdown("**Ticker**"); h[1].markdown("**Display name**")
        for i in range(int(n_assets)):
            c1,c2 = st.columns([1.1,1.8])
            ticker = c1.text_input("",value=default_tickers[i],key=f"ticker_{i}",label_visibility="collapsed").upper().strip()
            name   = c2.text_input("",value=default_names[i],  key=f"ticker_name_{i}",label_visibility="collapsed")
            ticker_rows.append({"ticker":ticker,"name":name or ticker,"manual_esg":None})

    valid_tickers = [r["ticker"] for r in ticker_rows if r["ticker"]]
    if valid_tickers:
        esg_preview = {r["ticker"]: lookup_esg(r["ticker"]) for r in ticker_rows if r["ticker"]}
        missing_esg = [t for t, res in esg_preview.items() if not res["has_esg"]]

        # Check which tickers are not on Yahoo Finance (price data unavailable)
        # We do a lightweight .info call — if it returns no regularMarketPrice the ticker is bad
        bad_tickers = []
        for r in ticker_rows:
            t = r["ticker"]
            try:
                info = yf.Ticker(t).fast_info
                price = getattr(info, "last_price", None)
                if price is None:
                    bad_tickers.append(t)
            except Exception:
                bad_tickers.append(t)

        manual_overrides = {}   # ESG overrides
        manual_ret_vol   = {}   # {ticker: {"ret": float, "vol": float}}

        if bad_tickers:
            st.markdown(
                f'<div class="warn-box"><strong>Ticker(s) not found on Yahoo Finance:</strong> '
                f'{", ".join(bad_tickers)}. '
                f'Enter expected return and volatility manually below.</div>',
                unsafe_allow_html=True)
            st.markdown("**Manual return / volatility inputs (annualised):**")
            for t in bad_tickers:
                def_idx = default_tickers.index(t) if t in default_tickers else 0
                bc1, bc2, bc3 = st.columns(3)
                bc1.markdown(f"**{t}**")
                m_ret = bc2.number_input(f"{t} E[R] (%)", value=default_ret[def_idx],
                                         min_value=-50.0, max_value=200.0, step=0.5, format="%.1f",
                                         key=f"manual_ret_{t}")
                m_vol = bc3.number_input(f"{t} σ (%)", value=default_vol[def_idx],
                                         min_value=0.1, max_value=200.0, step=0.5, format="%.1f",
                                         key=f"manual_vol_{t}")
                manual_ret_vol[t] = {"ret": m_ret / 100.0, "vol": m_vol / 100.0}

        if missing_esg:
            st.markdown(
                f'<div class="warn-box"><strong>Not in ESG CSV:</strong> '
                f'{", ".join(missing_esg)}. Enter ESG scores below.</div>',
                unsafe_allow_html=True)
            st.markdown("**Manual ESG scores (0–10):**")
            fcols = st.columns(min(len(missing_esg), 5))
            for idx, t in enumerate(missing_esg):
                def_idx = default_tickers.index(t) if t in default_tickers else 0
                manual_overrides[t] = fcols[idx % len(fcols)].number_input(
                    f"{t} ESG", value=float(default_esg[def_idx]),
                    min_value=0.0, max_value=10.0, step=0.1, format="%.1f",
                    key=f"manual_esg_{t}")

        if not missing_esg and not bad_tickers:
            st.markdown('<div class="info-box">All ticker data and ESG scores found.</div>',
                        unsafe_allow_html=True)

        for row in ticker_rows:
            t = row["ticker"]
            row["manual_esg"]     = manual_overrides.get(t, None)
            row["manual_ret_vol"] = manual_ret_vol.get(t, None)

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
run_col, _ = st.columns([1,3])
with run_col:
    run = st.button("Optimise Portfolio")

if run:
    # ── Build mu, cov, esg arrays ────────────────────────────────────────────
    if input_mode == "Manual input":
        names      = [d["name"] for d in asset_data]
        mu         = np.array([d["ret"] for d in asset_data], dtype=float)
        vols       = np.array([d["vol"] for d in asset_data], dtype=float)
        esg_scores = np.array([d["esg"] for d in asset_data], dtype=float)
        n          = len(names)
        try:
            corr_np = corr_df.values.astype(float)
        except Exception:
            st.error("Please make sure all correlation values are numeric."); st.stop()
        corr_np = (corr_np + corr_np.T) / 2
        np.fill_diagonal(corr_np, 1.0)
        corr_np = np.clip(corr_np, -0.999, 0.999)
        cov = np.outer(vols, vols) * corr_np
        esg_letters = {}
        ticker_data_display = None

    else:
        tickers = [r["ticker"] for r in ticker_rows if r["ticker"]]
        if len(tickers) < 2:
            st.error("Please enter at least two valid ticker symbols."); st.stop()
        try:
            prices, returns, mu_series, vols_series, cov_df, corr_df_market = \
                fetch_market_data(tickers, period=lookback_period)
        except Exception as e:
            st.error(f"Failed to fetch ticker data: {e}"); st.stop()

        # Combine fetched tickers with any manually-specified bad tickers
        manual_rv_map = {r["ticker"]: r.get("manual_ret_vol") for r in ticker_rows}
        manual_rv_map = {t: v for t, v in manual_rv_map.items() if v is not None}

        # Tickers where we have price data from Yahoo
        available = [t for t in tickers if t in mu_series.index]
        # Tickers where user provided manual return/vol
        manual_price_tickers = [r["ticker"] for r in ticker_rows
                                 if r["ticker"] not in available and r.get("manual_ret_vol")]
        all_tickers = available + manual_price_tickers

        if len(all_tickers) < 2:
            st.error("Not enough valid tickers. Check symbols or provide manual return/vol inputs.")
            st.stop()

        filtered_rows = [r for r in ticker_rows if r["ticker"] in all_tickers]
        esg_map = {t: lookup_esg(t) for t in all_tickers}
        resolved = []; used_manual_esg = []; esg_letters = {}

        for row in filtered_rows:
            t    = row["ticker"]
            meta = esg_map[t]
            if meta["has_esg"]:
                fe = float(meta["app_esg"]); esg_src = meta["source"]
                esg_letters[t] = meta.get("letter", "")
            else:
                fe = float(row.get("manual_esg") or 5.0)
                esg_src = "Manual"; used_manual_esg.append(t)
            resolved.append({"ticker": t, "name": row["name"], "final_esg": fe,
                              "src": esg_src, "letter": meta.get("letter"), "year": meta.get("year")})

        if used_manual_esg:
            st.markdown(f'<div class="error-box"><strong>Manual ESG used for:</strong> '
                        f'{", ".join(used_manual_esg)}.</div>', unsafe_allow_html=True)

        names      = [r["name"] for r in resolved]
        esg_scores = np.array([r["final_esg"] for r in resolved], dtype=float)
        n          = len(all_tickers)

        # Build mu, vols, cov — mixing Yahoo data with manual overrides
        mu_list   = []
        vols_list = []
        for t in all_tickers:
            if t in available:
                mu_list.append(float(mu_series.loc[t]))
                vols_list.append(float(vols_series.loc[t]))
            else:
                rv = manual_rv_map[t]
                mu_list.append(rv["ret"])
                vols_list.append(rv["vol"])
        mu   = np.array(mu_list, dtype=float)
        vols = np.array(vols_list, dtype=float)

        # Covariance: use Yahoo cov for available pairs, assume zero correlation for manual
        cov = np.zeros((n, n))
        idx_map = {t: i for i, t in enumerate(all_tickers)}
        for i, ti in enumerate(all_tickers):
            for j, tj in enumerate(all_tickers):
                if ti in available and tj in available:
                    cov[i, j] = float(cov_df.loc[ti, tj])
                elif i == j:
                    cov[i, j] = vols[i] ** 2
                # off-diagonal cross terms with manual tickers = 0 (no correlation data)

        corr_np = np.zeros((n, n))
        for i, ti in enumerate(all_tickers):
            for j, tj in enumerate(all_tickers):
                if ti in available and tj in available:
                    corr_np[i, j] = float(corr_df_market.loc[ti, tj])
                elif i == j:
                    corr_np[i, j] = 1.0

        ticker_data_display = pd.DataFrame({
            "Ticker":           all_tickers,
            "Name":             names,
            "E[R] (%)":         (mu * 100).round(2),
            "σ (%)":            (vols * 100).round(2),
            "ESG Score (0–10)": [r["final_esg"] for r in resolved],
            "LSEG Letter":      [r["letter"]    for r in resolved],
            "ESG Year":         [r["year"]      for r in resolved],
            "ESG Source":       [r["src"]       for r in resolved],
            "Return Source":    ["Yahoo Finance" if t in available else "Manual input"
                                 for t in all_tickers],
        })
        loaded_msg = ", ".join(available) if available else "none"
        manual_msg = (f" Manual inputs for: {', '.join(manual_price_tickers)}."
                      if manual_price_tickers else "")
        st.markdown(f'<div class="info-box">Market data loaded for: {loaded_msg} '
                    f'over {lookback_period}.{manual_msg}</div>', unsafe_allow_html=True)

    # PSD fix
    if np.any(np.linalg.eigvalsh(cov) < -1e-8):
        st.markdown('<div class="warn-box">Covariance matrix adjusted to PSD.</div>', unsafe_allow_html=True)
        cov = nearest_psd(cov)

    # ESG threshold for green frontier
    esg_thresh = min_esg_filter if use_exclusion else 0.0
    active_mask = esg_scores >= esg_thresh
    active_idx  = np.where(active_mask)[0]
    excluded = [names[i] for i in range(n) if not active_mask[i]]
    if excluded:
        st.markdown(f'<div class="warn-box">Excluded from ESG frontier: {", ".join(excluded)} '
                    f'(ESG < {esg_thresh:.1f})</div>', unsafe_allow_html=True)
    if len(active_idx) < 2:
        st.error("Need ≥ 2 assets passing ESG screen. Relax the filter."); st.stop()

    mu_a    = mu[active_idx]; cov_a = cov[np.ix_(active_idx, active_idx)]
    esg_a   = esg_scores[active_idx]; names_a = [names[i] for i in active_idx]
    vols_a  = vols[active_idx]

    # Bounds for green frontier: only ESG-passing assets can have weight > 0
    bounds_green = [(0.,1.) if active_mask[i] else (0.,0.) for i in range(n)]

    # ── Portfolios ────────────────────────────────────────────────────────────
    w_tan_all, ep_tan_all, sp_tan_all, sr_tan_all = find_tangency(mu, cov, rf)
    w_tan_esg, ep_tan_esg, sp_tan_esg, sr_tan_esg = find_tangency(mu, cov, rf, bounds=bounds_green)
    w_opt_a = find_optimal(mu_a, cov_a, esg_a, rf, gamma, lam)
    w_opt   = np.zeros(n)
    for idx, wi in zip(active_idx, w_opt_a): w_opt[idx] = wi
    ep, sp, sr, esg_bar = port_stats(w_opt_a, mu_a, cov_a, esg_a, rf)

    # ── Build frontiers ───────────────────────────────────────────────────────
    with st.spinner("Building mean-variance frontiers…"):
        std_blue,  ret_blue  = build_mv_frontier(mu, cov, n_points=100)
        std_green, ret_green = build_mv_frontier(mu, cov, bounds=bounds_green, n_points=100)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Optimal Portfolio</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    for col,label,val,unit in [
        (m1,"Expected Return",f"{ep*100:.2f}","%"),
        (m2,"Volatility (σ)", f"{sp*100:.2f}","%"),
        (m3,"Sharpe Ratio",   f"{sr:.3f}",    ""),
        (m4,"ESG Score",      f"{esg_bar:.2f}","/ 10"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                        f'<div class="metric-value">{val}<span class="metric-unit">{unit}</span>'
                        f'</div></div>', unsafe_allow_html=True)

    u_val = ep - gamma/2*sp**2 + lam*esg_bar
    st.markdown(
        f'<div class="info-box">U = E[Rp] − (γ/2)σ² + λs̄ = <strong>{u_val:.4f}</strong>'
        f' &nbsp;|&nbsp; γ={gamma}, λ={lam}, r_f={rf*100:.1f}%'
        f' &nbsp;|&nbsp; Tangency Sharpe (all assets) = {sr_tan_all:.3f}'
        f' &nbsp;|&nbsp; Tangency Sharpe (ESG screen) = {sr_tan_esg:.3f}</div>',
        unsafe_allow_html=True)

    st.markdown("#### Portfolio Weights")
    st.dataframe(pd.DataFrame({
        "Asset":              names,
        "Weight (%)":         [f"{w*100:.2f}"  for w in w_opt],
        "E[R] (%)":           [f"{r*100:.2f}"  for r in mu],
        "σ (%)":              [f"{v*100:.2f}"  for v in vols],
        "ESG (0–10)":         [f"{s:.2f}"      for s in esg_scores],
        "In ESG frontier":    ["" if m else "No" for m in active_mask],
    }), use_container_width=True, hide_index=True)

    if input_mode == "Ticker-based input":
        st.markdown("#### Ticker Data Used")
        st.dataframe(ticker_data_display, use_container_width=True, hide_index=True)
        st.markdown("#### Correlation Matrix")
        st.dataframe(pd.DataFrame(corr_np,index=names,columns=names).round(3),
                     use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    BG     = '#f5f2ec'
    BLUE   = '#1a66cc'
    GREEN  = '#2d8a2d'
    ORANGE = '#c76b2f'
    GREY   = '#777777'

    st.markdown('<div class="section-header">ESG-Efficient Frontier</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # ── Chart 1: Mean-Variance Frontier (matches lecture slide) ──────────────
    with c1:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

        # Blue frontier — all assets
        if len(std_blue) > 2:
            ax.plot(std_blue, ret_blue, color=BLUE, lw=2.4, zorder=4,
                    label='Mean-variance frontier\n(all assets)')

        # Green frontier — ESG-screened
        if len(std_green) > 2:
            ax.plot(std_green, ret_green, color=GREEN, lw=2.4, zorder=4,
                    label=f'Mean-variance frontier\n(ESG ≥ {esg_thresh:.1f})')

        # CML for all-assets tangency (blue dashed)
        if sp_tan_all > 1e-9 and len(std_blue) > 0:
            cml_max = max(np.nanmax(std_blue), sp_tan_all*100) * 1.5
            sd_cml  = np.linspace(0, cml_max, 300)
            ax.plot(sd_cml, rf*100 + (ep_tan_all-rf)/sp_tan_all*sd_cml,
                    color=BLUE, lw=1.5, linestyle='--', zorder=3,
                    label='CML (all assets)')

        # CML for ESG-constrained tangency (green dashed)
        if sp_tan_esg > 1e-9 and len(std_green) > 0:
            cml_max2 = max(np.nanmax(std_green), sp_tan_esg*100) * 1.5
            sd_cml2  = np.linspace(0, cml_max2, 300)
            ax.plot(sd_cml2, rf*100 + (ep_tan_esg-rf)/sp_tan_esg*sd_cml2,
                    color=GREEN, lw=1.5, linestyle='--', zorder=3,
                    label=f'CML (ESG ≥ {esg_thresh:.1f})')

        # Tangency: all assets (blue star)
        ax.scatter(sp_tan_all*100, ep_tan_all*100, color=BLUE, s=160,
                   zorder=9, edgecolors='white', lw=1.5, marker='*')
        ax.annotate('tangency portfolio\n(all assets)',
                    (sp_tan_all*100, ep_tan_all*100),
                    textcoords="offset points", xytext=(8, 2),
                    fontsize=7, color=BLUE, fontstyle='italic')

        # Tangency: ESG-constrained (green star)
        if len(std_green) > 2:
            ax.scatter(sp_tan_esg*100, ep_tan_esg*100, color=GREEN, s=160,
                       zorder=9, edgecolors='white', lw=1.5, marker='*')
            ax.annotate('tangency portfolio\n(ESG screen)',
                        (sp_tan_esg*100, ep_tan_esg*100),
                        textcoords="offset points", xytext=(8, -20),
                        fontsize=7, color=GREEN, fontstyle='italic')

        # Risk-free
        ax.scatter(0, rf*100, color=GREY, s=70, zorder=8,
                   edgecolors='white', lw=1, marker='s')

        # ESG-optimal
        ax.scatter(sp*100, ep*100, color=ORANGE, s=180, zorder=10,
                   edgecolors='white', lw=2, marker='*', label='ESG-Optimal portfolio')

        # Individual assets — blue dots if excluded from ESG frontier, green if included
        for i in range(n):
            col_pt = GREEN if active_mask[i] else BLUE
            ax.scatter(vols[i]*100, mu[i]*100, color=col_pt, s=50, zorder=6,
                       edgecolors='white', lw=0.8, alpha=0.85)
            ax.annotate(names[i], (vols[i]*100, mu[i]*100),
                        textcoords="offset points", xytext=(4,3),
                        fontsize=7, color='#2d4a2d')

        ax.set_xlabel("Std (%)", fontsize=9, color='#2d4a2d')
        ax.set_ylabel("Expected Return (%)", fontsize=9, color='#2d4a2d')
        ax.set_title("Mean-Variance Frontier", fontsize=11, fontweight='bold',
                     color='#1a2e1a', pad=10)
        ax.set_xlim(left=0)
        ax.tick_params(colors='#5a7a5a', labelsize=8)
        for sp_ in ax.spines.values(): sp_.set_color('#c8d8b8')
        ax.legend(fontsize=7, framealpha=0.92, facecolor=BG, edgecolor='#c8d8b8',
                  loc='upper left')
        ax.grid(True, alpha=0.3, color='#c8d8b8', linestyle='--')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── Chart 2: ESG-SR Frontier — matching the lecture slide ──────────────────
    # Lecture: single curve = max Sharpe for each ESG level (sweep ESG constraint).
    # Two labelled dots:
    #   - "Tangency portfolio using ESG information"   = peak of the curve (ESG-aware tangency)
    #   - "Tangency portfolio ignoring ESG information" = unconstrained tangency plotted at its ESG score
    # Individual assets shown as dots below the curve.
    with c2:
        # Sweep minimum-ESG constraint from min to max; record (achieved_ESG, Sharpe)
        esg_min_val = float(np.min(esg_a))
        esg_max_val = float(np.max(esg_a))
        esg_sweep   = np.linspace(esg_min_val, esg_max_val, 150)
        sw_esg, sw_sr = [], []
        for et in esg_sweep:
            res = minimize(
                lambda w: -port_sr(w, mu_a, cov_a, rf),
                np.ones(len(mu_a)) / len(mu_a), method="SLSQP",
                bounds=[(0., 1.)] * len(mu_a),
                constraints=[
                    {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
                    {"type": "ineq", "fun": lambda w, t=et: float(w @ esg_a) - t},
                ],
                options={"ftol": 1e-9, "maxiter": 400})
            if res.success:
                sw_esg.append(float(res.x @ esg_a))
                sw_sr.append(port_sr(res.x, mu_a, cov_a, rf))

        # "Tangency using ESG info" = unconstrained tangency among ESG assets (peak of curve)
        # This is w_tan_esg restricted to active assets
        esg_tan_using    = float(w_tan_esg[active_mask] @ esg_a) if active_mask.any() else esg_bar
        sr_tan_using     = sr_tan_esg

        # "Tangency ignoring ESG info" = unconstrained tangency across ALL assets
        # plotted at its own ESG score — sits BELOW the frontier (exactly as in lecture)
        esg_tan_ignoring = float(w_tan_all @ esg_scores)   # its actual ESG score
        sr_tan_ignoring  = sr_tan_all                       # its Sharpe ratio

        fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
        fig2.patch.set_facecolor(BG); ax2.set_facecolor(BG)

        # Frontier curve
        if sw_esg:
            ax2.plot(sw_esg, sw_sr, color=BLUE, lw=2.5, zorder=4,
                     label="ESG-SR frontier")
            ax2.fill_between(sw_esg, sw_sr,
                             alpha=0.08, color=BLUE)

        # Individual assets (dots below curve)
        for i in range(len(mu_a)):
            sr_i = (mu_a[i] - rf) / vols_a[i]
            ax2.scatter(esg_a[i], sr_i, color=BLUE, s=55, zorder=5,
                        edgecolors="white", lw=0.8, alpha=0.85)
            ax2.annotate(names_a[i], (esg_a[i], sr_i),
                         textcoords="offset points", xytext=(5, 4),
                         fontsize=7.5, color="#2d4a2d")

        # Tangency using ESG info (on or near peak of curve)
        ax2.scatter(esg_tan_using, sr_tan_using, color=BLUE, s=140, zorder=9,
                    edgecolors="white", lw=1.5)
        ax2.annotate("Tangency portfolio\nusing ESG information",
                     (esg_tan_using, sr_tan_using),
                     textcoords="offset points", xytext=(8, 4),
                     fontsize=7.5, color="#1a2e1a",
                     arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))

        # Tangency ignoring ESG info (below curve — same as lecture)
        ax2.scatter(esg_tan_ignoring, sr_tan_ignoring, color=BLUE, s=100, zorder=8,
                    edgecolors="white", lw=1.5)
        ax2.annotate("Tangency portfolio\nignoring ESG information",
                     (esg_tan_ignoring, sr_tan_ignoring),
                     textcoords="offset points", xytext=(8, -28),
                     fontsize=7.5, color="#1a2e1a",
                     arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))

        # ESG-optimal portfolio (orange, same as lecture's red dot)
        ax2.scatter(esg_bar, sr, color=ORANGE, s=120, zorder=10,
                    edgecolors="white", lw=2)
        ax2.annotate("Optimal portfolio",
                     (esg_bar, sr),
                     textcoords="offset points", xytext=(-80, 8),
                     fontsize=7.5, color=ORANGE,
                     arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))

        ax2.set_xlabel("ESG Score (0–10)", fontsize=9, color="#2d4a2d")
        ax2.set_ylabel("Sharpe Ratio",     fontsize=9, color="#2d4a2d")
        ax2.set_title("ESG-SR Frontier", fontsize=11, fontweight="bold",
                      color="#1a2e1a", pad=10)
        ax2.tick_params(colors="#5a7a5a", labelsize=8)
        for sp_ in ax2.spines.values(): sp_.set_color("#c8d8b8")
        ax2.legend(fontsize=8, framealpha=0.9, facecolor=BG, edgecolor="#c8d8b8")
        ax2.grid(True, alpha=0.3, color="#c8d8b8", linestyle="--")
        fig2.tight_layout(); st.pyplot(fig2); plt.close()

    # ── Allocation charts ─────────────────────────────────────────────────────
    st.markdown("#### Portfolio Allocation")
    pc, bc = st.columns(2)
    nz = [(names[i],w_opt[i],esg_scores[i]) for i in range(n) if w_opt[i]>0.005]
    if nz:
        plabels=[x[0] for x in nz]; pvals=[x[1] for x in nz]; pesg=[x[2] for x in nz]
        greens=['#1a4a1a','#2d6a2d','#4a8a3a','#6aaa5a','#8aba7a',
                '#a8cc98','#c4deb8','#d4e8c8','#e4f0d8','#f0f8ec']
        with pc:
            f3,a3 = plt.subplots(figsize=(5,4))
            f3.patch.set_facecolor(BG); a3.set_facecolor(BG)
            a3.pie(pvals,labels=plabels,autopct='%1.1f%%',colors=greens[:len(pvals)],
                   startangle=140,textprops={'fontsize':8,'color':'#1a2e1a'},
                   wedgeprops={'edgecolor':'white','linewidth':1.5})
            a3.set_title("Weight Allocation",fontsize=11,fontweight='bold',color='#1a2e1a',pad=10)
            f3.tight_layout(); st.pyplot(f3); plt.close()
        with bc:
            f4,a4 = plt.subplots(figsize=(5,4))
            f4.patch.set_facecolor(BG); a4.set_facecolor(BG)
            bcols = [plt.cm.YlGn(s/10) for s in pesg]
            bars  = a4.barh(plabels,[v*100 for v in pvals],color=bcols,edgecolor='white',height=0.6)
            for bar,ev in zip(bars,pesg):
                a4.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
                        f'ESG {ev:.1f}',va='center',fontsize=7.5,color='#2d4a2d')
            a4.set_xlabel("Weight (%)",fontsize=9,color='#2d4a2d')
            a4.set_title("Weights with ESG Scores",fontsize=11,fontweight='bold',color='#1a2e1a',pad=10)
            a4.tick_params(colors='#5a7a5a',labelsize=8)
            for sp_ in a4.spines.values(): sp_.set_color('#c8d8b8')
            a4.grid(True,alpha=0.3,color='#c8d8b8',axis='x',linestyle='--')
            f4.tight_layout(); st.pyplot(f4); plt.close()

    # ── Sensitivity ───────────────────────────────────────────────────────────
    with st.expander("Sensitivity Analysis — ESG Preference (λ)"):
        lam_vals  = np.linspace(0, 5, 20)
        sens_rows = []
        for lv in lam_vals:
            ww = find_optimal(mu_a, cov_a, esg_a, rf, gamma, lv)
            ep2,sp2,sr2,esg2 = port_stats(ww,mu_a,cov_a,esg_a,rf)
            sens_rows.append({"λ":round(float(lv),2),"E[R](%)":round(ep2*100,2),
                              "σ(%)":round(sp2*100,2),"Sharpe":round(sr2,3),"ESG":round(esg2,2)})
        sens_df = pd.DataFrame(sens_rows)
        f5,axes = plt.subplots(1,3,figsize=(12,3.5)); f5.patch.set_facecolor(BG)
        for ax_,col_,c_,yl_,tl_ in [
            (axes[0],"Sharpe",GREEN,"Sharpe Ratio","Sharpe vs λ"),
            (axes[1],"ESG",   GREEN,"ESG Score",   "ESG Score vs λ"),
        ]:
            ax_.set_facecolor(BG); ax_.plot(sens_df["λ"],sens_df[col_],color=c_,lw=2)
            ax_.set_title(tl_,fontsize=10,color='#1a2e1a')
            ax_.set_xlabel("λ",fontsize=9); ax_.set_ylabel(yl_,fontsize=9)
            ax_.tick_params(colors='#5a7a5a',labelsize=8)
            for sp_ in ax_.spines.values(): sp_.set_color('#c8d8b8')
            ax_.grid(True,alpha=0.3,color='#c8d8b8',linestyle='--')
        axes[2].set_facecolor(BG)
        axes[2].plot(sens_df["λ"],sens_df["E[R](%)"],color='#6aaa5a',lw=2,label='E[R]')
        axes[2].plot(sens_df["λ"],sens_df["σ(%)"],color=ORANGE,lw=2,linestyle='--',label='σ')
        axes[2].set_title("Return & Risk vs λ",fontsize=10,color='#1a2e1a')
        axes[2].set_xlabel("λ",fontsize=9); axes[2].set_ylabel("%",fontsize=9)
        axes[2].legend(fontsize=8,facecolor=BG,edgecolor='#c8d8b8')
        axes[2].tick_params(colors='#5a7a5a',labelsize=8)
        for sp_ in axes[2].spines.values(): sp_.set_color('#c8d8b8')
        axes[2].grid(True,alpha=0.3,color='#c8d8b8',linestyle='--')
        f5.tight_layout(); st.pyplot(f5); plt.close()
        st.dataframe(sens_df,use_container_width=True,hide_index=True)

    st.markdown("---")
    st.markdown(
        '<div class="info-box"><strong>Methodology:</strong> '
        'Utility U = E[Rp] − (γ/2)σ²p + λs̄, maximised via SLSQP (no short-selling). '
        '<strong>Blue frontier</strong>: unconstrained MV frontier across all assets. '
        '<strong>Green frontier</strong>: MV frontier restricted to assets passing the ESG screen '
        '— lies to the right of blue (same return costs more risk), matching the lecture diagram. '
        'Both CMLs drawn from r_f through their respective tangency portfolios. '
        'ESG data: LSEG ESGCombinedScore CSV, most recent year per ticker, '
        'scaled 0–1 → 0–10 (higher = better).</div>',
        unsafe_allow_html=True)

    # Store portfolio context in session state for the chatbot (rendered below)
    st.session_state["portfolio_ctx"] = build_portfolio_context(
        names, mu, vols, esg_scores, w_opt,
        ep, sp, sr, esg_bar, gamma, lam, rf,
        ep_tan_all, sp_tan_all, sr_tan_all,
        ep_tan_esg, sp_tan_esg, sr_tan_esg,
        active_mask, esg_thresh,
    )
    # Reset chat history each time a new optimisation runs
    st.session_state["chat_history"] = []

else:
    st.markdown('<div class="warn-box">Configure the asset universe and click '
                '<strong>Optimise Portfolio</strong> to generate results.</div>',
                unsafe_allow_html=True)
    with st.expander("How does the model work?"):
        st.markdown(r"""
**Utility Function**

$$U = E[R_p] - \frac{\gamma}{2}\sigma_p^2 + \lambda \bar{s}$$

**Frontier Construction — matching the lecture diagram**

Two mean-variance frontiers are built by minimising portfolio standard deviation for each target return level:

- **Blue**: No ESG constraints — uses all assets. This is the standard Markowitz frontier.
- **Green**: ESG-constrained — only assets passing the minimum ESG threshold can receive non-zero weight. Because this restricts the feasible set, the green frontier lies *to the right* of the blue (higher σ for the same E[R]), exactly as shown in the lecture slide.

Both frontiers show their Capital Market Line (dashed), tangency portfolio (★), and the risk-free asset.

**ESG Data Source**

Scores come from the uploaded LSEG ESGCombinedScore CSV. The `valuescore` column (0–1 scale, higher = better) is multiplied by 10 to give a 0–10 display scale. The most recent available year is used per ticker. If a ticker is not in the CSV, a manual score input appears.
""")


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO EXPLAINER CHATBOT
# Rendered outside if run: so it persists across Streamlit reruns triggered
# by chat interactions. Portfolio context is read from session_state.
# ══════════════════════════════════════════════════════════════════════════════

if "portfolio_ctx" in st.session_state:
    st.markdown("---")
    st.markdown('<div class="section-header">Portfolio Explainer</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    _SYSTEM_PROMPT = (
        "You are GreenPort's portfolio explainer — a knowledgeable assistant embedded inside "
        "an ESG portfolio optimisation tool built for a university sustainable finance course "
        "(ECN316). Your only job is to explain this specific portfolio and the model behind it. "
        "Always refer to the actual numbers in the portfolio summary below. "
        "Be concise (2–4 short paragraphs), use plain English, and never give personalised "
        "financial advice. When helpful, write equations inline (e.g. U = E[Rp] - (γ/2)σ² + λs̄). "
        "If a question is unrelated to this portfolio or model, politely redirect the user.\n\n"
        + st.session_state["portfolio_ctx"]
    )

    # Header
    st.markdown(
        '''<div class="chat-wrap">
  <div class="chat-header">
    <div>
      <p class="chat-header-title">GreenPort Portfolio Explainer</p>
      <p class="chat-header-sub">Ask anything about your portfolio — weights, ESG scores, frontiers, or the model</p>
    </div>
  </div>
  <div class="chat-body">''',
        unsafe_allow_html=True)

    # Suggested question buttons
    st.markdown("**Try asking:**")
    pill_cols = st.columns(4)
    for idx, q in enumerate(SUGGESTED_QUESTIONS):
        with pill_cols[idx % 4]:
            if st.button(q, key=f"pill_{idx}", use_container_width=True):
                st.session_state["chat_history"].append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    reply = call_claude(_SYSTEM_PROMPT, st.session_state["chat_history"])
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    # Conversation history
    for msg in st.session_state["chat_history"]:
        css_class = "chat-msg-user" if msg["role"] == "user" else "chat-msg-assistant"
        st.markdown(f'<div class="{css_class}">{msg["content"]}</div>',
                    unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Free-text input — use a form so Enter key submits without rerun loop
    with st.form(key="chat_form", clear_on_submit=True):
        fi_col, fb_col = st.columns([5, 1])
        user_input = fi_col.text_input(
            "Your question",
            placeholder="e.g. Why does my portfolio hold so much of this asset?",
            label_visibility="collapsed",
        )
        submitted = fb_col.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_input.strip()})
        with st.spinner("Thinking..."):
            reply = call_claude(_SYSTEM_PROMPT, st.session_state["chat_history"])
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.get("chat_history"):
        if st.button("Clear conversation", key="chat_clear"):
            st.session_state["chat_history"] = []
            st.rerun()
