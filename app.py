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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Syne', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.01em;
}

/* ── Frosted glass background ── */
.stApp {
    background: #0c0c0e;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(255,255,255,0.03) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(255,255,255,0.02) 0%, transparent 60%);
    color: #f0f0f2;
}
.block-container {
    padding-top: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px !important;
}

/* ── Sidebar — frosted ── */
[data-testid="stSidebar"] {
    background: rgba(14,14,16,0.85) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: rgba(240,240,242,0.5) !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: rgba(240,240,242,0.9) !important;
    font-size: 0.65rem !important; font-weight: 700 !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
}
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 1.2rem 0 !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.68rem !important; font-weight: 600 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    color: rgba(240,240,242,0.35) !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: #f0f0f2 !important;
    border: none !important;
    width: 14px !important; height: 14px !important;
    box-shadow: 0 0 0 3px rgba(240,240,242,0.12) !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.04) !important;
    color: rgba(240,240,242,0.9) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 6px !important; font-size: 0.88rem !important;
}
[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.78rem !important;
    color: rgba(240,240,242,0.5) !important;
    letter-spacing: 0.04em !important;
}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 { color: #f0f0f2 !important; }
p, div, label, span { color: rgba(240,240,242,0.55); }
strong { color: rgba(240,240,242,0.85) !important; }

/* ── Wordmark ── */
.gp-wordmark {
    font-size: 1.6rem; font-weight: 800; letter-spacing: -0.04em;
    color: #f0f0f2 !important; margin-bottom: 0.2rem;
    display: inline-block;
}
.gp-subtitle {
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.14em;
    text-transform: uppercase; color: rgba(240,240,242,0.3) !important;
    margin-bottom: 2rem;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: rgba(240,240,242,0.28) !important;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 0.6rem; margin: 2rem 0 1.2rem;
}

/* ── Frosted metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.25rem 1.4rem 1.1rem;
    margin-bottom: 0.75rem;
    transition: background 0.2s, border-color 0.2s;
}
.metric-card:hover {
    background: rgba(255,255,255,0.055);
    border-color: rgba(255,255,255,0.12);
}
.metric-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: rgba(240,240,242,0.3) !important;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2rem; font-weight: 700; letter-spacing: -0.04em;
    color: #f0f0f2 !important; line-height: 1;
    font-family: 'DM Mono', monospace;
}
.metric-unit {
    font-size: 0.78rem; font-weight: 400;
    color: rgba(240,240,242,0.3) !important;
    margin-left: 3px; letter-spacing: 0;
}
.metric-pos { color: #f0f0f2 !important; }
.metric-neg { color: rgba(240,240,242,0.4) !important; }

/* ── Status boxes ── */
.info-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.8rem 1.1rem; margin: 0.6rem 0;
    font-size: 0.82rem; color: rgba(240,240,242,0.6) !important;
    line-height: 1.55;
}
.warn-box {
    background: rgba(255,200,100,0.04);
    border: 1px solid rgba(255,200,100,0.15);
    border-radius: 10px; padding: 0.8rem 1.1rem; margin: 0.6rem 0;
    font-size: 0.82rem; color: rgba(255,210,120,0.8) !important;
    line-height: 1.55;
}
.error-box {
    background: rgba(255,100,100,0.04);
    border: 1px solid rgba(255,100,100,0.15);
    border-radius: 10px; padding: 0.8rem 1.1rem; margin: 0.6rem 0;
    font-size: 0.82rem; color: rgba(255,140,140,0.8) !important;
    line-height: 1.55;
}

/* ── Buttons ── */
div.stButton > button {
    background: rgba(240,240,242,0.9) !important;
    color: #0c0c0e !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.62rem 1.6rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.84rem !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
    width: 100% !important; transition: opacity 0.15s, transform 0.1s !important;
}
div.stButton > button:hover {
    opacity: 0.82 !important; transform: translateY(-1px) !important;
}
div.stButton > button:active {
    opacity: 0.65 !important; transform: translateY(0) !important;
}

/* ── Inputs ── */
.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    color: rgba(240,240,242,0.9) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; font-size: 0.88rem !important;
    transition: border-color 0.15s !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: rgba(255,255,255,0.25) !important;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.06) !important;
    outline: none !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    color: rgba(240,240,242,0.9) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
.stRadio label { color: rgba(240,240,242,0.55) !important; }
.stRadio div[role="radiogroup"] label {
    font-size: 0.86rem !important; font-weight: 500 !important;
}
.stCheckbox div[data-testid="stMarkdownContainer"] p {
    color: rgba(240,240,242,0.55) !important;
}

/* ── Tables ── */
.stDataFrame, [data-testid="stDataEditor"] {
    border-radius: 12px !important; overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stDataEditor"] * {
    color: rgba(240,240,242,0.8) !important;
    background: rgba(255,255,255,0.02) !important;
}
[data-testid="stTable"] * { color: rgba(240,240,242,0.8) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.02) !important;
}
[data-testid="stExpander"] summary {
    color: rgba(240,240,242,0.5) !important;
    font-size: 0.86rem !important; font-weight: 600 !important;
}
[data-testid="stExpander"] p {
    color: rgba(240,240,242,0.55) !important;
    line-height: 1.6 !important;
}

/* ── Tables in markdown ── */
table { color: #f0f0f2 !important; border-collapse: collapse; width: 100%; }
thead tr th {
    color: rgba(240,240,242,0.3) !important;
    font-size: 0.65rem !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    padding: 0.55rem 0.8rem !important; font-weight: 700 !important;
}
tbody tr td {
    color: rgba(240,240,242,0.65) !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    padding: 0.55rem 0.8rem !important;
}
tbody tr:hover td { background: rgba(255,255,255,0.025) !important; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 2rem 0 !important;
}

/* ── Chatbot ── */
.chat-wrap {
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; overflow: hidden;
    margin-top: 0.75rem;
}
.chat-header {
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 1.1rem 1.4rem;
}
.chat-header-title {
    color: rgba(240,240,242,0.9) !important;
    font-size: 0.88rem !important; font-weight: 700 !important;
    letter-spacing: 0.02em !important; margin: 0 0 0.15rem !important;
}
.chat-header-sub {
    color: rgba(240,240,242,0.3) !important;
    font-size: 0.74rem !important; margin: 0 !important;
    letter-spacing: 0.01em !important;
}
.chat-body { padding: 1.1rem 1.4rem 0.8rem; }
.chat-msg-user {
    background: rgba(240,240,242,0.9);
    color: #0c0c0e !important;
    border-radius: 14px 14px 3px 14px;
    padding: 0.65rem 1rem; margin: 0.5rem 0 0.5rem 22%;
    font-size: 0.84rem; font-weight: 600; display: block;
    text-align: right; line-height: 1.45;
}
.chat-msg-assistant {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    color: rgba(240,240,242,0.75) !important;
    border-radius: 14px 14px 14px 3px;
    padding: 0.75rem 1rem; margin: 0.5rem 22% 0.5rem 0;
    font-size: 0.84rem; display: block; line-height: 1.7;
    font-family: 'DM Mono', monospace;
}
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


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO EXPLAINER — expert simulation engine
# No API key needed. Handles both preset questions and free-form custom input.
# Built as a 30-year portfolio construction expert would reason.
# ══════════════════════════════════════════════════════════════════════════════

SUGGESTED_QUESTIONS = [
    "Why does my portfolio have these weights?",
    "What is the cost of the ESG constraint?",
    "Which asset has the highest Sharpe ratio?",
    "Explain the utility function and what it means",
    "Why is the ESG frontier to the right of the MV frontier?",
    "How does risk aversion affect my allocation?",
    "What is the Capital Market Line telling me?",
    "Which assets are dragging down my ESG score?",
    "How does lambda affect the portfolio?",
    "Why is my Sharpe ratio lower than the tangency?",
    "What does diversification contribute here?",
    "How sensitive is my portfolio to the risk-free rate?",
]


def _fmt(v): return f"{v:.4f}"
def _pct(v): return f"{v*100:.2f}%"
def _pct1(v): return f"{v*100:.1f}%"


def _portfolio_answer(question: str, d: dict) -> str:
    """
    Expert-level portfolio construction engine.
    Handles free-form questions with keyword routing + numeric fallback.
    All answers computed directly from portfolio data — precise, grounded.
    """
    q = question.lower().strip()

    names       = d["names"]
    mu          = d["mu"]
    vols        = d["vols"]
    esg_scores  = d["esg_scores"]
    w_opt       = d["w_opt"]
    ep          = d["ep"]
    sp          = d["sp"]
    sr          = d["sr"]
    esg_bar     = d["esg_bar"]
    gamma       = d["gamma"]
    lam         = d["lam"]
    rf          = d["rf"]
    ep_tan_all  = d["ep_tan_all"]
    sp_tan_all  = d["sp_tan_all"]
    sr_tan_all  = d["sr_tan_all"]
    ep_tan_esg  = d["ep_tan_esg"]
    sp_tan_esg  = d["sp_tan_esg"]
    sr_tan_esg  = d["sr_tan_esg"]
    active_mask = d["active_mask"]
    esg_thresh  = d["esg_thresh"]
    cov         = d["cov"]
    n           = d["n"]

    # Derived quantities used across many answers
    ind_sr     = [(mu[i] - rf) / vols[i] if vols[i] > 0 else 0.0 for i in range(n)]
    ind_sharpe = {names[i]: ind_sr[i] for i in range(n)}
    by_w       = sorted(range(n), key=lambda i: w_opt[i], reverse=True)
    by_esg     = sorted(range(n), key=lambda i: esg_scores[i])
    by_sr      = sorted(range(n), key=lambda i: ind_sr[i], reverse=True)
    by_vol     = sorted(range(n), key=lambda i: vols[i])
    u_val      = ep - gamma/2 * sp**2 + lam * esg_bar
    sharpe_cost = sr_tan_all - sr_tan_esg
    ret_cost    = ep_tan_all - ep_tan_esg
    # Variance-covariance contribution of each asset to portfolio variance
    w   = w_opt
    cov_contrib = [2 * w[i] * sum(w[j] * cov[i,j] for j in range(n)) for i in range(n)]
    pct_var_contrib = [cov_contrib[i] / max(sp**2, 1e-12) * 100 for i in range(n)]
    # Marginal contribution to Sharpe (how much each asset improves SR)
    port_cov_vec = [sum(w[j] * cov[i,j] for j in range(n)) for i in range(n)]

    # ──────────────────────────────────────────────────────────────────────────
    # KEYWORD ROUTING — ordered from most specific to most general
    # ──────────────────────────────────────────────────────────────────────────

    # ── 1. Portfolio weights ─────────────────────────────────────────────────
    if any(k in q for k in ["weight", "allocat", "holding", "position",
                              "why does my portfolio", "why is my", "why hold",
                              "why so much", "why so little"]):
        lines = [
            f"PORTFOLIO WEIGHTS — {n} assets, γ={gamma}, λ={lam}",
            "─" * 50,
            "",
            "The optimiser maximises U = E[Rp] - (γ/2)σ²p + λ·ESG, subject to",
            "Σwᵢ = 1 and wᵢ ≥ 0 (long-only). Each weight reflects the",
            "marginal contribution of that asset to utility across three dimensions:",
            "return, risk reduction, and ESG quality.",
            "",
        ]
        for i in by_w:
            w_i = w_opt[i]
            if w_i < 0.001:
                # Explain why excluded
                if not active_mask[i]:
                    reason = f"ESG screen: score {esg_scores[i]:.2f} < threshold {esg_thresh:.1f}"
                elif ind_sr[i] < min(ind_sr[j] for j in range(n) if w_opt[j] > 0.001):
                    reason = "lowest risk-adjusted return — optimizer finds no utility contribution"
                else:
                    reason = "dominated: another asset offers superior return/risk/ESG per unit weight"
                lines.append(f"  {names[i]:20s}  0.00%  — excluded ({reason})")
            else:
                # Explain why included and at this size
                drivers = []
                if esg_scores[i] >= max(esg_scores[j] for j in range(n) if w_opt[j]>0.001) - 0.5:
                    drivers.append(f"top ESG {esg_scores[i]:.2f}/10")
                if ind_sr[i] >= max(ind_sr[j] for j in range(n) if w_opt[j]>0.001) - 0.05:
                    drivers.append(f"best risk-adjusted SR={ind_sr[i]:.3f}")
                if vols[i] <= sorted(vols)[min(1, n-1)]:
                    drivers.append(f"low vol {_pct1(vols[i])}")
                # Diversification: low correlation with dominant asset?
                if n > 1 and len(by_w) > 1:
                    top = by_w[0] if i != by_w[0] else by_w[1] if len(by_w)>1 else -1
                    if top >= 0:
                        corr_ij = cov[i,top] / max(vols[i]*vols[top], 1e-12)
                        if corr_ij < 0.3 and w_opt[top] > 0.1:
                            drivers.append(f"diversifier (ρ={corr_ij:.2f} with {names[top]})")
                driver_str = f"  [{', '.join(drivers)}]" if drivers else ""
                lines.append(
                    f"  {names[i]:20s}  {w_i*100:5.1f}%"
                    f"  E[R]={_pct1(mu[i])} σ={_pct1(vols[i])}"
                    f" ESG={esg_scores[i]:.2f} SR={ind_sr[i]:.3f}{driver_str}"
                )
        lines += [
            "",
            f"Portfolio summary:  E[R]={_pct(ep)}  σ={_pct(sp)}  SR={sr:.3f}  ESG={esg_bar:.2f}/10",
            f"Utility achieved:   U = {_pct(ep)} - {gamma/2:.2f}×{sp**2:.5f} + {lam}×{esg_bar:.3f} = {u_val:.5f}",
        ]
        return "\n".join(lines)

    # ── 2. ESG constraint cost ────────────────────────────────────────────────
    if any(k in q for k in ["cost", "sacrifice", "price", "penalty", "esg constraint",
                              "esg screen", "tradeoff", "trade-off", "give up", "lose"]):
        excluded = [names[i] for i in range(n) if not active_mask[i]]
        lines = [
            "ESG CONSTRAINT COST ANALYSIS",
            "─" * 50,
            "",
            "Imposing ESG screens restricts the feasible investment set.",
            "The cost is measured as the reduction in maximum attainable Sharpe ratio:",
            "",
            f"  Unconstrained tangency SR:    {sr_tan_all:.4f}",
            f"  ESG-constrained tangency SR:  {sr_tan_esg:.4f}",
            f"  Sharpe ratio cost:            -{sharpe_cost:.4f}  ({sharpe_cost/max(sr_tan_all,0.001)*100:.1f}% reduction)",
            "",
            f"  Unconstrained tangency E[R]:  {_pct(ep_tan_all)}",
            f"  ESG-constrained tangency E[R]:{_pct(ep_tan_esg)}",
            f"  Expected return cost:         -{_pct(ret_cost)} per year",
            "",
        ]
        if esg_thresh > 0 and excluded:
            lines += [
                f"  Assets excluded by ESG screen (min score {esg_thresh:.1f}/10):",
            ]
            for nm in excluded:
                i = names.index(nm)
                lines.append(f"    {nm}: ESG={esg_scores[i]:.2f}, E[R]={_pct1(mu[i])}, σ={_pct1(vols[i])}, SR={ind_sr[i]:.3f}")
            lines.append("")
        lines += [
            f"  ESG benefit received:",
            f"    Portfolio ESG score: {esg_bar:.3f}/10",
            f"    ESG utility contribution: λ×ESG = {lam}×{esg_bar:.3f} = {lam*esg_bar:.4f}",
            "",
            "The ESG-SR frontier chart shows this tradeoff continuously:",
            "read off any ESG level and see the maximum Sharpe achievable.",
            f"With λ={lam}, the optimizer deems this cost {'acceptable' if lam > 1.5 else 'marginal — consider raising λ if ESG matters more'}.",
        ]
        return "\n".join(lines)

    # ── 3. Sharpe / risk-adjusted returns ────────────────────────────────────
    if any(k in q for k in ["sharpe", "risk-adjust", "risk adjusted", "best sharpe",
                              "highest sharpe", "individual sharpe", "mine good"]):
        best_i = by_sr[0]
        lines = [
            "SHARPE RATIO ANALYSIS",
            "─" * 50,
            "",
            "Sharpe ratio = (E[Rp] - rf) / σp — reward per unit of total risk.",
            f"Risk-free rate: {_pct(rf)}",
            "",
            "Individual asset Sharpe ratios (ranked):",
        ]
        for i in by_sr:
            active_flag = "" if active_mask[i] else " [ESG screened out]"
            lines.append(
                f"  {names[i]:20s}  SR={ind_sr[i]:.4f}"
                f"  ({_pct1(mu[i])} - {_pct1(rf)}) / {_pct1(vols[i])}{active_flag}"
            )
        lines += [
            "",
            "Portfolio-level:",
            f"  Your ESG-optimal portfolio:         SR = {sr:.4f}",
            f"  Tangency (unconstrained):           SR = {sr_tan_all:.4f}",
            f"  Tangency (ESG-screened):            SR = {sr_tan_esg:.4f}",
            "",
        ]
        if sr >= sr_tan_all * 0.95:
            lines.append(f"Your portfolio SR is close to the unconstrained tangency — the ESG and risk-aversion constraints impose minimal Sharpe cost.")
        elif sr >= sr_tan_esg * 0.95:
            lines.append(f"Your portfolio SR is close to the ESG-constrained tangency — the ESG preference λ={lam} shifts weights slightly but the Sharpe cost is modest.")
        else:
            lines.append(
                f"Your portfolio SR ({sr:.4f}) is {(sr_tan_esg - sr):.4f} below the ESG tangency ({sr_tan_esg:.4f})."
                f" The gap is driven by λ={lam} pulling weight toward higher-ESG assets at the expense of pure Sharpe maximisation."
            )
        return "\n".join(lines)

    # ── 4. Utility function ───────────────────────────────────────────────────
    if any(k in q for k in ["utility", "objective", "formula", "model", "maximis",
                              "optimi", "u =", "what is the function", "how does it work"]):
        lines = [
            "UTILITY FUNCTION — HOW THE MODEL WORKS",
            "─" * 50,
            "",
            "The optimizer maximises:",
            "",
            "  U = E[Rp]  -  (γ/2) × σ²p  +  λ × ESG_bar",
            "",
            "Each term, with your current values:",
            "",
            f"  E[Rp]          = {_pct(ep)}     ← reward: expected portfolio return",
            f"  -(γ/2)×σ²p    = -{gamma/2:.3f}×{sp**2:.5f} = {-gamma/2*sp**2:.5f}  ← penalty: scaled variance",
            f"  λ×ESG_bar      = {lam}×{esg_bar:.4f}    = {lam*esg_bar:.5f}  ← ESG premium",
            f"  ─────────────────────────────────────────",
            f"  U              = {u_val:.6f}",
            "",
            "Parameter interpretation:",
            f"  γ = {gamma}  → risk aversion. The variance penalty is γ/2 times σ².",
            f"         A 1% rise in σ costs {gamma/2 * 2 * sp * 0.01:.4f} utility units.",
            f"         {'Very risk-averse' if gamma > 6 else 'Moderately risk-averse' if gamma > 3 else 'Relatively risk-tolerant'} investor profile.",
            f"  λ = {lam}  → ESG preference weight. Each 1-point increase in portfolio ESG",
            f"         (0–10 scale) adds {lam:.3f} to utility,",
            f"         equivalent to ≈{lam/10*100:.1f}bp of additional expected return.",
            "",
            "The three terms trade off continuously:",
            "higher return → more risk → variance penalty increases.",
            "ESG preference → constraints reduce feasible set → Sharpe falls.",
            "The weights are the exact solution that balances all three.",
        ]
        return "\n".join(lines)

    # ── 5. Frontier / ESG frontier to the right ───────────────────────────────
    if any(k in q for k in ["frontier", "right of", "constrained", "feasible",
                              "why is the esg", "two curve", "blue curve", "white curve",
                              "efficient frontier"]):
        excluded = [names[i] for i in range(n) if not active_mask[i]]
        lines = [
            "THE TWO FRONTIERS — WHY THE ESG FRONTIER LIES TO THE RIGHT",
            "─" * 50,
            "",
            "The mean-variance frontier traces the minimum-risk portfolio for",
            "each level of expected return. Two versions are shown:",
            "",
            "  Frontier 1 (lighter): all assets, no ESG restriction",
            "    → maximum feasible set → global minimum variance for each E[R]",
            "",
            f"  Frontier 2 (darker):  ESG-screened assets only (ESG ≥ {esg_thresh:.1f}/10)",
            "    → restricted feasible set → higher minimum variance for same E[R]",
            "",
            "This is a fundamental result: adding constraints cannot improve",
            "and generally reduces portfolio efficiency. The rightward shift is",
            "the geometric representation of the ESG cost.",
            "",
            "Tangency portfolios (CML touch-points):",
            f"  All assets:   E[R]={_pct(ep_tan_all)}  σ={_pct(sp_tan_all)}  SR={sr_tan_all:.4f}",
            f"  ESG-screened: E[R]={_pct(ep_tan_esg)}  σ={_pct(sp_tan_esg)}  SR={sr_tan_esg:.4f}",
            f"  Sharpe cost:  -{sharpe_cost:.4f}",
            "",
        ]
        if excluded:
            lines += [
                "Assets removed by the ESG screen:",
                *[f"  {nm}: ESG={esg_scores[names.index(nm)]:.2f}/10  (contributes high return but fails screen)"
                  for nm in excluded],
                "",
            ]
        lines += [
            "The CML (dashed line) from rf through each tangency shows all",
            "achievable return/risk combinations with leverage.",
            "The slope of each CML equals its tangency Sharpe ratio.",
        ]
        return "\n".join(lines)

    # ── 6. Risk aversion ─────────────────────────────────────────────────────
    if any(k in q for k in ["risk aversion", "gamma", "γ", "aversion",
                              "risk toleran", "how does risk", "risk appetite"]):
        sorted_vol_asc = sorted(range(n), key=lambda i: vols[i])
        lines = [
            f"RISK AVERSION  γ = {gamma}",
            "─" * 50,
            "",
            "γ scales the variance penalty in the utility function:",
            "  U = E[Rp] - (γ/2)σ²p + λ·ESG",
            "",
            f"Current variance penalty: (γ/2)σ²p = {gamma/2:.2f} × {sp**2*100:.4f}% = {gamma/2*sp**2*100:.4f}%",
            f"This represents {abs(gamma/2*sp**2/ep)*100:.1f}% of expected return, surrendered to risk management.",
            "",
            "Effect on weights:",
            f"  γ > {gamma}: more averse → shifts weight toward low-σ assets",
            f"  γ < {gamma}: less averse → accepts more volatility for higher return",
            "",
            "Lowest-volatility assets in your universe (most favoured by high γ):",
        ]
        for i in sorted_vol_asc[:min(3,n)]:
            lines.append(f"  {names[i]:20s}  σ={_pct1(vols[i])}  weight={_pct1(w_opt[i])}  E[R]={_pct1(mu[i])}")
        lines += [
            "",
            "Highest-volatility assets (penalised most by high γ):",
        ]
        for i in sorted(range(n), key=lambda i: vols[i], reverse=True)[:min(3,n)]:
            lines.append(f"  {names[i]:20s}  σ={_pct1(vols[i])}  weight={_pct1(w_opt[i])}  E[R]={_pct1(mu[i])}")
        lines += [
            "",
            f"At γ={gamma}, the optimal portfolio has σ={_pct(sp)}.",
            f"The tangency portfolio (max Sharpe, γ-independent) has σ={_pct(sp_tan_esg)}.",
            f"The gap reflects {'risk aversion pulling the portfolio toward lower-vol assets' if sp < sp_tan_esg else 'ESG constraints forcing higher-vol assets into the mix'}.",
        ]
        return "\n".join(lines)

    # ── 7. Capital Market Line ────────────────────────────────────────────────
    if any(k in q for k in ["capital market", "cml", "market line", "leverage",
                              "risk free", "risk-free", "borrowing"]):
        lines = [
            "CAPITAL MARKET LINE",
            "─" * 50,
            "",
            "The CML is the set of all portfolios formed by combining the",
            "risk-free asset with the tangency (maximum Sharpe) portfolio.",
            "",
            "  E[R]_CML(σ) = rf + SR_tangency × σ",
            "",
            "Two CMLs are drawn in your chart:",
            "",
            f"  CML 1 (lighter) — unconstrained tangency:",
            f"    rf={_pct(rf)}  E[Rt]={_pct(ep_tan_all)}  σt={_pct(sp_tan_all)}",
            f"    Slope = SR = {sr_tan_all:.4f}",
            f"    At σ=10%:  implied E[R] = {rf*100 + sr_tan_all*10:.2f}%",
            "",
            f"  CML 2 (darker) — ESG-screened tangency:",
            f"    rf={_pct(rf)}  E[Rt]={_pct(ep_tan_esg)}  σt={_pct(sp_tan_esg)}",
            f"    Slope = SR = {sr_tan_esg:.4f}",
            f"    At σ=10%:  implied E[R] = {rf*100 + sr_tan_esg*10:.2f}%",
            "",
            f"Your optimal portfolio: E[R]={_pct(ep)}, σ={_pct(sp)}, SR={sr:.4f}",
            f"  It lies {'on' if abs(sr-sr_tan_esg)<0.005 else 'below'} CML 2 because",
            f"  λ={lam} introduces ESG into the objective, moving weights away from",
            "  pure Sharpe maximisation toward higher-ESG assets.",
            "",
            "Any point above a CML is unachievable (requires superior information).",
            "Any point below it is suboptimal relative to a CML + rf combination.",
        ]
        return "\n".join(lines)

    # ── 8. ESG drag / worst ESG assets ───────────────────────────────────────
    if any(k in q for k in ["drag", "worst esg", "bad esg", "lowest esg",
                              "esg score", "which asset", "esg contribution"]):
        weighted_contrib = sorted(
            [(i, esg_scores[i], w_opt[i], esg_scores[i]*w_opt[i]) for i in range(n) if w_opt[i]>0.001],
            key=lambda x: x[3]
        )
        lines = [
            "ESG SCORE ANALYSIS",
            "─" * 50,
            "",
            f"Portfolio weighted ESG: {esg_bar:.4f}/10",
            "",
            "Asset ESG scores and weighted contributions to portfolio ESG:",
            "",
        ]
        for i in by_esg:
            flag = "  [screened out]" if not active_mask[i] else ""
            in_port = w_opt[i] > 0.001
            contrib = esg_scores[i] * w_opt[i]
            lines.append(
                f"  {names[i]:20s}  ESG={esg_scores[i]:.3f}/10"
                f"  weight={_pct1(w_opt[i])}"
                f"  contribution={contrib:.4f}{flag}"
            )
        lines += [
            "",
            "Assets dragging portfolio ESG lowest (by weighted contribution):",
        ]
        for i, esg, w_i, contrib in weighted_contrib[:3]:
            share = contrib / max(esg_bar, 0.001) * 100
            lines.append(f"  {names[i]}: {_pct1(w_i)} weight × {esg:.3f}/10 = {contrib:.4f} ({share:.1f}% of portfolio ESG)")
        best_esg_i = by_esg[-1]
        worst_held = next((i for i,_,w_i,_ in weighted_contrib if w_i > 0.001), None)
        if worst_held is not None:
            esg_improvement = (esg_scores[best_esg_i] - esg_scores[worst_held]) * w_opt[worst_held]
            lines += [
                "",
                f"Replacing {names[worst_held]} (ESG={esg_scores[worst_held]:.3f})",
                f"with {names[best_esg_i]} (ESG={esg_scores[best_esg_i]:.3f}) at the same weight",
                f"would improve portfolio ESG by ≈{esg_improvement:.4f} points.",
                f"This would add λ×Δesg = {lam}×{esg_improvement:.4f} = {lam*esg_improvement:.5f} utility.",
            ]
        return "\n".join(lines)

    # ── 9. Lambda / ESG preference ────────────────────────────────────────────
    if any(k in q for k in ["lambda", "λ", "esg preference", "esg weight",
                              "esg parameter", "esg tilting", "how does lambda"]):
        lines = [
            f"ESG PREFERENCE  λ = {lam}",
            "─" * 50,
            "",
            "λ scales the ESG premium in utility:",
            "  U = E[Rp] - (γ/2)σ²p + λ × ESG_bar",
            "",
            f"ESG contribution to utility: {lam} × {esg_bar:.4f} = {lam*esg_bar:.5f}",
            f"Financial contribution:      E[Rp] - (γ/2)σ² = {ep - gamma/2*sp**2:.5f}",
            f"Total utility:               {u_val:.6f}",
            "",
            "Calibration — what λ means in return-equivalent terms:",
            f"  A 1-point rise in ESG (0–10 scale) adds λ/10 = {lam/10:.4f} to utility.",
            f"  This is equivalent to {lam/10 * 100:.2f}bp of additional expected return.",
            f"  At your portfolio size, ESG accounts for {lam*esg_bar/(u_val if u_val!=0 else 1)*100:.1f}% of utility.",
            "",
            "Sensitivity to λ changes:",
            f"  λ = 0:   pure Markowitz, ESG irrelevant. Tangency SR = {sr_tan_all:.4f}.",
            f"  λ = {lam}: current. Portfolio SR = {sr:.4f}. ESG = {esg_bar:.3f}/10.",
            f"  λ = 5:   maximum ESG weight. Portfolio would tilt heavily to ESG={max(esg_scores):.1f}/10 assets.",
            "",
            "The sensitivity analysis section below the chart shows exactly how",
            "E[R], σ, Sharpe, and ESG score change as λ moves from 0 to 5.",
        ]
        return "\n".join(lines)

    # ── 10. Tangency portfolio ────────────────────────────────────────────────
    if any(k in q for k in ["tangency", "tangent", "market portfolio",
                              "maximum sharpe", "max sharpe"]):
        lines = [
            "TANGENCY PORTFOLIO",
            "─" * 50,
            "",
            "The tangency portfolio is the unique risky portfolio that maximises",
            "the Sharpe ratio: max (E[Rp] - rf) / σp over all feasible portfolios.",
            "In the chart it sits where the CML is tangent to the efficient frontier.",
            "",
            "Two tangency portfolios:",
            "",
            f"  Unconstrained (all assets):",
            f"    E[R] = {_pct(ep_tan_all)}   σ = {_pct(sp_tan_all)}   SR = {sr_tan_all:.4f}",
            "",
            f"  ESG-constrained (screened assets, ESG ≥ {esg_thresh:.1f}):",
            f"    E[R] = {_pct(ep_tan_esg)}   σ = {_pct(sp_tan_esg)}   SR = {sr_tan_esg:.4f}",
            "",
            f"  Sharpe ratio cost of ESG constraint: -{sharpe_cost:.4f}",
            "",
            f"Your ESG-optimal portfolio:  SR = {sr:.4f}",
            f"  Gap vs ESG tangency: -{sr_tan_esg - sr:.4f}",
            "",
            f"This gap exists because λ={lam} introduces an ESG objective that",
            "conflicts with pure Sharpe maximisation. As λ→0, your portfolio",
            "converges to the ESG tangency. As λ increases, it tilts further",
            "toward high-ESG assets even at the cost of Sharpe ratio.",
            "",
            "Classic mean-variance theory: every investor holds the risk-free asset",
            "plus the tangency portfolio in proportions determined by risk aversion.",
            "Here, the ESG utility term breaks this two-fund separation.",
        ]
        return "\n".join(lines)

    # ── 11. Diversification ───────────────────────────────────────────────────
    if any(k in q for k in ["diversif", "correlation", "covariance", "benefit",
                              "risk reduction", "contribute"]):
        lines = [
            "DIVERSIFICATION ANALYSIS",
            "─" * 50,
            "",
            f"Portfolio σ = {_pct(sp)}",
            "Variance contribution by asset (% of portfolio variance):",
            "",
        ]
        for i in sorted(range(n), key=lambda i: pct_var_contrib[i], reverse=True):
            if w_opt[i] > 0.001:
                lines.append(
                    f"  {names[i]:20s}  w={_pct1(w_opt[i])}  var contrib={pct_var_contrib[i]:.1f}%"
                    f"  (σ={_pct1(vols[i])})"
                )
        weighted_avg_vol = sum(w_opt[i]*vols[i] for i in range(n))
        div_ratio = sp / weighted_avg_vol if weighted_avg_vol > 0 else 1
        lines += [
            "",
            f"Weighted-average individual σ:   {_pct(weighted_avg_vol)}",
            f"Portfolio σ:                      {_pct(sp)}",
            f"Diversification ratio:            {div_ratio:.4f}",
            f"Volatility saved by mixing:       {_pct(weighted_avg_vol - sp)}",
            "",
            f"A diversification ratio < 1 ({div_ratio:.4f}) confirms that combining these",
            "assets reduces risk below the weighted sum of individual volatilities.",
            "The benefit is driven by imperfect correlations between assets.",
        ]
        return "\n".join(lines)

    # ── 12. Risk-free rate sensitivity ───────────────────────────────────────
    if any(k in q for k in ["risk-free rate", "risk free rate", "rf", "interest rate",
                              "sensitive", "sensitivity"]):
        # approximate: dSR/drf = -1/σ for each asset
        lines = [
            f"RISK-FREE RATE SENSITIVITY  rf = {_pct(rf)}",
            "─" * 50,
            "",
            "The risk-free rate enters through the Sharpe ratio and utility.",
            "A higher rf reduces excess returns (E[R]-rf) for all assets equally,",
            "but its effect on optimal weights depends on relative risk-adjusted returns.",
            "",
            "Current Sharpe ratios at rf={:.1f}%:".format(rf*100),
        ]
        for i in by_sr:
            lines.append(f"  {names[i]:20s}  SR={ind_sr[i]:.4f}")
        # Recompute SRs at rf + 1%
        rf_high = rf + 0.01
        ind_sr_high = [(mu[i] - rf_high) / vols[i] for i in range(n)]
        by_sr_high  = sorted(range(n), key=lambda i: ind_sr_high[i], reverse=True)
        lines += [
            "",
            "If rf rises by 1% (to {:.1f}%):".format(rf_high*100),
        ]
        for i in by_sr_high:
            delta = ind_sr_high[i] - ind_sr[i]
            lines.append(f"  {names[i]:20s}  new SR={ind_sr_high[i]:.4f}  (change={delta:+.4f})")
        lines += [
            "",
            "Higher rf tends to favour lower-volatility assets because their",
            "Sharpe ratios shrink proportionally less.",
        ]
        return "\n".join(lines)

    # ── 13. Remove ESG / what-if ──────────────────────────────────────────────
    if any(k in q for k in ["remove", "without esg", "no esg", "ignore esg",
                              "what if", "what would", "screen off", "hypothetical",
                              "if i didn", "if there was no"]):
        lines = [
            "WHAT-IF: NO ESG CONSTRAINT",
            "─" * 50,
            "",
            f"If λ=0 and no minimum ESG threshold:",
            "",
            f"  Achievable tangency SR:   {sr_tan_all:.4f}  (vs {sr_tan_esg:.4f} currently)",
            f"  Sharpe gain:              +{sharpe_cost:.4f}  (+{sharpe_cost/max(sr_tan_esg,0.001)*100:.1f}%)",
            f"  Expected return gain:     +{_pct(ret_cost)} per year",
            "",
        ]
        excluded = [names[i] for i in range(n) if not active_mask[i]]
        if excluded:
            lines += [
                "Assets that would become available:",
            ]
            for nm in excluded:
                i = names.index(nm)
                lines.append(f"  {nm}: ESG={esg_scores[i]:.2f}/10  E[R]={_pct1(mu[i])}  SR={ind_sr[i]:.3f}")
            lines.append("")
        lines += [
            f"ESG score that would be lost: {esg_bar:.3f}/10 → lower (unconstrained portfolio)",
            f"Utility impact: +{sharpe_cost*sp:.4f} financial, -{lam*esg_bar:.4f} ESG premium",
            "",
            "This is the core tradeoff the ESG-SR frontier chart visualises.",
            "The choice of λ represents how much the investor values ESG",
            "relative to financial performance.",
        ]
        return "\n".join(lines)

    # ── 14. General portfolio question — free-form fallback ──────────────────
    # Parse the question for any asset name mentioned
    mentioned = [i for i in range(n) if names[i].lower() in q or
                 (len(names[i]) <= 5 and names[i].lower() in q.split())]
    if mentioned:
        lines = [f"ANALYSIS FOR: {', '.join(names[i] for i in mentioned)}", "─"*50, ""]
        for i in mentioned:
            corr_with_others = []
            for j in range(n):
                if j != i and vols[i]>0 and vols[j]>0:
                    rho = cov[i,j] / (vols[i]*vols[j])
                    corr_with_others.append((names[j], rho))
            corr_with_others.sort(key=lambda x: abs(x[1]), reverse=True)
            lines += [
                f"{names[i]}:",
                f"  Expected return:     {_pct(mu[i])}",
                f"  Volatility:          {_pct(vols[i])}",
                f"  ESG score:           {esg_scores[i]:.3f}/10",
                f"  Individual Sharpe:   {ind_sr[i]:.4f}",
                f"  Portfolio weight:    {_pct(w_opt[i])}",
                f"  Variance contribution: {pct_var_contrib[i]:.1f}% of portfolio variance",
                f"  In ESG frontier:     {'Yes' if active_mask[i] else 'No — ESG score below threshold'}",
                "",
                "  Correlations with other assets:",
                *[f"    {nm}: ρ = {rho:.3f}" for nm, rho in corr_with_others[:min(5,n-1)]],
                "",
            ]
        return "\n".join(lines)

    # ── 15. Absolute fallback ─────────────────────────────────────────────────
    lines = [
        "PORTFOLIO OVERVIEW",
        "─" * 50,
        "",
        f"Assets ({n}):  " + ", ".join(f"{names[i]} ({_pct1(w_opt[i])})" for i in by_w if w_opt[i]>0.001),
        f"Expected return:  {_pct(ep)}",
        f"Volatility:       {_pct(sp)}",
        f"Sharpe ratio:     {sr:.4f}",
        f"ESG score:        {esg_bar:.3f}/10",
        f"Utility:          {u_val:.5f}",
        f"Parameters:       γ={gamma}, λ={lam}, rf={_pct(rf)}",
        "",
        "I can answer questions about:",
        "  weights & allocation · Sharpe ratios · ESG scores & costs",
        "  utility function · efficient frontiers · Capital Market Line",
        "  risk aversion · diversification · specific assets",
        "",
        "Try asking about a specific asset by name, or use one of the",
        "suggested questions above.",
    ]
    return "\n".join(lines)


def answer_question(question: str) -> str:
    d = st.session_state.get("chat_data")
    if d is None:
        return "Run the portfolio optimiser first — click Optimise Portfolio."
    return _portfolio_answer(question, d)


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
    metric_data = [
        (m1, "Expected Return", f"{ep*100:.2f}",  "%",    "metric-pos"),
        (m2, "Volatility",      f"{sp*100:.2f}",  "%",    ""),
        (m3, "Sharpe Ratio",    f"{sr:.3f}",       "",     "metric-pos" if sr > 0 else "metric-neg"),
        (m4, "ESG Score",       f"{esg_bar:.2f}", "/ 10", "metric-pos" if esg_bar >= 5 else ""),
    ]
    for col, label, val, unit, cls in metric_data:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value {cls}">{val}<span class="metric-unit">{unit}</span></div>'
                f'</div>', unsafe_allow_html=True)

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
    BG     = '#0c0c0e'
    BLUE   = '#a0a0aa'
    GREEN  = '#d0d0d4'
    ORANGE = '#f0f0f2'
    GREY   = '#555560'

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
                       edgecolors='rgba(12,12,14,0.6)', lw=0.5, alpha=0.9)
            ax.annotate(names[i], (vols[i]*100, mu[i]*100),
                        textcoords="offset points", xytext=(4,3),
                        fontsize=7, color='rgba(240,240,242,0.45)')

        ax.set_xlabel("Std (%)", fontsize=9, color='rgba(240,240,242,0.45)')
        ax.set_ylabel("Expected Return (%)", fontsize=9, color='rgba(240,240,242,0.45)')
        ax.set_title("Mean-Variance Frontier", fontsize=11, fontweight='bold',
                     color='rgba(240,240,242,0.85)', pad=10)
        ax.set_xlim(left=0)
        ax.tick_params(colors='#505058', labelsize=7.5)
        for sp_ in ax.spines.values(): sp_.set_color('#1a1a1e')
        ax.legend(fontsize=7, framealpha=0.92, facecolor='#161622', edgecolor='#1e1e22',
                  loc='upper left')
        ax.grid(True, alpha=0.3, color='#252535', linestyle='--')
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
        ax2.legend(fontsize=8, framealpha=0.9, facecolor='#161622', edgecolor="#c8d8b8")
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
                   startangle=140,textprops={'fontsize':7.5,'color':'rgba(240,240,242,0.8)'},
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
                        f'ESG {ev:.2f}',va='center',fontsize=7,color='rgba(240,240,242,0.35)')
            a4.set_xlabel("Weight (%)",fontsize=9,color='rgba(240,240,242,0.45)')
            a4.set_title("Weights with ESG Scores",fontsize=11,fontweight='bold',color='#1a2e1a',pad=10)
            a4.tick_params(colors='#5a7a5a',labelsize=8)
            for sp_ in a4.spines.values(): sp_.set_color('#1a1a1e')
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
            (axes[0],"Sharpe",'#d0d0d4',"Sharpe Ratio","Sharpe vs λ"),
            (axes[1],"ESG",   '#a0a0aa',"ESG Score",   "ESG Score vs λ"),
        ]:
            ax_.set_facecolor(BG); ax_.plot(sens_df["λ"],sens_df[col_],color=c_,lw=1.8)
            ax_.set_title(tl_,fontsize=10,color='#1a2e1a')
            ax_.set_xlabel("λ",fontsize=9); ax_.set_ylabel(yl_,fontsize=9)
            ax_.tick_params(colors='#5a7a5a',labelsize=8)
            for sp_ in ax_.spines.values(): sp_.set_color('#1a1a1e')
            ax_.grid(True,alpha=0.3,color='#c8d8b8',linestyle='--')
        axes[2].set_facecolor(BG)
        axes[2].plot(sens_df["λ"],sens_df["E[R](%)"],color='#d0d0d4',lw=1.8,label='E[R]')
        axes[2].plot(sens_df["λ"],sens_df["σ(%)"],color='#707078',lw=1.8,linestyle='--',label='σ')
        axes[2].set_title("Return & Risk vs λ",fontsize=10,color='#1a2e1a')
        axes[2].set_xlabel("λ",fontsize=9); axes[2].set_ylabel("%",fontsize=9)
        axes[2].legend(fontsize=8,facecolor='#161622',edgecolor='#1e1e22')
        axes[2].tick_params(colors='#5a7a5a',labelsize=8)
        for sp_ in axes[2].spines.values(): sp_.set_color('#1a1a1e')
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

    # Store all portfolio data for the simulated chatbot engine
    st.session_state["chat_data"] = {
        "names":      names,
        "mu":         mu,
        "vols":       vols,
        "esg_scores": esg_scores,
        "w_opt":      w_opt,
        "ep":         ep,
        "sp":         sp,
        "sr":         sr,
        "esg_bar":    esg_bar,
        "gamma":      gamma,
        "lam":        lam,
        "rf":         rf,
        "ep_tan_all": ep_tan_all,
        "sp_tan_all": sp_tan_all,
        "sr_tan_all": sr_tan_all,
        "ep_tan_esg": ep_tan_esg,
        "sp_tan_esg": sp_tan_esg,
        "sr_tan_esg": sr_tan_esg,
        "active_mask":active_mask,
        "esg_thresh": esg_thresh,
        "cov":        cov,
        "n":          n,
    }
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
# Fully simulated — no API key needed. All answers computed from portfolio data.
# ══════════════════════════════════════════════════════════════════════════════

if "chat_data" in st.session_state:
    st.markdown("---")
    st.markdown('<div class="section-header">Portfolio Explainer</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Header
    st.markdown(
        '''<div class="chat-wrap">
  <div class="chat-header">
    <div>
      <p class="chat-header-title">GreenPort Portfolio Explainer</p>
      <p class="chat-header-sub">Ask anything about your portfolio — weights, ESG scores, the frontier, or the model. All answers are computed directly from your results.</p>
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
                reply = answer_question(q)
                st.session_state["chat_history"].append({"role": "user",      "content": q})
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()

    # Conversation history
    for msg in st.session_state["chat_history"]:
        css_class = "chat-msg-user" if msg["role"] == "user" else "chat-msg-assistant"
        # Render newlines as <br> so multi-line answers display correctly
        content = msg["content"].replace("\n", "<br>")
        st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Free-text input
    with st.form(key="chat_form", clear_on_submit=True):
        fi_col, fb_col = st.columns([5, 1])
        user_input = fi_col.text_input(
            "Your question",
            placeholder="e.g. Why does my portfolio hold so much of this asset?",
            label_visibility="collapsed",
        )
        submitted = fb_col.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        reply = answer_question(user_input.strip())
        st.session_state["chat_history"].append({"role": "user",      "content": user_input.strip()})
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.get("chat_history"):
        if st.button("Clear conversation", key="chat_clear"):
            st.session_state["chat_history"] = []
            st.rerun()
