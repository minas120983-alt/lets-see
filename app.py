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

# ── CSS — Barebone-inspired finance SaaS ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Tokens ── */
:root {
  --black:   #080808;
  --surface: #111111;
  --card:    #161616;
  --card2:   #1a1a1a;
  --border:  rgba(255,255,255,0.08);
  --border2: rgba(255,255,255,0.13);
  --t1: rgba(255,255,255,0.95);
  --t2: rgba(255,255,255,0.55);
  --t3: rgba(255,255,255,0.28);
  --cyan: #4af0e4;
  --cyan-dim: rgba(74,240,228,0.12);
  --red:   #ff453a;
  --green: #30d158;
  --r-sm:  8px;
  --r-md:  12px;
  --r-lg:  16px;
  --r-xl:  22px;
  --r-pill:100px;
}

/* ── Base ── */
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Helvetica Neue',sans-serif;
  -webkit-font-smoothing:antialiased;
  font-size:14px; line-height:1.5;
  background: var(--black) !important;
}
.stApp {
  background: var(--black) !important;
  color: var(--t1);
}
.block-container {
  padding: 0 3rem 6rem !important;
  max-width: 1320px !important;
}

/* ── Sidebar — dark nav panel ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--t2) !important; }
[data-testid="stSidebar"] hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 14px 0 !important;
}
[data-testid="stSidebar"] label {
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--t3) !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
  background: var(--cyan) !important;
  border: none !important;
  width: 16px !important; height: 16px !important;
  box-shadow: 0 0 10px rgba(74,240,228,0.4) !important;
}
[data-testid="stSidebar"] .stSlider p {
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--t1) !important;
}
[data-testid="stSidebar"] .stNumberInput input {
  background: var(--card) !important;
  color: var(--t1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  font-size: 13px !important;
}
[data-testid="stSidebar"] .stCheckbox label {
  font-size: 13px !important;
  color: var(--t2) !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
}
[data-testid="stSidebar"] small { color: var(--t3) !important; font-size: 11px !important; }
[data-testid="stSidebar"] p    { color: var(--t3) !important; font-size: 12px !important; }

/* ── Typography ── */
h1,h2,h3,h4,h5,h6 { color: var(--t1) !important; }
p, div, label, span { color: var(--t2); }
strong, b { color: var(--t1) !important; }

/* ── Nav bar (sidebar brand) ── */
.nav-brand {
  padding: 24px 20px 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 6px;
}
.nav-logo {
  display: inline-flex;
  align-items: center; gap: 10px;
  margin-bottom: 2px;
}
.nav-logo-mark {
  width: 28px; height: 28px;
  background: var(--t1);
  border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 900;
  color: var(--black) !important;
  letter-spacing: -0.06em;
  flex-shrink: 0;
}
.nav-logo-name {
  font-size: 16px; font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--t1) !important;
}
.nav-tagline {
  font-size: 11px; color: var(--t3) !important;
  letter-spacing: 0.04em; text-transform: uppercase;
}

/* ── Page header ── */
.page-header {
  padding: 52px 0 36px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 40px;
}
.header-eyebrow {
  font-size: 11px; font-weight: 600;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--cyan) !important;
  margin-bottom: 14px;
}
.header-title {
  font-size: 42px; font-weight: 800;
  letter-spacing: -0.04em; line-height: 1.06;
  color: var(--t1) !important;
  margin-bottom: 10px;
}
.header-subtitle {
  font-size: 15px; font-weight: 400;
  color: var(--t2) !important;
  letter-spacing: -0.01em; line-height: 1.55;
}
.header-status-pill {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--cyan-dim);
  border: 1px solid rgba(74,240,228,0.25);
  border-radius: var(--r-pill);
  padding: 4px 12px; margin-bottom: 18px;
  font-size: 11px; font-weight: 600;
  color: var(--cyan) !important;
  letter-spacing: 0.04em; text-transform: uppercase;
}

/* ── Section label ── */
.section-header {
  font-size: 10px; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--t3) !important;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
  margin: 48px 0 24px;
}

/* ── Feature cards (Barebone-style dark glass) ── */
.metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r-xl);
  padding: 24px 26px 20px;
  margin-bottom: 12px;
  transition: background 0.2s, border-color 0.2s;
  position: relative; overflow: hidden;
}
.metric-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg,
    transparent 0%, rgba(255,255,255,0.08) 50%, transparent 100%);
}
.metric-card:hover {
  background: var(--card2);
  border-color: rgba(255,255,255,0.13);
}
.metric-label {
  font-size: 10px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--t3) !important; margin-bottom: 10px;
}
.metric-value {
  font-size: 34px; font-weight: 300;
  letter-spacing: -0.05em; line-height: 1;
  color: var(--t1) !important;
  font-variant-numeric: tabular-nums;
}
.metric-unit {
  font-size: 15px; font-weight: 400;
  color: var(--t3) !important; margin-left: 2px;
}

/* ── Status / alert boxes ── */
.info-box {
  background: rgba(74,240,228,0.05);
  border: 1px solid rgba(74,240,228,0.18);
  border-radius: var(--r-md);
  padding: 12px 16px; margin: 10px 0;
  font-size: 13px; color: rgba(74,240,228,0.85) !important;
  line-height: 1.55;
}
.warn-box {
  background: rgba(255,159,10,0.05);
  border: 1px solid rgba(255,159,10,0.2);
  border-radius: var(--r-md);
  padding: 12px 16px; margin: 10px 0;
  font-size: 13px; color: rgba(255,159,10,0.9) !important;
  line-height: 1.55;
}
.error-box {
  background: rgba(255,69,58,0.05);
  border: 1px solid rgba(255,69,58,0.2);
  border-radius: var(--r-md);
  padding: 12px 16px; margin: 10px 0;
  font-size: 13px; color: rgba(255,100,90,0.9) !important;
  line-height: 1.55;
}

/* ── Buttons ── */
div.stButton > button {
  background: var(--card) !important;
  color: var(--t1) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r-pill) !important;
  padding: 10px 22px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: -0.01em !important;
  width: 100% !important;
  transition: all 0.15s ease !important;
}
div.stButton > button:hover {
  background: var(--card2) !important;
  border-color: rgba(255,255,255,0.22) !important;
  transform: translateY(-1px) !important;
}
div.stButton > button:active {
  transform: scale(0.97) !important;
  opacity: 0.65 !important;
}
/* Primary CTA — cyan outline like Barebone */
div[data-testid="stHorizontalBlock"] div.stButton > button {
  background: transparent !important;
  color: var(--cyan) !important;
  border: 1px solid var(--cyan) !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 0 20px rgba(74,240,228,0.08) !important;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
  background: var(--cyan-dim) !important;
  box-shadow: 0 0 28px rgba(74,240,228,0.16) !important;
}

/* ── Inputs ── */
.stNumberInput input, .stTextInput input, .stTextArea textarea {
  background: var(--card) !important;
  color: var(--t1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  font-size: 13px !important;
  padding: 9px 13px !important;
  transition: border-color 0.15s !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
  border-color: rgba(74,240,228,0.4) !important;
  box-shadow: 0 0 0 3px rgba(74,240,228,0.07) !important;
  outline: none !important;
}
.stSelectbox div[data-baseweb="select"] > div {
  background: var(--card) !important;
  color: var(--t1) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
}
.stRadio label { color: var(--t2) !important; }
.stRadio div[role="radiogroup"] label {
  font-size: 13px !important;
  font-weight: 400 !important;
}
.stCheckbox div[data-testid="stMarkdownContainer"] p {
  font-size: 13px !important; color: var(--t2) !important;
}

/* ── DataFrames / tables ── */
.stDataFrame, [data-testid="stDataEditor"] {
  border-radius: var(--r-lg) !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stDataEditor"] * {
  color: var(--t1) !important;
  background: var(--card) !important;
}
table { color: var(--t1) !important; border-collapse: collapse; width: 100%; }
thead tr th {
  background: var(--surface) !important;
  color: var(--t3) !important;
  font-size: 10px !important; font-weight: 600 !important;
  letter-spacing: 0.1em !important; text-transform: uppercase !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 10px 16px !important;
}
tbody tr td {
  color: var(--t2) !important;
  border-bottom: 1px solid rgba(255,255,255,0.04) !important;
  padding: 10px 16px !important; font-size: 13px !important;
}
tbody tr:hover td { background: rgba(255,255,255,0.025) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  background: var(--card) !important;
}
[data-testid="stExpander"] summary {
  color: var(--t2) !important; font-size: 13px !important;
  font-weight: 500 !important; padding: 14px 18px !important;
}
[data-testid="stExpander"] p {
  color: var(--t2) !important; line-height: 1.65 !important;
  font-size: 13px !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 2.5rem 0 !important;
}

/* ── Chat — Barebone "Ask Me a Question" style ── */
.chat-shell {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r-xl);
  overflow: hidden;
  margin-top: 16px;
}
.chat-topbar {
  display: flex; align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}
.chat-topbar-left { display: flex; align-items: center; gap: 12px; }
.chat-dot {
  width: 34px; height: 34px; border-radius: 50%;
  background: var(--cyan-dim);
  border: 1px solid rgba(74,240,228,0.25);
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700;
  color: var(--cyan) !important;
  flex-shrink: 0;
}
.chat-topbar-title {
  font-size: 14px; font-weight: 600;
  color: var(--t1) !important;
  letter-spacing: -0.02em;
}
.chat-topbar-sub {
  font-size: 11px; color: var(--t3) !important;
  margin-top: 1px;
}
.chat-online-pill {
  display: flex; align-items: center; gap: 5px;
  background: rgba(48,209,88,0.08);
  border: 1px solid rgba(48,209,88,0.2);
  border-radius: var(--r-pill);
  padding: 3px 10px;
  font-size: 10px; font-weight: 600;
  color: rgba(48,209,88,0.9) !important;
  letter-spacing: 0.06em; text-transform: uppercase;
}
.chat-online-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: #30d158;
  box-shadow: 0 0 6px rgba(48,209,88,0.8);
  display: inline-block;
}
/* Message area */
.chat-messages {
  max-height: 460px; overflow-y: auto;
  padding: 20px 20px 12px;
  scrollbar-width: thin;
  scrollbar-color: rgba(255,255,255,0.06) transparent;
}
.chat-messages::-webkit-scrollbar { width: 3px; }
.chat-messages::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.06); border-radius: 3px;
}
.chat-empty {
  text-align: center; padding: 40px 20px;
  font-size: 13px; color: var(--t3) !important;
  line-height: 1.6;
}
.chat-empty-icon {
  font-size: 28px; margin-bottom: 12px;
  opacity: 0.3; display: block;
}
.msg-row-user { display: flex; justify-content: flex-end; margin: 8px 0; }
.msg-row-bot  { display: flex; justify-content: flex-start; margin: 8px 0; }
.bubble-u {
  background: var(--t1);
  color: var(--black) !important;
  border-radius: 18px 18px 4px 18px;
  padding: 10px 15px; max-width: 68%;
  font-size: 13px; font-weight: 500; line-height: 1.5;
}
.bubble-b {
  background: var(--card2);
  border: 1px solid var(--border);
  color: var(--t2) !important;
  border-radius: 18px 18px 18px 4px;
  padding: 13px 16px; max-width: 86%;
  font-size: 12.5px; line-height: 1.78;
  font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace;
}
/* Suggestion chips */
.chat-chips {
  padding: 12px 20px 10px;
  border-top: 1px solid var(--border);
  background: var(--surface);
}
.chat-chips-label {
  font-size: 10px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--t3) !important; margin-bottom: 10px;
}
/* Input bar */
.chat-input-bar {
  padding: 12px 18px 16px;
  border-top: 1px solid var(--border);
}
.chat-input-bar .stTextInput input {
  border-radius: var(--r-pill) !important;
  padding: 11px 20px !important;
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  font-size: 13px !important;
}
.chat-input-bar .stTextInput input:focus {
  border-color: rgba(74,240,228,0.35) !important;
  box-shadow: 0 0 0 3px rgba(74,240,228,0.06) !important;
}

/* ── Footer ── */
.site-footer {
  margin-top: 80px; padding: 28px 0;
  border-top: 1px solid var(--border);
  display: flex; justify-content: space-between;
  align-items: center; flex-wrap: wrap; gap: 8px;
}
.site-footer span { font-size: 12px; color: var(--t3) !important; }
.site-footer a {
  color: var(--cyan) !important; text-decoration: none;
  font-size: 12px;
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
    "Is my Sharpe ratio good?",
    "Explain the utility function",
    "Why is the ESG frontier to the right?",
    "How does my risk aversion affect the portfolio?",
    "Which asset is dragging down my ESG score?",
    "What does lambda actually do here?",
    "Should I increase or decrease my ESG preference?",
    "What would happen without the ESG screen?",
    "Why is the tangency portfolio different from mine?",
    "How much return am I sacrificing for ESG?",
]


def _portfolio_answer(question: str, d: dict) -> str:
    """
    Expert portfolio construction commentary — interprets the numbers,
    draws conclusions, and gives professional judgement. Does not just
    restate the data. Handles both preset and completely free-form questions.
    """
    import numpy as np
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

    # Core derived quantities
    ind_sr        = [(mu[i]-rf)/vols[i] if vols[i]>0 else 0. for i in range(n)]
    by_w          = sorted(range(n), key=lambda i: w_opt[i], reverse=True)
    by_sr         = sorted(range(n), key=lambda i: ind_sr[i], reverse=True)
    by_esg        = sorted(range(n), key=lambda i: esg_scores[i])
    by_vol_asc    = sorted(range(n), key=lambda i: vols[i])
    u_val         = ep - gamma/2*sp**2 + lam*esg_bar
    sharpe_cost   = sr_tan_all - sr_tan_esg
    ret_cost_ann  = (ep_tan_all - ep_tan_esg) * 100
    held          = [i for i in range(n) if w_opt[i] > 0.005]
    excluded      = [i for i in range(n) if not active_mask[i]]
    variance_pct  = [2*w_opt[i]*sum(w_opt[j]*cov[i,j] for j in range(n))/max(sp**2,1e-14)*100
                     for i in range(n)]

    # ── Helpers ──────────────────────────────────────────────────────────────
    def p(v):  return f"{v*100:.2f}%"
    def p1(v): return f"{v*100:.1f}%"
    def sr_band(s):
        if s > 1.2:   return "exceptional"
        if s > 0.9:   return "strong"
        if s > 0.6:   return "decent"
        if s > 0.3:   return "modest"
        return "weak"
    def gamma_label(g):
        if g >= 7:  return "highly risk-averse"
        if g >= 4:  return "moderately risk-averse"
        if g >= 2:  return "balanced"
        return "risk-tolerant"
    def lam_label(l):
        if l >= 3.5: return "strongly ESG-driven"
        if l >= 1.5: return "moderately ESG-tilted"
        if l >= 0.5: return "lightly ESG-aware"
        return "essentially ESG-indifferent"
    def esg_label(s):
        if s >= 8: return "excellent"
        if s >= 6: return "good"
        if s >= 4: return "average"
        if s >= 2: return "poor"
        return "very poor"

    # ── 1. WEIGHTS — actual reasoning, not just a table ──────────────────────
    if any(k in q for k in ["weight", "allocat", "holding", "position",
                              "why does my portfolio", "why hold", "why so much",
                              "why is my", "why is there", "what drives"]):
        top = by_w[0]
        second = by_w[1] if len(by_w) > 1 else top
        bottom_held = by_w[-1] if held else top

        # What's actually dominating?
        top_drivers = []
        if ind_sr[top] == max(ind_sr): top_drivers.append("the best risk-adjusted return")
        if esg_scores[top] == max(esg_scores): top_drivers.append("the highest ESG score")
        if vols[top] == min(vols): top_drivers.append("the lowest volatility")
        driver_str = " and ".join(top_drivers) if top_drivers else "a strong combination of return, risk, and ESG"

        # Diversification check
        if len(held) >= 2:
            top2 = held[:2]
            rho = cov[top2[0],top2[1]] / max(vols[top2[0]]*vols[top2[1]], 1e-12)
            div_note = (
                f"{names[top2[1]]} ({p1(w_opt[top2[1]])}) complements it well "
                f"with a correlation of only {rho:.2f} — that low correlation is doing real "
                f"diversification work, pulling portfolio volatility down to {p(sp)} "
                f"versus a naively blended {p1(sum(w_opt[i]*vols[i] for i in range(n)))}."
                if abs(rho) < 0.5 else
                f"{names[top2[1]]} ({p1(w_opt[top2[1]])}) has a correlation of {rho:.2f} "
                f"with {names[top]} — moderately correlated, so the diversification benefit is limited."
            )
        else:
            div_note = "With only one asset held at meaningful weight, there is no diversification benefit."

        # Zero-weight explanation
        zero_held = [i for i in range(n) if w_opt[i] <= 0.005]
        zero_parts = []
        for i in zero_held[:2]:
            if not active_mask[i]:
                zero_parts.append(f"{names[i]} is excluded entirely by the ESG screen "
                                  f"(score {esg_scores[i]:.1f}/10 < threshold {esg_thresh:.1f})")
            else:
                # Find what dominates it
                better = [j for j in held if ind_sr[j] > ind_sr[i] and esg_scores[j] >= esg_scores[i]]
                if better:
                    zero_parts.append(f"{names[i]} receives zero weight because {names[better[0]]} "
                                      f"offers superior risk-adjusted return ({ind_sr[better[0]]:.2f} vs "
                                      f"{ind_sr[i]:.2f}) with {'equal or better' if esg_scores[better[0]]>=esg_scores[i] else 'similar'} ESG quality — "
                                      f"it is dominated and the optimizer correctly ignores it")
                else:
                    zero_parts.append(f"{names[i]} is zeroed out because its Sharpe of {ind_sr[i]:.2f} "
                                      f"and ESG of {esg_scores[i]:.1f}/10 add insufficient marginal utility "
                                      f"once the other assets are already in the mix")

        lines = [
            f"The portfolio is dominated by {names[top]} at {p1(w_opt[top])} "
            f"because it offers {driver_str}.",
            "",
            f"{div_note}",
            "",
        ]
        if zero_parts:
            lines += [z + "." for z in zero_parts]
            lines.append("")
        lines += [
            f"Your risk aversion (γ={gamma}, {gamma_label(gamma)}) "
            f"{'is heavily penalising high-volatility assets' if gamma > 5 else 'is allowing a moderate spread across assets' if gamma > 2 else 'is tolerating more volatility in pursuit of return'}. "
            f"Your ESG preference (λ={lam}, {lam_label(lam)}) "
            f"{'is materially tilting allocation toward high-ESG names' if lam > 2 else 'is adding a modest ESG tilt without drastically changing the financial allocation' if lam > 0.5 else 'is having minimal impact on where the weight lands'}.",
        ]
        return "\n".join(lines)

    # ── 2. ESG CONSTRAINT COST — interpret the actual magnitude ──────────────
    if any(k in q for k in ["cost", "sacrifice", "give up", "lose", "penalty",
                              "esg constraint", "esg screen", "tradeoff",
                              "trade-off", "return am i sacrific", "how much return",
                              "price of esg"]):
        # Interpret the magnitude
        if sharpe_cost < 0.02:
            cost_verdict = (
                f"Honestly, the ESG constraint here is almost free. "
                f"You are giving up just {sharpe_cost:.4f} Sharpe ratio points — "
                f"that is statistically indistinguishable from noise and well within "
                f"any reasonable estimation error. This is the ideal ESG portfolio situation: "
                f"good ESG and essentially no financial cost."
            )
        elif sharpe_cost < 0.08:
            cost_verdict = (
                f"The ESG constraint costs {sharpe_cost:.3f} Sharpe ratio points, "
                f"which translates to roughly {ret_cost_ann:.1f}% in expected annual return "
                f"at the tangency level. That is a real but manageable cost — "
                f"broadly in line with what academic research finds for typical ESG screens. "
                f"Whether it is worth it depends entirely on how much you value "
                f"the ESG improvement of {esg_bar:.1f}/10 in portfolio score."
            )
        elif sharpe_cost < 0.20:
            cost_verdict = (
                f"The ESG constraint is inflicting genuine financial pain: "
                f"{sharpe_cost:.3f} Sharpe ratio points lost, {ret_cost_ann:.1f}% "
                f"in expected annual return foregone. This is a meaningful cost. "
                f"It usually means one or more high-Sharpe assets are being excluded by the screen. "
                f"You should ask whether the ESG quality gained genuinely justifies this — "
                f"or whether relaxing the minimum threshold from {esg_thresh:.1f} to something lower "
                f"would recover most of the return at little ESG cost."
            )
        else:
            cost_verdict = (
                f"This ESG constraint is extremely expensive: {sharpe_cost:.3f} Sharpe points lost "
                f"and {ret_cost_ann:.1f}% in annual expected return. "
                f"At this level you are likely excluding your best-performing assets. "
                f"Practically speaking, this portfolio construction is not financially sound "
                f"unless the ESG mandate is non-negotiable. Strongly consider relaxing the threshold."
            )

        excl_str = ""
        if excluded:
            excl_names = [f"{names[i]} (SR={ind_sr[i]:.2f}, ESG={esg_scores[i]:.1f}/10)"
                          for i in excluded]
            excl_str = (
                f"\n\nThe assets being excluded are: {', '.join(excl_names)}. "
                f"{'The excluded assets have strong Sharpe ratios — their absence is the direct cause of the cost.' if any(ind_sr[i] > sr_tan_esg*0.8 for i in excluded) else 'Their Sharpe ratios are modest, so excluding them is not the main driver of the cost — the screen is simply restricting the feasible set.'}"
            )

        return cost_verdict + excl_str

    # ── 3. SHARPE RATIO — interpret, compare, give verdict ───────────────────
    if any(k in q for k in ["sharpe", "risk-adjust", "risk adjusted", "is mine good",
                              "how good", "how is my", "rate my", "assess"]):
        best_ind = by_sr[0]
        gap_to_tan = sr_tan_esg - sr
        gap_pct    = gap_to_tan / sr_tan_esg * 100 if sr_tan_esg > 0 else 0

        # Is the portfolio SR good?
        verdict = sr_band(sr)
        if gap_pct < 3:
            position = (f"Your Sharpe of {sr:.3f} is essentially at the ESG-efficient frontier — "
                        f"you are within {gap_pct:.1f}% of the maximum achievable Sharpe "
                        f"given your ESG constraints. This is excellent portfolio construction.")
        elif gap_pct < 12:
            position = (f"Your Sharpe of {sr:.3f} sits {gap_pct:.1f}% below the ESG tangency ({sr_tan_esg:.3f}). "
                        f"This gap is driven by your ESG preference λ={lam} tilting weight "
                        f"toward higher-ESG assets beyond what pure Sharpe maximisation would dictate. "
                        f"That is a deliberate, rational tradeoff — not a portfolio construction mistake.")
        else:
            position = (f"Your Sharpe of {sr:.3f} is {gap_pct:.1f}% below the ESG tangency ({sr_tan_esg:.3f}). "
                        f"This is a significant gap. With λ={lam}, your ESG preference is "
                        f"materially overriding financial efficiency. Consider whether this "
                        f"reflects your actual preferences or whether λ should be reduced.")

        # Compare to individual assets
        dominated = [i for i in range(n) if ind_sr[i] > sr and active_mask[i]]
        if dominated:
            dom_str = (f"\n\nInterestingly, {names[dominated[0]]} has a higher individual Sharpe "
                       f"({ind_sr[dominated[0]]:.3f}) than your portfolio ({sr:.3f}). "
                       f"This happens when ESG or risk-aversion constraints force weight "
                       f"away from the single best risk-adjusted asset. In a fully unconstrained "
                       f"mean-variance world, no individual asset should beat the portfolio Sharpe — "
                       f"but your constraints make this possible.")
        else:
            dom_str = (f"\n\nYour portfolio Sharpe ({sr:.3f}) exceeds all individual asset Sharpe ratios "
                       f"(best individual: {names[best_ind]} at {ind_sr[best_ind]:.3f}). "
                       f"This is the correct outcome — diversification is working as intended.")

        return f"Your Sharpe ratio of {sr:.3f} is {verdict} by typical standards. {position}{dom_str}"

    # ── 4. UTILITY FUNCTION — explain what it actually means ─────────────────
    if any(k in q for k in ["utility", "objective", "formula", "model",
                              "how does it work", "how does the model", "what is the model",
                              "explain the", "u ="]):
        fin_part = ep - gamma/2*sp**2
        esg_part = lam * esg_bar
        fin_pct  = fin_part / u_val * 100 if u_val != 0 else 100
        esg_pct  = esg_part / u_val * 100 if u_val != 0 else 0

        return (
            f"The model picks portfolio weights by maximising:\n\n"
            f"  U = E[Rp]  −  (γ/2)·σ²  +  λ·ESG\n\n"
            f"Think of it as three competing forces. The first term rewards return — "
            f"the model wants {ep*100:.2f}% expected annual return. "
            f"The second term punishes variance — with γ={gamma} you are {gamma_label(gamma)}, "
            f"so the model subtracts {gamma/2:.1f}×σ² = {gamma/2*sp**2*100:.3f}% for bearing "
            f"volatility of {sp*100:.2f}%. The third term rewards ESG quality — "
            f"with λ={lam} you are {lam_label(lam)}, adding {lam*esg_bar:.4f} to utility "
            f"for your portfolio ESG score of {esg_bar:.2f}/10.\n\n"
            f"Right now the financial component (return minus risk penalty) accounts for "
            f"{fin_pct:.0f}% of your total utility, and the ESG component accounts for "
            f"{esg_pct:.0f}%. "
            f"{'The ESG term is dominant — your portfolio construction is being driven more by sustainability preferences than by financial metrics.' if esg_pct > 40 else 'The financial terms dominate — ESG is a tilt, not the primary driver.' if esg_pct < 20 else 'Return, risk, and ESG are roughly balanced in driving your allocation.'}\n\n"
            f"The key insight is that γ and λ are not just sliders — they determine "
            f"the exchange rate between return, risk, and ESG. Every basis point of return "
            f"you could theoretically earn but do not is a deliberate choice made by these parameters."
        )

    # ── 5. FRONTIER INTERPRETATION ────────────────────────────────────────────
    if any(k in q for k in ["frontier", "right of", "why is the esg",
                              "two curve", "efficient frontier", "why does the"]):
        if sharpe_cost < 0.02:
            interp = (f"In this case the two frontiers are nearly coincident — the ESG screen "
                      f"is barely restricting your investment set. Either your excluded assets "
                      f"had poor Sharpe ratios anyway, or the screen threshold of {esg_thresh:.1f} "
                      f"is low enough that it is not biting. This is actually a good outcome: "
                      f"you get ESG alignment essentially for free.")
        elif sharpe_cost < 0.10:
            interp = (f"The gap between the two frontiers is modest — {sharpe_cost:.3f} Sharpe points "
                      f"at the tangency. This means the ESG screen is removing some useful assets "
                      f"but not the core of your opportunity set. The green frontier is visibly "
                      f"to the right but not dramatically so.")
        else:
            interp = (f"The two frontiers are noticeably separated — {sharpe_cost:.3f} Sharpe points "
                      f"at the tangency, {ret_cost_ann:.1f}% in expected return. "
                      f"This means the ESG screen is excluding assets that materially "
                      f"contribute to portfolio efficiency. The rightward shift of the green "
                      f"frontier is large and represents a real financial constraint.")

        return (
            f"The ESG frontier (brighter curve) sits to the right of the unconstrained frontier "
            f"(dimmer curve). This is not a coincidence or a chart quirk — it is a mathematical "
            f"certainty. Adding any constraint to an optimisation problem can only reduce or maintain "
            f"efficiency, never improve it.\n\n"
            f"What you are seeing is the geometric cost of sustainable investing: to achieve the "
            f"same expected return as the unconstrained portfolio, the ESG-screened portfolio must "
            f"accept higher volatility — because it cannot hold the same assets.\n\n"
            f"{interp}"
        )

    # ── 6. RISK AVERSION ─────────────────────────────────────────────────────
    if any(k in q for k in ["risk aversion", "gamma", "aversion", "risk toleran",
                              "how does risk", "risk appetite", "risk prefer"]):
        low_vol_weight = sum(w_opt[i] for i in by_vol_asc[:max(1,n//3)])
        vol_penalty    = gamma/2 * sp**2 * 100

        if gamma >= 6:
            aversion_comment = (
                f"At γ={gamma} you are highly risk-averse. The model is subtracting "
                f"{vol_penalty:.2f}% from utility purely as a variance penalty — "
                f"this is pulling weight heavily toward the lowest-volatility assets "
                f"({'%s at %s' % (names[by_vol_asc[0]], p1(vols[by_vol_asc[0]]))}) "
                f"even at the expense of return and ESG quality. "
                f"The top {round(low_vol_weight*100)}% of your portfolio "
                f"sits in the lowest-volatility third of your asset universe."
            )
        elif gamma >= 3:
            aversion_comment = (
                f"At γ={gamma} you are moderately risk-averse — roughly consistent "
                f"with a long-term institutional investor. The variance penalty of "
                f"{vol_penalty:.2f}% is meaningful but not dominant; "
                f"the allocation balances return-seeking with risk management. "
                f"If you increased γ to 8, the portfolio would shift toward "
                f"{names[by_vol_asc[0]]} and away from {names[by_sr[0]]}. "
                f"If you decreased γ to 1, it would concentrate in the highest-return assets."
            )
        else:
            aversion_comment = (
                f"At γ={gamma} you are relatively risk-tolerant. The model is barely "
                f"penalising variance, which means the weights are being driven almost "
                f"entirely by expected return and ESG score. This can lead to "
                f"concentrated positions in high-return/high-volatility assets — "
                f"check whether the resulting volatility of {p(sp)} feels right for "
                f"your actual investment horizon."
            )
        return aversion_comment

    # ── 7. LAMBDA / ESG PREFERENCE ───────────────────────────────────────────
    if any(k in q for k in ["lambda", "λ", "esg preference", "esg weight",
                              "esg parameter", "what does lambda", "how does lambda",
                              "should i increase", "should i decrease", "should i raise",
                              "should i lower"]):
        esg_contribution_pct = lam*esg_bar / abs(u_val) * 100 if u_val != 0 else 0
        bp_equiv = lam / 10 * 100  # basis points per ESG point

        assessment = ""
        if "should i" in q or "increase" in q or "decrease" in q or "raise" in q or "lower" in q:
            if sr < sr_tan_esg * 0.8 and lam > 2:
                assessment = (
                    f"\n\nGiven that your portfolio Sharpe ({sr:.3f}) is significantly below "
                    f"the ESG tangency ({sr_tan_esg:.3f}), λ={lam} appears too high for "
                    f"financially balanced construction. Reducing it toward 1.0–1.5 would "
                    f"recover Sharpe without abandoning your ESG tilt."
                )
            elif esg_bar < 5 and lam < 1:
                assessment = (
                    f"\n\nYour portfolio ESG score of {esg_bar:.2f}/10 is below average. "
                    f"If ESG quality matters to you, increasing λ toward 2.0 would meaningfully "
                    f"improve it — the sensitivity analysis shows exactly how much at each level."
                )
            else:
                assessment = (
                    f"\n\nYour current λ={lam} appears broadly appropriate given your Sharpe "
                    f"({sr:.3f} vs tangency {sr_tan_esg:.3f}) and ESG score ({esg_bar:.2f}/10). "
                    f"The sensitivity analysis expander below the charts shows the full tradeoff curve."
                )

        return (
            f"λ={lam} means that each 1-point improvement in portfolio ESG score (on the 0–10 scale) "
            f"is worth {bp_equiv:.0f} basis points of expected return to you. Put differently, "
            f"you are willing to accept {bp_equiv:.0f}bp less annual return in exchange for "
            f"one ESG point of improvement.\n\n"
            f"Right now the ESG term contributes {esg_contribution_pct:.0f}% of your total utility. "
            f"{'This is a substantial fraction — your portfolio construction is genuinely ESG-driven, not just ESG-aware.' if esg_contribution_pct > 35 else 'This is a moderate contribution — ESG is influencing but not dominating your allocation.' if esg_contribution_pct > 15 else 'This is a small fraction — at λ=' + str(lam) + ' the ESG term is a mild tilt. Raising λ would give it more influence.'}"
            + assessment
        )

    # ── 8. ESG SCORE DRAG ────────────────────────────────────────────────────
    if any(k in q for k in ["drag", "esg score", "worst esg", "bad esg",
                              "lowest esg", "which asset", "esg contribution",
                              "pulling down", "hurting"]):
        # Find what's actually dragging
        held_by_esg = sorted([i for i in held], key=lambda i: esg_scores[i])
        worst_held  = held_by_esg[0] if held_by_esg else by_esg[0]
        best_held   = held_by_esg[-1] if held_by_esg else by_esg[-1]
        drag_mag    = esg_scores[worst_held] * w_opt[worst_held]

        # What would swapping do?
        if len(held_by_esg) >= 2:
            esg_if_removed = (esg_bar - esg_scores[worst_held]*w_opt[worst_held]) / max(1 - w_opt[worst_held], 0.01)
            swap_gain      = esg_if_removed - esg_bar
        else:
            swap_gain = 0

        verdict = (
            f"{names[worst_held]} is the biggest ESG drag on this portfolio. "
            f"It carries an ESG score of {esg_scores[worst_held]:.2f}/10 — "
            f"{esg_label(esg_scores[worst_held])} — and its {p1(w_opt[worst_held])} weight "
            f"contributes {drag_mag:.4f} to your portfolio ESG score.\n\n"
        )
        if swap_gain > 0.3:
            verdict += (
                f"If you removed {names[worst_held]} entirely, your portfolio ESG "
                f"would improve by roughly {swap_gain:.2f} points — a meaningful gain. "
                f"The question is whether its financial contribution justifies keeping it: "
                f"it has a Sharpe ratio of {ind_sr[worst_held]:.3f}"
                f"{' which is above the portfolio Sharpe — removing it would hurt your risk-adjusted return' if ind_sr[worst_held] > sr else ' which is below the portfolio Sharpe — from a purely financial standpoint it is already a marginal holding'}."
            )
        elif esg_thresh > 0 and esg_scores[worst_held] < esg_thresh + 1:
            verdict += (
                f"{names[worst_held]} is close to the ESG screen threshold of {esg_thresh:.1f}. "
                f"It is currently just above the cutoff. Raising the threshold slightly "
                f"would exclude it and improve portfolio ESG, but at the Sharpe cost already quantified."
            )
        else:
            verdict += (
                f"Replacing it with {names[best_held]} (ESG={esg_scores[best_held]:.2f}/10) "
                f"would improve portfolio ESG, but you would need to re-optimise to see "
                f"the impact on Sharpe — the optimizer may already be constraining "
                f"{names[best_held]}'s weight for financial reasons."
            )
        return verdict

    # ── 9. TANGENCY DIFFERENCE ────────────────────────────────────────────────
    if any(k in q for k in ["tangency", "tangent", "why is the tangency",
                              "different from mine", "tangency different"]):
        gap_sr  = sr_tan_esg - sr
        gap_pct = gap_sr / sr_tan_esg * 100 if sr_tan_esg > 0 else 0

        if gap_pct < 2:
            return (
                f"Your portfolio is essentially at the tangency — within {gap_pct:.1f}% "
                f"of the maximum Sharpe achievable under your ESG screen. "
                f"With λ={lam} close to zero the utility function reduces to near-pure "
                f"Sharpe maximisation, which is why they nearly coincide. "
                f"If you raise λ, the two will diverge."
            )
        else:
            return (
                f"The tangency portfolio maximises Sharpe with no preference for ESG — "
                f"it is purely financially optimal. Your portfolio sits {gap_pct:.1f}% "
                f"below it in Sharpe terms ({sr:.3f} vs {sr_tan_esg:.3f}).\n\n"
                f"This gap is not a mistake — it is the direct, intended consequence "
                f"of λ={lam}. At that ESG preference level, the optimizer deliberately "
                f"moves weight toward {names[by_esg[-1]]} (ESG={esg_scores[by_esg[-1]]:.1f}/10) "
                f"and away from {names[by_sr[0]]} (SR={ind_sr[by_sr[0]]:.3f}) "
                f"even though that reduces Sharpe. Whether the {gap_sr:.3f} Sharpe sacrifice "
                f"is worth the ESG gain of moving from the tangency's implicit ESG level "
                f"to your portfolio's {esg_bar:.2f}/10 is a question only you can answer."
            )

    # ── 10. NO ESG / WHAT-IF ─────────────────────────────────────────────────
    if any(k in q for k in ["without esg", "no esg", "remove esg", "ignore esg",
                              "what if", "what would", "esg screen off",
                              "hypothetical", "if there was no", "without any"]):
        if sharpe_cost < 0.03:
            verdict = (
                f"Practically nothing. Removing the ESG screen would gain you {sharpe_cost:.4f} "
                f"Sharpe ratio points — {ret_cost_ann:.2f}% in annual expected return. "
                f"That is well within estimation error and not economically meaningful. "
                f"Your ESG constraints are essentially free in this portfolio."
            )
        else:
            verdict = (
                f"Removing all ESG constraints (λ=0, no screen) would deliver a tangency "
                f"Sharpe of {sr_tan_all:.3f} versus your current {sr:.3f} — "
                f"a gain of {sr_tan_all-sr:.3f} Sharpe points, equivalent to roughly "
                f"{ret_cost_ann:.1f}% more expected return per year at the tangency level.\n\n"
                f"Whether that is worth it depends on your perspective. "
                f"{'At ' + str(round(ret_cost_ann,1)) + '% annual return sacrifice, this is a substantial cost that most institutional mandates would struggle to justify.' if ret_cost_ann > 1.5 else 'At ' + str(round(ret_cost_ann,2)) + '% annual return sacrifice, this is the kind of cost that is typically considered acceptable for a genuine ESG mandate.'}"
            )
        if excluded:
            excl_names = [names[i] for i in excluded]
            verdict += (
                f"\n\nThe assets currently excluded that would re-enter are: "
                f"{', '.join(excl_names)}. "
                f"{'Their individual Sharpe ratios suggest they are contributing meaningfully to the unconstrained frontier.' if any(ind_sr[i] > sr for i in excluded) else 'Their Sharpe ratios are modest, so their absence is not the primary driver of the constraint cost.'}"
            )
        return verdict

    # ── 11. Asset-specific question ───────────────────────────────────────────
    mentioned = [i for i in range(n) if names[i].lower() in q
                 or any(w.lower() == names[i].lower() for w in q.split())]
    if mentioned:
        i = mentioned[0]
        w_i = w_opt[i]
        corrs = sorted(
            [(j, cov[i,j]/(vols[i]*vols[j]) if vols[i]*vols[j]>0 else 0)
             for j in range(n) if j != i],
            key=lambda x: abs(x[1]), reverse=True
        )
        corr_str = ", ".join(f"{names[j]} (ρ={r:.2f})" for j,r in corrs[:3])

        held_str = (
            f"It holds {p1(w_i)} of your portfolio. "
            if w_i > 0.005 else
            f"It holds zero weight — "
            + ("excluded by the ESG screen." if not active_mask[i]
               else "dominated by better risk/return/ESG combinations.")
        )

        assessment = (
            f"With a Sharpe of {ind_sr[i]:.3f} ({sr_band(ind_sr[i])}) and ESG score of "
            f"{esg_scores[i]:.2f}/10 ({esg_label(esg_scores[i])}), "
        )
        if ind_sr[i] > sr and esg_scores[i] > esg_bar:
            assessment += f"this asset is above average on both financial and ESG dimensions — it is earning its place."
        elif ind_sr[i] > sr and esg_scores[i] < esg_bar:
            assessment += f"it has strong financial merit (SR above portfolio) but below-average ESG. Your λ={lam} is reducing its weight relative to a purely financial portfolio."
        elif ind_sr[i] < sr and esg_scores[i] > esg_bar:
            assessment += f"it has above-average ESG quality but weak financial performance. It is in the portfolio primarily because of your ESG preference (λ={lam})."
        else:
            assessment += f"it is below average on both dimensions — its presence (or absence) reflects the holistic optimisation across all assets simultaneously."

        return f"{names[i]}: {held_str}\n\n{assessment}\n\nIts highest correlations are with {corr_str} — relevant for understanding the diversification contribution."

    # ── 12. Free-form fallback — extract key numbers and give a genuine read ──
    lines = [
        f"This portfolio holds {len(held)} assets: "
        f"{', '.join(names[i] + ' (' + p1(w_opt[i]) + ')' for i in by_w if w_opt[i]>0.005)}.\n",
        f"The headline numbers are E[R]={p(ep)}, σ={p(sp)}, Sharpe={sr:.3f}, ESG={esg_bar:.2f}/10 — "
        f"a {sr_band(sr)} risk-adjusted profile with {esg_label(esg_bar)} ESG quality.\n",
        f"The ESG constraint costs {sharpe_cost:.3f} Sharpe points versus the unconstrained frontier. "
        f"{'That is essentially free.' if sharpe_cost < 0.02 else 'That is a real but acceptable cost.' if sharpe_cost < 0.08 else 'That is a meaningful cost worth scrutinising.'}\n",
        f"You can ask me about specific assets by name, or about weights, costs, Sharpe, "
        f"the utility function, the frontier, risk aversion, or lambda.",
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
    st.markdown('''<div class="nav-brand">
  <div class="nav-logo">
    <div class="nav-logo-mark">GP</div>
    <span class="nav-logo-name">GreenPort</span>
  </div>
  <div class="nav-tagline">ESG Portfolio Optimiser</div>
</div>''', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Parameters")
    gamma = st.slider("Risk Aversion  γ", 0.5, 10.0, 3.0, 0.5,
                      help="Higher γ penalises portfolio variance. Typical range: 2–6.")
    lam   = st.slider("ESG Preference  λ", 0.0, 5.0, 1.0, 0.1,
                      help="Each λ unit ≈ 10bp of return per ESG point gained.")
    rf    = st.number_input("Risk-Free Rate  %", 0.0, 20.0, 4.0, 0.1, format="%.1f") / 100
    st.markdown("---")
    st.markdown("### ESG Screen")
    use_exclusion  = st.checkbox("Apply minimum ESG threshold", value=False)
    min_esg_filter = 0.0
    if use_exclusion:
        min_esg_filter = st.slider("Min ESG score (0–10)", 0.0, 10.0, 4.0, 0.5)
    st.markdown("---")
    st.markdown("<small>ECN316 · Sustainable Finance · 2026</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('''
<div class="page-header">
  <div class="header-eyebrow">ESG Portfolio Optimisation</div>
  <div class="header-title">GreenPort</div>
  <div class="header-subtitle">
    Institutional-grade mean-variance optimisation with ESG constraints.<br>
    ECN316 Sustainable Finance &nbsp;&middot;&nbsp; LSEG Data
  </div>
</div>
''', unsafe_allow_html=True)

if _ESG_DB:
    st.markdown(
        f'<div class="info-box"><strong>{len(_ESG_DB):,} tickers</strong> loaded — LSEG ESGCombinedScore, most recent year per ticker, scaled 0–10.</div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="error-box">Could not load ESG data. Check internet connection or verify CSV path.</div>',
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
    # CHARTS — iOS Stocks aesthetic
    # ══════════════════════════════════════════════════════════════════════════

    # Palette — monochrome, iOS Stocks inspired
    BG      = '#111114'
    PLOT_BG = '#161618'
    C1      = '#e8e8ec'   # primary line / curve (bright)
    C2      = '#606068'   # secondary line / curve (dim)
    C_DOT   = '#a8a8b0'   # individual asset dots
    C_OPT   = '#ffffff'   # optimal portfolio marker
    C_ANN   = '#6e6e78'   # annotation text
    C_GRID  = '#1e1e22'   # grid lines
    C_TICK  = '#48484e'   # tick labels
    C_TITLE = '#e8e8ec'   # chart title

    def _style_ax(ax, fig, title=""):
        """Apply iOS Stocks styling to an axes."""
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PLOT_BG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(C_GRID)
        ax.spines['bottom'].set_color(C_GRID)
        ax.tick_params(colors=C_TICK, labelsize=8, length=3)
        ax.grid(True, color=C_GRID, linewidth=0.5, linestyle='-', alpha=0.8)
        ax.set_axisbelow(True)
        if title:
            ax.set_title(title, fontsize=11, fontweight='600',
                         color=C_TITLE, pad=12, loc='left')

    st.markdown('<div class="section-header">Efficient Frontier</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # ── Chart 1: Mean-Variance Frontier ─────────────────────────────────────
    with c1:
        fig, ax = plt.subplots(figsize=(6, 5))
        _style_ax(ax, fig, "Mean-Variance Frontier")

        # Unconstrained frontier — dim
        if len(std_blue) > 2:
            ax.plot(std_blue, ret_blue, color=C2, lw=1.8, zorder=4,
                    label='All assets', alpha=0.7)

        # ESG frontier — bright
        if len(std_green) > 2:
            ax.plot(std_green, ret_green, color=C1, lw=2.2, zorder=5,
                    label=f'ESG ≥ {esg_thresh:.1f}')
            ax.fill_between(std_green, ret_green,
                            alpha=0.06, color=C1, zorder=2)

        # CMLs — hairline dashed
        if sp_tan_all > 1e-9 and len(std_blue) > 0:
            cml_max = max(float(np.nanmax(std_blue)), sp_tan_all*100) * 1.6
            sd_cml  = np.linspace(0, cml_max, 200)
            ax.plot(sd_cml, rf*100 + (ep_tan_all-rf)/sp_tan_all*sd_cml,
                    color=C2, lw=1, linestyle=(0,(4,3)), zorder=3, alpha=0.5)

        if sp_tan_esg > 1e-9 and len(std_green) > 0:
            cml_max2 = max(float(np.nanmax(std_green)), sp_tan_esg*100) * 1.6
            sd_cml2  = np.linspace(0, cml_max2, 200)
            ax.plot(sd_cml2, rf*100 + (ep_tan_esg-rf)/sp_tan_esg*sd_cml2,
                    color=C1, lw=1, linestyle=(0,(4,3)), zorder=3, alpha=0.4)

        # Tangency markers
        ax.scatter(sp_tan_all*100, ep_tan_all*100,
                   color=C2, s=60, zorder=9, edgecolors='none', marker='o')
        if len(std_green) > 2:
            ax.scatter(sp_tan_esg*100, ep_tan_esg*100,
                       color=C1, s=60, zorder=9, edgecolors='none', marker='o')

        # Risk-free dot
        ax.scatter(0, rf*100, color=C_TICK, s=40, zorder=8,
                   edgecolors='none', marker='o')

        # Individual assets
        for i in range(n):
            ax.scatter(vols[i]*100, mu[i]*100,
                       color=C_DOT if active_mask[i] else C2,
                       s=36, zorder=6, edgecolors='none', alpha=0.85)
            ax.annotate(names[i], (vols[i]*100, mu[i]*100),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=C_ANN, va='bottom')

        # Optimal portfolio — white circle
        ax.scatter(sp*100, ep*100, color=C_OPT, s=110, zorder=10,
                   edgecolors='none', marker='o')
        ax.annotate(f'  Optimal  SR={sr:.2f}',
                    (sp*100, ep*100),
                    textcoords="offset points", xytext=(8, -4),
                    fontsize=7.5, color=C_OPT, fontweight='600')

        ax.set_xlabel("Volatility (%)", fontsize=8.5, color=C_ANN, labelpad=6)
        ax.set_ylabel("Expected Return (%)", fontsize=8.5, color=C_ANN, labelpad=6)
        ax.set_xlim(left=0)

        # Minimal legend
        handles = [
            plt.Line2D([0],[0], color=C1, lw=2, label='ESG frontier'),
            plt.Line2D([0],[0], color=C2, lw=1.5, alpha=0.7, label='All-assets frontier'),
            plt.scatter([],[], color=C_OPT, s=50, label='Optimal', edgecolors='none'),
        ]
        ax.legend(handles=handles, fontsize=7.5,
                  framealpha=0, edgecolor='none',
                  labelcolor=C_ANN, loc='upper left')

        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Chart 2: ESG-SR Frontier ─────────────────────────────────────────────
    with c2:
        # Compute ESG-SR sweep
        esg_min_val = float(np.min(esg_a))
        esg_max_val = float(np.max(esg_a))
        esg_sweep   = np.linspace(esg_min_val, esg_max_val, 120)
        sw_esg, sw_sr_vals = [], []
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
                sw_sr_vals.append(port_sr(res.x, mu_a, cov_a, rf))

        esg_tan_using    = float(w_tan_esg[active_mask] @ esg_a) if active_mask.any() else esg_bar
        esg_tan_ignoring = float(w_tan_all @ esg_scores)

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        _style_ax(ax2, fig2, "ESG-SR Frontier")

        # Frontier curve with gradient fill
        if sw_esg:
            ax2.plot(sw_esg, sw_sr_vals, color=C1, lw=2.2, zorder=4)
            ax2.fill_between(sw_esg, sw_sr_vals,
                             min(sw_sr_vals) - 0.02,
                             alpha=0.07, color=C1, zorder=2)

        # Individual asset dots
        for i in range(len(mu_a)):
            sr_i = (mu_a[i] - rf) / vols_a[i]
            ax2.scatter(esg_a[i], sr_i, color=C_DOT, s=36,
                        zorder=5, edgecolors='none', alpha=0.85)
            ax2.annotate(names_a[i], (esg_a[i], sr_i),
                         textcoords="offset points", xytext=(5, 3),
                         fontsize=7, color=C_ANN)

        # Tangency using ESG — bright dot on curve
        ax2.scatter(esg_tan_using, sr_tan_esg, color=C1, s=80,
                    zorder=9, edgecolors='none')
        ax2.annotate('Tangency (ESG)',
                     (esg_tan_using, sr_tan_esg),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=7.5, color=C1, fontweight='600')

        # Tangency ignoring ESG — dim dot below curve
        ax2.scatter(esg_tan_ignoring, sr_tan_all, color=C2, s=60,
                    zorder=8, edgecolors='none')
        ax2.annotate('Tangency (no ESG)',
                     (esg_tan_ignoring, sr_tan_all),
                     textcoords="offset points", xytext=(6, -16),
                     fontsize=7.5, color=C2)

        # Optimal portfolio
        ax2.scatter(esg_bar, sr, color=C_OPT, s=100,
                    zorder=10, edgecolors='none')
        ax2.annotate(f'  Optimal  SR={sr:.2f}',
                     (esg_bar, sr),
                     textcoords="offset points", xytext=(7, 3),
                     fontsize=7.5, color=C_OPT, fontweight='600')

        ax2.set_xlabel("ESG Score (0–10)", fontsize=8.5, color=C_ANN, labelpad=6)
        ax2.set_ylabel("Sharpe Ratio",     fontsize=8.5, color=C_ANN, labelpad=6)

        fig2.tight_layout(pad=1.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

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
            f3.patch.set_facecolor('#111114'); a3.set_facecolor('#161618')
            a3.pie(pvals,labels=plabels,autopct='%1.1f%%',colors=greens[:len(pvals)],
                   startangle=140,textprops={'fontsize':7.5,'color':'#c8c8d0'},
                   wedgeprops={'edgecolor':'#111114','linewidth':1.5})
            a3.set_title('Weight Allocation',fontsize=11,fontweight='600',color='#e8e8ec',pad=12)
            f3.tight_layout(); st.pyplot(f3); plt.close()
        with bc:
            f4,a4 = plt.subplots(figsize=(5,4))
            f4.patch.set_facecolor('#111114'); a4.set_facecolor('#161618')
            bcols = ['#e8e8ec' if s >= 7 else '#a0a0a8' if s >= 5 else '#60606a' for s in pesg]
            bars  = a4.barh(plabels,[v*100 for v in pvals],color=bcols,edgecolor='white',height=0.6)
            for bar,ev in zip(bars,pesg):
                a4.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
                        f'ESG {ev:.2f}',va='center',fontsize=7,color='#5a5a68')
            a4.set_xlabel("Weight (%)",fontsize=9,color='#6e6e80')
            a4.set_title('Weights with ESG Scores',fontsize=11,fontweight='600',color='#e8e8ec',pad=12)
            a4.tick_params(colors='#5a7a5a',labelsize=8)
            for sp_ in a4.spines.values(): sp_.set_color('#1e1e22')
            a4.grid(True,alpha=0.5,color='#1e1e22',axis='x',linestyle='-')
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
        f5,axes = plt.subplots(1,3,figsize=(12,3.5)); f5.patch.set_facecolor('#111114')
        for ax_,col_,c_,yl_,tl_ in [
            (axes[0],"Sharpe",'#d0d0d4',"Sharpe Ratio","Sharpe vs λ"),
            (axes[1],"ESG",   '#a0a0aa',"ESG Score",   "ESG Score vs λ"),
        ]:
            ax_.set_facecolor('#161618'); ax_.plot(sens_df["λ"],sens_df[col_],color=c_,lw=1.8)
            ax_.set_title(tl_,fontsize=10,color='#e8e8ec')
            ax_.set_xlabel('λ',fontsize=9,color='#6e6e78'); ax_.set_ylabel(yl_,fontsize=9,color='#6e6e78')
            ax_.tick_params(colors='#5a7a5a',labelsize=8)
            for sp_ in ax_.spines.values(): sp_.set_color('#1e1e22')
            ax_.grid(True,alpha=0.5,color='#1e1e22',linestyle='-')
        axes[2].set_facecolor('#161618')
        axes[2].plot(sens_df["λ"],sens_df["E[R](%)"],color='#d0d0d4',lw=1.8,label='E[R]')
        axes[2].plot(sens_df["λ"],sens_df["σ(%)"],color='#707078',lw=1.8,linestyle='--',label='σ')
        axes[2].set_title('Return & Risk vs λ',fontsize=10,color='#e8e8ec')
        axes[2].set_xlabel('λ',fontsize=9,color='#6e6e78'); axes[2].set_ylabel('%',fontsize=9,color='#6e6e78')
        axes[2].legend(fontsize=8,facecolor='#161622',edgecolor='#1e1e22')
        axes[2].tick_params(colors='#5a7a5a',labelsize=8)
        for sp_ in axes[2].spines.values(): sp_.set_color('#1e1e22')
        axes[2].grid(True,alpha=0.5,color='#1e1e22',linestyle='-')
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

    # ── Chat window ─────────────────────────────────────────────────────────────
    st.markdown('''<div class="chat-shell">
  <div class="chat-topbar">
    <div class="chat-topbar-left">
      <div class="chat-dot">GP</div>
      <div>
        <div class="chat-topbar-title">Portfolio Analyst</div>
        <div class="chat-topbar-sub">Powered by your live portfolio data</div>
      </div>
    </div>
    <div class="chat-online-pill">
      <span class="chat-online-dot"></span>Live
    </div>
  </div>''', unsafe_allow_html=True)

    # Messages
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    if not st.session_state["chat_history"]:
        st.markdown('''<div class="chat-empty">
  <span class="chat-empty-icon">◎</span>
  Ask me anything about your portfolio — weights, ESG costs,<br>
  the frontier, Sharpe ratios, or specific assets by name.
</div>''', unsafe_allow_html=True)
    for msg in st.session_state["chat_history"]:
        content = msg["content"].replace("\n", "<br>")
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-row-user"><div class="bubble-u">{content}</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-row-bot"><div class="bubble-b">{content}</div></div>',
                        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Suggestion chips
    st.markdown('<div class="chat-chips"><div class="chat-chips-label">Suggested questions</div>',
                unsafe_allow_html=True)
    chip_cols = st.columns(3)
    for idx, q in enumerate(SUGGESTED_QUESTIONS[:9]):
        with chip_cols[idx % 3]:
            if st.button(q, key=f"pill_{idx}", use_container_width=True):
                reply = answer_question(q)
                st.session_state["chat_history"].append({"role": "user", "content": q})
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Input bar
    st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        inp_c, btn_c = st.columns([6, 1])
        user_input = inp_c.text_input("msg", placeholder="Ask about your portfolio...",
                                       label_visibility="collapsed")
        sent = btn_c.form_submit_button("Send", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if sent and user_input.strip():
        reply = answer_question(user_input.strip())
        st.session_state["chat_history"].append({"role": "user", "content": user_input.strip()})
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.get("chat_history"):
        clr, _ = st.columns([1, 5])
        with clr:
            if st.button("Clear", key="chat_clear", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('''
<div class="site-footer">
  <span>GreenPort &copy; 2026 &nbsp;&middot;&nbsp; ECN316 Sustainable Finance &nbsp;&middot;&nbsp; LSEG ESGCombinedScore Data</span>
  <span>Built with Streamlit &nbsp;&middot;&nbsp; Mean-Variance Optimisation &nbsp;&middot;&nbsp; SLSQP</span>
</div>''', unsafe_allow_html=True)
