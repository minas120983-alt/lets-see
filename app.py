import warnings
import io
import json
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TerraVest · ESG Portfolio Optimiser",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');
:root {
  --bg: #080808; --bg-card: #111111; --bg-elevated: #181818; --bg-3: #222222;
  --bg-input: rgba(255,255,255,0.07); --text-1: #f2f2f2;
  --text-2: rgba(242,242,242,0.60); --text-3: rgba(242,242,242,0.36);
  --accent: #22c55e; --accent-hover: #4ade80; --accent-light: rgba(34,197,94,0.14);
  --accent-dark: rgba(34,197,94,0.07); --accent-on: #000000;
  --sys-red: #f87171; --sys-orange: #fb923c; --sys-indigo: #818cf8;
  --sep: rgba(255,255,255,0.08); --sep-strong: rgba(255,255,255,0.16);
  --border: rgba(255,255,255,0.05);
  --chat-bg: #080808; --chat-header-bg: #111111;
  --chat-chip-bg: #1a1a1a; --chat-chip-color: #22c55e;
  --bubble-user-bg: #22c55e; --bubble-user-text: #000000;
  --bubble-bot-bg: #1a1a1a; --bubble-bot-text: rgba(242,242,242,0.88);
  --input-bar-bg: #111111;
  --font: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  --font-mono: "SF Mono", SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  --ease: cubic-bezier(0.25, 0.46, 0.45, 0.94);
  --ease-out: cubic-bezier(0.0, 0.0, 0.2, 1.0);
  --r-xs: 4px; --r-sm: 8px; --r-md: 12px; --r-lg: 16px; --r-xl: 20px;
}
html, body, [class*="css"] { font-family: var(--font) !important; -webkit-font-smoothing: antialiased; }
.stApp { background: var(--bg) !important; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
.stDeployButton { display: none !important; }
.block-container { padding-top: 0 !important; padding-bottom: 4rem !important; max-width: 1200px !important; padding-left: 2rem !important; padding-right: 2rem !important; }
.gp-nav { display: flex; align-items: center; padding: 0; min-height: 62px; }
.gp-logo-row { display: flex; align-items: center; gap: 10px; }
.gp-logo-mark { width: 30px; height: 30px; border-radius: 7px; background: var(--accent); display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.gp-wordmark { font-size: 1.05rem; font-weight: 800; letter-spacing: -0.03em; color: var(--text-1) !important; font-family: var(--font) !important; }
.gp-wordmark span { color: var(--accent) !important; }
.gp-badge { font-size: 0.58rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-3) !important; border: 1px solid var(--sep); border-radius: 100px; padding: 2px 7px; margin-left: 2px; vertical-align: middle; }
div[data-testid="stHorizontalBlock"]:first-of-type { border-bottom: 1px solid var(--sep) !important; padding-bottom: 0 !important; margin-bottom: 2.5rem !important; align-items: center !important; }
div[data-testid="stHorizontalBlock"]:first-of-type div.stButton { display: flex !important; justify-content: flex-end !important; }
div[data-testid="stHorizontalBlock"]:first-of-type div.stButton > button { background: transparent !important; color: var(--text-2) !important; border: 1px solid var(--sep) !important; border-radius: var(--r-sm) !important; font-size: 0.78rem !important; font-weight: 600 !important; padding: 0.38rem 1rem !important; width: auto !important; white-space: nowrap !important; }
.gp-hero { padding: 3.5rem 0 2.5rem; }
.gp-eyebrow { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent) !important; margin-bottom: 0.85rem; }
.gp-title { font-size: 3.4rem; font-weight: 800; letter-spacing: -0.05em; line-height: 1.0; color: var(--text-1) !important; margin-bottom: 1rem; }
.gp-subtitle { font-size: 1.05rem; font-weight: 400; color: var(--text-2) !important; line-height: 1.7; max-width: 500px; }
.gp-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3) !important; display: flex; align-items: center; gap: 0.75rem; margin: 2.5rem 0 1rem; }
.gp-label::after { content: ""; flex: 1; height: 1px; background: var(--sep); }
.section-header { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3) !important; display: flex; align-items: center; gap: 0.75rem; margin: 2.5rem 0 1.25rem; }
.section-header::after { content: ""; flex: 1; height: 1px; background: var(--sep); }
.gp-card { background: var(--bg-card); border: 1px solid var(--sep); border-radius: var(--r-lg); padding: 1.4rem 1.5rem; margin-bottom: 0.75rem; }
.results-hero { padding: 2rem 0 1.5rem; }
.results-title { font-size: 2.0rem; font-weight: 800; letter-spacing: -0.04em; color: var(--text-1) !important; margin-bottom: 0.4rem; }
.results-meta { font-size: 0.82rem; color: var(--text-3) !important; }
.metric-card { background: var(--bg-card); border: 1px solid var(--sep); border-radius: var(--r-lg); padding: 1.4rem 1.5rem 1.3rem; position: relative; overflow: hidden; }
.metric-card.card-ret { border-top: 2px solid var(--accent); }
.metric-card.card-vol { border-top: 2px solid var(--sys-orange); }
.metric-card.card-sr  { border-top: 2px solid var(--sys-indigo); }
.metric-card.card-esg { border-top: 2px solid var(--accent); }
.metric-label { font-size: 0.60rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3) !important; margin-bottom: 0.65rem; }
.metric-value { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.04em; color: var(--text-1) !important; line-height: 1; }
.metric-unit { font-size: 0.78rem; color: var(--text-3) !important; margin-left: 2px; font-weight: 400; }
.metric-pos { color: var(--accent) !important; }
.metric-neg { color: var(--sys-red) !important; }
.info-box  { background: var(--accent-dark); border: 1px solid rgba(34,197,94,0.18); border-radius: var(--r-sm); padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.81rem; color: var(--accent) !important; line-height: 1.6; }
.warn-box  { background: rgba(251,146,60,0.07); border: 1px solid rgba(251,146,60,0.20); border-radius: var(--r-sm); padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.81rem; color: var(--sys-orange) !important; line-height: 1.6; }
.error-box { background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.20); border-radius: var(--r-sm); padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.81rem; color: var(--sys-red) !important; line-height: 1.6; }
div.stButton > button { background: var(--accent) !important; color: var(--accent-on) !important; border: none !important; border-radius: var(--r-sm) !important; padding: 0.6rem 1.8rem !important; font-family: var(--font) !important; font-weight: 700 !important; font-size: 0.88rem !important; width: 100% !important; cursor: pointer !important; }
div.stButton > button:hover { background: var(--accent-hover) !important; transform: translateY(-1px) !important; }
.stNumberInput input, .stTextInput input, .stTextArea textarea { background: var(--bg-input) !important; color: #000000 !important; border: 1px solid var(--sep) !important; border-radius: var(--r-sm) !important; font-family: var(--font) !important; font-size: 0.88rem !important; }
.stSelectbox div[data-baseweb="select"] > div { background: var(--bg-input) !important; color: var(--text-1) !important; border: 1px solid var(--sep) !important; border-radius: var(--r-sm) !important; }
.stRadio label { color: var(--text-2) !important; font-size: 0.88rem !important; }
.stSlider [role="slider"] { background: var(--accent) !important; border: none !important; box-shadow: 0 0 0 3px var(--accent-light) !important; }
label, .stSlider label { color: var(--text-3) !important; font-size: 0.68rem !important; font-weight: 700 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
h1, h2, h3, h4, h5, h6 { color: var(--text-1) !important; font-family: var(--font) !important; }
p, div, span, li { font-family: var(--font) !important; color: var(--text-2); }
code { font-family: var(--font-mono) !important; background: var(--bg-input); color: var(--accent); border-radius: 4px; padding: 1px 6px; font-size: 0.84em; }
.stDataFrame thead th { background: var(--bg-elevated) !important; color: var(--text-3) !important; font-size: 0.64rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; font-weight: 700 !important; }
.stDataFrame tbody tr { border-bottom: 1px solid var(--sep) !important; }
table { color: var(--text-1) !important; border-collapse: collapse; width: 100%; }
thead tr th { color: var(--text-3) !important; font-size: 0.64rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; border-bottom: 1px solid var(--sep) !important; padding: 0.55rem 0.8rem !important; font-weight: 700 !important; }
tbody tr td { color: var(--text-2) !important; border-bottom: 1px solid var(--sep) !important; padding: 0.5rem 0.8rem !important; }
hr { border: none !important; border-top: 1px solid var(--sep) !important; margin: 2rem 0 !important; }
.chat-page { background: var(--chat-bg); border: 1px solid var(--sep); border-radius: var(--r-xl); overflow: hidden; margin-top: 1rem; }
.chat-header { background: var(--chat-header-bg); border-bottom: 1px solid var(--sep); padding: 1rem 1.5rem; display: flex; align-items: center; gap: 0.9rem; }
.chat-avatar { width: 38px; height: 38px; border-radius: 50%; background: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 800; color: #000; flex-shrink: 0; }
.chat-name { font-size: 0.92rem; font-weight: 700; color: var(--text-1) !important; margin: 0; line-height: 1.2; letter-spacing: -0.01em; }
.chat-status { font-size: 0.68rem; color: var(--accent) !important; margin: 0; display: flex; align-items: center; gap: 4px; }
.chat-status::before { content: ""; display: inline-block; width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }
.chips-row { display: flex; gap: 6px; flex-wrap: nowrap; overflow-x: auto; padding: 0.65rem 1.25rem; border-bottom: 1px solid var(--sep); scrollbar-width: none; }
.chip { background: var(--chat-chip-bg); color: var(--chat-chip-color) !important; border: 1px solid var(--sep); border-radius: 100px; padding: 0.25rem 0.75rem; font-size: 0.72rem; font-weight: 600; white-space: nowrap; flex-shrink: 0; }
.messages-scroll { height: 440px; overflow-y: auto; display: flex; flex-direction: column-reverse; padding: 1rem 1.25rem; background: var(--chat-bg); }
.messages-inner { display: flex; flex-direction: column; gap: 0.2rem; }
.bubble-row { display: flex; margin-bottom: 0.25rem; }
.user-row { justify-content: flex-end; }
.bot-row  { justify-content: flex-start; align-items: flex-end; gap: 0.4rem; }
.bot-mini-avatar { width: 22px; height: 22px; border-radius: 50%; background: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 0.48rem; font-weight: 800; color: #000; flex-shrink: 0; margin-bottom: 2px; }
.bubble { max-width: 70%; padding: 0.55rem 0.9rem; font-size: 0.84rem; line-height: 1.55; white-space: pre-wrap; word-break: break-word; }
.bubble-u { background: var(--bubble-user-bg); color: var(--bubble-user-text) !important; border-radius: 16px 16px 4px 16px; }
.bubble-b { background: var(--bubble-bot-bg); color: var(--bubble-bot-text) !important; border-radius: 16px 16px 16px 4px; }
.chat-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 0.5rem; text-align: center; color: var(--text-3) !important; font-size: 0.8rem; line-height: 1.65; padding: 2rem 1rem; }
.chat-input-bar { background: var(--input-bar-bg); border-top: 1px solid var(--sep); padding: 0.6rem 1.1rem 0.7rem; }
.gp-rw-outer { display: flex; align-items: center; height: 62px; overflow: visible; }
.gp-rw-text { font-size: 0.88rem; font-weight: 600; color: var(--text-2); white-space: nowrap; display: inline-flex; align-items: center; gap: 5px; }
.gp-rw-wrap { position: relative; display: inline-block; width: 72px; height: 1.2em; overflow: hidden; vertical-align: middle; }
@keyframes gp-rw-out { 0%,42%{transform:translateY(0);opacity:1} 50%,92%{transform:translateY(-160%);opacity:0} 100%{transform:translateY(0);opacity:1} }
@keyframes gp-rw-in  { 0%,42%{transform:translateY(160%);opacity:0} 50%,92%{transform:translateY(0);opacity:1} 100%{transform:translateY(160%);opacity:0} }
.gp-rw-a { position:absolute;top:0;left:0;white-space:nowrap;animation:gp-rw-out 5s cubic-bezier(0.16,1,0.3,1) infinite;color:var(--accent);font-weight:700; }
.gp-rw-b { position:absolute;top:0;left:0;white-space:nowrap;animation:gp-rw-in 5s cubic-bezier(0.16,1,0.3,1) infinite;color:var(--accent);font-weight:700; }
[data-testid="stBaseButton-primary"] { background: #22c55e !important; color: #ffffff !important; border: none !important; border-radius: 50px !important; font-weight: 700 !important; }
[data-testid="stBaseButton-primary"]:hover { background: #16a34a !important; color: #ffffff !important; }
/* ── Page-transition animations ───────────────────────────────────────── */
@keyframes gp-fade-up { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
@keyframes gp-fade-in { from{opacity:0} to{opacity:1} }
/* Results page — staggered fade-in */
.results-hero { animation: gp-fade-up 0.45s cubic-bezier(0.16,1,0.3,1) both; }
.metric-card  { animation: gp-fade-up 0.45s cubic-bezier(0.16,1,0.3,1) 0.08s both; }
.section-header { animation: gp-fade-in 0.4s ease 0.05s both; }
.info-box, .warn-box, .error-box { animation: gp-fade-in 0.35s ease both; }
.gp-card { animation: gp-fade-up 0.4s cubic-bezier(0.16,1,0.3,1) both; }
/* Charts, tables, expanders */
[data-testid="stImage"] { animation: gp-fade-up 0.55s cubic-bezier(0.16,1,0.3,1) 0.18s both; }
[data-testid="stDataFrame"] { animation: gp-fade-up 0.45s cubic-bezier(0.16,1,0.3,1) 0.12s both; }
[data-testid="stExpander"] { animation: gp-fade-in 0.4s ease 0.1s both; }
/* Spinner fade-in */
[data-testid="stSpinner"] > div { animation: gp-fade-in 0.3s ease both; }
[data-testid="stExpander"] { border: 1px solid var(--sep) !important; border-radius: var(--r-md) !important; background: var(--bg-card) !important; margin-bottom: 0.75rem !important; }
[data-testid="stExpander"] summary { color: transparent !important; font-weight: 500 !important; font-family: var(--font) !important; list-style: none !important; }
[data-testid="stExpander"] summary::-webkit-details-marker { display: none !important; }
[data-testid="stExpander"] summary p { color: var(--text-2) !important; font-size: 0.88rem !important; font-weight: 500 !important; }
[data-testid="stExpander"] summary svg { display: none !important; }
[data-testid="stExpander"] summary span { color: transparent !important; font-size: 0 !important; }
</style>
""", unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
# ESG DATABASE
# ══════════════════════════════════════════════════════════════════════════════
_ESG_CSV_URL = "https://raw.githubusercontent.com/minas120983-alt/lets-see/main/ESG%20data%202026.csv"
_ESG_CSV_LOCAL = "/mnt/user-data/uploads/ESG data 2026.csv"
def _parse_esg_df(df: pd.DataFrame) -> dict:
    df = df[df["fieldname"] == "ESGCombinedScore"].copy()
    df["valuescore"] = pd.to_numeric(df["valuescore"], errors="coerce")
    df = df.dropna(subset=["valuescore", "ticker"])
    df["ticker"] = df["ticker"].str.upper().str.strip()
    latest = df.sort_values("year").groupby("ticker").last().reset_index()
    return {
        row["ticker"]: {
            "app_esg": round(float(row["valuescore"]) * 10, 3),
            "letter": str(row["value"]),
            "year": int(row["year"]),
            "source": f"LSEG ESGCombinedScore ({int(row['year'])})",
            "has_esg": True,
        }
        for _, row in latest.iterrows()
    }
@st.cache_data(show_spinner=False)
def load_esg_db() -> dict:
    try:
        resp = requests.get(_ESG_CSV_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        result = _parse_esg_df(df)
        if result:
            return result
    except Exception:
        pass
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
            "source": None, "has_esg": False, "error": f"'{t}' not found in ESG CSV."}
# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO MATH
# ══════════════════════════════════════════════════════════════════════════════
def port_ret(w, mu):   return float(np.asarray(w) @ np.asarray(mu))
def port_var(w, cov):  return float(np.asarray(w) @ np.asarray(cov) @ np.asarray(w))
def port_sd(w, cov):   return float(max(port_var(w, cov), 1e-14) ** 0.5)
def port_sr(w, mu, cov, rf):
    ep = port_ret(w, mu); sp = port_sd(w, cov)
    return (ep - rf) / sp if sp > 1e-9 else 0.0
def port_stats(w, mu, cov, esg, rf):
    w = np.asarray(w)
    w_sum = float(np.sum(w))
    ep_full = rf * (1 - w_sum) + port_ret(w, mu)
    sp = port_sd(w, cov)
    sr = (ep_full - rf) / sp if sp > 1e-9 else 0.0
    esg_bar = float(w @ esg) / max(w_sum, 1e-9)
    return ep_full, sp, sr, esg_bar
def find_tangency(mu, cov, rf, bounds=None):
    n = len(mu)
    b = bounds or [(0., 1.)] * n
    res = minimize(
        lambda w: -port_sr(w, mu, cov, rf),
        np.ones(n) / n, method="SLSQP", bounds=b,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"ftol": 1e-10, "maxiter": 800})
    wt = res.x if res.success else np.ones(n) / n
    return wt, port_ret(wt, mu), port_sd(wt, cov), port_sr(wt, mu, cov, rf)
def find_optimal(mu, cov, esg, rf, gamma, lam):
    n = len(mu)
    mu_adj = np.asarray(mu) + (lam / max(gamma, 1e-9)) * np.asarray(esg)
    res = minimize(
        lambda w: -port_sr(w, mu_adj, cov, rf),
        np.ones(n) / n, method="SLSQP",
        bounds=[(0., 1.)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"ftol": 1e-10, "maxiter": 1000})
    w_tan = res.x if res.success else np.ones(n) / n
    ret_t = port_ret(w_tan, mu)
    sd_t  = port_sd(w_tan, cov)
    w_star = (ret_t - rf) / (gamma * sd_t ** 2) if sd_t > 1e-9 else 0.0
    w_star = float(np.clip(w_star, 0.0, 1.0))
    return w_tan * w_star
def build_mv_frontier(mu, cov, bounds=None, n_points=100):
    n = len(mu)
    b = bounds or [(0., 1.)] * n
    w_mv = minimize(lambda w: port_sd(w, cov), np.ones(n) / n,
                    method="SLSQP", bounds=b,
                    constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                    options={"ftol": 1e-10, "maxiter": 800}).x
    ret_max = float(np.max([port_ret(np.eye(n)[i], mu) for i in range(n) if b[i][1] > 0]))
    ret_min = float(np.min([port_ret(np.eye(n)[i], mu) for i in range(n) if b[i][1] > 0]))
    targets = np.linspace(ret_min, ret_max, n_points)
    stds, rets = [], []
    for rt in targets:
        res = minimize(lambda w: port_sd(w, cov), np.ones(n) / n,
                       method="SLSQP", bounds=b,
                       constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                                    {"type": "eq", "fun": lambda w, r=rt: port_ret(w, mu) - r}],
                       options={"ftol": 1e-10, "maxiter": 500})
        if res.success:
            stds.append(port_sd(res.x, cov) * 100)
            rets.append(port_ret(res.x, mu) * 100)
    return np.array(stds), np.array(rets)
def nearest_psd(matrix):
    ev, evec = np.linalg.eigh(matrix)
    ev[ev < 1e-8] = 1e-8
    return evec @ np.diag(ev) @ evec.T
# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
SUGGESTED_QUESTIONS = [
    "Why does my portfolio have these weights?",
    "What is the Sharpe ratio and is mine good?",
    "Why does the green frontier sit to the right of blue?",
    "What does the ESG preference (λ) actually do?",
    "How does increasing risk aversion (γ) change things?",
    "Which asset drags down my ESG score the most?",
    "What is the cost of the ESG constraint here?",
    "Explain the utility function used here.",
    "How does the Capital Market Line work?",
]
_CHIP_LABELS = [
    "Why these weights?", "Is my Sharpe ratio good?", "Why green frontier right of blue?",
    "What does λ do?", "How does γ change things?", "Worst ESG asset?",
    "Cost of ESG constraint?", "Explain the utility function", "How does the CML work?",
]
def _portfolio_answer(question: str, d: dict) -> str:
    q = question.lower()
    names = d["names"]; mu = d["mu"]; vols = d["vols"]
    esg_scores = d["esg_scores"]; w_opt = d["w_opt"]
    ep = d["ep"]; sp = d["sp"]; sr = d["sr"]; esg_bar = d["esg_bar"]
    gamma = d["gamma"]; lam = d["lam"]; rf = d["rf"]
    ep_tan_all = d["ep_tan_all"]; sp_tan_all = d["sp_tan_all"]; sr_tan_all = d["sr_tan_all"]
    ep_tan_esg = d["ep_tan_esg"]; sp_tan_esg = d["sp_tan_esg"]; sr_tan_esg = d["sr_tan_esg"]
    active_mask = d["active_mask"]; esg_thresh = d["esg_thresh"]
    cov = d["cov"]; n = d["n"]
    ind_sr = [(mu[i] - rf) / vols[i] for i in range(n)]
    sorted_by_w   = sorted(range(n), key=lambda i: w_opt[i], reverse=True)
    sorted_by_esg = sorted(range(n), key=lambda i: esg_scores[i])
    sorted_by_sr  = sorted(range(n), key=lambda i: ind_sr[i], reverse=True)
    w_sum = float(np.sum(w_opt))
    u_val = port_ret(w_opt, mu) - gamma / 2 * sp ** 2 + lam * (float(np.dot(w_opt, esg_scores)) / max(w_sum, 1e-9))
    sharpe_cost = sr_tan_all - sr_tan_esg
    ret_cost    = ep_tan_all - ep_tan_esg
    top_w_name  = names[sorted_by_w[0]]
    top_w_pct   = w_opt[sorted_by_w[0]] * 100
    worst_esg_i = sorted_by_esg[0]
    best_sr_i   = sorted_by_sr[0]
    if any(k in q for k in ["utility", "objective", "maximis", "optimi", "formula"]):
        esg_term = lam * (esg_bar / 100); var_term = gamma / 2 * sp ** 2
        return "\n".join([
            "The model maximises investor utility:",
            " U = E[Rp] − (γ/2)·σ²p + λ·(ESG̅/100)", "",
            f"For your portfolio (γ={gamma}, λ={lam}, rf={rf*100:.1f}%):",
            f" E[Rp] = {ep*100:.2f}%", f" −(γ/2)σ² = −{var_term*100:.3f}%",
            f" λ·(ESG̅/100) = {lam}·({esg_bar:.2f}/100) = +{esg_term:.4f}",
            f" Total U = {u_val:.5f}", "",
            f"With λ={lam}, each ESG point is worth {lam/100*100:.2f}% return equivalent.",
        ])
    if any(k in q for k in ["weight", "allocation", "holding", "position", "why does my portfolio"]):
        lines = [f"Weights maximise U = E[Rp] − (γ/2)σ² + λ·(ESG/100) with γ={gamma}, λ={lam}.", ""]
        for i in sorted_by_w:
            w = w_opt[i]; sr_i = ind_sr[i]
            tag = "" if w > 0.001 else " (zero weight)"
            lines.append(f" {names[i]} ({w*100:.1f}%{tag}): E[R]={mu[i]*100:.1f}%, σ={vols[i]*100:.1f}%, ESG={esg_scores[i]:.1f}/10, SR={sr_i:.3f}")
        return "\n".join(lines)
    if any(k in q for k in ["sharpe", "risk-adjusted", "risk adjusted"]):
        verdict = "excellent" if sr > 1.0 else "good" if sr > 0.6 else "moderate" if sr > 0.3 else "low"
        lines = [f"Sharpe = ({ep*100:.2f}% − {rf*100:.1f}%) / {sp*100:.2f}% = {sr:.3f} — {verdict}.", "",
                 f"Unconstrained tangency: {sr_tan_all:.3f} | ESG-screened tangency: {sr_tan_esg:.3f}",
                 "", "Individual SRs:"]
        for i in sorted_by_sr:
            lines.append(f" {names[i]}: {ind_sr[i]:.3f}")
        return "\n".join(lines)
    if any(k in q for k in ["cost", "constraint", "penalty", "sacrifice", "tradeoff", "price of esg"]):
        pct_loss = sharpe_cost / max(sr_tan_all, 0.001) * 100
        return "\n".join([
            f"ESG screen (min ESG ≥ {esg_thresh:.1f}) costs:",
            f" Sharpe loss: {sharpe_cost:.4f} ({pct_loss:.1f}% reduction)",
            f" Return loss: {ret_cost*100:.2f}%/yr",
            f" ESG gained: {esg_bar:.2f}/10", "",
            f"Unconstrained SR = {sr_tan_all:.4f} | ESG-screened SR = {sr_tan_esg:.4f}",
        ])
    if any(k in q for k in ["lambda", "λ", "esg preference", "what does the esg"]):
        esg_term = lam * (esg_bar / 100)
        return "\n".join([
            f"λ = {lam} is your ESG preference. It enters utility as +λ·(ESG̅/100) = {esg_term:.4f}.",
            f"Each 1-point improvement in ESG is worth {lam/100*100:.2f}% of expected return.",
            f"At λ=0 you'd hold the tangency portfolio (SR={sr_tan_all:.3f}).",
        ])
    if any(k in q for k in ["gamma", "γ", "risk aversion", "how does increasing risk"]):
        var_pen = gamma / 2 * sp ** 2
        sorted_by_vol = sorted(range(n), key=lambda i: vols[i])
        lines = [f"γ = {gamma}: penalises variance by −{var_pen*100:.3f}% in utility.", "Assets by volatility:"]
        for i in sorted_by_vol:
            lines.append(f" {names[i]}: σ={vols[i]*100:.1f}%, weight={w_opt[i]*100:.1f}%")
        return "\n".join(lines)
    if any(k in q for k in ["drags", "drag", "worst esg", "lowest esg", "which asset"]):
        lines = [f"Lowest ESG: {names[worst_esg_i]} ({esg_scores[worst_esg_i]:.2f}/10), weight={w_opt[worst_esg_i]*100:.1f}%", "", "All assets by ESG:"]
        for i in sorted_by_esg:
            flag = " [excluded]" if not active_mask[i] else ""
            lines.append(f" {names[i]}: {esg_scores[i]:.2f}/10{flag}")
        return "\n".join(lines)
    if any(k in q for k in ["capital market line", "cml", "market line"]):
        return "\n".join([
            "The CML is a straight line from the risk-free asset through the tangency portfolio.",
            f"Blue CML (base — all assets): SR={sr_tan_all:.4f} — E[R]={ep_tan_all*100:.2f}%, σ={sp_tan_all*100:.2f}%",
            f"Green CML (ESG utility-max): SR={sr_tan_esg:.4f} — E[R]={ep_tan_esg*100:.2f}%, σ={sp_tan_esg*100:.2f}%",
        ])
    if any(k in q for k in ["green frontier", "right of blue", "frontier sit"]):
        excluded = [names[i] for i in range(n) if not active_mask[i]]
        lines = [
            "Blue = all assets (largest feasible set). Green = ESG utility-max (smaller set).",
            "A smaller set can never beat a larger one → green sits to the RIGHT of blue.",
            f"Sharpe cost: {sharpe_cost:.4f} ({sharpe_cost/max(sr_tan_all,0.001)*100:.1f}% reduction)",
        ]
        if excluded: lines.append(f"Excluded: {', '.join(excluded)}")
        return "\n".join(lines)
    # fallback
    active = [(names[i], w_opt[i]) for i in range(n) if w_opt[i] > 0.001]
    lines = [f"Portfolio (γ={gamma}, λ={lam}): E[R]={ep*100:.2f}%, σ={sp*100:.2f}%, SR={sr:.3f}, ESG={esg_bar:.2f}/10", "", "Holdings:"]
    for nm, wt in active:
        lines.append(f" {nm}: {wt*100:.1f}%")
    return "\n".join(lines)
def answer_question(question: str) -> str:
    d = st.session_state.get("chat_data")
    if d is None:
        return "Please run the portfolio optimiser first."
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
    ret = close.pct_change().dropna(how="all")
    if ret.empty or ret.shape[1] < 2:
        raise ValueError("Not enough return data.")
    return close, ret, ret.mean() * 252, ret.std() * np.sqrt(252), ret.cov() * 252, ret.corr()
# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
if "page" not in st.session_state: st.session_state["page"] = "home"
if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
_page = st.session_state["page"]
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if _page == "home":
    st.markdown("""<style>
    .block-container { padding-top:0 !important; padding-bottom:0 !important; max-width:100% !important; padding-left:0 !important; padding-right:0 !important; }
    .stApp, [data-testid="stAppViewContainer"], section.main > div { background:#000000 !important; }
    div[data-testid="stHorizontalBlock"]:first-of-type { border-bottom:none !important; margin-bottom:0 !important; padding-bottom:0 !important; }
    /* Enter TerraVest — pill button, centred */
    div.stButton { display:flex !important; justify-content:center !important; }
    div.stButton > button { width:auto !important; min-width:210px !important; border-radius:50px !important; font-size:0.95rem !important; letter-spacing:-0.01em !important; padding:0.7rem 2.4rem !important; min-height:48px !important; animation:gp-fade-up 0.5s cubic-bezier(0.16,1,0.3,1) 0.35s both !important; }
    div.stButton > button:hover { background:#4ade80 !important; transform:none !important; }
    </style>""", unsafe_allow_html=True)
    _HOME_HTML = """<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
    *,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
    html,body{width:100%;height:100%;overflow:hidden;background:#000;font-family:"Plus Jakarta Sans",system-ui,sans-serif}
    canvas{position:absolute;top:0;left:0;width:100%;height:100%;display:block}
    .overlay{position:absolute;top:0;left:0;width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:20;text-align:center;padding:2rem;pointer-events:none}
    .overlay>*{opacity:0;transform:translateY(18px);animation:fadeUp .65s cubic-bezier(.16,1,.3,1) forwards}
    @keyframes fadeUp{to{opacity:1;transform:translateY(0)}}
    .logo-mark{width:52px;height:52px;border-radius:14px;background:rgba(255,255,255,.11);border:1px solid rgba(255,255,255,.24);display:flex;align-items:center;justify-content:center;margin:0 auto 1.3rem;animation-delay:.05s}
    .badge{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.18);border-radius:100px;padding:5px 18px;font-size:9.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.72);margin-bottom:1.5rem;animation-delay:.15s}
    .title{font-size:clamp(3.8rem,12vw,8rem);font-weight:800;letter-spacing:-.055em;line-height:.9;color:white;margin-bottom:1.1rem;animation-delay:.25s}
    .title .dim{opacity:.42}
    .subtitle{font-size:clamp(.88rem,1.6vw,1.02rem);color:rgba(255,255,255,.5);max-width:400px;line-height:1.72;animation-delay:.35s}
    </style></head><body>
    <canvas id="canvas"></canvas>
    <div class="overlay">
      <div class="logo-mark"><svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 2C8.13 2 5 5.58 5 10C5 14.15 7.9 17.55 11.75 18V22H12.25V18C16.1 17.55 19 14.15 19 10C19 5.58 15.87 2 12 2Z" fill="rgba(255,255,255,0.9)"/></svg></div>
      <div class="badge">ECN316 &middot; Sustainable Finance &middot; 2026</div>
      <h1 class="title">Terra<span class="dim">Vest</span></h1>
      <p class="subtitle">ESG-integrated portfolio optimisation. Build and analyse sustainable investments with live LSEG data and mean-variance theory.</p>
    </div>
    <script>
    var canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),W=0,H=0,mouse={x:.5,y:.5},sm={x:.5,y:.5},frame=0;
    var BLIND_COUNT=20,SPOT_R=.5,DAMP=.15,COLOR_A='#4ade80',COLOR_B='#000000';
    function lerp(a,b,t){return a+(b-a)*t}function clamp(v,lo,hi){return v<lo?lo:v>hi?hi:v}
    function resize(){W=canvas.width=canvas.offsetWidth;H=canvas.height=canvas.offsetHeight}
    window.addEventListener('resize',resize);
    document.addEventListener('mousemove',function(e){var r=canvas.getBoundingClientRect();mouse.x=(e.clientX-r.left)/W;mouse.y=(e.clientY-r.top)/H});
    function draw(){frame++;sm.x=lerp(sm.x,mouse.x,DAMP);sm.y=lerp(sm.y,mouse.y,DAMP);ctx.clearRect(0,0,W,H);
    var bg=ctx.createLinearGradient(0,0,W,0);bg.addColorStop(0,COLOR_A);bg.addColorStop(1,COLOR_B);ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
    var bw=W/BLIND_COUNT;
    for(var i=0;i<BLIND_COUNT;i++){var nx=(i+.5)/BLIND_COUNT,dist=Math.abs(nx-sm.x),spot=Math.max(0,1-dist/SPOT_R),wave=.03*Math.sin(frame*.02+i*.9),s=clamp(.42*(1-spot*.72)+wave,.02,.6);
    var g=ctx.createLinearGradient(i*bw,0,(i+1)*bw,0);
    g.addColorStop(0,'rgba(0,0,0,'+clamp(s+.32,0,.85)+')');g.addColorStop(.5,'rgba(0,0,0,'+clamp(s-.15,0,.45)+')');g.addColorStop(1,'rgba(0,0,0,'+clamp(s+.32,0,.85)+')');
    ctx.fillStyle=g;ctx.fillRect(i*bw,0,bw,H);ctx.strokeStyle='rgba(0,0,0,.55)';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(Math.round(i*bw)+.5,0);ctx.lineTo(Math.round(i*bw)+.5,H);ctx.stroke()}
    var sx=sm.x*W,sy=sm.y*H,sr2=SPOT_R*Math.min(W,H),sp2=ctx.createRadialGradient(sx,sy,0,sx,sy,sr2);
    sp2.addColorStop(0,'rgba(255,255,255,.36)');sp2.addColorStop(.4,'rgba(255,255,255,.1)');sp2.addColorStop(1,'rgba(255,255,255,0)');
    ctx.globalCompositeOperation='lighten';ctx.fillStyle=sp2;ctx.fillRect(0,0,W,H);ctx.globalCompositeOperation='source-over';
    var fd=ctx.createLinearGradient(0,H*.68,0,H);fd.addColorStop(0,'rgba(0,0,0,0)');fd.addColorStop(1,'rgba(0,0,0,.94)');ctx.fillStyle=fd;ctx.fillRect(0,0,W,H);
    requestAnimationFrame(draw)}resize();draw();
    </script></body></html>"""
    components.html(_HOME_HTML, height=620, scrolling=False)
    # Native Streamlit button — no URL change, CSS handles centering
    if st.button("Enter TerraVest →", key="home_enter_btn"):
        st.session_state["page"] = "input"
        st.rerun()
    st.stop()
# ══════════════════════════════════════════════════════════════════════════════
# DOT GRID BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════
if _page != "home":
    st.markdown("""<style>
    .stApp,[data-testid="stAppViewContainer"],section.main>div,.block-container,
    [data-testid="stVerticalBlock"],[data-testid="stVerticalBlockBorderWrapper"]{background:transparent!important}
    </style>""", unsafe_allow_html=True)
    components.html("""<!DOCTYPE html><html><head>
    <style>*{margin:0;padding:0;box-sizing:border-box}html,body{width:100%;height:100%;overflow:hidden;background:transparent}canvas{display:block;width:100%;height:100%}</style>
    </head><body><canvas id="c"></canvas><script>
    (function(){var fe=window.frameElement;if(fe){fe.style.cssText='position:fixed!important;top:0!important;left:0!important;width:100vw!important;height:100vh!important;z-index:0!important;pointer-events:none!important;border:none!important;';var el=fe.parentElement;while(el&&el.tagName!=='BODY'){el.style.overflow='visible';el.style.padding='0';el.style.margin='0';el=el.parentElement}}
    var DOT_R=2.5,GAP=15,BASE_HEX='#271E37',ACT_HEX='#22c55e',PROX=120,SPD_TRIG=160,SHK_R=180,SHK_STR=1.2,MAX_SPD=5000,DAMP=.88,SPRING=.045;
    function hexRGB(h){var v=parseInt(h.slice(1),16);return[(v>>16)&255,(v>>8)&255,v&255]}
    var BASE=hexRGB(BASE_HEX),ACT=hexRGB(ACT_HEX);
    function lerp(t){t=Math.min(Math.max(t,0),1);return'rgb('+Math.round(BASE[0]+(ACT[0]-BASE[0])*t)+','+Math.round(BASE[1]+(ACT[1]-BASE[1])*t)+','+Math.round(BASE[2]+(ACT[2]-BASE[2])*t)+')'}
    var cv=document.getElementById('c'),ctx=cv.getContext('2d'),dots=[],W=0,H=0,mx=-9999,my=-9999,pmx=-9999,pmy=-9999;
    function pw(){return window.parent||window}
    function buildGrid(){W=cv.width=pw().innerWidth||window.innerWidth;H=cv.height=pw().innerHeight||window.innerHeight;dots=[];for(var r=0;r*GAP<=H+GAP;r++)for(var c=0;c*GAP<=W+GAP;c++)dots.push({ox:c*GAP,oy:r*GAP,x:c*GAP,y:r*GAP,vx:0,vy:0})}
    try{pw().addEventListener('mousemove',function(e){mx=e.clientX;my=e.clientY},{passive:true});pw().addEventListener('resize',function(){buildGrid()},{passive:true})}catch(e){}
    function frame(){var sdx=mx-pmx,sdy=my-pmy,spd=Math.sqrt(sdx*sdx+sdy*sdy);
    if(spd>SPD_TRIG){var sf=(spd/SPD_TRIG)*SHK_STR;for(var i=0;i<dots.length;i++){var d=dots[i],dx=d.ox-mx,dy=d.oy-my,dist=Math.sqrt(dx*dx+dy*dy);if(dist<SHK_R&&dist>0){var f=(1-dist/SHK_R)*sf;d.vx+=(dx/dist)*f;d.vy+=(dy/dist)*f}}}
    pmx=mx;pmy=my;ctx.fillStyle='#0a080f';ctx.fillRect(0,0,W,H);
    for(var i=0;i<dots.length;i++){var d=dots[i],ox2mx=d.ox-mx,oy2my=d.oy-my,od=Math.sqrt(ox2mx*ox2mx+oy2my*oy2my);
    if(od<PROX&&od>0){var push=(1-od/PROX)*.55;d.vx+=(ox2mx/od)*push;d.vy+=(oy2my/od)*push}
    d.vx+=(d.ox-d.x)*SPRING;d.vy+=(d.oy-d.y)*SPRING;d.vx*=DAMP;d.vy*=DAMP;
    var sp=Math.sqrt(d.vx*d.vx+d.vy*d.vy);if(sp>MAX_SPD){d.vx=d.vx/sp*MAX_SPD;d.vy=d.vy/sp*MAX_SPD}
    d.x+=d.vx;d.y+=d.vy;var disp=Math.sqrt((d.x-d.ox)*(d.x-d.ox)+(d.y-d.oy)*(d.y-d.oy));
    ctx.beginPath();ctx.arc(d.x,d.y,DOT_R,0,6.2832);ctx.fillStyle=lerp(Math.min(disp/18,1));ctx.fill()}
    requestAnimationFrame(frame)}buildGrid();frame()})();
    </script></body></html>""", height=2, scrolling=False)
# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
if _page != "home":
    _n_logo, _n_rot, _n_gap, _n_back = st.columns([3.2, 3.8, 0.3, 4.0])
    with _n_logo:
        st.markdown("""<div class="gp-nav"><div class="gp-logo-row">
        <div class="gp-logo-mark"><svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 1C5.2 1 3 3.5 3 6.5C3 9.3 4.9 11.6 7.5 12V15H8.5V12C11.1 11.6 13 9.3 13 6.5C13 3.5 10.8 1 8 1Z" fill="#000"/></svg></div>
        <span class="gp-wordmark">Terra<span>Vest</span></span><span class="gp-badge">ESG</span>
        </div></div>""", unsafe_allow_html=True)
    with _n_rot:
        st.markdown("""<div class="gp-rw-outer"><span class="gp-rw-text">Sustainable&nbsp;<span class="gp-rw-wrap"><span class="gp-rw-a">Portfolio</span><span class="gp-rw-b">Life</span></span></span></div>""", unsafe_allow_html=True)
    with _n_back:
        if _page == "results":
            _nb_c1, _nb_c2 = st.columns(2)
            with _nb_c1:
                if st.button("← Setup", key="nav_back", use_container_width=True):
                    st.session_state["page"] = "input"; st.rerun()
            with _nb_c2:
                if st.button("Home", key="nav_home", type="primary", use_container_width=True):
                    st.session_state["page"] = "home"; st.rerun()
        else:
            _, _nb_c2 = st.columns([3, 1])
            with _nb_c2:
                if st.button("Home", key="nav_home", type="primary", use_container_width=True):
                    st.session_state["page"] = "home"; st.rerun()
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INPUT
# ══════════════════════════════════════════════════════════════════════════════
if _page == "input":
    st.markdown("""<style>.block-container{max-width:740px!important}</style>""", unsafe_allow_html=True)
    st.markdown("""<div class="gp-hero">
    <p class="gp-eyebrow">ECN316 · Sustainable Finance</p>
    <h1 class="gp-title">Build Your<br>ESG Portfolio</h1>
    <p class="gp-subtitle">Construct and optimise a mean-variance portfolio with integrated ESG scoring, drawn from live LSEG data.</p>
    </div>""", unsafe_allow_html=True)
    if _ESG_DB:
        st.markdown(f'<div class="info-box">ESG database loaded — <strong>{len(_ESG_DB):,} tickers</strong> from LSEG ESGCombinedScore CSV.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="error-box">Could not load ESG data.</div>', unsafe_allow_html=True)
    st.markdown('<div class="gp-label">Asset Universe</div>', unsafe_allow_html=True)
    _im_col, _ = st.columns([2, 3])
    with _im_col:
        input_mode = st.radio("Input method", ["Manual input", "Ticker-based input"], index=1, horizontal=False)
    default_names   = ["Tech ETF","Green Bond","Energy Stock","Healthcare","Consumer ETF","Infra Fund","EM Equity","Gov Bond","Real Estate","Commodity"]
    default_ret     = [9.0, 4.5, 7.0, 7.5, 6.5, 5.5, 10.0, 3.0, 6.0, 5.0]
    default_vol     = [18.0, 5.0, 22.0, 15.0, 14.0, 10.0, 25.0, 4.0, 13.0, 20.0]
    default_esg     = [6.5, 8.5, 2.0, 7.0, 5.5, 7.5, 4.0, 6.0, 5.0, 3.5]
    default_tickers = ["AAPL","MSFT","XOM","JNJ","SPY","TLT","NVDA","VWO","GLD","META"]
    asset_data = []; ticker_rows = []; corr_df = None; lookback_period = "3y"
    if input_mode == "Manual input":
        cl, cr = st.columns([2, 1])
        with cr:
            n_assets = st.number_input("Number of assets", 2, 10, 2, 1)
        st.markdown('<div class="info-box">Enter expected return, volatility and ESG score (0–10).</div>', unsafe_allow_html=True)
        with cl:
            h = st.columns([2, 1.2, 1.2, 1.2])
            h[0].markdown("**Asset name**"); h[1].markdown("**E[R] (%)**")
            h[2].markdown("**Vol (%)**"); h[3].markdown("**ESG (0–10)**")
            for i in range(int(n_assets)):
                c0, c1, c2, c3 = st.columns([2, 1.2, 1.2, 1.2])
                name = c0.text_input("", value=default_names[i], key=f"name_{i}", label_visibility="collapsed")
                ret  = c1.number_input("", value=default_ret[i],  key=f"ret_{i}",  label_visibility="collapsed", format="%.1f")
                vol  = c2.number_input("", value=default_vol[i],  key=f"vol_{i}",  label_visibility="collapsed", format="%.1f", min_value=0.1)
                esg  = c3.number_input("", value=default_esg[i],  key=f"esg_{i}",  label_visibility="collapsed", format="%.1f", min_value=0.0, max_value=10.0)
                asset_data.append({"name": name, "ret": ret / 100, "vol": vol / 100, "esg": esg})
        st.markdown('<div class="gp-label">Correlation Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Enter pairwise correlations (−1 to 1). Diagonal fixed at 1.</div>', unsafe_allow_html=True)
        n = int(n_assets)
        ci = pd.DataFrame(np.eye(n), columns=[asset_data[i]["name"] for i in range(n)], index=[asset_data[i]["name"] for i in range(n)])
        for r in range(n):
            for c in range(n):
                if r != c: ci.iloc[r, c] = 0.25
        corr_df = st.data_editor(ci, use_container_width=True, key="corr_matrix")
    else:
        cl, cr = st.columns([2, 1])
        with cr:
            n_assets = st.number_input("Number of assets", 2, 10, 2, 1, key="n_ticker_assets")
            lookback_period = st.selectbox("History window", ["1y","3y","5y","10y"], index=1)
        with cl:
            h = st.columns([1.1, 1.8])
            h[0].markdown("**Ticker**"); h[1].markdown("**Display name**")
            for i in range(int(n_assets)):
                c1, c2 = st.columns([1.1, 1.8])
                ticker = c1.text_input("", value=default_tickers[i], key=f"ticker_{i}", label_visibility="collapsed").upper().strip()
                _cache_key = f"fetched_name_{ticker}"
                if ticker and _cache_key not in st.session_state:
                    try:
                        _info = yf.Ticker(ticker).info
                        _fetched = _info.get("longName") or _info.get("shortName") or ticker
                        st.session_state[_cache_key] = _fetched
                    except Exception:
                        st.session_state[_cache_key] = ticker
                _default_name = st.session_state.get(_cache_key, default_names[i])
                name = c2.text_input("", value=_default_name, key=f"ticker_name_{i}", label_visibility="collapsed")
                ticker_rows.append({"ticker": ticker, "name": name or ticker, "manual_esg": None})
        valid_tickers = [r["ticker"] for r in ticker_rows if r["ticker"]]
        if valid_tickers:
            esg_preview  = {r["ticker"]: lookup_esg(r["ticker"]) for r in ticker_rows if r["ticker"]}
            missing_esg  = [t for t, res in esg_preview.items() if not res["has_esg"]]
            bad_tickers  = []
            for r in ticker_rows:
                t = r["ticker"]
                try:
                    info = yf.Ticker(t).fast_info
                    price = getattr(info, "last_price", None)
                    if price is None: bad_tickers.append(t)
                except Exception: bad_tickers.append(t)
            manual_overrides = {}; manual_ret_vol = {}
            if bad_tickers:
                st.markdown(f'<div class="warn-box"><strong>Tickers not found on Yahoo Finance:</strong> {", ".join(bad_tickers)}.</div>', unsafe_allow_html=True)
                for t in bad_tickers:
                    def_idx = default_tickers.index(t) if t in default_tickers else 0
                    bc1, bc2, bc3 = st.columns(3)
                    bc1.markdown(f"**{t}**")
                    m_ret = bc2.number_input(f"{t} E[R] (%)", value=default_ret[def_idx], min_value=-50.0, max_value=200.0, step=0.5, format="%.1f", key=f"manual_ret_{t}")
                    m_vol = bc3.number_input(f"{t} Vol (%)",  value=default_vol[def_idx], min_value=0.1,   max_value=200.0, step=0.5, format="%.1f", key=f"manual_vol_{t}")
                    manual_ret_vol[t] = {"ret": m_ret / 100.0, "vol": m_vol / 100.0}
            if missing_esg:
                st.markdown(f'<div class="warn-box"><strong>Not in ESG CSV:</strong> {", ".join(missing_esg)}.</div>', unsafe_allow_html=True)
                fcols = st.columns(min(len(missing_esg), 5))
                for idx, t in enumerate(missing_esg):
                    def_idx = default_tickers.index(t) if t in default_tickers else 0
                    manual_overrides[t] = fcols[idx % len(fcols)].number_input(
                        f"{t} ESG", value=float(default_esg[def_idx]), min_value=0.0, max_value=10.0, step=0.1, format="%.1f", key=f"manual_esg_{t}")
            if not missing_esg and not bad_tickers:
                st.markdown('<div class="info-box">All ticker data and ESG scores resolved.</div>', unsafe_allow_html=True)
            for row in ticker_rows:
                t = row["ticker"]
                row["manual_esg"]     = manual_overrides.get(t, None)
                row["manual_ret_vol"] = manual_ret_vol.get(t, None)
    st.markdown('<div class="gp-label">Investor Preferences</div>', unsafe_allow_html=True)
    _pref_mode = st.radio("How would you like to set your preferences?", ["Manual sliders", "Take the quiz"], horizontal=True, index=0, key="pref_mode")
    if _pref_mode == "Take the quiz":
        st.markdown('<div class="info-box"><strong>Quick Preference Quiz</strong></div>', unsafe_allow_html=True)
        _rq1 = st.radio("1 · Your portfolio falls 25%. What do you do?",
            ["Sell everything (3)", "Sell some (2)", "Hold (1)", "Buy more (0)"], key="quiz_r1", index=2)
        _rq2 = st.radio("2 · Which profile appeals to you?",
            ["4% ret, ~3% vol (3)", "8% ret, ~10% vol (2)", "13% ret, ~20% vol (1)", "20%+ ret, ~35% vol (0)"], key="quiz_r2", index=1)
        _rq3 = st.radio("3 · Investment time horizon?",
            ["Under 2 years (3)", "2–5 years (2)", "5–10 years (1)", "10+ years (0)"], key="quiz_r3", index=1)
        _eq1 = st.radio("4 · How important is sustainability?",
            ["Not important (0)", "Somewhat (1)", "Very important (2)", "Central mandate (3)"], key="quiz_e1", index=1)
        _eq2 = st.radio("5 · Accept lower return for higher ESG?",
            ["Never (0)", "Up to ~0.5%/yr (1)", "Up to ~2%/yr (2)", "Yes, ESG first (3)"], key="quiz_e2", index=1)
        _eq3 = st.radio("6 · View on ESG screening?",
            ["Ignore ESG (0)", "Nice to have (1)", "Meaningful tilts (2)", "Hard minimum (3)"], key="quiz_e3", index=1)
        def _score(opt_str): return int(opt_str.strip()[-2])
        _risk_score = _score(_rq1) + _score(_rq2) + _score(_rq3)
        _esg_score  = _score(_eq1) + _score(_eq2) + _score(_eq3)
        _gamma_map  = [(2, 1.0), (4, 2.5), (6, 4.5), (8, 7.0), (9, 9.5)]
        _lam_map    = [(1, 0.0), (3, 0.5), (5, 1.5), (7, 3.0), (9, 5.0)]
        gamma = next(v for threshold, v in _gamma_map if _risk_score <= threshold)
        lam   = next(v for threshold, v in _lam_map   if _esg_score  <= threshold)
        _qc1, _qc2 = st.columns(2)
        with _qc1:
            st.markdown(f'<div class="metric-card card-vol" style="margin-top:1rem;"><div class="metric-label">Derived Risk Aversion (γ)</div><div class="metric-value">{gamma}</div></div>', unsafe_allow_html=True)
        with _qc2:
            st.markdown(f'<div class="metric-card card-esg" style="margin-top:1rem;"><div class="metric-label">Derived ESG Preference (λ)</div><div class="metric-value">{lam}</div></div>', unsafe_allow_html=True)
    else:
        pref_c1, pref_c2 = st.columns(2)
        with pref_c1: gamma = st.slider("Risk Aversion (γ)", 0.5, 10.0, 3.0, 0.5)
        with pref_c2: lam   = st.slider("ESG Preference (λ)", 0.0, 5.0, 1.0, 0.1)
    pref_rf_col, _ = st.columns([1, 2])
    with pref_rf_col:
        rf = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 4.0, 0.1, format="%.1f") / 100
    st.markdown('<div class="gp-label">ESG Screen</div>', unsafe_allow_html=True)
    use_exclusion = st.checkbox("Apply minimum ESG exclusion screen", value=False)
    min_esg_filter = 0.0
    if use_exclusion:
        min_esg_filter = st.slider("Minimum ESG score (0–10)", 0.0, 10.0, 4.0, 0.5)
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
    _, cta_col, _ = st.columns([3, 2, 3])
    with cta_col:
        run = st.button("Run Optimisation", use_container_width=True)
    if run:
        ticker_data_display = None; esg_letters = {}; corr_np = None
        if input_mode == "Manual input":
            names  = [d["name"] for d in asset_data]
            mu     = np.array([d["ret"] for d in asset_data], dtype=float)
            vols   = np.array([d["vol"] for d in asset_data], dtype=float)
            esg_scores = np.array([d["esg"] for d in asset_data], dtype=float)
            n = len(names)
            try:
                corr_np = corr_df.values.astype(float)
            except Exception:
                st.error("Please make sure all correlation values are numeric."); st.stop()
            corr_np = (corr_np + corr_np.T) / 2
            np.fill_diagonal(corr_np, 1.0)
            corr_np = np.clip(corr_np, -0.999, 0.999)
            cov = np.outer(vols, vols) * corr_np
        else:
            tickers = [r["ticker"] for r in ticker_rows if r["ticker"]]
            if len(tickers) < 2:
                st.error("Please enter at least two valid ticker symbols."); st.stop()
            try:
                prices, returns, mu_series, vols_series, cov_df, corr_df_market = fetch_market_data(tickers, period=lookback_period)
            except Exception as e:
                st.error(f"Failed to fetch ticker data: {e}"); st.stop()
            manual_rv_map = {r["ticker"]: r.get("manual_ret_vol") for r in ticker_rows}
            manual_rv_map = {t: v for t, v in manual_rv_map.items() if v is not None}
            available = [t for t in tickers if t in mu_series.index]
            manual_price_tickers = [r["ticker"] for r in ticker_rows if r["ticker"] not in available and r.get("manual_ret_vol")]
            all_tickers = available + manual_price_tickers
            if len(all_tickers) < 2:
                st.error("Not enough valid tickers."); st.stop()
            filtered_rows = [r for r in ticker_rows if r["ticker"] in all_tickers]
            esg_map = {t: lookup_esg(t) for t in all_tickers}
            resolved = []; used_manual_esg = []
            for row in filtered_rows:
                t = row["ticker"]; meta = esg_map[t]
                if meta["has_esg"]:
                    final_esg = meta["app_esg"]; src_label = meta["source"]; letter = meta["letter"]; year = meta["year"]
                elif row.get("manual_esg") is not None:
                    final_esg = row["manual_esg"]; src_label = "Manual input"; letter = "N/A"; year = 2026; used_manual_esg.append(t)
                else:
                    final_esg = 5.0; src_label = "Default (5.0)"; letter = "N/A"; year = 2026
                resolved.append({"ticker": t, "name": row["name"], "final_esg": final_esg, "src": src_label, "letter": letter, "year": year})
                esg_letters[t] = letter
            names      = [r["name"] for r in resolved]
            esg_scores = np.array([r["final_esg"] for r in resolved], dtype=float)
            n = len(names)
            mu_list = []; vols_list = []
            for row in filtered_rows:
                t = row["ticker"]
                if t in available:
                    mu_list.append(float(mu_series[t])); vols_list.append(float(vols_series[t]))
                else:
                    rv = manual_rv_map.get(t, {"ret": 0.07, "vol": 0.20})
                    mu_list.append(rv["ret"]); vols_list.append(rv["vol"])
            mu = np.array(mu_list, dtype=float); vols = np.array(vols_list, dtype=float)
            cov    = np.zeros((n, n)); corr_np = np.zeros((n, n))
            for i, ti in enumerate(all_tickers):
                for j, tj in enumerate(all_tickers):
                    if ti in available and tj in available:
                        cov[i, j]    = float(cov_df.loc[ti, tj])
                        corr_np[i,j] = float(corr_df_market.loc[ti, tj])
                    elif i == j:
                        cov[i, j] = vols[i] ** 2; corr_np[i, j] = 1.0
            ticker_data_display = pd.DataFrame({
                "Ticker": all_tickers, "Name": names,
                "E[R] (%)": (mu * 100).round(2), "Vol (%)": (vols * 100).round(2),
                "ESG Score (0–10)": [r["final_esg"] for r in resolved],
                "LSEG Letter": [r["letter"] for r in resolved],
                "ESG Year": [r["year"] for r in resolved], "ESG Source": [r["src"] for r in resolved],
                "Return Source": ["Yahoo Finance" if t in available else "Manual" for t in all_tickers],
            })
            st.markdown(f'<div class="info-box">Market data loaded for: {", ".join(available)} over {lookback_period}.</div>', unsafe_allow_html=True)
        if np.any(np.linalg.eigvalsh(cov) < -1e-8): cov = nearest_psd(cov)
        esg_thresh  = min_esg_filter if use_exclusion else 0.0
        active_mask = esg_scores >= esg_thresh
        active_idx  = np.where(active_mask)[0]
        excluded    = [names[i] for i in range(n) if not active_mask[i]]
        if excluded:
            st.markdown(f'<div class="warn-box">Excluded: {", ".join(excluded)} (ESG &lt; {esg_thresh:.1f})</div>', unsafe_allow_html=True)
        if len(active_idx) < 2:
            st.error("Need at least 2 assets passing the ESG screen."); st.stop()
        mu_a  = mu[active_idx]; cov_a = cov[np.ix_(active_idx, active_idx)]; esg_a = esg_scores[active_idx]
        bounds_green = [(0., 1.) if active_mask[i] else (0., 0.) for i in range(n)]
        w_tan_all, ep_tan_all, sp_tan_all, sr_tan_all = find_tangency(mu, cov, rf)
        w_tan_esg, ep_tan_esg, sp_tan_esg, sr_tan_esg = find_tangency(mu, cov, rf, bounds=bounds_green)
        w_opt_a = find_optimal(mu_a, cov_a, esg_a, rf, gamma, lam)
        w_opt   = np.zeros(n)
        for idx, wi in zip(active_idx, w_opt_a): w_opt[idx] = wi
        ep, sp, sr, esg_bar = port_stats(w_opt_a, mu_a, cov_a, esg_a, rf)
        with st.spinner("Building efficient frontiers..."):
            std_blue,  ret_blue  = build_mv_frontier(mu, cov, n_points=100)
            std_green, ret_green = build_mv_frontier(mu, cov, bounds=bounds_green, n_points=100)
        st.session_state["opt_results"] = {
            "names": names, "mu": mu, "vols": vols, "esg_scores": esg_scores,
            "w_opt": w_opt, "ep": ep, "sp": sp, "sr": sr, "esg_bar": esg_bar,
            "gamma": gamma, "lam": lam, "rf": rf, "n": n,
            "ep_tan_all": ep_tan_all, "sp_tan_all": sp_tan_all, "sr_tan_all": sr_tan_all,
            "ep_tan_esg": ep_tan_esg, "sp_tan_esg": sp_tan_esg, "sr_tan_esg": sr_tan_esg,
            "active_mask": active_mask, "esg_thresh": esg_thresh, "cov": cov,
            "std_blue": std_blue, "ret_blue": ret_blue, "std_green": std_green, "ret_green": ret_green,
            "ticker_data_display": ticker_data_display, "corr_np": corr_np,
            "input_mode": input_mode, "esg_letters": esg_letters,
        }
        st.session_state["chat_data"]    = st.session_state["opt_results"]
        st.session_state["chat_history"] = []
        st.session_state["page"] = "results"
        st.rerun()
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif _page == "results":
    if "opt_results" not in st.session_state:
        st.markdown('<div class="warn-box">No results found. Return to setup.</div>', unsafe_allow_html=True)
        if st.button("Back to Setup"): st.session_state["page"] = "input"; st.rerun()
        st.stop()
    R = st.session_state["opt_results"]
    names = R["names"]; mu = R["mu"]; vols = R["vols"]; esg_scores = R["esg_scores"]
    w_opt = R["w_opt"]; ep = R["ep"]; sp = R["sp"]; sr = R["sr"]; esg_bar = R["esg_bar"]
    gamma = R["gamma"]; lam = R["lam"]; rf = R["rf"]; n = R["n"]
    ep_tan_all = R["ep_tan_all"]; sp_tan_all = R["sp_tan_all"]; sr_tan_all = R["sr_tan_all"]
    ep_tan_esg = R["ep_tan_esg"]; sp_tan_esg = R["sp_tan_esg"]; sr_tan_esg = R["sr_tan_esg"]
    active_mask = R["active_mask"]; esg_thresh = R["esg_thresh"]; cov = R["cov"]
    std_blue = R["std_blue"]; ret_blue = R["ret_blue"]
    std_green = R["std_green"]; ret_green = R["ret_green"]
    ticker_data_display = R["ticker_data_display"]; corr_np = R["corr_np"]
    input_mode = R["input_mode"]
    active_idx = np.where(active_mask)[0]
    mu_a = mu[active_idx]; cov_a = cov[np.ix_(active_idx, active_idx)]; esg_a = esg_scores[active_idx]
    CHART_BG = "#080808"; BLUE = "#3b82f6"; GREEN = "#22c55e"; ORANGE = "#fb923c"
    GREY = "#6b7280"; LABEL_C = "#f2f2f2"; LEG_BG = "#111111"; LEG_ED = "#222222"
    TICK_C = "#6b7280"; SPINE_C = "#222222"; GRID_C = "#1a1a1a"
    def _style_ax(ax, title):
        ax.set_facecolor(CHART_BG)
        ax.set_title(title, fontsize=10, fontweight="bold", color=LABEL_C, pad=10)
        ax.tick_params(colors=TICK_C, labelsize=8)
        for sp_ in ax.spines.values(): sp_.set_color(SPINE_C)
        ax.grid(True, alpha=0.3, color=GRID_C, linestyle="--", linewidth=0.6)
    u_val = port_ret(w_opt, mu) - gamma / 2 * sp ** 2 + lam * esg_bar
    st.markdown(f"""<div class="results-hero">
    <p class="gp-eyebrow">Optimisation Complete</p>
    <h2 class="results-title">Your Optimal Portfolio</h2>
    <p class="results-meta">γ = {gamma} &nbsp;·&nbsp; λ = {lam} &nbsp;·&nbsp; rf = {rf*100:.1f}% &nbsp;·&nbsp; U = {u_val:.4f}</p>
    </div>""", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, unit, cls, card_cls in [
        (m1, "Expected Return", f"{ep*100:.2f}", "%", "metric-pos", "card-ret"),
        (m2, "Volatility",      f"{sp*100:.2f}", "%", "",            "card-vol"),
        (m3, "Sharpe Ratio",    f"{sr:.3f}",      "",  "metric-pos" if sr > 0 else "metric-neg", "card-sr"),
        (m4, "ESG Score",       f"{esg_bar:.2f}", "/ 10", "metric-pos" if esg_bar >= 5 else "", "card-esg"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card {card_cls}"><div class="metric-label">{label}</div><div class="metric-value {cls}">{val}<span class="metric-unit">{unit}</span></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">Tangency Sharpe — Base (all assets): <strong>{sr_tan_all:.3f}</strong> &nbsp;|&nbsp; ESG Utility-Max: <strong>{sr_tan_esg:.3f}</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Portfolio Weights</div>', unsafe_allow_html=True)
    _rf_weight = max(0.0, 1.0 - float(np.sum(w_opt)))
    _display_names  = names + ["Risk-Free Asset"]
    _display_w      = [f"{w*100:.2f}" for w in w_opt] + [f"{_rf_weight*100:.2f}"]
    _display_ret    = [f"{r*100:.2f}" for r in mu] + [f"{rf*100:.2f}"]
    _display_vol    = [f"{v*100:.2f}" for v in vols] + ["0.00"]
    _display_esg    = [f"{s:.2f}" for s in esg_scores] + ["N/A"]
    st.dataframe(pd.DataFrame({
        "Asset": _display_names,
        "Weight (%)": _display_w,
        "E[R] (%)": _display_ret,
        "Vol (%)": _display_vol,
        "ESG (0–10)": _display_esg,
    }), use_container_width=True, hide_index=True)
    if input_mode == "Ticker-based input" and ticker_data_display is not None:
        with st.expander("Ticker data used"):
            st.dataframe(ticker_data_display, use_container_width=True, hide_index=True)
    if corr_np is not None:
        with st.expander("Correlation matrix"):
            st.dataframe(pd.DataFrame(corr_np, index=names, columns=names).round(3), use_container_width=True)
    st.markdown('<div class="section-header">Efficient Frontier</div>', unsafe_allow_html=True)
    # ── collect axis limit data ───────────────────────────────────────────────
    all_stds = list(std_blue) + list(std_green) + [sp * 100, sp_tan_all * 100, sp_tan_esg * 100]
    all_rets = list(ret_blue) + list(ret_green) + [ep * 100, ep_tan_all * 100, ep_tan_esg * 100, rf * 100]
    x_pad = max(all_stds) * 0.08 if all_stds else 5
    y_pad = ((max(all_rets) - min(all_rets)) * 0.12) if len(all_rets) > 1 else 1
    _c1, _c2 = st.columns(2)
    # ════════════════════════════════════════════════════════════════════════
    # GRAPH 1  —  Mean-Variance Frontier
    #   Blue  = Base portfolio      (all assets, unconstrained): frontier + CML
    #   Green = ESG Utility-Max portfolio (ESG-screened):        frontier + CML
    # ════════════════════════════════════════════════════════════════════════
    with _c1:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.patch.set_facecolor(CHART_BG)

        # ── Green FIRST (lower z) so blue always renders on top ──────────────
        # Green: ESG UTILITY-MAX portfolio frontier (ESG-screened)
        if len(std_green) > 2:
            ax.plot(std_green, ret_green, color=GREEN, lw=2.2, zorder=3,
                    label=f"Efficient Frontier — ESG Utility-Max (ESG ≥ {esg_thresh:.1f})")

        # Blue: BASE portfolio frontier (all assets, unconstrained) — drawn on top
        if len(std_blue) > 2:
            ax.plot(std_blue, ret_blue, color=BLUE, lw=2.5, zorder=5,
                    label="Efficient Frontier — Base (all assets)")

        # ── Capital Market Lines ──────────────────────────────────────────────
        cml_max = max(all_stds) + x_pad if all_stds else 50
        sd_cml  = np.linspace(0, cml_max, 300)

        # Green CML — ESG UTILITY-MAX (drawn first / lower z)
        if sp_tan_esg > 1e-9:
            ax.plot(sd_cml, rf * 100 + (ep_tan_esg - rf) / sp_tan_esg * sd_cml,
                    color=GREEN, lw=1.6, linestyle="--", zorder=4,
                    label=f"CML — ESG Utility-Max (SR={sr_tan_esg:.3f})")

        # Blue CML — BASE portfolio (drawn on top so always visible)
        if sp_tan_all > 1e-9:
            ax.plot(sd_cml, rf * 100 + (ep_tan_all - rf) / sp_tan_all * sd_cml,
                    color=BLUE, lw=1.6, linestyle="--", zorder=6,
                    label=f"CML — Base (SR={sr_tan_all:.3f})")

        # ── Tangency markers ──────────────────────────────────────────────────
        # Green tangency first (lower z) — ESG Utility-Max
        if sp_tan_esg > 1e-9:
            ax.scatter(sp_tan_esg * 100, ep_tan_esg * 100, color=GREEN, s=110, zorder=9,
                       edgecolors="white", lw=1.4, marker="o")
            ax.annotate("ESG Utility-Max\ntangency",
                        (sp_tan_esg * 100, ep_tan_esg * 100),
                        textcoords="offset points", xytext=(7, -20),
                        fontsize=7, color=GREEN, fontstyle="italic")

        # Blue tangency on top — Base
        ax.scatter(sp_tan_all * 100, ep_tan_all * 100, color=BLUE, s=110, zorder=11,
                   edgecolors="white", lw=1.4, marker="o")
        ax.annotate("Base tangency\n(all assets)",
                    (sp_tan_all * 100, ep_tan_all * 100),
                    textcoords="offset points", xytext=(-72, 8),
                    fontsize=7, color=BLUE, fontstyle="italic")

        # ── Risk-free asset ───────────────────────────────────────────────────
        ax.scatter(0, rf * 100, color=GREY, s=60, zorder=8,
                   edgecolors="white", lw=1, marker="s")

        # ── ESG-optimal portfolio (user's chosen portfolio) ───────────────────
        ax.scatter(sp * 100, ep * 100, color=ORANGE, s=160, zorder=10,
                   edgecolors="white", lw=2, marker="*", label="Your ESG-Optimal portfolio")

        # ── Individual assets ─────────────────────────────────────────────────
        for i in range(n):
            ax.scatter(vols[i] * 100, mu[i] * 100,
                       color=GREEN if active_mask[i] else BLUE,
                       s=45, zorder=6, edgecolors="white", lw=0.7, alpha=0.8)
            ax.annotate(names[i], (vols[i] * 100, mu[i] * 100),
                        textcoords="offset points", xytext=(4, 3), fontsize=7, color=GREY)

        ax.set_xlabel("Std (%)", fontsize=9, color=GREY)
        ax.set_ylabel("Expected Return (%)", fontsize=9, color=GREY)
        ax.set_xlim(0, max(all_stds) + x_pad)
        ax.set_ylim(rf * 100 - y_pad, max(all_rets) + y_pad)
        ax.legend(fontsize=7, framealpha=0.9, facecolor=LEG_BG, edgecolor=LEG_ED, labelcolor=LABEL_C)
        _style_ax(ax, "Mean-Variance Frontier")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ════════════════════════════════════════════════════════════════════════
    # GRAPH 2  —  ESG–Sharpe Frontier  (arch curve, x-axis from min ESG)
    # ════════════════════════════════════════════════════════════════════════
    with _c2:
        _w0   = np.ones(n) / n
        _ures = minimize(lambda w: -port_sr(w, mu, cov, rf), _w0, method="SLSQP",
                         bounds=[(0., 1.)] * n,
                         constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                         options={"ftol": 1e-10, "maxiter": 800})
        _w_unc  = _ures.x if _ures.success else _w0
        _sr_unc = port_sr(_w_unc, mu, cov, rf)
        _esg_unc = float(np.dot(_w_unc, esg_scores))
        _bounds_scr = [(0., 1.) if active_mask[i] else (0., 0.) for i in range(n)]
        _w_esgt, _ep_esgt, _sp_esgt, _sr_esgt = find_tangency(mu, cov, rf, bounds=_bounds_scr)
        _esg_esgt    = float(np.dot(_w_esgt, esg_scores))
        _scr_differs = (abs(_esg_esgt - _esg_unc) > 0.05 or abs(_sr_esgt - _sr_unc) > 0.005)
        _esg_min = float(np.min(esg_scores))
        _esg_max = float(np.max(esg_scores)) * 0.999
        _esg_pts, _sr_pts = [], []
        # Left side: ESG ≤ τ (rising toward peak)
        for _tau in np.linspace(_esg_min, _esg_unc, 45):
            try:
                _r = minimize(lambda w: -port_sr(w, mu, cov, rf), _w0, method="SLSQP",
                              bounds=[(0., 1.)] * n,
                              constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                                           {"type": "ineq", "fun": lambda w, t=_tau: t - float(np.dot(w, esg_scores))}],
                              options={"ftol": 1e-9, "maxiter": 500})
                if _r.success and port_sd(_r.x, cov) > 1e-9:
                    _esg_pts.append(float(np.dot(_r.x, esg_scores)))
                    _sr_pts.append(port_sr(_r.x, mu, cov, rf))
            except Exception:
                continue
        # Right side: ESG ≥ τ (falling from peak)
        for _tau in np.linspace(_esg_unc, _esg_max, 55):
            try:
                _r = minimize(lambda w: -port_sr(w, mu, cov, rf), _w0, method="SLSQP",
                              bounds=[(0., 1.)] * n,
                              constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                                           {"type": "ineq", "fun": lambda w, t=_tau: float(np.dot(w, esg_scores)) - t}],
                              options={"ftol": 1e-9, "maxiter": 500})
                if _r.success and port_sd(_r.x, cov) > 1e-9:
                    _esg_pts.append(float(np.dot(_r.x, esg_scores)))
                    _sr_pts.append(port_sr(_r.x, mu, cov, rf))
            except Exception:
                continue
        if _esg_pts:
            _pairs = sorted(set(zip([round(x, 4) for x in _esg_pts], [round(s, 5) for s in _sr_pts])))
            _esg_sorted = [p[0] for p in _pairs]
            _sr_sorted  = [p[1] for p in _pairs]
        else:
            _esg_sorted, _sr_sorted = [], []
        _indiv_sr = (mu - rf) / np.maximum(vols, 1e-9)
        _esg_opt  = float(np.dot(w_opt, esg_scores))
        # Axis limits: x starts from minimum asset ESG
        _all_sr_y = (_sr_sorted + list(_indiv_sr) + [_sr_unc, sr]
                     + ([_sr_esgt] if _scr_differs else []))
        _x_lo = max(0,   _esg_min - 0.25)
        _x_hi = min(10,  _esg_max + 0.25)
        _sr_range = max(_all_sr_y) - min(_all_sr_y) if len(_all_sr_y) > 1 else 0.1
        _y_lo = min(_all_sr_y) - _sr_range * 0.22
        _y_hi = max(_all_sr_y) + _sr_range * 0.30
        fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
        fig2.patch.set_facecolor(CHART_BG)
        ax2.set_facecolor(CHART_BG)
        if len(_esg_sorted) >= 2:
            ax2.plot(_esg_sorted, _sr_sorted, color=GREEN, lw=2.4, zorder=4, label="ESG–SR frontier")
            ax2.fill_between(_esg_sorted, max(0, _y_lo), _sr_sorted, alpha=0.07, color=GREEN, zorder=2)
        _ann_offsets = [(6, 6), (6, -16), (-60, 6), (-60, -16), (6, 16), (-60, 16), (14, -4)]
        for _i in range(n):
            ax2.scatter(esg_scores[_i], _indiv_sr[_i],
                        color=GREEN if active_mask[_i] else GREY,
                        s=55, zorder=6, edgecolors="white", lw=0.8, alpha=0.9)
            ax2.annotate(names[_i], (esg_scores[_i], _indiv_sr[_i]),
                         textcoords="offset points", xytext=_ann_offsets[_i % len(_ann_offsets)],
                         fontsize=7, color=GREY)
        ax2.scatter(_esg_unc, _sr_unc, color=BLUE, s=140, zorder=9,
                    edgecolors="white", lw=1.5, marker="D",
                    label=f"Tangency — ignoring ESG (SR={_sr_unc:.3f})")
        ax2.annotate(f"Tangency portfolio\nignoring ESG information",
                     (_esg_unc, _sr_unc), textcoords="offset points", xytext=(8, 10),
                     fontsize=7, color=BLUE, fontstyle="italic",
                     bbox=dict(boxstyle="round,pad=0.25", fc=CHART_BG, ec=BLUE, alpha=0.85, lw=0.6))
        if _sp_esgt > 1e-9 and _scr_differs:
            ax2.scatter(_esg_esgt, _sr_esgt, color=GREEN, s=170, zorder=10,
                        edgecolors="white", lw=2, marker="*",
                        label=f"Tangency — ESG screened (SR={_sr_esgt:.3f})")
            ax2.annotate(f"Tangency portfolio\nusing ESG information",
                         (_esg_esgt, _sr_esgt), textcoords="offset points", xytext=(8, -32),
                         fontsize=7, color=GREEN, fontstyle="italic",
                         bbox=dict(boxstyle="round,pad=0.25", fc=CHART_BG, ec=GREEN, alpha=0.85, lw=0.6))
        ax2.scatter(_esg_opt, sr, color=ORANGE, s=180, zorder=11,
                    edgecolors="white", lw=2, marker="*",
                    label=f"Your portfolio (SR={sr:.3f})")
        _ann_x = 10 if _esg_opt < (_x_lo + _x_hi) / 2 else -95
        ax2.annotate(f"Your portfolio\nSR = {sr:.3f}",
                     (_esg_opt, sr), textcoords="offset points", xytext=(_ann_x, -24),
                     fontsize=7, color=ORANGE, fontstyle="italic",
                     bbox=dict(boxstyle="round,pad=0.25", fc=CHART_BG, ec=ORANGE, alpha=0.85, lw=0.6))
        ax2.set_xlim(_x_lo, _x_hi)
        ax2.set_ylim(_y_lo, _y_hi)
        ax2.set_xlabel("ESG Score (0–10)", fontsize=9, color=GREY)
        ax2.set_ylabel("Sharpe Ratio", fontsize=9, color=GREY)
        ax2.legend(fontsize=7, framealpha=0.92, facecolor=LEG_BG, edgecolor=LEG_ED, labelcolor=LABEL_C,
                   loc="upper left" if _esg_unc > (_x_lo + _x_hi) / 2 else "upper right")
        _style_ax(ax2, "ESG–Sharpe Frontier")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()
    # ── Portfolio Breakdown ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Portfolio Breakdown</div>', unsafe_allow_html=True)
    _c3, _c4 = st.columns(2)
    with _c3:
        nonzero = [(w_opt[i], names[i]) for i in range(n) if w_opt[i] > 1e-4]
        if nonzero:
            w_nz, n_nz = zip(*nonzero)
            fig3, ax3 = plt.subplots(figsize=(5.5, 5.5))
            fig3.patch.set_facecolor(CHART_BG)
            _palette = [GREEN, BLUE, ORANGE, "#818cf8","#fb923c","#f472b6","#34d399","#60a5fa","#a78bfa","#fbbf24"]
            ax3.pie(w_nz, labels=n_nz, colors=_palette[:len(w_nz)], autopct="%1.1f%%", pctdistance=0.82,
                    textprops={"fontsize": 8, "color": LABEL_C},
                    wedgeprops={"linewidth": 1.5, "edgecolor": CHART_BG})
            ax3.set_title("Weight Allocation", fontsize=10, fontweight="bold", color=LABEL_C, pad=10)
            fig3.patch.set_facecolor(CHART_BG)
            fig3.tight_layout(); st.pyplot(fig3); plt.close()
    with _c4:
        risk_contrib = np.array([w_opt[i] * np.dot(cov[i], w_opt) for i in range(n)])
        total_var    = float(np.dot(w_opt, np.dot(cov, w_opt)))
        if total_var > 1e-10:
            risk_pct = (risk_contrib / total_var * 100)
            pos_mask = risk_pct > 0.01
            if pos_mask.sum() > 0:
                fig4, ax4 = plt.subplots(figsize=(5.5, 5.5))
                fig4.patch.set_facecolor(CHART_BG); ax4.set_facecolor(CHART_BG)
                ax4.barh([names[i] for i in range(n) if pos_mask[i]],
                         risk_pct[pos_mask],
                         color=[GREEN if active_mask[i] else BLUE for i in range(n) if pos_mask[i]],
                         edgecolor=CHART_BG, linewidth=1.2, height=0.55)
                ax4.set_xlabel("Risk Contribution (%)", fontsize=9, color=GREY)
                _style_ax(ax4, "Risk Contribution by Asset")
                fig4.tight_layout(); st.pyplot(fig4); plt.close()
    # ── Sensitivity Analysis ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    _gamma_range = np.linspace(0.5, 10, 30); _lam_range = np.linspace(0, 5, 30)
    sens_g = []; sens_l = []
    for g_ in _gamma_range:
        try:
            w_ = find_optimal(mu_a, cov_a, esg_a, rf, g_, lam)
            ep_, sp_, sr_, _ = port_stats(w_, mu_a, cov_a, esg_a, rf)
            sens_g.append({"γ": round(float(g_), 2), "E[R](%)": round(ep_ * 100, 3), "σ(%)": round(sp_ * 100, 3), "Sharpe": round(sr_, 4)})
        except Exception: pass
    for l_ in _lam_range:
        try:
            w_ = find_optimal(mu_a, cov_a, esg_a, rf, gamma, l_)
            ep_, sp_, sr_, _ = port_stats(w_, mu_a, cov_a, esg_a, rf)
            sens_l.append({"λ": round(float(l_), 2), "E[R](%)": round(ep_ * 100, 3), "σ(%)": round(sp_ * 100, 3), "Sharpe": round(sr_, 4)})
        except Exception: pass
    sens_g_df = pd.DataFrame(sens_g); sens_l_df = pd.DataFrame(sens_l)
    _s1, _s2 = st.columns(2)
    with _s1:
        if not sens_g_df.empty:
            fig5, ax5 = plt.subplots(figsize=(6.5, 4))
            fig5.patch.set_facecolor(CHART_BG); ax5.set_facecolor(CHART_BG)
            ax5.plot(sens_g_df["γ"], sens_g_df["E[R](%)"], color=GREEN, lw=2, label="E[R]")
            ax5.plot(sens_g_df["γ"], sens_g_df["σ(%)"],    color=BLUE,  lw=2, linestyle="--", label="Vol")
            ax5.set_xlabel("γ (Risk Aversion)", fontsize=9, color=GREY); ax5.set_ylabel("%", fontsize=9, color=GREY)
            ax5.legend(fontsize=8, facecolor=LEG_BG, edgecolor=LEG_ED, labelcolor=LABEL_C)
            _style_ax(ax5, "Return & Risk vs Risk Aversion (γ)")
            fig5.tight_layout(); st.pyplot(fig5); plt.close()
    with _s2:
        if not sens_l_df.empty:
            fig6, ax6 = plt.subplots(figsize=(6.5, 4))
            fig6.patch.set_facecolor(CHART_BG); ax6.set_facecolor(CHART_BG)
            ax6.plot(sens_l_df["λ"], sens_l_df["E[R](%)"], color=GREEN,  lw=2, label="E[R]")
            ax6.plot(sens_l_df["λ"], sens_l_df["σ(%)"],    color=ORANGE, lw=2, linestyle="--", label="Vol")
            ax6.set_xlabel("λ (ESG Preference)", fontsize=9, color=GREY); ax6.set_ylabel("%", fontsize=9, color=GREY)
            ax6.legend(fontsize=8, facecolor=LEG_BG, edgecolor=LEG_ED, labelcolor=LABEL_C)
            _style_ax(ax6, "Return & Risk vs ESG Preference (λ)")
            fig6.tight_layout(); st.pyplot(fig6); plt.close()
    st.markdown("""<div class="info-box">
    <strong>Methodology:</strong> Utility U = E[R<sub>p</sub>] &minus; (&gamma;/2)&sigma;&sup2;<sub>p</sub> + &lambda;s&#772;,
    maximised via SLSQP (no short-selling). Blue frontier + CML: unconstrained base portfolio (all assets).
    Green frontier + CML: ESG utility-max portfolio (restricted to assets passing the ESG screen). ESG data: LSEG ESGCombinedScore CSV, scaled 0–10.
    </div>""", unsafe_allow_html=True)
    # ── Chatbot ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Portfolio Explainer</div>', unsafe_allow_html=True)
    if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
    _t1_c = "#f2f2f2"
    if not st.session_state["chat_history"]:
        msgs_html = '<div class="chat-empty">Ask about weights, the Sharpe ratio, ESG scores,<br>the utility function, or any part of your portfolio.</div>'
    else:
        msgs_html = '<div class="messages-inner">'
        for msg in st.session_state["chat_history"]:
            safe = msg["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            if msg["role"] == "user":
                msgs_html += f'<div class="bubble-row user-row"><div class="bubble bubble-u">{safe}</div></div>'
            else:
                msgs_html += f'<div class="bubble-row bot-row"><div class="bot-mini-avatar">GP</div><div class="bubble bubble-b">{safe}</div></div>'
        msgs_html += '</div>'
    st.markdown(f"""<div class="chat-page">
    <div class="chat-header">
      <div class="chat-avatar">GP</div>
      <div style="flex:1;"><p class="chat-name" style="color:{_t1_c};">Portfolio Explainer</p><p class="chat-status">Active</p></div>
      <div style="font-size:.68rem;color:rgba(128,128,128,.6);text-align:right;line-height:1.7;">Powered by TerraVest<br>No API key required</div>
    </div>
    <div class="chips-row">{"".join(f'<span class="chip">{q}</span>' for q in SUGGESTED_QUESTIONS)}</div>
    <div class="messages-scroll">{msgs_html}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        _fi, _fs = st.columns([8, 1])
        with _fi:
            user_input = st.text_input("msg", placeholder="Ask about your portfolio...", label_visibility="collapsed")
        with _fs:
            submitted = st.form_submit_button("Send", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    if submitted and user_input.strip():
        reply = answer_question(user_input.strip())
        st.session_state["chat_history"].append({"role": "user",      "content": user_input.strip()})
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()
    st.markdown("<div style='margin-top:1.25rem;'>", unsafe_allow_html=True)
    st.caption("Suggested questions:")
    _pc = st.columns(3)
    for _i, _q in enumerate(SUGGESTED_QUESTIONS):
        _label = _CHIP_LABELS[_i] if _i < len(_CHIP_LABELS) else _q
        with _pc[_i % 3]:
            if st.button(_label, key=f"chip_{_i}", use_container_width=True):
                _r = answer_question(_q)
                st.session_state["chat_history"].append({"role": "user",      "content": _q})
                st.session_state["chat_history"].append({"role": "assistant", "content": _r})
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    if st.session_state.get("chat_history"):
        _, _clr_col, _ = st.columns([3, 1, 3])
        with _clr_col:
            if st.button("Clear chat", key="chat_clear"):
                st.session_state["chat_history"] = []; st.rerun()
    st.markdown("""<div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid var(--sep);text-align:center;font-size:.65rem;color:var(--text-3);letter-spacing:.06em;text-transform:uppercase;">
    TerraVest &nbsp;·&nbsp; ECN316 Sustainable Finance &nbsp;·&nbsp; 2026
    </div>""", unsafe_allow_html=True)
