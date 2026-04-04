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

# ── CSS: Apple Support aesthetic + day/night mode ────────────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════════════════════
   DESIGN TOKENS  —  light (day) is default; dark overrides
   ══════════════════════════════════════════════════════════ */
:root {
  /* Light / Day (Apple Support palette) */
  --bg:           #f5f5f7;
  --bg-card:      #ffffff;
  --bg-elevated:  #ffffff;
  --bg-3:         #e8e8ed;
  --bg-input:     rgba(118,118,128,0.12);

  --text-1:       #1d1d1f;
  --text-2:       #6e6e73;
  --text-3:       #86868b;
  --text-inv:     #ffffff;

  --accent:       #0071e3;
  --accent-hover: #0077ed;
  --accent-light: rgba(0,113,227,0.10);

  --sys-green:    #1d8348;
  --sys-orange:   #c84b00;
  --sys-red:      #c0392b;
  --sys-indigo:   #5e5ce6;

  --border:       rgba(0,0,0,0.08);
  --sep:          rgba(0,0,0,0.12);
  --sep-strong:   rgba(0,0,0,0.18);

  --sb-bg:        #f5f5f7;
  --sb-border:    rgba(0,0,0,0.10);

  --chat-bg:          #ffffff;
  --chat-header-bg:   #f5f5f7;
  --chat-chip-bg:     #e8e8ed;
  --chat-chip-color:  #0071e3;
  --chat-msg-bg:      #f0f0f5;
  --bubble-user-bg:   #0071e3;
  --bubble-user-text: #ffffff;
  --bubble-bot-bg:    #e8e8ed;
  --bubble-bot-text:  #1d1d1f;
  --input-bar-bg:     #f5f5f7;

  --chart-bg:    #ffffff;
  --chart-bg2:   #f5f5f7;

  /* Typography */
  --font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display",
          "Helvetica Neue", Arial, sans-serif;
  --font-mono: "SF Mono", SFMono-Regular, Menlo, Monaco, Consolas, monospace;

  /* Motion */
  --ease: cubic-bezier(0.25, 0.46, 0.45, 0.94);

  /* Radii */
  --r-xs: 6px;
  --r-sm: 8px;
  --r-md: 12px;
  --r-lg: 16px;
  --r-xl: 20px;
}

/* ── Dark / Night overrides ── */
[data-theme="dark"] {
  --bg:           #000000;
  --bg-card:      #1c1c1e;
  --bg-elevated:  #2c2c2e;
  --bg-3:         #3a3a3c;
  --bg-input:     rgba(118,118,128,0.24);

  --text-1:       rgba(255,255,255,1.00);
  --text-2:       rgba(235,235,245,0.60);
  --text-3:       rgba(235,235,245,0.30);
  --text-inv:     #000000;

  --accent:       #0a84ff;
  --accent-hover: #409cff;
  --accent-light: rgba(10,132,255,0.18);

  --sys-green:    #30d158;
  --sys-orange:   #ff9f0a;
  --sys-red:      #ff453a;
  --sys-indigo:   #5e5ce6;

  --border:       rgba(255,255,255,0.08);
  --sep:          rgba(84,84,88,0.65);
  --sep-strong:   rgba(84,84,88,0.90);

  --sb-bg:        #1c1c1e;
  --sb-border:    rgba(255,255,255,0.08);

  --chat-bg:          #000000;
  --chat-header-bg:   #1c1c1e;
  --chat-chip-bg:     #2c2c2e;
  --chat-chip-color:  #0a84ff;
  --chat-msg-bg:      #1c1c1e;
  --bubble-user-bg:   #0a84ff;
  --bubble-user-text: #ffffff;
  --bubble-bot-bg:    #2c2c2e;
  --bubble-bot-text:  rgba(255,255,255,0.88);
  --input-bar-bg:     #1c1c1e;

  --chart-bg:    #000000;
  --chart-bg2:   #1c1c1e;
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: var(--font);
  -webkit-font-smoothing: antialiased;
  font-feature-settings: "kern" 1;
}
.stApp { background: var(--bg) !important; transition: background 0.3s var(--ease); }
.block-container {
  padding-top: 2.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 1200px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--sb-bg) !important;
  border-right: 0.5px solid var(--sb-border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-2) !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: var(--text-1) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
}
[data-testid="stSidebar"] hr {
  border: none !important;
  border-top: 0.5px solid var(--sep) !important;
  margin: 0.85rem 0 !important;
}
[data-testid="stSidebar"] label {
  font-size: 0.7rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--text-3) !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
  background: var(--accent) !important;
  border: none !important;
  box-shadow: 0 0 0 3px var(--accent-light) !important;
}
[data-testid="stSidebar"] .stSlider div[data-testid="stTickBar"] { display: none; }
[data-testid="stSidebar"] .stNumberInput input {
  background: var(--bg-input) !important;
  color: var(--text-1) !important;
  border: 0.5px solid var(--sep) !important;
  border-radius: var(--r-sm) !important;
}
[data-testid="stSidebar"] .stCheckbox label {
  font-size: 0.85rem !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  color: var(--text-2) !important;
}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 { color: var(--text-1) !important; }
p, div, span { color: var(--text-2); }
code {
  font-family: var(--font-mono);
  background: var(--bg-input);
  color: var(--accent);
  border-radius: 4px;
  padding: 1px 6px;
  font-size: 0.84em;
}

/* ── Hero ── */
.hero-title {
  font-size: 2.5rem;
  font-weight: 700;
  letter-spacing: -0.025em;
  color: var(--text-1);
  line-height: 1.06;
  margin-bottom: 0.25rem;
}
.hero-sub {
  font-size: 0.92rem;
  color: var(--text-3);
  font-weight: 400;
  letter-spacing: -0.01em;
  margin-bottom: 0;
}

/* ── Section headers ── */
.section-header {
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-3) !important;
  border-bottom: 0.5px solid var(--sep);
  padding-bottom: 0.5rem;
  margin: 2rem 0 1.25rem;
}

/* ── Metric cards ── */
.metric-card {
  background: var(--bg-card);
  border: 0.5px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1.2rem 1.4rem 1.1rem;
  margin-bottom: 0.75rem;
  position: relative;
  overflow: hidden;
  transition: box-shadow 0.2s var(--ease), transform 0.2s var(--ease);
  box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}
.metric-card:hover {
  box-shadow: 0 4px 16px rgba(0,0,0,0.10);
  transform: translateY(-2px);
}
.metric-card.card-ret  { border-top: 2px solid var(--sys-green); }
.metric-card.card-vol  { border-top: 2px solid var(--sys-orange); }
.metric-card.card-sr   { border-top: 2px solid var(--accent); }
.metric-card.card-esg  { border-top: 2px solid var(--sys-indigo); }
.metric-label {
  font-size: 0.66rem;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--text-3) !important;
  margin-bottom: 0.4rem;
}
.metric-value {
  font-size: 2rem;
  font-weight: 600;
  letter-spacing: -0.03em;
  color: var(--text-1) !important;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.metric-unit { font-size: 0.76rem; color: var(--text-3) !important; margin-left: 2px; font-weight: 400; }
.metric-pos  { color: var(--sys-green) !important; }
.metric-neg  { color: var(--sys-red) !important; }

/* ── Status boxes ── */
.info-box {
  background: var(--accent-light);
  border: 0.5px solid rgba(0,113,227,0.22);
  border-radius: var(--r-sm);
  padding: 0.8rem 1.1rem;
  margin: 0.6rem 0;
  font-size: 0.83rem;
  color: var(--accent) !important;
  line-height: 1.55;
}
.warn-box {
  background: rgba(255,159,10,0.08);
  border: 0.5px solid rgba(255,159,10,0.22);
  border-radius: var(--r-sm);
  padding: 0.8rem 1.1rem;
  margin: 0.6rem 0;
  font-size: 0.83rem;
  color: var(--sys-orange) !important;
  line-height: 1.55;
}
.error-box {
  background: rgba(255,69,58,0.08);
  border: 0.5px solid rgba(255,69,58,0.22);
  border-radius: var(--r-sm);
  padding: 0.8rem 1.1rem;
  margin: 0.6rem 0;
  font-size: 0.83rem;
  color: var(--sys-red) !important;
  line-height: 1.55;
}

/* ── Primary button ── */
div.stButton > button {
  background: var(--accent) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 980px !important;
  padding: 0.55rem 1.6rem !important;
  font-family: var(--font) !important;
  font-weight: 500 !important;
  font-size: 0.92rem !important;
  letter-spacing: -0.01em !important;
  width: 100% !important;
  transition: background 0.15s var(--ease), transform 0.1s var(--ease) !important;
}
div.stButton > button:hover {
  background: var(--accent-hover) !important;
  transform: scale(1.02) !important;
}
div.stButton > button:active { transform: scale(0.98) !important; }

/* ── Theme toggle button (ghost/outline style) ── */
div[data-testid="stSidebar"] div.stButton > button {
  background: transparent !important;
  color: var(--text-2) !important;
  border: 0.5px solid var(--sep) !important;
  border-radius: var(--r-sm) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  padding: 0.4rem 1rem !important;
  margin-bottom: 1rem !important;
  opacity: 0.85;
}
div[data-testid="stSidebar"] div.stButton > button:hover {
  background: var(--bg-3) !important;
  opacity: 1;
  transform: none !important;
}

/* ── Inputs ── */
.stNumberInput input, .stTextInput input, .stTextArea textarea {
  background: var(--bg-input) !important;
  color: var(--text-1) !important;
  border: 0.5px solid var(--sep) !important;
  border-radius: var(--r-sm) !important;
  font-family: var(--font) !important;
  font-size: 0.9rem !important;
  transition: border-color 0.15s !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-light) !important;
}
.stSelectbox div[data-baseweb="select"] > div {
  background: var(--bg-input) !important;
  color: var(--text-1) !important;
  border: 0.5px solid var(--sep) !important;
}
.stRadio label { color: var(--text-2) !important; }
.stRadio div[role="radiogroup"] label { font-size: 0.88rem !important; }
.stCheckbox div[data-testid="stMarkdownContainer"] p { color: var(--text-2) !important; }

/* ── Tables ── */
.stDataFrame, [data-testid="stDataEditor"] {
  border-radius: var(--r-md);
  overflow: hidden;
  border: 0.5px solid var(--sep) !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stDataEditor"] * { color: var(--text-1) !important; background: var(--bg-card) !important; }
.stDataFrame thead th {
  background: var(--bg-3) !important; color: var(--text-3) !important;
  font-size: 0.68rem !important; letter-spacing: 0.07em !important;
  text-transform: uppercase !important; font-weight: 600 !important;
}
.stDataFrame tbody tr { border-bottom: 0.5px solid var(--sep) !important; }
.stDataFrame tbody tr:hover { background: var(--bg-3) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  border: 0.5px solid var(--sep) !important;
  border-radius: var(--r-md) !important;
  background: var(--bg-card) !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  margin-bottom: 0.75rem !important;
}
[data-testid="stExpander"] summary { color: var(--text-2) !important; font-size: 0.9rem !important; font-weight: 500 !important; }
[data-testid="stExpander"] summary * { color: var(--text-2) !important; }
[data-testid="stExpander"] p { color: var(--text-2) !important; }

/* ── Markdown tables ── */
table { color: var(--text-1) !important; border-collapse: collapse; width: 100%; }
thead tr th { color: var(--text-3) !important; font-size: 0.68rem !important; letter-spacing: 0.07em !important; text-transform: uppercase !important; border-bottom: 0.5px solid var(--sep) !important; padding: 0.55rem 0.8rem !important; font-weight: 600 !important; }
tbody tr td { color: var(--text-2) !important; border-bottom: 0.5px solid var(--sep) !important; padding: 0.5rem 0.8rem !important; }
tbody tr:hover td { background: var(--bg-3) !important; }

/* ── Divider ── */
hr { border: none !important; border-top: 0.5px solid var(--sep) !important; margin: 2rem 0 !important; }

/* ── Tabs (segmented control) ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  background: var(--bg-input);
  border-radius: 10px;
  padding: 3px;
  width: fit-content;
  margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  padding: 0.35rem 1.1rem;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text-3) !important;
  background: transparent !important;
  border: none !important;
  letter-spacing: -0.01em;
  transition: color 0.15s, background 0.15s;
}
.stTabs [aria-selected="true"] {
  background: var(--bg-card) !important;
  color: var(--text-1) !important;
  font-weight: 600 !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ── iMessage chat ── */
.chat-page {
  background: var(--chat-bg);
  border: 0.5px solid var(--sep);
  border-radius: var(--r-xl);
  overflow: hidden;
  box-shadow: 0 4px 32px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.06);
}
.chat-header {
  background: var(--chat-header-bg);
  border-bottom: 0.5px solid var(--sep);
  padding: 1rem 1.5rem;
  display: flex; align-items: center; gap: 0.9rem;
  backdrop-filter: blur(20px);
}
.chat-avatar {
  width: 40px; height: 40px; border-radius: 50%;
  background: linear-gradient(135deg, var(--sys-green) 0%, var(--accent) 100%);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.88rem; font-weight: 700; color: #fff;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(0,113,227,0.28);
}
.chat-name {
  font-size: 0.95rem; font-weight: 600;
  color: var(--text-1); letter-spacing: -0.01em;
  margin: 0; line-height: 1.2;
}
.chat-status {
  font-size: 0.7rem; color: var(--sys-green);
  margin: 0; display: flex; align-items: center; gap: 4px;
}
.chat-status::before {
  content: ""; display: inline-block;
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--sys-green);
}

/* Suggestion chips */
.chips-row {
  display: flex; gap: 7px; flex-wrap: nowrap; overflow-x: auto;
  padding: 0.75rem 1.25rem 0.65rem;
  border-bottom: 0.5px solid var(--sep);
  scrollbar-width: none; background: var(--chat-bg);
}
.chips-row::-webkit-scrollbar { display: none; }
.chip {
  background: var(--chat-chip-bg); color: var(--chat-chip-color);
  border: 0.5px solid var(--sep);
  border-radius: 100px;
  padding: 0.28rem 0.8rem;
  font-size: 0.75rem; font-weight: 500;
  white-space: nowrap; flex-shrink: 0;
}

/* Scrollable message area — column-reverse so newest message is always visible */
.messages-scroll {
  height: 460px;
  overflow-y: auto;
  display: flex;
  flex-direction: column-reverse;
  padding: 1.1rem 1.25rem;
  background: var(--chat-bg);
  scrollbar-width: thin;
  scrollbar-color: var(--sep) transparent;
}
.messages-inner {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

/* Bubble rows */
.bubble-row { display: flex; margin-bottom: 0.25rem; }
.user-row  { justify-content: flex-end; }
.bot-row   { justify-content: flex-start; align-items: flex-end; gap: 0.45rem; }
.bot-mini-avatar {
  width: 24px; height: 24px; border-radius: 50%;
  background: linear-gradient(135deg, var(--sys-green), var(--accent));
  display: flex; align-items: center; justify-content: center;
  font-size: 0.52rem; font-weight: 700; color: #fff;
  flex-shrink: 0; margin-bottom: 2px;
}
.bubble {
  max-width: 70%;
  padding: 0.58rem 0.9rem;
  font-size: 0.85rem; line-height: 1.55;
  white-space: pre-wrap; word-break: break-word;
}
.bubble-u {
  background: var(--bubble-user-bg);
  color: var(--bubble-user-text) !important;
  border-radius: 18px 18px 5px 18px;
}
.bubble-b {
  background: var(--bubble-bot-bg);
  color: var(--bubble-bot-text) !important;
  border-radius: 18px 18px 18px 5px;
}
.chat-empty {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; height: 100%; gap: 0.6rem; text-align: center;
  color: var(--text-3); font-size: 0.8rem; line-height: 1.65;
  padding: 2rem 1rem;
}

/* Input bar */
.chat-input-bar {
  background: var(--input-bar-bg);
  border-top: 0.5px solid var(--sep);
  padding: 0.65rem 1.1rem 0.75rem;
}
.chat-input-bar .stTextInput input {
  border-radius: 20px !important;
  padding: 0.5rem 1rem !important;
  font-size: 0.88rem !important;
  background: var(--bg-card) !important;
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


# ── Simulated chatbot — no API key required ──────────────────────────────────
# All answers are computed directly from the portfolio numbers stored in
# st.session_state["chat_data"]. The engine matches keywords in the user
# question and returns a precise, numerically-grounded response.

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
    "What would happen if I removed the ESG screen?",
    "Which asset has the best individual Sharpe ratio?",
    "Why is the tangency portfolio important?",
]


def _portfolio_answer(question: str, d: dict) -> str:
    """
    Generate a precise, data-driven answer from portfolio data dict d.
    d keys: names, mu, vols, esg_scores, w_opt, ep, sp, sr, esg_bar,
            gamma, lam, rf, ep_tan_all, sp_tan_all, sr_tan_all,
            ep_tan_esg, sp_tan_esg, sr_tan_esg, active_mask, esg_thresh,
            cov (np array), n
    """
    import numpy as np
    q = question.lower()

    names      = d["names"]
    mu         = d["mu"]
    vols       = d["vols"]
    esg_scores = d["esg_scores"]
    w_opt      = d["w_opt"]
    ep         = d["ep"]
    sp         = d["sp"]
    sr         = d["sr"]
    esg_bar    = d["esg_bar"]
    gamma      = d["gamma"]
    lam        = d["lam"]
    rf         = d["rf"]
    ep_tan_all = d["ep_tan_all"]
    sp_tan_all = d["sp_tan_all"]
    sr_tan_all = d["sr_tan_all"]
    ep_tan_esg = d["ep_tan_esg"]
    sp_tan_esg = d["sp_tan_esg"]
    sr_tan_esg = d["sr_tan_esg"]
    active_mask= d["active_mask"]
    esg_thresh = d["esg_thresh"]
    cov        = d["cov"]
    n          = d["n"]

    # Individual Sharpe ratios
    ind_sr = [(mu[i] - rf) / vols[i] for i in range(n)]
    # Sort assets by weight descending
    sorted_by_w = sorted(range(n), key=lambda i: w_opt[i], reverse=True)
    # Sort by ESG ascending (worst first)
    sorted_by_esg = sorted(range(n), key=lambda i: esg_scores[i])
    # Sort by Sharpe descending
    sorted_by_sr = sorted(range(n), key=lambda i: ind_sr[i], reverse=True)

    u_val = ep - gamma/2 * sp**2 + lam * esg_bar
    sharpe_cost = sr_tan_all - sr_tan_esg
    ret_cost    = ep_tan_all - ep_tan_esg
    esg_gain    = esg_bar  # portfolio ESG vs unconstrained

    # ── Utility function ──────────────────────────────────────────────────────
    if any(k in q for k in ["utility", "objective", "maximis", "optimi", "formula", "u ="]):
        lines = [
            "The model maximises investor utility defined as:",
            "",
            "    U = E[Rp] - (γ/2) × σ²p + λ × ESG_bar",
            "",
            f"With your current parameters (γ={gamma}, λ={lam}, rf={rf*100:.1f}%):",
            f"  • E[Rp] = {ep*100:.2f}% — rewards higher expected return",
            f"  • -(γ/2)σ² = -{gamma/2:.2f} × {sp**2*100:.4f} — penalises variance; "
            f"γ={gamma} means moderate{'ly high' if gamma>5 else 'ly low' if gamma<2 else ''} risk aversion",
            f"  • λ × ESG_bar = {lam} × {esg_bar:.3f} = {lam*esg_bar:.4f} — rewards ESG quality",
            f"  • Total utility U = {u_val:.5f}",
            "",
            "The three terms are in direct tension: chasing higher return increases risk,"
            " and imposing ESG constraints limits the feasible set. The weights you see are"
            " the exact combination that resolves this tradeoff optimally given your γ and λ.",
        ]
        return "\n".join(lines)

    # ── Weights explanation ───────────────────────────────────────────────────
    if any(k in q for k in ["weight", "allocation", "holding", "position", "why does my portfolio"]):
        lines = [
            "Portfolio weights are determined by maximising U = E[Rp] - (γ/2)σ² + λ×ESG_bar.",
            "",
            "Here is why each asset received its weight:",
            "",
        ]
        for i in sorted_by_w:
            w = w_opt[i]
            if w < 0.001:
                reason = (f"excluded — its return/risk profile adds no marginal utility "
                          f"(E[R]={mu[i]*100:.1f}%, σ={vols[i]*100:.1f}%, "
                          f"ESG={esg_scores[i]:.1f}/10, individual Sharpe={ind_sr[i]:.3f})")
            else:
                drivers = []
                if ind_sr[i] == max(ind_sr): drivers.append("highest individual Sharpe ratio")
                if esg_scores[i] == max(esg_scores): drivers.append("highest ESG score")
                if vols[i] == min(vols): drivers.append("lowest volatility")
                if mu[i] == max(mu): drivers.append("highest expected return")
                reason_str = (", ".join(drivers) + " — "
                              if drivers else "balance of return, risk and ESG — ")
                reason = (f"{reason_str}E[R]={mu[i]*100:.1f}%, σ={vols[i]*100:.1f}%, "
                          f"ESG={esg_scores[i]:.1f}/10, Sharpe={ind_sr[i]:.3f}")
            lines.append(f"  {names[i]} ({w*100:.1f}%): {reason}")
        lines += [
            "",
            f"Risk aversion γ={gamma} {'heavily ' if gamma>6 else ''}penalises variance, "
            f"so {'low-volatility' if gamma>4 else 'balanced'} assets receive higher weights.",
            f"ESG preference λ={lam} {'strongly ' if lam>3 else ''}tilts allocations toward "
            f"higher-ESG assets; portfolio ESG = {esg_bar:.2f}/10.",
        ]
        return "\n".join(lines)

    # ── Sharpe ratio ──────────────────────────────────────────────────────────
    if any(k in q for k in ["sharpe", "risk-adjusted", "risk adjusted"]):
        best_i = sorted_by_sr[0]
        lines = [
            "The Sharpe ratio measures return earned per unit of risk above the risk-free rate:",
            "",
            "    SR = (E[Rp] - rf) / σp",
            "",
            f"Your optimal portfolio: SR = ({ep*100:.2f}% - {rf*100:.1f}%) / {sp*100:.2f}% = {sr:.3f}",
            "",
            f"Benchmark comparisons:",
            f"  • Unconstrained tangency portfolio:   SR = {sr_tan_all:.3f}",
            f"  • ESG-constrained tangency portfolio: SR = {sr_tan_esg:.3f}",
            f"  • Your ESG-optimal portfolio:         SR = {sr:.3f}",
            "",
            "Individual asset Sharpe ratios:",
        ]
        for i in sorted_by_sr:
            lines.append(f"  {names[i]}: ({mu[i]*100:.1f}% - {rf*100:.1f}%) / {vols[i]*100:.1f}% = {ind_sr[i]:.3f}")
        lines += [
            "",
            f"Note: your portfolio SR ({sr:.3f}) {'exceeds' if sr > ind_sr[sorted_by_sr[0]] else 'is below'} "
            f"the best individual asset SR ({ind_sr[sorted_by_sr[0]]:.3f}) because diversification "
            f"{'reduces portfolio variance below any single asset.' if sr > ind_sr[sorted_by_sr[0]] else 'is constrained by ESG and risk-aversion requirements.'}",
        ]
        return "\n".join(lines)

    # ── ESG cost / constraint cost ────────────────────────────────────────────
    if any(k in q for k in ["cost", "constraint", "penalty", "sacrifice", "tradeoff", "trade-off",
                              "price of esg", "esg screen", "what is the cost"]):
        lines = [
            "The ESG constraint restricts the feasible investment set, which has a measurable cost:",
            "",
            "    Sharpe ratio cost of ESG constraint:",
            f"      Unconstrained tangency SR:   {sr_tan_all:.4f}",
            f"      ESG-constrained tangency SR: {sr_tan_esg:.4f}",
            f"      Cost:                        {sharpe_cost:.4f} ({sharpe_cost/sr_tan_all*100:.1f}% reduction)",
            "",
            "    Return cost at the tangency level:",
            f"      Unconstrained E[R]:   {ep_tan_all*100:.2f}%",
            f"      ESG-constrained E[R]: {ep_tan_esg*100:.2f}%",
            f"      Cost:                 {ret_cost*100:.2f}% per year",
            "",
        ]
        if esg_thresh > 0:
            excluded = [names[i] for i in range(n) if not active_mask[i]]
            if excluded:
                lines += [
                    f"    Assets excluded by ESG screen (min score {esg_thresh:.1f}):",
                    f"      {', '.join(excluded)}",
                    "",
                ]
        lines += [
            f"    ESG benefit gained:",
            f"      Portfolio ESG score: {esg_bar:.2f}/10",
            f"      This is the social/environmental premium the investor accepts lower",
            f"      financial return to achieve — the ESG-SR frontier quantifies exactly",
            f"      how much Sharpe ratio is given up at each ESG level.",
            "",
            f"With λ={lam}, the utility model values each ESG point at λ×(1/10)={lam/10:.3f} "
            f"utility units, making the tradeoff {'worthwhile' if lam > 1 else 'marginal'} "
            f"at the current ESG level.",
        ]
        return "\n".join(lines)

    # ── ESG preference lambda ─────────────────────────────────────────────────
    if any(k in q for k in ["lambda", "λ", "esg preference", "what does the esg", "esg weight",
                              "esg parameter"]):
        lines = [
            f"λ (lambda) = {lam} is the ESG preference parameter in the utility function:",
            "",
            "    U = E[Rp] - (γ/2)σ² + λ × ESG_bar",
            "",
            f"It scales the ESG term relative to financial return. Concretely:",
            f"  • λ=0: ESG is completely ignored — pure mean-variance optimisation",
            f"  • λ={lam}: each 1-point increase in portfolio ESG score (0–10 scale) "
            f"adds {lam:.2f} to utility, equivalent to ~{lam/10*100:.1f}bps of extra return",
            f"  • λ=5: maximum setting — strong ESG tilt, significant return sacrifice",
            "",
            f"With your λ={lam}:",
            f"  • ESG contribution to utility: {lam} × {esg_bar:.3f} = {lam*esg_bar:.4f}",
            f"  • Financial contribution: E[Rp] - (γ/2)σ² = {ep - gamma/2*sp**2:.4f}",
            f"  • Total utility: {u_val:.4f}",
            "",
            "Increasing λ would shift weights toward higher-ESG assets, reducing Sharpe "
            "but improving the sustainability profile. The sensitivity analysis in the "
            "expander below the charts shows exactly how each metric changes with λ.",
        ]
        return "\n".join(lines)

    # ── Risk aversion gamma ───────────────────────────────────────────────────
    if any(k in q for k in ["gamma", "γ", "risk aversion", "risk-aversion", "aversion",
                              "how does increasing risk"]):
        lines = [
            f"γ (gamma) = {gamma} is the risk aversion coefficient in the utility function:",
            "",
            "    U = E[Rp] - (γ/2) × σ²p + λ × ESG_bar",
            "",
            f"The term -(γ/2)σ² penalises portfolio variance. With γ={gamma}:",
            f"  • Variance penalty = -{gamma/2:.2f} × {sp**2:.6f} = {-gamma/2*sp**2:.5f}",
            f"  • This represents {abs(gamma/2*sp**2)/ep*100:.1f}% of your expected return, lost to risk",
            "",
            "Effect of changing γ on optimal weights:",
            f"  • γ < {gamma}: less risk averse → higher allocation to high-return/high-risk assets",
            f"  • γ > {gamma}: more risk averse → higher allocation to low-volatility assets",
            "",
            "Highest-volatility assets in your universe:",
        ]
        sorted_vol = sorted(range(n), key=lambda i: vols[i], reverse=True)
        for i in sorted_vol[:3]:
            lines.append(f"  {names[i]}: σ={vols[i]*100:.1f}%, weight={w_opt[i]*100:.1f}%")
        lines += [
            "",
            f"Lowest-volatility assets (beneficiaries of high γ):",
        ]
        for i in sorted(range(n), key=lambda i: vols[i])[:3]:
            lines.append(f"  {names[i]}: σ={vols[i]*100:.1f}%, weight={w_opt[i]*100:.1f}%")
        return "\n".join(lines)

    # ── ESG drag ──────────────────────────────────────────────────────────────
    if any(k in q for k in ["drags", "drag", "worst esg", "lowest esg", "bad esg",
                              "which asset", "esg score the most"]):
        worst_i = sorted_by_esg[0]
        weighted_esg = [(names[i], esg_scores[i], w_opt[i], esg_scores[i]*w_opt[i])
                        for i in range(n) if w_opt[i] > 0.001]
        weighted_esg.sort(key=lambda x: x[3])
        lines = [
            "Assets ranked by ESG score (lowest first):",
            "",
        ]
        for i in sorted_by_esg:
            flag = " [excluded from ESG frontier]" if not active_mask[i] else ""
            lines.append(f"  {names[i]}: ESG={esg_scores[i]:.2f}/10, weight={w_opt[i]*100:.1f}%{flag}")
        lines += [
            "",
            f"Biggest drag on portfolio ESG score (weighted contribution):",
        ]
        for name, esg, w, contrib in weighted_esg:
            lines.append(f"  {name}: {esg:.2f}/10 × {w*100:.1f}% weight = {contrib:.4f} contribution")
        lines += [
            "",
            f"Portfolio weighted ESG = {esg_bar:.3f}/10",
            f"If {sorted_by_esg[0]} (ESG={esg_scores[sorted_by_esg[0]]:.2f}) were replaced "
            f"with the highest-ESG asset ({names[sorted_by_esg[-1]]}, ESG={esg_scores[sorted_by_esg[-1]]:.2f}), "
            f"portfolio ESG would increase significantly.",
        ]
        return "\n".join(lines)

    # ── CML / Capital Market Line ─────────────────────────────────────────────
    if any(k in q for k in ["capital market line", "cml", "market line"]):
        lines = [
            "The Capital Market Line (CML) connects the risk-free asset to the tangency portfolio.",
            "Any point on the CML represents a mix of the risk-free asset and the tangency portfolio.",
            "",
            "    E[R]_CML = rf + (E[Rt] - rf) / σt × σ",
            "",
            f"CML using all assets (blue dashed line):",
            f"  rf = {rf*100:.2f}%, Tangency: E[R]={ep_tan_all*100:.2f}%, σ={sp_tan_all*100:.2f}%",
            f"  Slope (Sharpe of tangency) = {sr_tan_all:.4f}",
            "",
            f"CML using ESG-screened assets (green dashed line):",
            f"  rf = {rf*100:.2f}%, Tangency: E[R]={ep_tan_esg*100:.2f}%, σ={sp_tan_esg*100:.2f}%",
            f"  Slope (Sharpe of tangency) = {sr_tan_esg:.4f}",
            "",
            f"The green CML has a {'lower' if sr_tan_esg < sr_tan_all else 'similar'} slope "
            f"({sr_tan_esg:.4f} vs {sr_tan_all:.4f}), reflecting the cost of the ESG constraint "
            f"— fewer assets means a less efficient tangency portfolio.",
            "",
            f"Your optimal portfolio (E[R]={ep*100:.2f}%, σ={sp*100:.2f}%) sits "
            f"{'on' if abs(sr - sr_tan_esg) < 0.01 else 'below'} the green CML because "
            f"the ESG preference λ={lam} shifts the optimal point away from pure Sharpe maximisation.",
        ]
        return "\n".join(lines)

    # ── Green frontier right of blue ──────────────────────────────────────────
    if any(k in q for k in ["green frontier", "right of blue", "right of the blue",
                              "frontier sit", "why does the green", "two frontier"]):
        lines = [
            "The green frontier lies to the RIGHT of the blue frontier.",
            "This is a fundamental result of constrained optimisation:",
            "",
            "  Blue frontier = mean-variance frontier using ALL assets",
            "    → the largest feasible set → lowest possible risk at each return level",
            "",
            "  Green frontier = mean-variance frontier using only ESG-screened assets",
            f"    → restricted to assets with ESG ≥ {esg_thresh:.1f}/10",
            "    → smaller feasible set → higher risk for the same expected return",
            "",
            "Mathematically: adding constraints can only reduce or maintain portfolio efficiency,"
            " never improve it. The distance between the curves is the ESG constraint cost.",
            "",
        ]
        if esg_thresh > 0:
            excluded = [names[i] for i in range(n) if not active_mask[i]]
            if excluded:
                lines.append(f"Assets excluded by your ESG screen (ESG < {esg_thresh:.1f}):")
                for name in excluded:
                    i = names.index(name)
                    lines.append(f"  {name}: ESG={esg_scores[i]:.2f}/10, E[R]={mu[i]*100:.1f}%, σ={vols[i]*100:.1f}%")
                lines.append("")
        lines += [
            f"The Sharpe ratio cost of this restriction:",
            f"  Blue tangency SR:  {sr_tan_all:.4f}",
            f"  Green tangency SR: {sr_tan_esg:.4f}",
            f"  Reduction:         {sharpe_cost:.4f} ({sharpe_cost/max(sr_tan_all,0.001)*100:.1f}%)",
        ]
        return "\n".join(lines)

    # ── Best individual Sharpe ────────────────────────────────────────────────
    if any(k in q for k in ["best individual", "best asset", "individual sharpe",
                              "which asset has the best"]):
        best_i = sorted_by_sr[0]
        lines = [
            "Individual Sharpe ratios (best to worst):",
            "",
        ]
        for i in sorted_by_sr:
            lines.append(
                f"  {names[i]}: E[R]={mu[i]*100:.1f}%, σ={vols[i]*100:.1f}%, "
                f"ESG={esg_scores[i]:.1f}/10 → SR = {ind_sr[i]:.3f}"
            )
        lines += [
            "",
            f"Best: {names[best_i]} with SR={ind_sr[best_i]:.3f}",
            f"Portfolio SR = {sr:.3f} — {'higher than any individual asset due to diversification benefits.'if sr > max(ind_sr) else 'lower than the best individual asset, constrained by ESG and risk-aversion requirements.'}",
        ]
        return "\n".join(lines)

    # ── Tangency portfolio ────────────────────────────────────────────────────
    if any(k in q for k in ["tangency", "tangent", "market portfolio", "why is the tangency"]):
        lines = [
            "The tangency portfolio is the portfolio with the highest Sharpe ratio.",
            "It is found by maximising SR = (E[Rp] - rf) / σp subject to weights summing to 1.",
            "It sits at the point where the Capital Market Line touches the efficient frontier.",
            "",
            f"Tangency portfolio (all assets, unconstrained):",
            f"  E[R] = {ep_tan_all*100:.2f}%",
            f"  σ    = {sp_tan_all*100:.2f}%",
            f"  SR   = {sr_tan_all:.4f}",
            "",
            f"Tangency portfolio (ESG-screened assets only):",
            f"  E[R] = {ep_tan_esg*100:.2f}%",
            f"  σ    = {sp_tan_esg*100:.2f}%",
            f"  SR   = {sr_tan_esg:.4f}",
            "",
            f"Your ESG-optimal portfolio has SR = {sr:.4f}. It differs from the ESG tangency "
            f"because λ={lam} introduces an ESG term into the objective, shifting weights "
            f"toward higher-ESG assets even when that reduces Sharpe.",
            "",
            "In classic mean-variance theory a rational investor holds a mix of the risk-free "
            "asset and the tangency portfolio. Here, the ESG utility term means the investor "
            "accepts a lower Sharpe in exchange for a higher ESG score.",
        ]
        return "\n".join(lines)

    # ── Remove ESG screen ─────────────────────────────────────────────────────
    if any(k in q for k in ["remove", "without esg", "no esg", "ignore esg", "esg screen off"]):
        lines = [
            "If the ESG screen were removed (λ=0, no minimum ESG threshold):",
            "",
            f"  Unconstrained tangency portfolio achieves SR = {sr_tan_all:.4f}",
            f"  vs current ESG-constrained tangency SR = {sr_tan_esg:.4f}",
            f"  Gain from removing ESG: +{sharpe_cost:.4f} Sharpe ratio points",
            "",
            f"  Return: {ep_tan_all*100:.2f}% vs {ep_tan_esg*100:.2f}% (gain of {ret_cost*100:.2f}%)",
            "",
            "However, removing ESG would:",
        ]
        excluded = [names[i] for i in range(n) if not active_mask[i]]
        if excluded:
            lines.append(f"  • Allow excluded assets back in: {', '.join(excluded)}")
        lines += [
            f"  • Reduce portfolio ESG score from {esg_bar:.2f}/10",
            f"  • Set λ contribution to zero (currently λ×ESG = {lam*esg_bar:.4f} utility units)",
            "",
            "The ESG-SR frontier chart shows the full Sharpe/ESG tradeoff curve — "
            "you can read off exactly how much Sharpe is sacrificed at any ESG level.",
        ]
        return "\n".join(lines)

    # ── Default / fallback ────────────────────────────────────────────────────
    lines = [
        f"Your portfolio summary:",
        f"  Assets: {', '.join(f'{names[i]} ({w_opt[i]*100:.1f}%)' for i in range(n) if w_opt[i]>0.001)}",
        f"  Expected return: {ep*100:.2f}%  |  Volatility: {sp*100:.2f}%  |  Sharpe: {sr:.3f}",
        f"  ESG score: {esg_bar:.2f}/10  |  Utility: {u_val:.4f}",
        f"  Parameters: γ={gamma}, λ={lam}, rf={rf*100:.1f}%",
        "",
        "Try one of the suggested questions above, or ask about:",
        "weights, Sharpe ratio, ESG scores, the utility function, the frontier,"
        " the Capital Market Line, risk aversion, or the cost of the ESG constraint.",
    ]
    return "\n".join(lines)


def answer_question(question: str) -> str:
    """Route a user question to the simulated chatbot engine."""
    d = st.session_state.get("chat_data")
    if d is None:
        return "Please run the portfolio optimiser first by clicking Optimise Portfolio."
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

# Initialise theme state early so toggle works on first render
if "dark" not in st.session_state:
    st.session_state["dark"] = True

with st.sidebar:
    _is_dark = st.session_state["dark"]
    _sep_col  = "#38383A" if _is_dark else "rgba(0,0,0,0.10)"
    _t1_col   = "rgba(255,255,255,1)" if _is_dark else "#1d1d1f"
    _t3_col   = "rgba(235,235,245,0.30)" if _is_dark else "#86868b"
    st.markdown(f"""
<div style="padding:0.4rem 0 1rem; border-bottom:0.5px solid {_sep_col}; margin-bottom:0.6rem;">
  <div style="display:flex; align-items:center; gap:0.6rem;">
    <div style="width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg,#30D158 0%,#0A84FF 100%); display:flex; align-items:center; justify-content:center; flex-shrink:0;">
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M9 2C6.2 2 4 4.5 4 7.5C4 10.3 5.9 12.6 8.5 13V16H9.5V13C12.1 12.6 14 10.3 14 7.5C14 4.5 11.8 2 9 2Z" fill="white"/></svg>
    </div>
    <div>
      <div style="font-size:1.0rem; font-weight:700; color:{_t1_col}; letter-spacing:-0.02em; line-height:1.1;">GreenPort</div>
      <div style="font-size:0.64rem; color:{_t3_col}; letter-spacing:0.05em; text-transform:uppercase; font-weight:500;">ESG Portfolio Optimiser</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    # ── Theme toggle ──────────────────────────────────────────────────────────
    _toggle_label = "☀️  Day mode" if _is_dark else "🌙  Night mode"
    if st.button(_toggle_label, key="theme_toggle", use_container_width=True):
        st.session_state["dark"] = not st.session_state["dark"]
        st.rerun()
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
    st.markdown(f"""
<div style="margin-top:1.5rem; padding-top:0.85rem; border-top:0.5px solid {_sep_col};">
  <div style="font-size:0.65rem; color:{_t3_col}; line-height:1.8; letter-spacing:0.03em;">
    ECN316 · Sustainable Finance · 2026
  </div>
</div>
""", unsafe_allow_html=True)




# Inject the [data-theme] attribute so CSS variables switch correctly
_theme_attr = "dark" if st.session_state["dark"] else "light"
st.markdown(f"""
<script>
  (function() {{
    var root = window.parent.document.querySelector('.stApp');
    if (root) root.setAttribute('data-theme', '{_theme_attr}');
    var all = window.parent.document.querySelectorAll('*[data-theme]');
    all.forEach(function(el) {{ el.setAttribute('data-theme', '{_theme_attr}'); }});
    document.documentElement.setAttribute('data-theme', '{_theme_attr}');
  }})();
</script>
""", unsafe_allow_html=True)

# Inline override CSS — simpler & more reliable than JS attribute in Streamlit
if not st.session_state["dark"]:
    st.markdown("""
<style>
:root {
  --bg: #f5f5f7 !important; --bg-card: #ffffff !important;
  --bg-elevated: #ffffff !important; --bg-3: #e8e8ed !important;
  --bg-input: rgba(118,118,128,0.12) !important;
  --text-1: #1d1d1f !important; --text-2: #6e6e73 !important;
  --text-3: #86868b !important; --text-inv: #ffffff !important;
  --accent: #0071e3 !important; --accent-hover: #0077ed !important;
  --accent-light: rgba(0,113,227,0.10) !important;
  --sys-green: #1d8348 !important; --sys-orange: #c84b00 !important;
  --sys-red: #c0392b !important;
  --border: rgba(0,0,0,0.08) !important; --sep: rgba(0,0,0,0.12) !important;
  --sep-strong: rgba(0,0,0,0.18) !important;
  --sb-bg: #f5f5f7 !important; --sb-border: rgba(0,0,0,0.10) !important;
  --chat-bg: #ffffff !important; --chat-header-bg: #f5f5f7 !important;
  --chat-chip-bg: #e8e8ed !important; --chat-chip-color: #0071e3 !important;
  --bubble-user-bg: #0071e3 !important; --bubble-user-text: #ffffff !important;
  --bubble-bot-bg: #e8e8ed !important; --bubble-bot-text: #1d1d1f !important;
  --input-bar-bg: #f5f5f7 !important;
  --chart-bg: #ffffff !important; --chart-bg2: #f5f5f7 !important;
}
.stApp { background: #f5f5f7 !important; }
[data-testid="stSidebar"] { background: #f5f5f7 !important; }
</style>
""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# TABS  — Portfolio Optimiser | Portfolio Explainer
# ══════════════════════════════════════════════════════════════════════════════

_tab1, _tab2 = st.tabs(["Portfolio Optimiser", "Portfolio Explainer"])

with _tab1:


    # ══════════════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════════════

    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.25rem;">
      <div style="width:46px; height:46px; border-radius:11px; background:linear-gradient(135deg,#30D158 0%,#0A84FF 100%); display:flex; align-items:center; justify-content:center; flex-shrink:0; box-shadow: 0 2px 12px rgba(10,132,255,0.3);">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 2C8.1 2 5 5.6 5 10C5 14.1 7.9 17.5 11.7 17.96V21H12.3V17.96C16.1 17.5 19 14.1 19 10C19 5.6 15.9 2 12 2Z" fill="white"/></svg>
      </div>
      <div class="hero-title">GreenPort</div>
    </div>
    <div class="hero-sub">ESG-aware portfolio optimiser · ECN316 Sustainable Finance</div>
    """, unsafe_allow_html=True)

    if _ESG_DB:
        st.markdown(
            f'<div class="info-box">ESG database loaded: <strong>{len(_ESG_DB):,} tickers</strong> '
            f'from LSEG ESGCombinedScore CSV — most recent year per ticker, scaled 0–10.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="error-box">Could not load ESG data. '
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
            (m1, "Expected Return", f"{ep*100:.2f}",  "%",    "metric-pos", "card-ret"),
            (m2, "Volatility",      f"{sp*100:.2f}",  "%",    "",            "card-vol"),
            (m3, "Sharpe Ratio",    f"{sr:.3f}",       "",     "metric-pos" if sr > 0 else "metric-neg", "card-sr"),
            (m4, "ESG Score",       f"{esg_bar:.2f}", "/ 10", "metric-pos" if esg_bar >= 5 else "", "card-esg"),
        ]
        for col, label, val, unit, cls, card_cls in metric_data:
            with col:
                st.markdown(
                    f'<div class="metric-card {card_cls}">'
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
        # CHARTS — iOS / macOS system palette
        # ══════════════════════════════════════════════════════════════════════════
        BG     = '#000000'   # iOS true black
        BG2    = '#1C1C1E'   # iOS system background 2
        BLUE   = '#0A84FF'   # iOS system blue   — "all assets" frontier
        GREEN  = '#30D158'   # iOS system green  — ESG frontier
        ORANGE = '#FF9F0A'   # iOS system orange — optimal portfolio point
        GREY   = '#636366'   # iOS system gray   — labels

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
                            fontsize=7, color=GREY)

            ax.set_xlabel("Std (%)", fontsize=9, color=GREY)
            ax.set_ylabel("Expected Return (%)", fontsize=9, color=GREY)
            ax.set_title("Mean-Variance Frontier", fontsize=11, fontweight='bold',
                         color='#F2F2F7', pad=10)
            ax.set_xlim(left=0)
            ax.tick_params(colors=GREY, labelsize=8)
            for sp_ in ax.spines.values(): sp_.set_color('#2C2C2E')
            ax.legend(fontsize=7, framealpha=0.92, facecolor='#1C1C1E', edgecolor='#3A3A3C',
                      loc='upper left', labelcolor='#8E8E93')
            ax.grid(True, alpha=0.25, color='#2C2C2E', linestyle='--', linewidth=0.7)
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
                             fontsize=7.5, color=GREY)

            # Tangency using ESG info (on or near peak of curve)
            ax2.scatter(esg_tan_using, sr_tan_using, color=GREEN, s=140, zorder=9,
                        edgecolors=BG, lw=1.5)
            ax2.annotate("Tangency portfolio\nusing ESG information",
                         (esg_tan_using, sr_tan_using),
                         textcoords="offset points", xytext=(8, 4),
                         fontsize=7.5, color=GREEN,
                         arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))

            # Tangency ignoring ESG info (below curve — same as lecture)
            ax2.scatter(esg_tan_ignoring, sr_tan_ignoring, color=BLUE, s=100, zorder=8,
                        edgecolors=BG, lw=1.5)
            ax2.annotate("Tangency portfolio\nignoring ESG information",
                         (esg_tan_ignoring, sr_tan_ignoring),
                         textcoords="offset points", xytext=(8, -28),
                         fontsize=7.5, color=BLUE,
                         arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))

            # ESG-optimal portfolio
            ax2.scatter(esg_bar, sr, color=ORANGE, s=120, zorder=10,
                        edgecolors=BG, lw=2)
            ax2.annotate("Optimal portfolio",
                         (esg_bar, sr),
                         textcoords="offset points", xytext=(-80, 8),
                         fontsize=7.5, color=ORANGE,
                         arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))

            ax2.set_xlabel("ESG Score (0–10)", fontsize=9, color=GREY)
            ax2.set_ylabel("Sharpe Ratio",     fontsize=9, color=GREY)
            ax2.set_title("ESG-SR Frontier", fontsize=11, fontweight="bold",
                          color='#F2F2F7', pad=10)
            ax2.tick_params(colors=GREY, labelsize=8)
            for sp_ in ax2.spines.values(): sp_.set_color('#2C2C2E')
            ax2.legend(fontsize=8, framealpha=0.92, facecolor='#1C1C1E', edgecolor='#3A3A3C',
                       labelcolor='#8E8E93')
            ax2.grid(True, alpha=0.25, color='#2C2C2E', linestyle='--', linewidth=0.7)
            fig2.tight_layout(); st.pyplot(fig2); plt.close()

        # ── Allocation charts ─────────────────────────────────────────────────────
        st.markdown("#### Portfolio Allocation")
        pc, bc = st.columns(2)
        nz = [(names[i],w_opt[i],esg_scores[i]) for i in range(n) if w_opt[i]>0.005]
        if nz:
            plabels=[x[0] for x in nz]; pvals=[x[1] for x in nz]; pesg=[x[2] for x in nz]
            greens=['#1E4D2A','#2D7A42','#3D9A56','#52B86A','#6ECC84',
                    '#90DAAA','#B0E8C4','#CAEFD8','#E0F7EB','#F0FCF4']
            with pc:
                f3,a3 = plt.subplots(figsize=(5,4))
                f3.patch.set_facecolor(BG); a3.set_facecolor(BG)
                a3.pie(pvals,labels=plabels,autopct='%1.1f%%',colors=greens[:len(pvals)],
                       startangle=140,textprops={'fontsize':8,'color':'#A8C8A8'},
                       wedgeprops={'edgecolor':BG,'linewidth':2})
                a3.set_title("Weight Allocation",fontsize=11,fontweight='bold',color='#F2F2F7',pad=10)
                f3.tight_layout(); st.pyplot(f3); plt.close()
            with bc:
                f4,a4 = plt.subplots(figsize=(5,4))
                f4.patch.set_facecolor(BG); a4.set_facecolor(BG)
                bcols = [plt.cm.YlGn(s/10) for s in pesg]
                bars  = a4.barh(plabels,[v*100 for v in pvals],color=bcols,edgecolor=BG,height=0.55)
                for bar,ev in zip(bars,pesg):
                    a4.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
                            f'ESG {ev:.1f}',va='center',fontsize=7.5,color=GREY)
                a4.set_xlabel("Weight (%)",fontsize=9,color=GREY)
                a4.set_title("Weights with ESG Scores",fontsize=11,fontweight='bold',color='#F2F2F7',pad=10)
                a4.tick_params(colors=GREY,labelsize=8)
                for sp_ in a4.spines.values(): sp_.set_color('#2C2C2E')
                a4.grid(True,alpha=0.25,color='#2C2C2E',axis='x',linestyle='--',linewidth=0.7)
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
                ax_.set_title(tl_,fontsize=10,color='#F2F2F7')
                ax_.set_xlabel("λ",fontsize=9); ax_.set_ylabel(yl_,fontsize=9)
                ax_.tick_params(colors='#5a7a5a',labelsize=8)
                for sp_ in ax_.spines.values(): sp_.set_color('#2C2C2E')
                ax_.grid(True,alpha=0.3,color='#c8d8b8',linestyle='--')
            axes[2].set_facecolor(BG)
            axes[2].plot(sens_df["λ"],sens_df["E[R](%)"],color='#00d4aa',lw=2,label='E[R]')
            axes[2].plot(sens_df["λ"],sens_df["σ(%)"],color='#f59e0b',lw=2,linestyle='--',label='σ')
            axes[2].set_title("Return & Risk vs λ",fontsize=10,color='#F2F2F7')
            axes[2].set_xlabel("λ",fontsize=9); axes[2].set_ylabel("%",fontsize=9)
            axes[2].legend(fontsize=8,facecolor='#1C1C1E',edgecolor='#3A3A3C')
            axes[2].tick_params(colors='#5a7a5a',labelsize=8)
            for sp_ in axes[2].spines.values(): sp_.set_color('#2C2C2E')
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



with _tab2:
    _dark = st.session_state.get("dark", True)
    _bg_val = "#000" if _dark else "#fff"
    _sep_val = "#3A3A3C" if _dark else "rgba(0,0,0,0.12)"

    if "chat_data" not in st.session_state:
        st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            min-height:460px;text-align:center;gap:1.2rem;
            background:{'#000' if _dark else '#fff'};border-radius:20px;
            border:0.5px solid {'#3A3A3C' if _dark else 'rgba(0,0,0,0.10)'};">
  <div style="width:64px;height:64px;border-radius:16px;
              background:linear-gradient(135deg,{'#30D158' if _dark else '#1d8348'},{'#0A84FF' if _dark else '#0071e3'});
              display:flex;align-items:center;justify-content:center;
              box-shadow:0 4px 20px rgba(0,113,227,0.30);">
    <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
      <path d="M15 3C10 3 6 7.5 6 13C6 18.2 9.7 22.4 14.5 22.9V27H15.5V22.9C20.3 22.4 24 18.2 24 13C24 7.5 20 3 15 3Z" fill="white"/>
    </svg>
  </div>
  <div>
    <div style="font-size:1.1rem;font-weight:600;color:{'rgba(255,255,255,1)' if _dark else '#1d1d1f'};margin-bottom:0.4rem;">
      Portfolio Explainer
    </div>
    <div style="font-size:0.84rem;color:{'rgba(235,235,245,0.40)' if _dark else '#86868b'};max-width:300px;line-height:1.6;">
      Run the optimiser first, then return here to ask questions about your results.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        _dark  = st.session_state.get("dark", True)
        _bg    = "#000000" if _dark else "#ffffff"
        _hbg   = "#1C1C1E" if _dark else "#f5f5f7"
        _sep   = "#3A3A3C" if _dark else "rgba(0,0,0,0.10)"
        _t1    = "rgba(255,255,255,1)" if _dark else "#1d1d1f"
        _t3    = "rgba(235,235,245,0.30)" if _dark else "#86868b"
        _green = "#30D158" if _dark else "#1d8348"
        _blue  = "#0A84FF" if _dark else "#0071e3"

        # ── Build ALL messages as single HTML block ───────────────────────
        if not st.session_state["chat_history"]:
            msgs_html = f"""
<div class="chat-empty">
  Ask about weights, the Sharpe ratio, ESG scores,<br>
  the utility function, or any part of your portfolio.
</div>"""
        else:
            msgs_html = '<div class="messages-inner">'
            for msg in st.session_state["chat_history"]:
                safe = (msg["content"]
                        .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
                if msg["role"] == "user":
                    msgs_html += (f'<div class="bubble-row user-row">'
                                  f'<div class="bubble bubble-u">{safe}</div></div>')
                else:
                    msgs_html += (f'<div class="bubble-row bot-row">'
                                  f'<div class="bot-mini-avatar">GP</div>'
                                  f'<div class="bubble bubble-b">{safe}</div></div>')
            msgs_html += '</div>'

        # ── Render the whole chat page as one block ───────────────────────
        st.markdown(f"""
<div class="chat-page">
  <div class="chat-header">
    <div class="chat-avatar" style="background:linear-gradient(135deg,{_green},{_blue});">GP</div>
    <div style="flex:1;">
      <p class="chat-name" style="color:{_t1};">Portfolio Explainer</p>
      <p class="chat-status" style="color:{_green};">Active now</p>
    </div>
    <div style="font-size:0.7rem;color:{_t3};text-align:right;line-height:1.65;">
      Powered by GreenPort<br>No API key needed
    </div>
  </div>
  <div class="chips-row" id="chips-row" style="background:{_bg};">
    {"".join(f'<span class="chip">{q}</span>' for q in SUGGESTED_QUESTIONS)}
  </div>
  <div class="messages-scroll" id="gp-msgs" style="background:{_bg};">
    {msgs_html}
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Input bar outside the static HTML so Streamlit form works ─────
        st.markdown(f'<div class="chat-input-bar" style="background:{_hbg};border:0.5px solid {_sep};border-top:none;border-radius:0 0 20px 20px;">', unsafe_allow_html=True)
        with st.form(key="imessage_form", clear_on_submit=True):
            _ci, _cs = st.columns([8, 1])
            with _ci:
                user_input = st.text_input("msg", placeholder="Message GreenPort...",
                                           label_visibility="collapsed")
            with _cs:
                submitted = st.form_submit_button("↑", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted and user_input.strip():
            reply = answer_question(user_input.strip())
            st.session_state["chat_history"].append({"role":"user","content":user_input.strip()})
            st.session_state["chat_history"].append({"role":"assistant","content":reply})
            st.rerun()

        # ── Suggestion quick-buttons (actual Streamlit buttons) ───────────
        st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
        st.caption("Suggested questions — click to send:")
        _pc = st.columns(3)
        for _i, _q in enumerate(SUGGESTED_QUESTIONS[:9]):
            with _pc[_i % 3]:
                if st.button(_q, key=f"chip_{_i}", use_container_width=True):
                    _r = answer_question(_q)
                    st.session_state["chat_history"].append({"role":"user","content":_q})
                    st.session_state["chat_history"].append({"role":"assistant","content":_r})
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.get("chat_history"):
            if st.button("Clear conversation", key="chat_clear"):
                st.session_state["chat_history"] = []
                st.rerun()
