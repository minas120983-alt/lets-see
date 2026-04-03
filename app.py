import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
from io import StringIO

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
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ESG DATABASE - Public GitHub Raw URL
# ══════════════════════════════════════════════════════════════════════════════

ESG_CSV_URL = "https://raw.githubusercontent.com/minas120983-alt/lets-see/refs/heads/main/ESG%20data%202026.csv"

@st.cache_data(show_spinner=False)
def load_esg_database(uploaded_file=None) -> dict:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(ESG_CSV_URL)
    
    df.columns = [str(c).strip() for c in df.columns]
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["fieldname"] == "ESGCombinedScore"].copy()
    df["valuescore"] = pd.to_numeric(df["valuescore"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["ticker", "valuescore", "year"])

    # Keep latest year per ticker
    latest = df.sort_values("year").groupby("ticker").last().reset_index()

    esg_db = {}
    for _, row in latest.iterrows():
        t = row["ticker"]
        esg_db[t] = {
            "app_esg": round(float(row["valuescore"]) * 10, 3),
            "letter": str(row.get("value", "N/A")),
            "year": int(row["year"]),
            "source": f"LSEG ESGCombinedScore ({int(row['year'])})",
            "has_esg": True,
        }
    return esg_db

# Load ESG data
with st.sidebar:
    st.markdown("## GreenPort")
    st.markdown("---")
    st.markdown("### ESG Data Source")
    uploaded_esg_file = st.file_uploader("Upload your own ESG CSV (optional)", type="csv")

    if uploaded_esg_file:
        _ESG_DB = load_esg_database(uploaded_esg_file)
        st.success(f"✅ Custom ESG file loaded — {len(_ESG_DB):,} tickers")
    else:
        _ESG_DB = load_esg_database()
        st.info(f"📊 Using public ESG database — **{len(_ESG_DB):,} tickers** (2026 data)")

def lookup_esg(ticker: str) -> dict:
    t = ticker.upper().strip()
    if t in _ESG_DB:
        return {"ticker": t, **_ESG_DB[t], "error": None}
    return {"ticker": t, "has_esg": False, "error": f"'{t}' not found in ESG database."}

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO MATH FUNCTIONS (unchanged from your original)
# ══════════════════════════════════════════════════════════════════════════════
def port_ret(w, mu):
    return float(np.asarray(w) @ np.asarray(mu))

def port_var(w, cov):
    return float(np.asarray(w) @ np.asarray(cov) @ np.asarray(w))

def port_sd(w, cov):
    return float(max(port_var(w, cov), 1e-14) ** 0.5)

def port_sr(w, mu, cov, rf):
    ep = port_ret(w, mu)
    sp = port_sd(w, cov)
    return (ep - rf) / sp if sp > 1e-9 else 0.

def port_stats(w, mu, cov, esg, rf):
    w = np.asarray(w)
    ep = port_ret(w, mu)
    sp = port_sd(w, cov)
    return ep, sp, (ep - rf) / sp if sp > 1e-9 else 0., float(w @ esg)

def find_tangency(mu, cov, rf, bounds=None):
    n = len(mu)
    b = bounds or [(0., 1.)] * n
    res = minimize(
        lambda w: -port_sr(w, mu, cov, rf),
        np.ones(n) / n,
        method="SLSQP",
        bounds=b,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"ftol": 1e-10, "maxiter": 800}
    )
    wt = res.x if res.success else np.ones(n) / n
    return wt, port_ret(wt, mu), port_sd(wt, cov), port_sr(wt, mu, cov, rf)

def find_optimal(mu, cov, esg, rf, gamma, lam):
    n = len(mu)
    res = minimize(
        lambda w: -(port_ret(w, mu) - gamma / 2 * port_var(w, cov) + lam * float(np.asarray(w) @ esg)),
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0., 1.)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"ftol": 1e-10, "maxiter": 1000}
    )
    return res.x if res.success else np.ones(n) / n

def build_mv_frontier(mu, cov, bounds=None, n_points=100):
    n = len(mu)
    b = bounds or [(0., 1.)] * n
    w_mv = minimize(lambda w: port_sd(w, cov), np.ones(n)/n, method="SLSQP",
                    bounds=b, constraints=[{"type": "eq", "fun": lambda w: np.sum(w)-1}]).x
    ret_min = port_ret(w_mv, mu)
    ret_max = max([port_ret(np.eye(n)[i], mu) for i in range(n) if b[i][1] > 0])
    targets = np.linspace(ret_min, ret_max, n_points)
    stds, rets = [], []
    for rt in targets:
        c = {"type": "eq", "fun": lambda w, r=rt: port_ret(w, mu) - r}
        res = minimize(lambda w: port_sd(w, cov), np.ones(n)/n, method="SLSQP",
                       bounds=b, constraints=[{"type": "eq", "fun": lambda w: np.sum(w)-1}, c])
        if res.success:
            stds.append(port_sd(res.x, cov) * 100)
            rets.append(port_ret(res.x, mu) * 100)
    return np.array(stds), np.array(rets)

# ══════════════════════════════════════════════════════════════════════════════
# MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def fetch_market_data(tickers, period="3y"):
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by="ticker")
    # ... (your original fetch_market_data function - keep it as is)
    # For brevity, I'm keeping a placeholder. Paste your full original function here if needed.
    # I'll assume you have it working.
    pass  # Replace with your full fetch_market_data function

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR INPUTS + MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
# Paste the rest of your original code here (Investor Preferences, Asset Universe, etc.)

# For now, here's the ESG Explorer section (add this after the header):

st.markdown('<div class="section-header">ESG Database Explorer</div>', unsafe_allow_html=True)

search = st.text_input("Search by ticker", placeholder="e.g. AAPL, TSLA...")

esg_list = []
for ticker, data in _ESG_DB.items():
    esg_list.append({
        "Ticker": ticker,
        "ESG Score": data["app_esg"],
        "Letter Grade": data["letter"],
        "Year": data["year"],
        "Source": data["source"]
    })

esg_df = pd.DataFrame(esg_list)

if search:
    esg_df = esg_df[esg_df["Ticker"].str.contains(search, case=False)]

esg_df = esg_df.sort_values("ESG Score", ascending=False)

st.dataframe(
    esg_df.style.format({"ESG Score": "{:.2f}"}),
    use_container_width=True,
    hide_index=True
)

st.caption(f"Showing {len(esg_df)} of {len(_ESG_DB)} companies • LSEG ESG 2026")

st.download_button(
    "📥 Download full ESG database",
    esg_df.to_csv(index=False).encode(),
    "ESG_data_2026.csv",
    "text/csv"
)

# Now continue with your original "INPUT MODE", "RUN" button, charts, etc.
