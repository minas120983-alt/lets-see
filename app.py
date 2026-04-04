import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenPort · ESG Portfolio Optimiser",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* ── Design tokens (OKLCH — perceptually uniform) ── */
:root {
  /* Surfaces — warm sage-tinted, never pure white or grey */
  --bg:           oklch(96.5% 0.009 140);
  --bg-card:      oklch(98.5% 0.005 138);
  --bg-card-alt:  oklch(95.5% 0.013 138);
  --bg-hover:     oklch(93.5% 0.014 140);
  --bg-inset:     oklch(92.0% 0.018 136);

  /* Text hierarchy */
  --text-1:  oklch(21% 0.022 145);
  --text-2:  oklch(38% 0.022 142);
  --text-3:  oklch(55% 0.016 140);
  --text-4:  oklch(68% 0.010 138);

  /* Accent — forest green (ESG-authentic, not AI-blue) */
  --accent:        oklch(44% 0.155 145);
  --accent-mid:    oklch(52% 0.145 143);
  --accent-light:  oklch(62% 0.120 142);
  --accent-wash:   oklch(89% 0.032 140);

  /* Amber — warm secondary (matches ESG warmth) */
  --amber:         oklch(68% 0.160 64);
  --amber-wash:    oklch(95% 0.028 80);

  /* Borders */
  --border:        oklch(88% 0.014 140);
  --border-strong: oklch(78% 0.020 140);

  /* Sidebar */
  --sb-bg:         oklch(17.5% 0.024 148);
  --sb-bg-hover:   oklch(22% 0.026 146);
  --sb-border:     oklch(24% 0.028 148);
  --sb-text:       oklch(82% 0.018 142);
  --sb-muted:      oklch(62% 0.016 140);
  --sb-label:      oklch(70% 0.022 140);

  /* Easing — ease-out-expo (never bounce or elastic) */
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-quick: cubic-bezier(0.22, 1, 0.36, 1);

  --radius-sm: 8px;
  --radius:    12px;
  --radius-lg: 16px;
}

/* ── Base typography ── */
html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
  color: var(--text-1);
  -webkit-font-smoothing: antialiased;
}

/* ── App background ── */
.stApp {
  background: var(--bg);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--sb-bg) !important;
  border-right: 1px solid var(--sb-border);
}
[data-testid="stSidebar"] * {
  color: var(--sb-text) !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-family: 'Fraunces', serif !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em;
}
[data-testid="stSidebar"] label {
  color: var(--sb-label) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.09em !important;
  text-transform: uppercase !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--sb-border) !important;
  opacity: 0.6;
}
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div {
  background: var(--sb-border);
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
  background: var(--accent-light) !important;
  border: 2px solid oklch(72% 0.130 142) !important;
}

/* ── Main container ── */
.block-container {
  padding-top: 2.5rem;
  max-width: 1200px;
}

/* ── Hero header ── */
.hero-title {
  font-family: 'Fraunces', serif;
  font-size: clamp(2.4rem, 5vw, 3.6rem);
  font-weight: 600;
  color: var(--text-1);
  line-height: 1.05;
  letter-spacing: -0.025em;
  margin-bottom: 0.35rem;
}
.hero-sub {
  font-size: 1.0rem;
  font-weight: 300;
  color: var(--text-3);
  margin-bottom: 2.2rem;
  letter-spacing: 0.01em;
}

/* ── Metric cards — each visually distinct ── */
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.4rem 1.1rem;
  margin-bottom: 0.75rem;
  position: relative;
  overflow: hidden;
  transition: border-color 0.22s var(--ease), transform 0.22s var(--ease);
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--accent);
  border-radius: var(--radius) var(--radius) 0 0;
}
.metric-card.card-ret::before { background: var(--accent); }
.metric-card.card-vol::before { background: var(--amber); }
.metric-card.card-sharpe::before { background: oklch(55% 0.16 260); }
.metric-card.card-esg::before  { background: var(--accent-mid); }

.metric-card:hover {
  border-color: var(--border-strong);
  transform: translateY(-2px);
}
.metric-label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--text-3);
  margin-bottom: 0.35rem;
}
.metric-value {
  font-family: 'Fraunces', serif;
  font-size: 2.1rem;
  font-weight: 400;
  color: var(--text-1);
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.metric-unit {
  font-size: 0.8rem;
  font-weight: 400;
  color: var(--text-3);
  margin-left: 3px;
  font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Section headers ── */
.section-header {
  font-family: 'Fraunces', serif;
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--text-1);
  letter-spacing: -0.015em;
  border-bottom: 1.5px solid var(--border);
  padding-bottom: 0.45rem;
  margin-bottom: 1.1rem;
}

/* ── Info / warning boxes ── */
.info-box {
  background: var(--accent-wash);
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 0.75rem 1rem;
  margin: 0.65rem 0;
  font-size: 0.875rem;
  font-weight: 400;
  color: var(--text-2);
  line-height: 1.55;
}
.warn-box {
  background: var(--amber-wash);
  border-left: 3px solid var(--amber);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 0.75rem 1rem;
  margin: 0.65rem 0;
  font-size: 0.875rem;
  color: oklch(38% 0.080 62);
  line-height: 1.55;
}

/* ── Asset tag ── */
.asset-tag {
  display: inline-block;
  background: var(--accent-wash);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 2px 10px;
  font-size: 0.76rem;
  font-weight: 600;
  color: var(--accent);
  margin-right: 5px;
  margin-bottom: 4px;
}

/* ── Primary button ── */
div.stButton > button {
  background: var(--accent);
  color: oklch(97% 0.006 140);
  border: none;
  border-radius: var(--radius-sm);
  padding: 0.65rem 2rem;
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-weight: 600;
  font-size: 0.95rem;
  letter-spacing: 0.02em;
  width: 100%;
  transition: background 0.2s var(--ease), transform 0.15s var(--ease-quick);
}
div.stButton > button:hover {
  background: var(--text-2);
  transform: translateY(-1px);
}
div.stButton > button:active {
  transform: translateY(0);
}

/* ── Expander ── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--bg-card) !important;
}
[data-testid="stExpander"] summary {
  font-weight: 600;
  color: var(--text-1) !important;
}

/* ── Data table ── */
.stDataFrame {
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border) !important;
}

/* ── Inputs ── */
.stNumberInput input,
.stTextInput input {
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--bg-card);
  color: var(--text-1);
  font-family: 'Plus Jakarta Sans', sans-serif;
  transition: border-color 0.18s var(--ease);
}
.stNumberInput input:focus,
.stTextInput input:focus {
  border-color: var(--accent-mid);
  outline: none;
}

/* ── Selectbox ── */
.stSelectbox select {
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}

/* ── Divider ── */
hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.5rem 0;
}

/* ── Streamlit default overrides ── */
h4 {
  font-family: 'Fraunces', serif;
  font-weight: 600;
  color: var(--text-1);
  letter-spacing: -0.01em;
}
p, li {
  color: var(--text-2);
  line-height: 1.6;
}
code {
  background: var(--bg-inset);
  border-radius: 4px;
  padding: 1px 5px;
  font-size: 0.87em;
  color: var(--accent);
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p {
  color: var(--text-3) !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {
  color: var(--sb-text) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_stats(weights, mu, cov, esg_scores, rf):
    """Return (E[Rp], sigma_p, sharpe, esg_bar)"""
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


def build_frontier(mu, cov, esg_scores, rf, n_points=300):
    """Build the ESG-efficient frontier: for each ESG target, find max Sharpe."""
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Range of reachable ESG scores
    esg_min = min(esg_scores)
    esg_max = max(esg_scores)
    esg_targets = np.linspace(esg_min, esg_max, n_points)

    frontier_sharpe = []
    frontier_ret = []
    frontier_std = []
    frontier_esg = []
    frontier_weights = []

    w0 = np.ones(n) / n

    for esg_t in esg_targets:
        esg_constraint = {
            "type": "ineq",
            "fun": lambda w, t=esg_t: w @ esg_scores - t,
        }
        # Maximise Sharpe subject to ESG >= esg_t
        def neg_sharpe(w):
            ep, sp, sr, _ = portfolio_stats(w, mu, cov, esg_scores, rf)
            return -sr

        res = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=[sum_constraint, esg_constraint],
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if res.success:
            ep, sp, sr, eg = portfolio_stats(res.x, mu, cov, esg_scores, rf)
            frontier_sharpe.append(sr)
            frontier_ret.append(ep)
            frontier_std.append(sp)
            frontier_esg.append(eg)
            frontier_weights.append(res.x)
        else:
            frontier_sharpe.append(np.nan)
            frontier_ret.append(np.nan)
            frontier_std.append(np.nan)
            frontier_esg.append(esg_t)
            frontier_weights.append(None)

    return (
        np.array(frontier_esg),
        np.array(frontier_sharpe),
        np.array(frontier_ret),
        np.array(frontier_std),
        frontier_weights,
    )


def find_optimal(mu, cov, esg_scores, rf, gamma, lam):
    """Find the utility-maximising portfolio."""
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0 = np.ones(n) / n

    res = minimize(
        utility,
        w0,
        args=(mu, cov, esg_scores, rf, gamma, lam),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    return res.x if res.success else w0


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  —  INVESTOR PREFERENCES
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
<div style="padding: 0.4rem 0 1.2rem; border-bottom: 1px solid oklch(24% 0.028 148);">
  <div style="display:flex; align-items:center; gap:0.55rem;">
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="14" cy="14" r="14" fill="oklch(44% 0.155 145)" opacity="0.18"/>
      <path d="M14 6C10.13 6 7 9.36 7 13.5C7 17.36 9.8 20.52 13.5 20.94V23.5H14.5V20.94C18.2 20.52 21 17.36 21 13.5C21 9.36 17.87 6 14 6Z" fill="oklch(72% 0.130 142)"/>
      <path d="M14 8.5C11.5 8.5 9.5 10.76 9.5 13.5C9.5 16.24 11.5 18.5 14 18.5C16.5 18.5 18.5 16.24 18.5 13.5" stroke="oklch(52% 0.145 143)" stroke-width="1.2" stroke-linecap="round" fill="none"/>
    </svg>
    <span style="font-family:'Fraunces',serif; font-size:1.25rem; font-weight:600; color:oklch(88% 0.020 142); letter-spacing:-0.015em;">GreenPort</span>
  </div>
  <div style="font-size:0.7rem; color:oklch(58% 0.014 140); margin-top:0.3rem; letter-spacing:0.06em; text-transform:uppercase; font-weight:500;">ESG Portfolio Optimiser</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Investor Preferences")
    gamma = st.slider(
        "Risk Aversion (γ)",
        min_value=0.5, max_value=10.0, value=3.0, step=0.5,
        help="Higher γ = more risk-averse. Standard range: 1–5."
    )
    lam = st.slider(
        "ESG Preference (λ)",
        min_value=0.0, max_value=5.0, value=1.0, step=0.1,
        help="λ = 0 → pure financial investor. Higher = cares more about ESG."
    )
    rf = st.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0, max_value=20.0, value=4.0, step=0.1,
        format="%.1f"
    ) / 100

    st.markdown("---")
    st.markdown("### ESG Filter")
    use_exclusion = st.checkbox("Apply exclusion screen", value=False,
        help="Exclude assets with ESG score below a minimum threshold.")
    min_esg = 0.0
    if use_exclusion:
        min_esg = st.slider("Minimum ESG score", 0.0, 10.0, 4.0, 0.5)

    st.markdown("---")
    st.markdown("""
<div style="margin-top:1.5rem; padding-top:1rem; border-top:1px solid oklch(24% 0.028 148);">
  <div style="font-size:0.68rem; color:oklch(52% 0.014 140); font-weight:500; letter-spacing:0.04em; line-height:1.8;">
    ECN316 · Sustainable Finance<br>Group Assignment 2026
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  —  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.25rem;">
  <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="20" cy="20" r="20" fill="oklch(44% 0.155 145)" opacity="0.12"/>
    <path d="M20 8C14.48 8 10 12.92 10 19C10 24.82 14.0 29.52 19.3 29.94V33H20.7V29.94C25.99 29.52 30 24.82 30 19C30 12.92 25.52 8 20 8Z" fill="oklch(44% 0.155 145)"/>
    <path d="M20 12C17.0 12 14.5 15.1 14.5 19C14.5 22.9 17.0 26 20 26C23.0 26 25.5 22.9 25.5 19" stroke="oklch(62% 0.120 142)" stroke-width="1.5" stroke-linecap="round" fill="none"/>
    <line x1="20" y1="19" x2="25" y2="14" stroke="oklch(62% 0.120 142)" stroke-width="1.3" stroke-linecap="round"/>
  </svg>
  <div class="hero-title">GreenPort</div>
</div>
<div class="hero-sub">ESG-aware portfolio optimiser — built for sustainable investing</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ASSET INPUT SECTION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Asset Universe</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_right:
    n_assets = st.number_input("Number of assets", min_value=2, max_value=10, value=3, step=1)
    st.markdown('<div class="info-box">Enter expected return, volatility and ESG score (0–10) for each asset. Correlation matrix is entered below.</div>', unsafe_allow_html=True)

with col_left:
    asset_data = []
    cols = st.columns([2, 1.2, 1.2, 1.2])
    cols[0].markdown("**Asset name**")
    cols[1].markdown("**E[R] (%)**")
    cols[2].markdown("**σ (%)**")
    cols[3].markdown("**ESG (0–10)**")

    default_names  = ["Tech ETF", "Green Bond", "Energy Stock", "Healthcare", "Consumer ETF",
                      "Infra Fund", "EM Equity", "Gov Bond", "Real Estate", "Commodity"]
    default_ret    = [9.0, 4.5, 7.0, 7.5, 6.5, 5.5, 10.0, 3.0, 6.0, 5.0]
    default_vol    = [18.0, 5.0, 22.0, 15.0, 14.0, 10.0, 25.0, 4.0, 13.0, 20.0]
    default_esg    = [6.5, 8.5, 2.0, 7.0, 5.5, 7.5, 4.0, 6.0, 5.0, 3.5]

    for i in range(int(n_assets)):
        c0, c1, c2, c3 = st.columns([2, 1.2, 1.2, 1.2])
        name = c0.text_input("", value=default_names[i], key=f"name_{i}", label_visibility="collapsed")
        ret  = c1.number_input("", value=default_ret[i], key=f"ret_{i}",  label_visibility="collapsed", format="%.1f")
        vol  = c2.number_input("", value=default_vol[i], key=f"vol_{i}",  label_visibility="collapsed", format="%.1f", min_value=0.1)
        esg  = c3.number_input("", value=default_esg[i], key=f"esg_{i}", label_visibility="collapsed", format="%.1f", min_value=0.0, max_value=10.0)
        asset_data.append({"name": name, "ret": ret / 100, "vol": vol / 100, "esg": esg})

# Correlation matrix
st.markdown("**Correlation Matrix**")
st.markdown('<div class="info-box">Enter pairwise correlations (−1 to 1). The matrix must be positive semi-definite. Diagonal is fixed at 1.</div>', unsafe_allow_html=True)

n = int(n_assets)
default_corr_val = 0.25

# Build editable correlation matrix using a DataFrame
corr_init = pd.DataFrame(
    np.eye(n),
    columns=[asset_data[i]["name"] for i in range(n)],
    index=[asset_data[i]["name"] for i in range(n)],
)
# Pre-fill off-diagonals
for r in range(n):
    for c in range(n):
        if r != c:
            corr_init.iloc[r, c] = default_corr_val

corr_df = st.data_editor(
    corr_init,
    use_container_width=True,
    key="corr_matrix",
)

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISE BUTTON
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
run_col, _ = st.columns([1, 3])
with run_col:
    run = st.button("⚡ Optimise Portfolio")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if run:
    # ── Parse inputs ──────────────────────────────────────────────────────────
    names      = [d["name"] for d in asset_data]
    mu         = np.array([d["ret"] for d in asset_data])
    vols       = np.array([d["vol"] for d in asset_data])
    esg_scores = np.array([d["esg"] for d in asset_data])

    # Build covariance matrix from correlation + vols
    corr_np = corr_df.values.astype(float)
    # Symmetrise
    corr_np = (corr_np + corr_np.T) / 2
    np.fill_diagonal(corr_np, 1.0)
    # Clamp off-diagonals
    corr_np = np.clip(corr_np, -0.999, 0.999)

    cov = np.outer(vols, vols) * corr_np

    # Check PSD
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < -1e-8):
        st.markdown('<div class="warn-box">⚠️ Covariance matrix is not positive semi-definite. Check your correlations — some combinations may be inconsistent. The optimiser will still run but results may be approximate.</div>', unsafe_allow_html=True)
        # Nearest PSD fix
        cov = cov + np.eye(n) * max(0, -eigvals.min() + 1e-8)

    # Apply exclusion screen
    active_mask = np.ones(n, dtype=bool)
    if use_exclusion:
        active_mask = esg_scores >= min_esg
        excluded = [names[i] for i in range(n) if not active_mask[i]]
        if excluded:
            st.markdown(f'<div class="warn-box">🚫 Excluded by ESG screen: {", ".join(excluded)}</div>', unsafe_allow_html=True)

    active_idx = np.where(active_mask)[0]
    if len(active_idx) < 2:
        st.error("At least 2 assets must pass the ESG screen. Relax the filter.")
        st.stop()

    mu_a   = mu[active_idx]
    cov_a  = cov[np.ix_(active_idx, active_idx)]
    esg_a  = esg_scores[active_idx]
    names_a = [names[i] for i in active_idx]
    n_a    = len(active_idx)

    # ── Find optimal portfolio ─────────────────────────────────────────────────
    w_opt_a = find_optimal(mu_a, cov_a, esg_a, rf, gamma, lam)

    # Map back to full weight vector
    w_opt = np.zeros(n)
    for idx, wi in zip(active_idx, w_opt_a):
        w_opt[idx] = wi

    ep, sp, sr, esg_bar = portfolio_stats(w_opt_a, mu_a, cov_a, esg_a, rf)

    # ── Build ESG-efficient frontier ───────────────────────────────────────────
    with st.spinner("Building ESG-efficient frontier…"):
        f_esg, f_sharpe, f_ret, f_std, f_weights = build_frontier(mu_a, cov_a, esg_a, rf, n_points=200)

    # Valid points only
    valid = ~np.isnan(f_sharpe)

    # ══════════════════════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown('<div class="section-header">Optimal Portfolio</div>', unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card card-ret"><div class="metric-label">Expected Return</div><div class="metric-value">{ep*100:.2f}<span class="metric-unit">%</span></div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card card-vol"><div class="metric-label">Volatility (σ)</div><div class="metric-value">{sp*100:.2f}<span class="metric-unit">%</span></div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card card-sharpe"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{sr:.3f}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card card-esg"><div class="metric-label">ESG Score</div><div class="metric-value">{esg_bar:.2f}<span class="metric-unit">/ 10</span></div></div>', unsafe_allow_html=True)

    # Utility value
    u_val = ep - gamma / 2 * sp**2 + lam * esg_bar
    st.markdown(f'<div class="info-box">🎯 Investor utility U = E[Rp] − (γ/2)σ² + λs̄ = <strong>{u_val:.4f}</strong> &nbsp;|&nbsp; Parameters: γ = {gamma}, λ = {lam}, r_f = {rf*100:.1f}%</div>', unsafe_allow_html=True)

    # ── Portfolio weights table ────────────────────────────────────────────────
    st.markdown("#### Portfolio Weights")
    weight_df = pd.DataFrame({
        "Asset": names,
        "Weight (%)": [f"{w*100:.2f}" for w in w_opt],
        "E[R] (%)": [f"{r*100:.1f}" for r in mu],
        "σ (%)": [f"{v*100:.1f}" for v in vols],
        "ESG Score": [f"{s:.1f}" for s in esg_scores],
        "Included": ["✅" if m else "❌ (screened)" for m in active_mask],
    })
    st.dataframe(weight_df, use_container_width=True, hide_index=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">ESG-Efficient Frontier</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    # Shared chart style helpers
    BG        = '#F7F5F0'   # warm cream (approx oklch 96.5% 0.009 140)
    BG_INSET  = '#EAE8E0'   # slightly deeper for axis area
    C_ACCENT  = '#2D6B3E'   # forest green (approx oklch 44% 0.155 145)
    C_MID     = '#4A8A5A'   # lighter green
    C_OPTIMAL = '#C75B1A'   # warm amber-orange for optimal point
    C_MUTED   = '#7A9A7A'   # muted green text
    C_GRID    = '#D8E0CE'   # soft grid

    def style_ax(ax, fig):
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG_INSET)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(C_GRID)
            spine.set_linewidth(0.8)
        ax.grid(True, alpha=0.35, color=C_GRID, linestyle='--', linewidth=0.7)

    # Chart 1: ESG vs Sharpe frontier
    with chart_col1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        style_ax(ax, fig)

        if valid.sum() > 1:
            ax.plot(f_esg[valid], f_sharpe[valid], color=C_ACCENT, lw=2.2,
                    label='ESG-Efficient Frontier', solid_capstyle='round')
            ax.fill_between(f_esg[valid], f_sharpe[valid],
                            alpha=0.10, color=C_ACCENT)

        # Individual assets
        for i in range(n_a):
            sr_i = (mu_a[i] - rf) / vols[active_idx[i]]
            ax.scatter(esg_a[i], sr_i, color=C_MID, zorder=5, s=55,
                       edgecolors=BG, lw=1.5)
            ax.annotate(names_a[i], (esg_a[i], sr_i), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=C_MUTED)

        # Optimal point
        ax.scatter(esg_bar, sr, color=C_OPTIMAL, zorder=10, s=110,
                   edgecolors=BG, lw=2, label=f'Optimal (λ={lam})')

        ax.set_xlabel("Average ESG Score", fontsize=9, color=C_MUTED)
        ax.set_ylabel("Sharpe Ratio", fontsize=9, color=C_MUTED)
        ax.set_title("ESG Score vs. Sharpe Ratio", fontsize=11,
                     fontweight='bold', color='#1C2A1C', pad=10)
        ax.legend(fontsize=8, framealpha=0.85, facecolor=BG, edgecolor=C_GRID,
                  labelcolor=C_MUTED)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 2: Mean-Variance frontier coloured by ESG
    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        style_ax(ax2, fig2)

        if valid.sum() > 1:
            sc = ax2.scatter(f_std[valid]*100, f_ret[valid]*100,
                             c=f_esg[valid], cmap='YlGn',
                             s=10, zorder=3, vmin=min(esg_a), vmax=max(esg_a))
            cb = fig2.colorbar(sc, ax=ax2, pad=0.02)
            cb.set_label("ESG Score", fontsize=8, color=C_MUTED)
            cb.ax.tick_params(labelsize=7, colors=C_MUTED)
            cb.outline.set_color(C_GRID)

        # Individual assets
        for i in range(n_a):
            ax2.scatter(vols[active_idx[i]]*100, mu_a[i]*100,
                        color=C_ACCENT, zorder=5, s=55,
                        edgecolors=BG, lw=1.5)
            ax2.annotate(names_a[i], (vols[active_idx[i]]*100, mu_a[i]*100),
                         textcoords="offset points", xytext=(5, 3),
                         fontsize=7, color=C_MUTED)

        # Optimal point
        ax2.scatter(sp*100, ep*100, color=C_OPTIMAL, zorder=10, s=130,
                    edgecolors=BG, lw=2, label='Optimal portfolio',
                    marker='*')

        ax2.set_xlabel("Volatility σ (%)", fontsize=9, color=C_MUTED)
        ax2.set_ylabel("Expected Return E[R] (%)", fontsize=9, color=C_MUTED)
        ax2.set_title("Mean-Variance Space (coloured by ESG)", fontsize=11,
                      fontweight='bold', color='#1C2A1C', pad=10)
        ax2.legend(fontsize=8, framealpha=0.85, facecolor=BG, edgecolor=C_GRID,
                   labelcolor=C_MUTED)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Chart 3: Pie chart of weights
    st.markdown("#### Portfolio Allocation")
    pie_col, bar_col = st.columns(2)

    nonzero = [(names[i], w_opt[i]) for i in range(n) if w_opt[i] > 0.005]
    if nonzero:
        pie_labels, pie_vals = zip(*nonzero)
        with pie_col:
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            fig3.patch.set_facecolor(BG)
            ax3.set_facecolor(BG)
            greens = ['#1E4D2A','#2D6B3E','#3D8A52','#52A86A','#6EC284',
                      '#92D4A2','#B4E2C0','#CEF0D8','#E2F7EB','#F0FCF4']
            ax3.pie(pie_vals, labels=pie_labels, autopct='%1.1f%%',
                    colors=greens[:len(pie_vals)], startangle=140,
                    textprops={'fontsize': 8, 'color': '#1C2A1C'},
                    wedgeprops={'edgecolor': BG, 'linewidth': 2})
            ax3.set_title("Weight Allocation", fontsize=11, fontweight='bold',
                          color='#1C2A1C', pad=10)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # ESG vs Weight bar
        with bar_col:
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            style_ax(ax4, fig4)
            bar_names = [names[i] for i in range(n) if w_opt[i] > 0.005]
            bar_weights = [w_opt[i]*100 for i in range(n) if w_opt[i] > 0.005]
            bar_esg = [esg_scores[i] for i in range(n) if w_opt[i] > 0.005]
            bar_colors = [plt.cm.YlGn(s / 10) for s in bar_esg]
            bars = ax4.barh(bar_names, bar_weights, color=bar_colors,
                            edgecolor=BG, height=0.55)
            for bar, esg_v in zip(bars, bar_esg):
                ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                         f'ESG {esg_v:.1f}', va='center', fontsize=7.5, color=C_MUTED)
            ax4.set_xlabel("Weight (%)", fontsize=9, color=C_MUTED)
            ax4.set_title("Weights with ESG Scores", fontsize=11,
                          fontweight='bold', color='#1C2A1C', pad=10)
            ax4.grid(True, alpha=0.3, color=C_GRID, axis='x', linestyle='--', linewidth=0.7)
            fig4.tight_layout()
            st.pyplot(fig4)
            plt.close()

    # ── Sensitivity: vary λ ───────────────────────────────────────────────────
    with st.expander("📊 Sensitivity Analysis — ESG Preference (λ)"):
        st.markdown("How does the portfolio change as you vary λ from 0 to 5?")
        lam_vals = np.linspace(0, 5, 20)
        sens_rows = []
        for lv in lam_vals:
            ww = find_optimal(mu_a, cov_a, esg_a, rf, gamma, lv)
            ep2, sp2, sr2, esg2 = portfolio_stats(ww, mu_a, cov_a, esg_a, rf)
            sens_rows.append({"λ": round(lv, 2), "E[R] (%)": round(ep2*100, 2),
                               "σ (%)": round(sp2*100, 2),
                               "Sharpe": round(sr2, 3), "ESG": round(esg2, 2)})
        sens_df = pd.DataFrame(sens_rows)

        fig5, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        fig5.patch.set_facecolor(BG)
        for ax_ in axes:
            style_ax(ax_, fig5)

        axes[0].plot(sens_df["λ"], sens_df["Sharpe"], color=C_ACCENT, lw=2,
                     solid_capstyle='round')
        axes[0].set_title("Sharpe vs λ", fontsize=10, color='#1C2A1C')
        axes[0].set_xlabel("λ", fontsize=9, color=C_MUTED)
        axes[0].set_ylabel("Sharpe Ratio", fontsize=9, color=C_MUTED)

        axes[1].plot(sens_df["λ"], sens_df["ESG"], color=C_MID, lw=2,
                     solid_capstyle='round')
        axes[1].set_title("ESG Score vs λ", fontsize=10, color='#1C2A1C')
        axes[1].set_xlabel("λ", fontsize=9, color=C_MUTED)
        axes[1].set_ylabel("ESG Score", fontsize=9, color=C_MUTED)

        axes[2].plot(sens_df["λ"], sens_df["E[R] (%)"], color=C_ACCENT, lw=2,
                     solid_capstyle='round', label='E[R]')
        axes[2].plot(sens_df["λ"], sens_df["σ (%)"], color=C_OPTIMAL, lw=2,
                     linestyle='--', solid_capstyle='round', label='σ')
        axes[2].set_title("Return & Risk vs λ", fontsize=10, color='#1C2A1C')
        axes[2].set_xlabel("λ", fontsize=9, color=C_MUTED)
        axes[2].set_ylabel("%", fontsize=9, color=C_MUTED)
        axes[2].legend(fontsize=8, facecolor=BG, edgecolor=C_GRID, labelcolor=C_MUTED)

        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close()

        st.dataframe(sens_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="info-box">📌 <strong>Methodology:</strong> Utility function U = E[Rp] − (γ/2)σ²p + λs̄ · ESG-efficient frontier computed by maximising Sharpe ratio subject to a minimum ESG constraint at each point. Optimisation via Sequential Least Squares Programming (SLSQP). No short-selling allowed.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="warn-box">👆 Configure assets and preferences in the sidebar and above, then click <strong>Optimise Portfolio</strong> to see results.</div>', unsafe_allow_html=True)

    # Show preview of the utility function
    with st.expander("📖 How does the model work?"):
        st.markdown("""
**Utility Function**

$$U = E[R_p] - \\frac{\\gamma}{2}\\sigma_p^2 + \\lambda \\bar{s}$$

| Symbol | Meaning |
|--------|---------|
| $E[R_p]$ | Expected portfolio return |
| $\\sigma_p^2$ | Portfolio variance |
| $\\gamma$ | Risk aversion (higher = more risk-averse) |
| $\\lambda$ | ESG preference intensity (0 = ignore ESG) |
| $\\bar{s}$ | Weighted average ESG score of the portfolio |

**ESG-Efficient Frontier**

For each level of ESG score, the app finds the portfolio with the highest Sharpe ratio subject to meeting that ESG target. The result is a frontier in (ESG, Sharpe) space — showing the trade-off between sustainability and financial performance.

**Reference:** Gantchev et al. (2023); Lecture 6 — ECN316 Sustainable Finance
        """)
