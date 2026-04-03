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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #f5f2ec;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a2e1a !important;
    border-right: 2px solid #2d4a2d;
}
[data-testid="stSidebar"] * {
    color: #d4e8c2 !important;
}
[data-testid="stSidebar"] .stSlider > div > div {
    background: #2d4a2d;
}
[data-testid="stSidebar"] label {
    color: #a8cc8c !important;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Main heading */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #1a2e1a;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #5a7a5a;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #d4e0c8;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(26,46,26,0.06);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7a9a6a;
    margin-bottom: 0.2rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #1a2e1a;
    line-height: 1;
}
.metric-unit {
    font-size: 0.85rem;
    color: #8aaa7a;
    margin-left: 2px;
}

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1a2e1a;
    border-bottom: 2px solid #c8dab8;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* Asset tag */
.asset-tag {
    display:inline-block;
    background:#e8f0e0;
    border:1px solid #c0d4a8;
    border-radius:6px;
    padding:2px 10px;
    font-size:0.78rem;
    font-weight:600;
    color:#2d4a2d;
    margin-right:6px;
    margin-bottom:4px;
}

/* Info box */
.info-box {
    background: #e8f5e0;
    border-left: 4px solid #4a8a3a;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: #2a4a2a;
}

/* Warning box */
.warn-box {
    background: #fff8e8;
    border-left: 4px solid #d4a020;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: #5a4010;
}

/* Button */
div.stButton > button {
    background: #2d6a2d;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    width: 100%;
    transition: background 0.2s;
}
div.stButton > button:hover {
    background: #1a4a1a;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #d4e0c8 !important;
    border-radius: 10px !important;
    background: white !important;
}

/* Table */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* Number input */
.stNumberInput input {
    border-radius: 6px;
    border: 1px solid #c8d8b8;
    background: white;
}

/* Selectbox */
.stSelectbox select {
    border-radius: 6px;
}

/* Remove default streamlit padding top */
.block-container {
    padding-top: 2rem;
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
    st.markdown("## 🌿 GreenPort")
    st.markdown("---")

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
    st.markdown("<small style='color:#7aaa6a'>ECN316 · Sustainable Finance<br>Group Assignment 2026</small>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  —  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">GreenPort</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">ESG-aware portfolio optimiser for retail investors</div>', unsafe_allow_html=True)

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
        st.markdown(f'<div class="metric-card"><div class="metric-label">Expected Return</div><div class="metric-value">{ep*100:.2f}<span class="metric-unit">%</span></div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Volatility (σ)</div><div class="metric-value">{sp*100:.2f}<span class="metric-unit">%</span></div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{sr:.3f}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">ESG Score</div><div class="metric-value">{esg_bar:.2f}<span class="metric-unit">/ 10</span></div></div>', unsafe_allow_html=True)

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

    # Chart 1: ESG vs Sharpe frontier
    with chart_col1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor('#f5f2ec')
        ax.set_facecolor('#f5f2ec')

        if valid.sum() > 1:
            ax.plot(f_esg[valid], f_sharpe[valid], color='#2d6a2d', lw=2.5, label='ESG-Efficient Frontier')
            ax.fill_between(f_esg[valid], f_sharpe[valid],
                            alpha=0.12, color='#4a9a3a')

        # Individual assets
        for i in range(n_a):
            sr_i = (mu_a[i] - rf) / vols[active_idx[i]]
            ax.scatter(esg_a[i], sr_i, color='#8ab87a', zorder=5, s=60, edgecolors='#2d6a2d', lw=1)
            ax.annotate(names_a[i], (esg_a[i], sr_i), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color='#2d4a2d')

        # Optimal point
        ax.scatter(esg_bar, sr, color='#d44a1a', zorder=10, s=120,
                   edgecolors='white', lw=2, label=f'Optimal (λ={lam})')

        ax.set_xlabel("Average ESG Score", fontsize=9, color='#2d4a2d')
        ax.set_ylabel("Sharpe Ratio", fontsize=9, color='#2d4a2d')
        ax.set_title("ESG Score vs. Sharpe Ratio", fontsize=11,
                     fontweight='bold', color='#1a2e1a', pad=10)
        ax.tick_params(colors='#5a7a5a', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#c8d8b8')
        ax.legend(fontsize=8, framealpha=0.7, facecolor='#f5f2ec', edgecolor='#c8d8b8')
        ax.grid(True, alpha=0.3, color='#c8d8b8', linestyle='--')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 2: Mean-Variance frontier coloured by ESG
    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        fig2.patch.set_facecolor('#f5f2ec')
        ax2.set_facecolor('#f5f2ec')

        if valid.sum() > 1:
            sc = ax2.scatter(f_std[valid]*100, f_ret[valid]*100,
                             c=f_esg[valid], cmap='YlGn',
                             s=10, zorder=3, vmin=min(esg_a), vmax=max(esg_a))
            cb = fig2.colorbar(sc, ax=ax2, pad=0.02)
            cb.set_label("ESG Score", fontsize=8, color='#2d4a2d')
            cb.ax.tick_params(labelsize=7, colors='#5a7a5a')

        # Individual assets
        for i in range(n_a):
            ax2.scatter(vols[active_idx[i]]*100, mu_a[i]*100,
                        color='#2d6a2d', zorder=5, s=60,
                        edgecolors='white', lw=1.2)
            ax2.annotate(names_a[i], (vols[active_idx[i]]*100, mu_a[i]*100),
                         textcoords="offset points", xytext=(5, 3),
                         fontsize=7, color='#2d4a2d')

        # Optimal point
        ax2.scatter(sp*100, ep*100, color='#d44a1a', zorder=10, s=140,
                    edgecolors='white', lw=2, label='Optimal portfolio',
                    marker='*')

        ax2.set_xlabel("Volatility σ (%)", fontsize=9, color='#2d4a2d')
        ax2.set_ylabel("Expected Return E[R] (%)", fontsize=9, color='#2d4a2d')
        ax2.set_title("Mean-Variance Space (coloured by ESG)", fontsize=11,
                      fontweight='bold', color='#1a2e1a', pad=10)
        ax2.tick_params(colors='#5a7a5a', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color('#c8d8b8')
        ax2.legend(fontsize=8, framealpha=0.7, facecolor='#f5f2ec', edgecolor='#c8d8b8')
        ax2.grid(True, alpha=0.3, color='#c8d8b8', linestyle='--')
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
            fig3.patch.set_facecolor('#f5f2ec')
            ax3.set_facecolor('#f5f2ec')
            greens = ['#1a4a1a','#2d6a2d','#4a8a3a','#6aaa5a','#8aba7a',
                      '#a8cc98','#c4deb8','#d4e8c8','#e4f0d8','#f0f8ec']
            ax3.pie(pie_vals, labels=pie_labels, autopct='%1.1f%%',
                    colors=greens[:len(pie_vals)], startangle=140,
                    textprops={'fontsize': 8, 'color': '#1a2e1a'},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
            ax3.set_title("Weight Allocation", fontsize=11, fontweight='bold',
                          color='#1a2e1a', pad=10)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # ESG vs Weight bar
        with bar_col:
            fig4, ax4 = plt.subplots(figsize=(5, 4))
            fig4.patch.set_facecolor('#f5f2ec')
            ax4.set_facecolor('#f5f2ec')
            bar_names = [names[i] for i in range(n) if w_opt[i] > 0.005]
            bar_weights = [w_opt[i]*100 for i in range(n) if w_opt[i] > 0.005]
            bar_esg = [esg_scores[i] for i in range(n) if w_opt[i] > 0.005]
            bar_colors = [plt.cm.YlGn(s / 10) for s in bar_esg]
            bars = ax4.barh(bar_names, bar_weights, color=bar_colors,
                            edgecolor='white', height=0.6)
            for bar, esg_v in zip(bars, bar_esg):
                ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                         f'ESG {esg_v:.1f}', va='center', fontsize=7.5, color='#2d4a2d')
            ax4.set_xlabel("Weight (%)", fontsize=9, color='#2d4a2d')
            ax4.set_title("Weights with ESG Scores", fontsize=11,
                          fontweight='bold', color='#1a2e1a', pad=10)
            ax4.tick_params(colors='#5a7a5a', labelsize=8)
            for spine in ax4.spines.values():
                spine.set_color('#c8d8b8')
            ax4.grid(True, alpha=0.3, color='#c8d8b8', axis='x', linestyle='--')
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
        fig5.patch.set_facecolor('#f5f2ec')
        for ax_ in axes:
            ax_.set_facecolor('#f5f2ec')
            ax_.tick_params(colors='#5a7a5a', labelsize=8)
            for sp_ in ax_.spines.values():
                sp_.set_color('#c8d8b8')
            ax_.grid(True, alpha=0.3, color='#c8d8b8', linestyle='--')

        axes[0].plot(sens_df["λ"], sens_df["Sharpe"], color='#2d6a2d', lw=2)
        axes[0].set_title("Sharpe vs λ", fontsize=10, color='#1a2e1a')
        axes[0].set_xlabel("λ", fontsize=9); axes[0].set_ylabel("Sharpe Ratio", fontsize=9)

        axes[1].plot(sens_df["λ"], sens_df["ESG"], color='#4a8a3a', lw=2)
        axes[1].set_title("ESG Score vs λ", fontsize=10, color='#1a2e1a')
        axes[1].set_xlabel("λ", fontsize=9); axes[1].set_ylabel("ESG Score", fontsize=9)

        axes[2].plot(sens_df["λ"], sens_df["E[R] (%)"], color='#6aaa5a', lw=2,
                     label='E[R]')
        axes[2].plot(sens_df["λ"], sens_df["σ (%)"], color='#d44a1a', lw=2,
                     linestyle='--', label='σ')
        axes[2].set_title("Return & Risk vs λ", fontsize=10, color='#1a2e1a')
        axes[2].set_xlabel("λ", fontsize=9); axes[2].set_ylabel("%", fontsize=9)
        axes[2].legend(fontsize=8, facecolor='#f5f2ec', edgecolor='#c8d8b8')

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
