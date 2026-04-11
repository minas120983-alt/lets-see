import warnings
import io
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
    page_title="GreenPort · ESG Portfolio Optimiser",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS (unchanged from your original) ───────────────────────────────────────
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
.metric-card { background: var(--bg-card); border: 1px solid var(--sep); border-radius: var(--r-lg); padding: 1.4rem 1.5rem 1.3rem; position: relative; overflow: hidden; }
.metric-card.card-ret { border-top: 2px solid var(--accent); }
.metric-card.card-vol { border-top: 2px solid var(--sys-orange); }
.metric-card.card-sr { border-top: 2px solid var(--sys-indigo); }
.metric-card.card-esg { border-top: 2px solid var(--accent); }
.metric-label { font-size: 0.60rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3) !important; margin-bottom: 0.65rem; }
.metric-value { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.04em; color: var(--text-1) !important; line-height: 1; }
.metric-unit { font-size: 0.78rem; color: var(--text-3) !important; margin-left: 2px; font-weight: 400; }
.info-box { background: var(--accent-dark); border: 1px solid rgba(34,197,94,0.18); border-radius: var(--r-sm); padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.81rem; color: var(--accent) !important; line-height: 1.6; }
.warn-box { background: rgba(251,146,60,0.07); border: 1px solid rgba(251,146,60,0.20); border-radius: var(--r-sm); padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.81rem; color: var(--sys-orange) !important; line-height: 1.6; }
.error-box { background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.20); border-radius: var(--r-sm); padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.81rem; color: var(--sys-red) !important; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── ESG DATABASE (unchanged) ────────────────────────────────────────────────
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

# ── PORTFOLIO MATH (unchanged) ───────────────────────────────────────────────
def port_ret(w, mu): return float(np.asarray(w) @ np.asarray(mu))
def port_var(w, cov): return float(np.asarray(w) @ np.asarray(cov) @ np.asarray(w))
def port_sd(w, cov): return float(max(port_var(w, cov), 1e-14) ** 0.5)
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
    sd_t = port_sd(w_tan, cov)
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

# ── CHATBOT, MARKET DATA, STATE, HOME, etc. (unchanged) ─────────────────────
# ... [All your original chatbot, market data fetching, home page, etc. remain the same]

# For brevity in this response, I'm showing only the critical updated RESULTS section.
# Replace your entire `elif _page == "results":` block with the code below.

# ── RESULTS PAGE (UPDATED) ───────────────────────────────────────────────────
elif _page == "results":
    if "opt_results" not in st.session_state:
        st.markdown('<div class="warn-box">No results found. Return to setup.</div>', unsafe_allow_html=True)
        if st.button("Back to Setup"): 
            st.session_state["page"] = "input"; st.rerun()
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
    ticker_data_display = R.get("ticker_data_display")
    corr_np = R.get("corr_np")
    input_mode = R["input_mode"]

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

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, unit, cls, card_cls in [
        (m1, "Expected Return", f"{ep*100:.2f}", "%", "metric-pos", "card-ret"),
        (m2, "Volatility", f"{sp*100:.2f}", "%", "", "card-vol"),
        (m3, "Sharpe Ratio", f"{sr:.3f}", "", "metric-pos" if sr > 0 else "metric-neg", "card-sr"),
        (m4, "ESG Score", f"{esg_bar:.2f}", "/ 10", "metric-pos" if esg_bar >= 5 else "", "card-esg"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card {card_cls}"><div class="metric-label">{label}</div><div class="metric-value {cls}">{val}<span class="metric-unit">{unit}</span></div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="info-box">Tangency Sharpe — all assets: <strong>{sr_tan_all:.3f}</strong> &nbsp;|&nbsp; ESG-screened: <strong>{sr_tan_esg:.3f}</strong></div>', unsafe_allow_html=True)

    # Portfolio Weights (unchanged)
    st.markdown('<div class="section-header">Portfolio Weights</div>', unsafe_allow_html=True)
    _rf_weight = max(0.0, 1.0 - float(np.sum(w_opt)))
    _display_names = names + ["Risk-Free Asset"]
    _display_w = [f"{w*100:.2f}" for w in w_opt] + [f"{_rf_weight*100:.2f}"]
    _display_ret = [f"{r*100:.2f}" for r in mu] + [f"{rf*100:.2f}"]
    _display_vol = [f"{v*100:.2f}" for v in vols] + ["0.00"]
    _display_esg = [f"{s:.2f}" for s in esg_scores] + ["N/A"]
    st.dataframe(pd.DataFrame({
        "Asset": _display_names,
        "Weight (%)": _display_w,
        "E[R] (%)": _display_ret,
        "Vol (%)": _display_vol,
        "ESG (0–10)": _display_esg,
    }), use_container_width=True, hide_index=True)

    # ── EFFICIENT FRONTIERS + TWO CMLs ───────────────────────────────────────
    st.markdown('<div class="section-header">Efficient Frontiers & Capital Market Lines</div>', unsafe_allow_html=True)

    all_stds = list(std_blue) + list(std_green) + [sp * 100, sp_tan_all * 100, sp_tan_esg * 100]
    all_rets = list(ret_blue) + list(ret_green) + [ep * 100, ep_tan_all * 100, ep_tan_esg * 100, rf * 100]
    x_pad = max(all_stds) * 0.08 if all_stds else 5
    y_pad = ((max(all_rets) - min(all_rets)) * 0.12) if len(all_rets) > 1 else 1

    _c1, _c2 = st.columns(2)

    # Graph 1: MV Frontiers + 2 CMLs
    with _c1:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.patch.set_facecolor(CHART_BG)

        # Blue frontier (all assets)
        if len(std_blue) > 2:
            ax.plot(std_blue, ret_blue, color=BLUE, lw=2.2, zorder=3,
                    label="MV Frontier (All Assets)")

        # Green frontier (ESG-screened)
        if len(std_green) > 2:
            ax.plot(std_green, ret_green, color=GREEN, lw=2.2, zorder=4,
                    label=f"MV Frontier (ESG ≥ {esg_thresh:.1f})")

        # TWO Capital Market Lines
        cml_max = max(all_stds) + x_pad if all_stds else 50
        sd_cml = np.linspace(0, cml_max, 300)

        if sp_tan_all > 1e-9:
            ax.plot(sd_cml, rf * 100 + (ep_tan_all - rf) / sp_tan_all * sd_cml,
                    color=BLUE, lw=1.6, linestyle="--", zorder=5,
                    label="CML — All Assets (Max Sharpe)")

        if sp_tan_esg > 1e-9 and len(std_green) > 0:
            ax.plot(sd_cml, rf * 100 + (ep_tan_esg - rf) / sp_tan_esg * sd_cml,
                    color=GREEN, lw=1.6, linestyle="--", zorder=5,
                    label="CML — ESG-Screened (Max Sharpe)")

        # Tangency points
        ax.scatter(sp_tan_all * 100, ep_tan_all * 100, color=BLUE, s=100, zorder=9,
                   edgecolors="white", lw=1.4, marker="o")
        ax.annotate("Tangency (All)", (sp_tan_all * 100, ep_tan_all * 100),
                    textcoords="offset points", xytext=(-75, 12), fontsize=7, color=BLUE)

        if len(std_green) > 2:
            ax.scatter(sp_tan_esg * 100, ep_tan_esg * 100, color=GREEN, s=100, zorder=9,
                       edgecolors="white", lw=1.4, marker="o")
            ax.annotate("Tangency (ESG)", (sp_tan_esg * 100, ep_tan_esg * 100),
                        textcoords="offset points", xytext=(10, -28), fontsize=7, color=GREEN)

        # Risk-free
        ax.scatter(0, rf * 100, color=GREY, s=60, zorder=8, edgecolors="white", lw=1, marker="s")

        # Your utility-max portfolio
        ax.scatter(sp * 100, ep * 100, color=ORANGE, s=180, zorder=10,
                   edgecolors="white", lw=2.5, marker="*",
                   label="Utility-Max Portfolio (Yours)")

        # Individual assets
        for i in range(n):
            color = GREEN if active_mask[i] else BLUE
            ax.scatter(vols[i] * 100, mu[i] * 100, color=color, s=45, zorder=6,
                       edgecolors="white", lw=0.7, alpha=0.85)
            ax.annotate(names[i], (vols[i] * 100, mu[i] * 100),
                        textcoords="offset points", xytext=(4, 3), fontsize=7, color=GREY)

        ax.set_xlabel("Standard Deviation (%)", fontsize=9, color=GREY)
        ax.set_ylabel("Expected Return (%)", fontsize=9, color=GREY)
        ax.set_xlim(0, max(all_stds) + x_pad)
        ax.set_ylim(rf * 100 - y_pad, max(all_rets) + y_pad)
        ax.legend(fontsize=7.5, framealpha=0.9, facecolor=LEG_BG, edgecolor=LEG_ED, labelcolor=LABEL_C)
        _style_ax(ax, "Mean-Variance Frontiers + Capital Market Lines")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Graph 2: ESG–Sharpe Frontier (unchanged from your original)
    with _c2:
        # [Your original ESG-Sharpe Frontier code goes here - unchanged]
        # For space, I'm omitting the full ESG-Sharpe code. Paste your original _c2 block here.
        st.info("ESG–Sharpe Frontier chart (unchanged)")

    # Rest of your code: Portfolio Breakdown, Sensitivity Analysis, Chatbot, etc.
    # (Keep everything after the charts exactly as in your original code)

    st.markdown("""<div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid var(--sep);text-align:center;font-size:.65rem;color:var(--text-3);letter-spacing:.06em;text-transform:uppercase;">
    GreenPort · ECN316 Sustainable Finance · 2026
    </div>""", unsafe_allow_html=True)
