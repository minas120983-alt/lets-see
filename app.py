"""
GREENPORT — Graph Fix Patch
===========================
Two bugs fixed:
  1. Graph 1 (Mean-Variance Frontier): std_blue / ret_blue were never plotted.
  2. Graph 2 (ESG–SR Frontier): x-axis was forced to 0–10 regardless of data;
     left-side (rising) arch segment was often empty because ESG ≤ τ feasibility
     breaks for very low τ values — replaced with a direct sweep that works.

HOW TO APPLY
------------
Replace the entire  `with _c1:` and  `with _c2:` blocks in your results page
with the code below.  Everything else in your app stays the same.
"""

# ── paste your existing variable definitions above this block ────────────────
# (CHART_BG, BLUE, GREEN, ORANGE, GREY, LABEL_C, LEG_BG, LEG_ED, TICK_C,
#  SPINE_C, GRID_C, _style_ax, and all R["..."] unpacking)
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════════════════
# GRAPH 1  —  Mean-Variance Efficient Frontier  (FIXED: blue frontier added)
# ════════════════════════════════════════════════════════════════════════════
with _c1:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor(CHART_BG)

    # ── collect all x/y values so we can set sensible axis limits ────────
    all_stds = (list(std_blue) + list(std_green)
                + [sp * 100, sp_tan_all * 100, sp_tan_esg * 100])
    all_rets = (list(ret_blue) + list(ret_green)
                + [ep * 100, ep_tan_all * 100, ep_tan_esg * 100, rf * 100])
    x_pad = max(all_stds) * 0.08 if all_stds else 5
    y_pad = ((max(all_rets) - min(all_rets)) * 0.12) if len(all_rets) > 1 else 1

    # ── BUG FIX: plot the BLUE (all-assets) frontier ─────────────────────
    if len(std_blue) > 2:
        ax.plot(std_blue, ret_blue,
                color=BLUE, lw=2.0, zorder=3,
                label="MV Frontier (all assets)")

    # ── green (ESG-screened) frontier ─────────────────────────────────────
    if len(std_green) > 2:
        ax.plot(std_green, ret_green,
                color=GREEN, lw=2.0, zorder=4,
                label=f"MV Frontier (ESG \u2265 {esg_thresh:.1f})")

    # ── Capital Market Lines ──────────────────────────────────────────────
    cml_max = max(all_stds) + x_pad if all_stds else 50
    sd_cml  = np.linspace(0, cml_max, 300)

    if sp_tan_all > 1e-9 and len(std_blue) > 0:
        ax.plot(sd_cml,
                rf * 100 + (ep_tan_all - rf) / sp_tan_all * sd_cml,
                color=BLUE, lw=1.4, linestyle="--", zorder=4,
                label="CML (all assets)")

    if sp_tan_esg > 1e-9 and len(std_green) > 0:
        ax.plot(sd_cml,
                rf * 100 + (ep_tan_esg - rf) / sp_tan_esg * sd_cml,
                color=GREEN, lw=1.4, linestyle="--", zorder=3,
                label=f"CML (ESG \u2265 {esg_thresh:.1f})")

    # ── tangency portfolios ───────────────────────────────────────────────
    ax.scatter(sp_tan_all * 100, ep_tan_all * 100,
               color=BLUE, s=140, zorder=9,
               edgecolors="white", lw=1.4, marker="*")
    ax.annotate("tangency (all assets)",
                (sp_tan_all * 100, ep_tan_all * 100),
                textcoords="offset points", xytext=(7, 2),
                fontsize=7, color=BLUE, fontstyle="italic")

    if len(std_green) > 2:
        ax.scatter(sp_tan_esg * 100, ep_tan_esg * 100,
                   color=GREEN, s=140, zorder=9,
                   edgecolors="white", lw=1.4, marker="*")
        ax.annotate("tangency (portfolios with given ESG)",
                    (sp_tan_esg * 100, ep_tan_esg * 100),
                    textcoords="offset points", xytext=(7, -18),
                    fontsize=7, color=GREEN, fontstyle="italic")

    # ── risk-free asset ───────────────────────────────────────────────────
    ax.scatter(0, rf * 100, color=GREY, s=60, zorder=8,
               edgecolors="white", lw=1, marker="s")

    # ── ESG-optimal portfolio (orange star) ───────────────────────────────
    ax.scatter(sp * 100, ep * 100,
               color=ORANGE, s=160, zorder=10,
               edgecolors="white", lw=2, marker="*",
               label="ESG-Optimal portfolio")

    # ── individual assets ─────────────────────────────────────────────────
    for i in range(n):
        col_pt = GREEN if active_mask[i] else BLUE
        ax.scatter(vols[i] * 100, mu[i] * 100,
                   color=col_pt, s=45, zorder=6,
                   edgecolors="white", lw=0.7, alpha=0.8)
        ax.annotate(names[i], (vols[i] * 100, mu[i] * 100),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=7, color=GREY)

    ax.set_xlabel("Std (%)", fontsize=9, color=GREY)
    ax.set_ylabel("Expected Return (%)", fontsize=9, color=GREY)
    ax.set_xlim(0, max(all_stds) + x_pad)
    ax.set_ylim(rf * 100 - y_pad, max(all_rets) + y_pad)
    ax.legend(fontsize=7, framealpha=0.9,
              facecolor=LEG_BG, edgecolor=LEG_ED, labelcolor=LABEL_C)
    _style_ax(ax, "Mean-Variance Frontier")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# GRAPH 2  —  ESG–Sharpe Ratio Frontier  (FIXED: arch curve + axis limits)
# ════════════════════════════════════════════════════════════════════════════
with _c2:
    # ── unconstrained (max-SR) tangency ──────────────────────────────────
    _w0    = np.ones(n) / n
    _ures  = minimize(lambda w: -port_sr(w, mu, cov, rf), _w0,
                      method="SLSQP", bounds=[(0., 1.)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                      options={"ftol": 1e-10, "maxiter": 800})
    _w_unc = _ures.x if _ures.success else _w0
    _sr_unc   = port_sr(_w_unc, mu, cov, rf)
    _esg_unc  = float(np.dot(_w_unc, esg_scores))   # ESG at peak of arch

    # ── ESG-screened tangency (only when screen is active) ───────────────
    _bounds_scr = [(0., 1.) if active_mask[i] else (0., 0.) for i in range(n)]
    _w_esgt, _ep_esgt, _sp_esgt, _sr_esgt = find_tangency(mu, cov, rf,
                                                            bounds=_bounds_scr)
    _esg_esgt   = float(np.dot(_w_esgt, esg_scores))
    _scr_differs = (abs(_esg_esgt - _esg_unc) > 0.05
                    or abs(_sr_esgt - _sr_unc) > 0.005)

    # ── Build the arch curve by sweeping ESG ≥ τ from min → max ──────────
    # The RIGHT-side (falling) branch: force ESG ≥ τ, maximise SR
    # The LEFT-side (rising) branch is the same problem in reverse: for each
    # τ below the unconstrained peak we ask "what is the max SR attainable
    # with ESG ≤ τ?"  Both sides are computed below.
    _esg_min  = float(np.min(esg_scores))
    _esg_max  = float(np.max(esg_scores)) * 0.999

    _esg_pts, _sr_pts = [], []

    # LEFT side  →  ESG ≤ τ  (rising toward peak)
    for _tau in np.linspace(_esg_min, _esg_unc, 45):
        try:
            _r = minimize(lambda w: -port_sr(w, mu, cov, rf), _w0,
                          method="SLSQP", bounds=[(0., 1.)] * n,
                          constraints=[
                              {"type": "eq",  "fun": lambda w: np.sum(w) - 1},
                              {"type": "ineq","fun": lambda w, t=_tau:
                               t - float(np.dot(w, esg_scores))},
                          ],
                          options={"ftol": 1e-9, "maxiter": 500})
            if _r.success and port_sd(_r.x, cov) > 1e-9:
                _esg_pts.append(float(np.dot(_r.x, esg_scores)))
                _sr_pts.append(port_sr(_r.x, mu, cov, rf))
        except Exception:
            continue

    # RIGHT side  →  ESG ≥ τ  (falling from peak)
    for _tau in np.linspace(_esg_unc, _esg_max, 55):
        try:
            _r = minimize(lambda w: -port_sr(w, mu, cov, rf), _w0,
                          method="SLSQP", bounds=[(0., 1.)] * n,
                          constraints=[
                              {"type": "eq",  "fun": lambda w: np.sum(w) - 1},
                              {"type": "ineq","fun": lambda w, t=_tau:
                               float(np.dot(w, esg_scores)) - t},
                          ],
                          options={"ftol": 1e-9, "maxiter": 500})
            if _r.success and port_sd(_r.x, cov) > 1e-9:
                _esg_pts.append(float(np.dot(_r.x, esg_scores)))
                _sr_pts.append(port_sr(_r.x, mu, cov, rf))
        except Exception:
            continue

    # sort left → right, deduplicate
    if _esg_pts:
        _pairs = sorted(set(zip(
            [round(x, 4) for x in _esg_pts],
            [round(s, 5) for s in _sr_pts]
        )))
        _esg_sorted = [p[0] for p in _pairs]
        _sr_sorted  = [p[1] for p in _pairs]
    else:
        _esg_sorted, _sr_sorted = [], []

    # ── individual asset Sharpe ratios ───────────────────────────────────
    _indiv_sr = (mu - rf) / np.maximum(vols, 1e-9)

    # ── user's ESG-optimal portfolio ──────────────────────────────────────
    _esg_opt  = float(np.dot(w_opt, esg_scores))

    # ── axis limits — driven by actual data, not forced 0–10 ─────────────
    _all_esg_x = (_esg_sorted + list(esg_scores)
                  + [_esg_unc, _esg_esgt, _esg_opt])
    _all_sr_y  = (_sr_sorted + list(_indiv_sr)
                  + [_sr_unc, sr]
                  + ([_sr_esgt] if _scr_differs else []))
    _x_lo = max(0,  min(_all_esg_x) - 0.4)
    _x_hi = min(10, max(_all_esg_x) + 0.4)
    _y_lo = min(_all_sr_y) - (max(_all_sr_y) - min(_all_sr_y)) * 0.22
    _y_hi = max(_all_sr_y) + (max(_all_sr_y) - min(_all_sr_y)) * 0.28

    fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
    fig2.patch.set_facecolor(CHART_BG)
    ax2.set_facecolor(CHART_BG)

    # ── arch frontier curve ───────────────────────────────────────────────
    if len(_esg_sorted) >= 2:
        ax2.plot(_esg_sorted, _sr_sorted,
                 color=GREEN, lw=2.4, zorder=4,
                 label="ESG–SR frontier")
        ax2.fill_between(_esg_sorted, max(0, _y_lo), _sr_sorted,
                         alpha=0.07, color=GREEN, zorder=2)

    # ── individual assets ─────────────────────────────────────────────────
    _ann_offsets = [(6, 6), (6, -16), (-60, 6), (-60, -16),
                    (6, 16), (-60, 16), (14, -4)]
    for _i in range(n):
        _col_i = GREEN if active_mask[_i] else GREY
        ax2.scatter(esg_scores[_i], _indiv_sr[_i],
                    color=_col_i, s=55, zorder=6,
                    edgecolors="white", lw=0.8, alpha=0.9)
        _ofs = _ann_offsets[_i % len(_ann_offsets)]
        ax2.annotate(names[_i], (esg_scores[_i], _indiv_sr[_i]),
                     textcoords="offset points", xytext=_ofs,
                     fontsize=7, color=GREY)

    # ── unconstrained tangency (blue diamond) ─────────────────────────────
    ax2.scatter(_esg_unc, _sr_unc,
                color=BLUE, s=140, zorder=9,
                edgecolors="white", lw=1.5, marker="D",
                label=f"Tangency portfolio\nignoring ESG (SR={_sr_unc:.3f})")
    ax2.annotate(f"Tangency portfolio\nignoring ESG information\nSR = {_sr_unc:.3f}",
                 (_esg_unc, _sr_unc),
                 textcoords="offset points", xytext=(8, 10),
                 fontsize=7, color=BLUE, fontstyle="italic",
                 bbox=dict(boxstyle="round,pad=0.25", fc=CHART_BG,
                           ec=BLUE, alpha=0.85, lw=0.6))

    # ── ESG-screened tangency (green star) ───────────────────────────────
    if _sp_esgt > 1e-9 and _scr_differs:
        ax2.scatter(_esg_esgt, _sr_esgt,
                    color=GREEN, s=170, zorder=10,
                    edgecolors="white", lw=2, marker="*",
                    label=f"Tangency portfolio\nusing ESG information (SR={_sr_esgt:.3f})")
        ax2.annotate(f"Tangency portfolio\nusing ESG information\nSR = {_sr_esgt:.3f}",
                     (_esg_esgt, _sr_esgt),
                     textcoords="offset points", xytext=(8, -32),
                     fontsize=7, color=GREEN, fontstyle="italic",
                     bbox=dict(boxstyle="round,pad=0.25", fc=CHART_BG,
                               ec=GREEN, alpha=0.85, lw=0.6))

    # ── user's ESG-optimal portfolio (orange star) ────────────────────────
    ax2.scatter(_esg_opt, sr,
                color=ORANGE, s=180, zorder=11,
                edgecolors="white", lw=2, marker="*",
                label=f"Your portfolio (SR={sr:.3f})")
    _ann_x = 10 if _esg_opt < (_x_lo + _x_hi) / 2 else -95
    ax2.annotate(f"Your portfolio\nSR = {sr:.3f}",
                 (_esg_opt, sr),
                 textcoords="offset points", xytext=(_ann_x, -24),
                 fontsize=7, color=ORANGE, fontstyle="italic",
                 bbox=dict(boxstyle="round,pad=0.25", fc=CHART_BG,
                           ec=ORANGE, alpha=0.85, lw=0.6))

    ax2.set_xlim(_x_lo, _x_hi)
    ax2.set_ylim(_y_lo, _y_hi)
    ax2.set_xlabel("ESG Score (0–10)", fontsize=9, color=GREY)
    ax2.set_ylabel("Sharpe Ratio", fontsize=9, color=GREY)
    ax2.legend(fontsize=7, framealpha=0.92,
               facecolor=LEG_BG, edgecolor=LEG_ED, labelcolor=LABEL_C,
               loc="upper left" if _esg_unc > (_x_lo + _x_hi) / 2 else "upper right")
    _style_ax(ax2, "ESG–Sharpe Frontier")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()
