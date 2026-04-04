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
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Apple Design System (iOS 18 / macOS 15 Sequoia) ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ══════════════════════════════════════════════
   APPLE DESIGN TOKENS
   ══════════════════════════════════════════════ */
:root {
  /* System Background Layers */
  --sys-bg:          #000000;
  --sys-bg-2:        #1C1C1E;
  --sys-bg-3:        #2C2C2E;
  --sys-bg-4:        #3A3A3C;

  /* Frosted glass */
  --glass-heavy:     rgba(22, 22, 24, 0.92);
  --glass-mid:       rgba(44, 44, 46, 0.80);
  --glass-light:     rgba(255, 255, 255, 0.05);
  --glass-hover:     rgba(255, 255, 255, 0.08);

  /* Apple System Colors — Dark Mode */
  --blue:            #0A84FF;
  --blue-bg:         rgba(10, 132, 255, 0.14);
  --blue-border:     rgba(10, 132, 255, 0.32);
  --green:           #30D158;
  --green-bg:        rgba(48, 209, 88, 0.12);
  --green-border:    rgba(48, 209, 88, 0.28);
  --orange:          #FF9F0A;
  --orange-bg:       rgba(255, 159, 10, 0.12);
  --orange-border:   rgba(255, 159, 10, 0.28);
  --red:             #FF453A;
  --red-bg:          rgba(255, 69, 58, 0.12);
  --red-border:      rgba(255, 69, 58, 0.28);

  /* Label Hierarchy */
  --label-1:  rgba(255, 255, 255, 0.92);
  --label-2:  rgba(255, 255, 255, 0.54);
  --label-3:  rgba(255, 255, 255, 0.30);
  --label-4:  rgba(255, 255, 255, 0.16);

  /* Separators */
  --sep:      rgba(60, 60, 67, 0.40);
  --sep-lt:   rgba(255, 255, 255, 0.08);
  --sep-md:   rgba(255, 255, 255, 0.13);

  /* Border Radius */
  --r-xs:   6px;
  --r-sm:   10px;
  --r-md:   13px;
  --r-lg:   16px;
  --r-xl:   20px;
  --r-2xl:  28px;
  --r-pill: 100px;

  /* SF Pro font stack */
  --font: -apple-system, 'SF Pro Display', 'SF Pro Text',
          BlinkMacSystemFont, 'Helvetica Neue', 'Inter', Arial, sans-serif;
}

/* ══════════════════════════════════════════════
   BASE
   ══════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: var(--font) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background: var(--sys-bg) !important;
  font-size: 14px;
  line-height: 1.5;
}

.stApp { background: var(--sys-bg) !important; }

.block-container {
  padding: 0 2.5rem 8rem !important;
  max-width: 1280px !important;
}

/* ══════════════════════════════════════════════
   SIDEBAR — macOS Settings Panel
   ══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: rgba(18, 18, 20, 0.97) !important;
  border-right: 0.5px solid var(--sep-lt) !important;
}

[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: var(--label-2) !important; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  color: var(--label-3) !important;
}

[data-testid="stSidebar"] hr {
  border: none !important;
  border-top: 0.5px solid var(--sep-lt) !important;
  margin: 14px 0 !important;
}

[data-testid="stSidebar"] label {
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.05em !important;
  color: var(--label-3) !important;
  text-transform: uppercase !important;
}

/* Slider */
[data-testid="stSidebar"] .stSlider [role="slider"] {
  background: var(--blue) !important;
  border: none !important;
  width: 18px !important;
  height: 18px !important;
  box-shadow: 0 2px 8px rgba(10,132,255,0.45), 0 0 0 1px rgba(10,132,255,0.5) !important;
}
[data-testid="stSidebar"] .stSlider p {
  font-size: 15px !important;
  font-weight: 500 !important;
  color: var(--label-1) !important;
}

/* Number input */
[data-testid="stSidebar"] .stNumberInput input {
  background: var(--sys-bg-4) !important;
  border: 0.5px solid var(--sep) !important;
  border-radius: var(--r-sm) !important;
  color: var(--label-1) !important;
  font-size: 14px !important;
  padding: 9px 12px !important;
}

/* Checkbox */
[data-testid="stSidebar"] .stCheckbox label {
  font-size: 13px !important;
  color: var(--label-2) !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  font-weight: 400 !important;
}

[data-testid="stSidebar"] small,
[data-testid="stSidebar"] p {
  font-size: 12px !important;
  color: var(--label-3) !important;
}

/* ══════════════════════════════════════════════
   TYPOGRAPHY
   ══════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 {
  color: var(--label-1) !important;
  letter-spacing: -0.025em !important;
}
p, div, label, span { color: var(--label-2); }
strong, b { color: var(--label-1) !important; font-weight: 600 !important; }

/* ══════════════════════════════════════════════
   SIDEBAR BRAND
   ══════════════════════════════════════════════ */
.sb-brand {
  padding: 28px 20px 22px;
  border-bottom: 0.5px solid var(--sep-lt);
}
.sb-logo-row {
  display: flex;
  align-items: center;
  gap: 13px;
}
.sb-icon {
  width: 40px;
  height: 40px;
  background: linear-gradient(145deg, #0d5a2a, #30D158);
  border-radius: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 4px 16px rgba(48,209,88,0.30);
}
.sb-name {
  font-size: 19px !important;
  font-weight: 700 !important;
  letter-spacing: -0.045em !important;
  color: var(--label-1) !important;
  line-height: 1.1 !important;
}
.sb-sub {
  font-size: 10px !important;
  color: var(--label-3) !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  margin-top: 3px !important;
}

/* ══════════════════════════════════════════════
   PAGE HEADER
   ══════════════════════════════════════════════ */
.pg-header {
  padding: 64px 0 52px;
}
.pg-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: var(--green-bg);
  border: 0.5px solid var(--green-border);
  border-radius: var(--r-pill);
  padding: 5px 14px;
  margin-bottom: 24px;
  font-size: 12px;
  font-weight: 600;
  color: var(--green) !important;
  letter-spacing: 0.03em;
}
.pg-title {
  font-size: 62px !important;
  font-weight: 700 !important;
  letter-spacing: -0.058em !important;
  line-height: 1.02 !important;
  color: var(--label-1) !important;
  margin-bottom: 18px !important;
}
.pg-subtitle {
  font-size: 17px !important;
  font-weight: 400 !important;
  color: var(--label-2) !important;
  line-height: 1.65 !important;
  letter-spacing: -0.01em !important;
  max-width: 520px !important;
}

/* ══════════════════════════════════════════════
   SECTION HEADERS
   ══════════════════════════════════════════════ */
.section-header {
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--label-3) !important;
  padding-bottom: 12px !important;
  border-bottom: 0.5px solid var(--sep-lt) !important;
  margin: 48px 0 24px !important;
}

/* ══════════════════════════════════════════════
   FROSTED GLASS METRIC CARDS
   ══════════════════════════════════════════════ */
.metric-card {
  background: var(--glass-heavy);
  border: 0.5px solid var(--sep-lt);
  border-radius: var(--r-xl);
  padding: 22px 22px 18px;
  position: relative;
  overflow: hidden;
  transition: background 0.2s, border-color 0.2s, transform 0.18s cubic-bezier(0.4,0,0.2,1);
  margin-bottom: 12px;
}
.metric-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; height: 0.5px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}
.metric-card:hover {
  background: var(--glass-mid);
  border-color: var(--sep-md);
  transform: translateY(-2px);
}
.metric-label {
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--label-3) !important;
  margin-bottom: 10px !important;
}
.metric-value {
  font-size: 40px !important;
  font-weight: 300 !important;
  letter-spacing: -0.06em !important;
  line-height: 1 !important;
  color: var(--label-1) !important;
  font-variant-numeric: tabular-nums !important;
}
.metric-unit {
  font-size: 17px !important;
  font-weight: 400 !important;
  color: var(--label-3) !important;
  margin-left: 2px !important;
}

/* ══════════════════════════════════════════════
   STATUS BOXES
   ══════════════════════════════════════════════ */
.info-box {
  background: var(--blue-bg) !important;
  border: 0.5px solid var(--blue-border) !important;
  border-radius: var(--r-md) !important;
  padding: 12px 16px !important;
  font-size: 13px !important;
  color: rgba(120, 190, 255, 0.9) !important;
  line-height: 1.6 !important;
  margin: 8px 0 !important;
}
.warn-box {
  background: var(--orange-bg) !important;
  border: 0.5px solid var(--orange-border) !important;
  border-radius: var(--r-md) !important;
  padding: 12px 16px !important;
  font-size: 13px !important;
  color: rgba(255, 185, 80, 0.9) !important;
  line-height: 1.6 !important;
  margin: 8px 0 !important;
}
.error-box {
  background: var(--red-bg) !important;
  border: 0.5px solid var(--red-border) !important;
  border-radius: var(--r-md) !important;
  padding: 12px 16px !important;
  font-size: 13px !important;
  color: rgba(255, 110, 100, 0.9) !important;
  line-height: 1.6 !important;
  margin: 8px 0 !important;
}

/* ══════════════════════════════════════════════
   BUTTONS — Apple Design
   ══════════════════════════════════════════════ */
div.stButton > button {
  background: var(--glass-light) !important;
  border: 0.5px solid var(--sep-md) !important;
  border-radius: var(--r-pill) !important;
  color: var(--label-1) !important;
  font-family: var(--font) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  letter-spacing: -0.01em !important;
  padding: 10px 22px !important;
  transition: all 0.15s cubic-bezier(0.4,0,0.2,1) !important;
  width: 100% !important;
}
div.stButton > button:hover {
  background: var(--glass-hover) !important;
  border-color: rgba(255,255,255,0.22) !important;
  transform: scale(1.015) !important;
}
div.stButton > button:active {
  transform: scale(0.96) !important;
  opacity: 0.65 !important;
}

/* Primary CTA — Apple Blue */
div[data-testid="stHorizontalBlock"] div.stButton > button {
  background: var(--blue) !important;
  border: none !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  font-size: 15px !important;
  padding: 13px 30px !important;
  box-shadow: 0 4px 20px rgba(10,132,255,0.38) !important;
  letter-spacing: -0.01em !important;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
  box-shadow: 0 6px 28px rgba(10,132,255,0.52) !important;
  transform: scale(1.02) !important;
}

/* ══════════════════════════════════════════════
   INPUTS
   ══════════════════════════════════════════════ */
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
  background: var(--sys-bg-3) !important;
  border: 0.5px solid var(--sep) !important;
  border-radius: var(--r-sm) !important;
  color: var(--label-1) !important;
  font-family: var(--font) !important;
  font-size: 14px !important;
  padding: 10px 14px !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stNumberInput input:focus,
.stTextInput input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(10,132,255,0.14) !important;
  outline: none !important;
}
.stSelectbox div[data-baseweb="select"] > div {
  background: var(--sys-bg-3) !important;
  border: 0.5px solid var(--sep) !important;
  border-radius: var(--r-sm) !important;
  color: var(--label-1) !important;
}
.stRadio label { color: var(--label-2) !important; }
.stRadio div[role="radiogroup"] label {
  font-size: 14px !important;
  color: var(--label-2) !important;
  font-weight: 400 !important;
}
.stCheckbox div[data-testid="stMarkdownContainer"] p {
  font-size: 14px !important;
  color: var(--label-2) !important;
}

/* ══════════════════════════════════════════════
   DATAFRAMES
   ══════════════════════════════════════════════ */
.stDataFrame, [data-testid="stDataEditor"] {
  border-radius: var(--r-lg) !important;
  overflow: hidden !important;
  border: 0.5px solid var(--sep-lt) !important;
}
table {
  color: var(--label-1) !important;
  border-collapse: collapse;
  width: 100%;
}
thead tr th {
  background: rgba(22, 22, 24, 0.97) !important;
  color: var(--label-3) !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border-bottom: 0.5px solid var(--sep-lt) !important;
  padding: 12px 16px !important;
}
tbody tr td {
  color: var(--label-2) !important;
  border-bottom: 0.5px solid rgba(255,255,255,0.04) !important;
  padding: 11px 16px !important;
  font-size: 13px !important;
}
tbody tr:hover td { background: rgba(255,255,255,0.025) !important; }

/* ══════════════════════════════════════════════
   EXPANDER
   ══════════════════════════════════════════════ */
[data-testid="stExpander"] {
  background: var(--glass-heavy) !important;
  border: 0.5px solid var(--sep-lt) !important;
  border-radius: var(--r-lg) !important;
}
[data-testid="stExpander"] summary {
  color: var(--label-2) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  padding: 16px 20px !important;
  letter-spacing: -0.01em !important;
}
[data-testid="stExpander"] p {
  color: var(--label-2) !important;
  line-height: 1.72 !important;
  font-size: 14px !important;
}

/* ══════════════════════════════════════════════
   DIVIDER
   ══════════════════════════════════════════════ */
hr {
  border: none !important;
  border-top: 0.5px solid var(--sep-lt) !important;
  margin: 2.5rem 0 !important;
}

/* ══════════════════════════════════════════════
   CHAT — iMessage / Apple Messages Design
   ══════════════════════════════════════════════ */
.chat-shell {
  background: var(--glass-heavy);
  border: 0.5px solid var(--sep-lt);
  border-radius: var(--r-2xl);
  overflow: hidden;
  margin-top: 20px;
}

/* Title bar — macOS Messages window style */
.chat-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  background: rgba(22, 22, 24, 0.96);
  border-bottom: 0.5px solid var(--sep-lt);
}
.chat-topbar-left { display: flex; align-items: center; gap: 12px; }
.chat-avatar {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  background: linear-gradient(145deg, #0c5425, #30D158);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 700;
  color: #fff !important;
  flex-shrink: 0;
  box-shadow: 0 2px 10px rgba(48,209,88,0.30);
}
.chat-name {
  font-size: 15px !important;
  font-weight: 600 !important;
  color: var(--label-1) !important;
  letter-spacing: -0.025em !important;
  line-height: 1.2 !important;
}
.chat-status-text {
  font-size: 11px !important;
  color: var(--label-3) !important;
  margin-top: 2px !important;
}
.live-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--green-bg);
  border: 0.5px solid var(--green-border);
  border-radius: var(--r-pill);
  padding: 4px 12px;
  font-size: 11px;
  font-weight: 600;
  color: var(--green) !important;
  letter-spacing: 0.04em;
}
.live-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px rgba(48,209,88,0.8);
  display: inline-block;
}

/* Messages scroll area */
.chat-messages {
  max-height: 480px;
  overflow-y: auto;
  padding: 24px 20px 16px;
  scrollbar-width: thin;
  scrollbar-color: rgba(255,255,255,0.08) transparent;
}
.chat-messages::-webkit-scrollbar { width: 3px; }
.chat-messages::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.08);
  border-radius: 3px;
}

/* Empty state */
.chat-empty {
  text-align: center;
  padding: 52px 24px;
}
.chat-empty-icon {
  font-size: 34px;
  display: block;
  margin-bottom: 14px;
  opacity: 0.22;
}
.chat-empty-title {
  font-size: 15px !important;
  font-weight: 500 !important;
  color: var(--label-2) !important;
  letter-spacing: -0.015em !important;
  margin-bottom: 6px !important;
}
.chat-empty-sub {
  font-size: 13px !important;
  color: var(--label-3) !important;
  line-height: 1.58 !important;
}

/* Bubble rows */
.msg-row-user {
  display: flex;
  justify-content: flex-end;
  margin: 4px 0 10px;
}
.msg-row-bot {
  display: flex;
  justify-content: flex-start;
  align-items: flex-end;
  gap: 8px;
  margin: 4px 0 10px;
}
.bot-mini-avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: linear-gradient(145deg, #0c5425, #30D158);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 9px;
  font-weight: 700;
  color: white !important;
  flex-shrink: 0;
  margin-bottom: 2px;
}

/* iMessage bubble — user (blue) */
.bubble-u {
  background: var(--blue);
  color: #ffffff !important;
  border-radius: 20px 20px 5px 20px;
  padding: 10px 16px;
  max-width: 70%;
  font-size: 14px;
  font-weight: 400;
  line-height: 1.5;
}

/* iMessage bubble — bot (dark gray) */
.bubble-b {
  background: var(--sys-bg-3);
  border: 0.5px solid var(--sep-lt);
  color: var(--label-2) !important;
  border-radius: 20px 20px 20px 5px;
  padding: 12px 16px;
  max-width: 84%;
  font-size: 13px;
  line-height: 1.78;
}

/* Quick reply chips bar */
.chat-chips {
  padding: 12px 20px 10px;
  border-top: 0.5px solid var(--sep-lt);
  background: rgba(22, 22, 24, 0.65);
}
.chat-chips-label {
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.09em !important;
  text-transform: uppercase !important;
  color: var(--label-4) !important;
  margin-bottom: 10px !important;
}

/* Suggestion chip buttons — Apple Quick Actions style */
.chat-chips div.stButton > button {
  background: var(--blue-bg) !important;
  border: 0.5px solid var(--blue-border) !important;
  color: var(--blue) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  padding: 7px 14px !important;
  border-radius: var(--r-pill) !important;
  text-align: left !important;
  white-space: normal !important;
  line-height: 1.4 !important;
  letter-spacing: 0 !important;
  transition: all 0.12s cubic-bezier(0.4,0,0.2,1) !important;
}
.chat-chips div.stButton > button:hover {
  background: rgba(10,132,255,0.22) !important;
  border-color: rgba(10,132,255,0.48) !important;
  transform: scale(1.02) !important;
}

/* Input compose bar */
.chat-input-bar {
  padding: 12px 16px 16px;
  border-top: 0.5px solid var(--sep-lt);
  background: rgba(22, 22, 24, 0.80);
}
.chat-input-bar .stTextInput input {
  border-radius: var(--r-pill) !important;
  background: var(--sys-bg-4) !important;
  border: 0.5px solid var(--sep) !important;
  padding: 12px 20px !important;
  font-size: 14px !important;
}
.chat-input-bar .stTextInput input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(10,132,255,0.13) !important;
}

/* Send button — Apple Blue */
.chat-input-bar div.stButton > button {
  background: var(--blue) !important;
  border: none !important;
  color: #fff !important;
  border-radius: var(--r-pill) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 11px 22px !important;
  box-shadow: 0 2px 14px rgba(10,132,255,0.32) !important;
  letter-spacing: 0 !important;
  width: 100% !important;
}
.chat-input-bar div.stButton > button:hover {
  box-shadow: 0 4px 22px rgba(10,132,255,0.50) !important;
  transform: scale(1.03) !important;
}

/* ══════════════════════════════════════════════
   FOOTER — Apple-style minimal
   ══════════════════════════════════════════════ */
.site-footer {
  margin-top: 80px;
  padding: 28px 0;
  border-top: 0.5px solid var(--sep-lt);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}
.site-footer span {
  font-size: 12px !important;
  color: var(--label-3) !important;
}
.site-footer a {
  color: var(--blue) !important;
  text-decoration: none !important;
  font-size: 12px !important;
}
.footer-dot {
  display: inline-block;
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: var(--label-4);
  margin: 0 6px;
  vertical-align: middle;
}

/* ══════════════════════════════════════════════
   SCROLLBARS
   ══════════════════════════════════════════════ */
* { scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.08) transparent; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 4px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ESG DATABASE — loaded from the uploaded LSEG CSV
# ══════════════════════════════════════════════════════════════════════════════
_ESG_CSV_URL = (
    "https://raw.githubusercontent.com/minas120983-alt/lets-see/main/ESG%20data%202026.csv"
)
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
            "letter":  str(row["value"]),
            "year":    int(row["year"]),
            "source":  f"LSEG ESGCombinedScore ({int(row['year'])})",
            "has_esg": True,
        }
        for _, row in latest.iterrows()
    }


@st.cache_data(show_spinner=False)
def load_esg_db() -> dict:
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
    n = len(mu)
    b = bounds or [(0.,1.)]*n
    w_mv = _minimise_sd(mu, cov, bounds=b)
    ret_min = port_ret(w_mv, mu)
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
# CHATBOT
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

    if any(k in q for k in ["weight", "allocat", "holding", "position",
                              "why does my portfolio", "why hold", "why so much",
                              "why is my", "why is there", "what drives"]):
        top = by_w[0]
        top_drivers = []
        if ind_sr[top] == max(ind_sr): top_drivers.append("the best risk-adjusted return")
        if esg_scores[top] == max(esg_scores): top_drivers.append("the highest ESG score")
        if vols[top] == min(vols): top_drivers.append("the lowest volatility")
        driver_str = " and ".join(top_drivers) if top_drivers else "a strong combination of return, risk, and ESG"
        if len(held) >= 2:
            top2 = held[:2]
            rho = cov[top2[0],top2[1]] / max(vols[top2[0]]*vols[top2[1]], 1e-12)
            div_note = (
                f"{names[top2[1]]} ({p1(w_opt[top2[1]])}) complements it well "
                f"with a correlation of only {rho:.2f} — that low correlation is doing real "
                f"diversification work, pulling portfolio volatility down to {p(sp)}."
                if abs(rho) < 0.5 else
                f"{names[top2[1]]} ({p1(w_opt[top2[1]])}) has a correlation of {rho:.2f} "
                f"with {names[top]} — moderately correlated, so diversification benefit is limited."
            )
        else:
            div_note = "With only one asset held at meaningful weight, there is no diversification benefit."
        zero_held = [i for i in range(n) if w_opt[i] <= 0.005]
        zero_parts = []
        for i in zero_held[:2]:
            if not active_mask[i]:
                zero_parts.append(f"{names[i]} is excluded entirely by the ESG screen "
                                  f"(score {esg_scores[i]:.1f}/10 < threshold {esg_thresh:.1f})")
            else:
                better = [j for j in held if ind_sr[j] > ind_sr[i] and esg_scores[j] >= esg_scores[i]]
                if better:
                    zero_parts.append(f"{names[i]} receives zero weight because {names[better[0]]} "
                                      f"offers superior risk-adjusted return ({ind_sr[better[0]]:.2f} vs "
                                      f"{ind_sr[i]:.2f}) with equal or better ESG quality")
                else:
                    zero_parts.append(f"{names[i]} is zeroed out because its Sharpe of {ind_sr[i]:.2f} "
                                      f"and ESG of {esg_scores[i]:.1f}/10 add insufficient marginal utility")
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
            f"{'is materially tilting allocation toward high-ESG names' if lam > 2 else 'is adding a modest ESG tilt' if lam > 0.5 else 'is having minimal impact on allocation'}.",
        ]
        return "\n".join(lines)

    if any(k in q for k in ["cost", "sacrifice", "give up", "lose", "penalty",
                              "esg constraint", "esg screen", "tradeoff",
                              "trade-off", "return am i sacrific", "how much return"]):
        if sharpe_cost < 0.02:
            cost_verdict = (
                f"The ESG constraint here is almost free. "
                f"You are giving up just {sharpe_cost:.4f} Sharpe ratio points — "
                f"statistically indistinguishable from noise. This is the ideal situation: "
                f"good ESG and essentially no financial cost."
            )
        elif sharpe_cost < 0.08:
            cost_verdict = (
                f"The ESG constraint costs {sharpe_cost:.3f} Sharpe ratio points, "
                f"roughly {ret_cost_ann:.1f}% in expected annual return at the tangency level. "
                f"Real but manageable — broadly in line with academic research on ESG screens."
            )
        elif sharpe_cost < 0.20:
            cost_verdict = (
                f"The ESG constraint is inflicting genuine financial pain: "
                f"{sharpe_cost:.3f} Sharpe ratio points lost, {ret_cost_ann:.1f}% "
                f"in expected annual return foregone. Consider relaxing the minimum threshold."
            )
        else:
            cost_verdict = (
                f"This ESG constraint is extremely expensive: {sharpe_cost:.3f} Sharpe points lost. "
                f"At this level you are likely excluding your best-performing assets. "
                f"Strongly consider relaxing the threshold from {esg_thresh:.1f}."
            )
        if excluded:
            excl_names = [f"{names[i]} (SR={ind_sr[i]:.2f}, ESG={esg_scores[i]:.1f}/10)"
                          for i in excluded]
            cost_verdict += f"\n\nExcluded assets: {', '.join(excl_names)}."
        return cost_verdict

    if any(k in q for k in ["sharpe", "risk-adjust", "risk adjusted", "is mine good",
                              "how good", "how is my", "rate my", "assess"]):
        verdict = sr_band(sr)
        gap_pct = (sr_tan_esg - sr) / sr_tan_esg * 100 if sr_tan_esg > 0 else 0
        if gap_pct < 3:
            position = (f"Your Sharpe of {sr:.3f} is essentially at the ESG-efficient frontier — "
                        f"within {gap_pct:.1f}% of the maximum achievable. Excellent portfolio construction.")
        elif gap_pct < 12:
            position = (f"Your Sharpe of {sr:.3f} sits {gap_pct:.1f}% below the ESG tangency ({sr_tan_esg:.3f}). "
                        f"This gap is driven by your ESG preference λ={lam} — a deliberate, rational tradeoff.")
        else:
            position = (f"Your Sharpe of {sr:.3f} is {gap_pct:.1f}% below the ESG tangency ({sr_tan_esg:.3f}). "
                        f"Your ESG preference is materially overriding financial efficiency. Consider reducing λ.")
        dominated = [i for i in range(n) if ind_sr[i] > sr and active_mask[i]]
        if dominated:
            dom_str = (f"\n\nInterestingly, {names[dominated[0]]} has a higher individual Sharpe "
                       f"({ind_sr[dominated[0]]:.3f}) than your portfolio — a consequence of your constraints.")
        else:
            dom_str = (f"\n\nYour portfolio Sharpe ({sr:.3f}) exceeds all individual asset Sharpe ratios — "
                       f"diversification is working correctly.")
        return f"Your Sharpe ratio of {sr:.3f} is {verdict} by typical standards. {position}{dom_str}"

    if any(k in q for k in ["utility", "objective", "formula", "model",
                              "how does it work", "explain the", "u ="]):
        fin_part = ep - gamma/2*sp**2
        esg_part = lam * esg_bar
        esg_pct  = esg_part / u_val * 100 if u_val != 0 else 0
        return (
            f"The model maximises:\n\n"
            f"  U = E[Rp]  −  (γ/2)·σ²  +  λ·ESG\n\n"
            f"Three forces: return reward (E[R]={ep*100:.2f}%), variance penalty "
            f"(γ={gamma}, {gamma_label(gamma)}), and ESG reward (λ={lam}, {lam_label(lam)}).\n\n"
            f"The ESG term contributes {esg_pct:.0f}% of total utility. "
            f"{'ESG is dominant — sustainability preferences are driving construction.' if esg_pct > 40 else 'Financial terms dominate — ESG is a tilt.' if esg_pct < 20 else 'Return, risk, and ESG are roughly balanced.'}"
        )

    if any(k in q for k in ["frontier", "right of", "why is the esg",
                              "efficient frontier", "two curve"]):
        return (
            f"The ESG frontier sits to the right of the unconstrained frontier — a mathematical "
            f"certainty. Adding any constraint can only reduce or maintain efficiency, never improve it.\n\n"
            f"To achieve the same expected return, the ESG-screened portfolio must accept higher volatility "
            f"because it cannot hold the same assets. The cost here is {sharpe_cost:.3f} Sharpe points "
            f"at the tangency — "
            f"{'essentially free.' if sharpe_cost < 0.02 else 'modest but real.' if sharpe_cost < 0.10 else 'significant and worth scrutinising.'}"
        )

    if any(k in q for k in ["risk aversion", "gamma", "aversion", "risk toleran"]):
        vol_penalty = gamma/2 * sp**2 * 100
        if gamma >= 6:
            return (f"At γ={gamma} you are highly risk-averse. The model subtracts "
                    f"{vol_penalty:.2f}% from utility as a variance penalty, "
                    f"pulling weight heavily toward the lowest-volatility assets.")
        elif gamma >= 3:
            return (f"At γ={gamma} you are moderately risk-averse — consistent with a long-term "
                    f"institutional investor. The variance penalty of {vol_penalty:.2f}% is meaningful "
                    f"but not dominant; the allocation balances return and risk management.")
        else:
            return (f"At γ={gamma} you are relatively risk-tolerant. The model barely penalises "
                    f"variance, so weights are driven almost entirely by expected return and ESG score. "
                    f"Check whether {p(sp)} portfolio volatility fits your investment horizon.")

    if any(k in q for k in ["lambda", "λ", "esg preference", "what does lambda",
                              "should i increase", "should i decrease"]):
        bp_equiv = lam / 10 * 100
        esg_pct = lam*esg_bar / abs(u_val) * 100 if u_val != 0 else 0
        return (
            f"λ={lam} means each 1-point ESG improvement is worth {bp_equiv:.0f}bp of expected return. "
            f"The ESG term contributes {esg_pct:.0f}% of total utility. "
            f"{'ESG is the dominant driver — consider whether this reflects your actual preferences.' if esg_pct > 35 else 'ESG influences but does not dominate allocation.' if esg_pct > 15 else 'At λ=' + str(lam) + ' the ESG term is a mild tilt. Raising λ gives it more influence.'}"
        )

    if any(k in q for k in ["drag", "esg score", "worst esg", "lowest esg", "which asset"]):
        held_by_esg = sorted([i for i in held], key=lambda i: esg_scores[i])
        if not held_by_esg:
            return "No assets are held at meaningful weight."
        worst = held_by_esg[0]
        best  = held_by_esg[-1]
        return (
            f"{names[worst]} is the biggest ESG drag — score {esg_scores[worst]:.2f}/10 "
            f"({esg_label(esg_scores[worst])}) at {p1(w_opt[worst])} weight.\n\n"
            f"Removing it would improve portfolio ESG toward {names[best]}'s level "
            f"({esg_scores[best]:.2f}/10). However its Sharpe of {ind_sr[worst]:.3f} "
            f"{'is above portfolio Sharpe — removing it would hurt risk-adjusted return.' if ind_sr[worst] > sr else 'is below portfolio Sharpe — from a financial standpoint it is already a marginal holding.'}"
        )

    if any(k in q for k in ["without esg", "no esg", "remove esg", "what if", "what would"]):
        if sharpe_cost < 0.03:
            return (f"Practically nothing. Removing the ESG screen would gain {sharpe_cost:.4f} "
                    f"Sharpe points — well within estimation error. Your ESG constraints are essentially free.")
        else:
            return (
                f"Removing all ESG constraints would deliver Sharpe {sr_tan_all:.3f} vs your {sr:.3f} — "
                f"a gain of {sr_tan_all-sr:.3f} points, roughly {ret_cost_ann:.1f}% more expected return annually. "
                f"{'A substantial cost that most mandates would struggle to justify.' if ret_cost_ann > 1.5 else 'An acceptable cost for a genuine ESG mandate.'}"
            )

    mentioned = [i for i in range(n) if names[i].lower() in q]
    if mentioned:
        i = mentioned[0]
        held_str = (f"Holds {p1(w_opt[i])} of portfolio." if w_opt[i] > 0.005 else
                    f"Zero weight — " + ("excluded by ESG screen." if not active_mask[i]
                                         else "dominated by better risk/return/ESG combinations."))
        return (f"{names[i]}: {held_str}\n\n"
                f"Sharpe: {ind_sr[i]:.3f} ({sr_band(ind_sr[i])}), "
                f"ESG: {esg_scores[i]:.2f}/10 ({esg_label(esg_scores[i])}), "
                f"E[R]: {p(mu[i])}, σ: {p(vols[i])}.")

    lines = [
        f"Portfolio: {len(held)} assets — "
        f"{', '.join(names[i] + ' (' + p1(w_opt[i]) + ')' for i in by_w if w_opt[i]>0.005)}.\n",
        f"E[R]={p(ep)}, σ={p(sp)}, Sharpe={sr:.3f}, ESG={esg_bar:.2f}/10.\n",
        f"ESG constraint cost: {sharpe_cost:.3f} Sharpe points — "
        f"{'essentially free.' if sharpe_cost < 0.02 else 'real but manageable.' if sharpe_cost < 0.08 else 'significant.'}\n",
        f"Ask me about weights, ESG costs, Sharpe ratio, the utility function, "
        f"the frontier, risk aversion, lambda, or any specific asset by name.",
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
    # ── GreenPort logo — leaf + chart SVG ────────────────────────────────────
    st.markdown('''
<div class="sb-brand">
  <div class="sb-logo-row">
    <div class="sb-icon">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 3.5C7.8 3.5 4.5 7 4.5 11.2C4.5 15.1 7.5 18 11.5 18.3C11.5 18.3 11.5 14.5 14.8 11.8C18 9.1 20.5 8.5 20.5 8.5C20.5 8.5 18.8 3.5 12 3.5Z" fill="white" opacity="0.95"/>
        <path d="M11.5 18.3L11.5 22" stroke="white" stroke-width="1.6" stroke-linecap="round" opacity="0.55"/>
        <path d="M3 21L6.5 17.5" stroke="white" stroke-width="1.4" stroke-linecap="round" opacity="0.35"/>
      </svg>
    </div>
    <div>
      <div class="sb-name">GreenPort</div>
      <div class="sb-sub">ESG Optimiser</div>
    </div>
  </div>
</div>
''', unsafe_allow_html=True)

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
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('''
<div class="pg-header">
  <div class="pg-pill">
    <svg width="7" height="7" viewBox="0 0 7 7"><circle cx="3.5" cy="3.5" r="3.5" fill="#30D158"/></svg>
    ESG Portfolio Optimisation
  </div>
  <div class="pg-title">GreenPort</div>
  <div class="pg-subtitle">
    Institutional-grade mean-variance optimisation with ESG constraints.
    ECN316 Sustainable Finance &nbsp;·&nbsp; LSEG Data
  </div>
</div>
''', unsafe_allow_html=True)

if _ESG_DB:
    st.markdown(
        f'<div class="info-box"><strong>{len(_ESG_DB):,} tickers</strong> loaded — '
        f'LSEG ESGCombinedScore, most recent year per ticker, scaled 0–10.</div>',
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

        manual_overrides = {}
        manual_ret_vol   = {}

        if bad_tickers:
            st.markdown(
                f'<div class="warn-box"><strong>Ticker(s) not found on Yahoo Finance:</strong> '
                f'{", ".join(bad_tickers)}. Enter expected return and volatility manually below.</div>',
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
# RUN — "Optimise Portfolio" CTA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
run_col, _ = st.columns([1,3])
with run_col:
    run = st.button("🌿  Optimise Portfolio")

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

        manual_rv_map = {r["ticker"]: r.get("manual_ret_vol") for r in ticker_rows}
        manual_rv_map = {t: v for t, v in manual_rv_map.items() if v is not None}

        available = [t for t in tickers if t in mu_series.index]
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

        mu_list = []; vols_list = []
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

        cov = np.zeros((n, n))
        idx_map = {t: i for i, t in enumerate(all_tickers)}
        for i, ti in enumerate(all_tickers):
            for j, tj in enumerate(all_tickers):
                if ti in available and tj in available:
                    cov[i, j] = float(cov_df.loc[ti, tj])
                elif i == j:
                    cov[i, j] = vols[i] ** 2

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
        st.markdown('<div class="warn-box">Covariance matrix adjusted to PSD.</div>',
                    unsafe_allow_html=True)
        cov = nearest_psd(cov)

    esg_thresh  = min_esg_filter if use_exclusion else 0.0
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
    bounds_green = [(0.,1.) if active_mask[i] else (0.,0.) for i in range(n)]

    # ── Portfolios ────────────────────────────────────────────────────────────
    w_tan_all, ep_tan_all, sp_tan_all, sr_tan_all = find_tangency(mu, cov, rf)
    w_tan_esg, ep_tan_esg, sp_tan_esg, sr_tan_esg = find_tangency(mu, cov, rf, bounds=bounds_green)
    w_opt_a = find_optimal(mu_a, cov_a, esg_a, rf, gamma, lam)
    w_opt   = np.zeros(n)
    for idx, wi in zip(active_idx, w_opt_a): w_opt[idx] = wi
    ep, sp, sr, esg_bar = port_stats(w_opt_a, mu_a, cov_a, esg_a, rf)

    # ── Frontiers ─────────────────────────────────────────────────────────────
    with st.spinner("Building mean-variance frontiers…"):
        std_blue,  ret_blue  = build_mv_frontier(mu, cov, n_points=100)
        std_green, ret_green = build_mv_frontier(mu, cov, bounds=bounds_green, n_points=100)

    # ── Metric Cards ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Optimal Portfolio</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    metric_data = [
        (m1, "Expected Return", f"{ep*100:.2f}",  "%"),
        (m2, "Volatility",      f"{sp*100:.2f}",  "%"),
        (m3, "Sharpe Ratio",    f"{sr:.3f}",       ""),
        (m4, "ESG Score",       f"{esg_bar:.2f}", "/ 10"),
    ]
    for col, label, val, unit in metric_data:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{val}<span class="metric-unit">{unit}</span></div>'
                f'</div>', unsafe_allow_html=True)

    u_val = ep - gamma/2*sp**2 + lam*esg_bar
    st.markdown(
        f'<div class="info-box">'
        f'U = E[Rp] − (γ/2)σ² + λs̄ = <strong>{u_val:.4f}</strong>'
        f' &nbsp;·&nbsp; γ={gamma}, λ={lam}, r<sub>f</sub>={rf*100:.1f}%'
        f' &nbsp;·&nbsp; Tangency Sharpe (all) = {sr_tan_all:.3f}'
        f' &nbsp;·&nbsp; Tangency Sharpe (ESG) = {sr_tan_esg:.3f}'
        f'</div>', unsafe_allow_html=True)

    st.markdown("#### Portfolio Weights")
    st.dataframe(pd.DataFrame({
        "Asset":           names,
        "Weight (%)":      [f"{w*100:.2f}"  for w in w_opt],
        "E[R] (%)":        [f"{r*100:.2f}"  for r in mu],
        "σ (%)":           [f"{v*100:.2f}"  for v in vols],
        "ESG (0–10)":      [f"{s:.2f}"      for s in esg_scores],
        "In ESG frontier": ["✓" if m else "—" for m in active_mask],
    }), use_container_width=True, hide_index=True)

    if input_mode == "Ticker-based input":
        st.markdown("#### Ticker Data Used")
        st.dataframe(ticker_data_display, use_container_width=True, hide_index=True)
        st.markdown("#### Correlation Matrix")
        st.dataframe(pd.DataFrame(corr_np,index=names,columns=names).round(3),
                     use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # CHARTS — Apple-dark aesthetic (refined from iOS Stocks)
    # ══════════════════════════════════════════════════════════════════════════
    BG      = '#000000'
    PLOT_BG = '#1C1C1E'
    C_ESG   = '#ffffff'        # ESG frontier — bright
    C_ALL   = '#48484A'        # Unconstrained — muted
    C_DOT   = '#8E8E93'        # Individual asset dots
    C_OPT   = '#0A84FF'        # Optimal portfolio — Apple blue
    C_TAN   = '#30D158'        # ESG tangency — Apple green
    C_ANN   = '#636366'        # Annotation text
    C_GRID  = '#2C2C2E'        # Grid lines
    C_TICK  = '#48484A'        # Tick labels
    C_TITLE = '#EBEBF5'        # Chart title

    def _style_ax(ax, fig, title=""):
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PLOT_BG)
        for spine in ax.spines.values():
            spine.set_color(C_GRID)
            spine.set_linewidth(0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors=C_TICK, labelsize=8.5, length=3, width=0.5)
        ax.grid(True, color=C_GRID, linewidth=0.5, linestyle='-', alpha=1)
        ax.set_axisbelow(True)
        if title:
            ax.set_title(title, fontsize=11.5, fontweight='600',
                         color=C_TITLE, pad=14, loc='left',
                         fontfamily='-apple-system, Helvetica Neue, Arial')

    st.markdown('<div class="section-header">Efficient Frontier</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # ── Chart 1: Mean-Variance Frontier ──────────────────────────────────────
    with c1:
        fig, ax = plt.subplots(figsize=(6, 5))
        _style_ax(ax, fig, "Mean-Variance Frontier")

        if len(std_blue) > 2:
            ax.plot(std_blue, ret_blue, color=C_ALL, lw=1.6, zorder=4,
                    label='All assets', alpha=0.75)

        if len(std_green) > 2:
            ax.plot(std_green, ret_green, color=C_ESG, lw=2.2, zorder=5,
                    label=f'ESG ≥ {esg_thresh:.1f}')
            ax.fill_between(std_green, ret_green, alpha=0.05, color=C_ESG, zorder=2)

        if sp_tan_all > 1e-9 and len(std_blue) > 0:
            cml_max = max(float(np.nanmax(std_blue)), sp_tan_all*100) * 1.6
            sd_cml  = np.linspace(0, cml_max, 200)
            ax.plot(sd_cml, rf*100 + (ep_tan_all-rf)/sp_tan_all*sd_cml,
                    color=C_ALL, lw=0.8, linestyle=(0,(5,4)), zorder=3, alpha=0.5)

        if sp_tan_esg > 1e-9 and len(std_green) > 0:
            cml_max2 = max(float(np.nanmax(std_green)), sp_tan_esg*100) * 1.6
            sd_cml2  = np.linspace(0, cml_max2, 200)
            ax.plot(sd_cml2, rf*100 + (ep_tan_esg-rf)/sp_tan_esg*sd_cml2,
                    color=C_ESG, lw=0.8, linestyle=(0,(5,4)), zorder=3, alpha=0.38)

        ax.scatter(sp_tan_all*100, ep_tan_all*100,
                   color=C_ALL, s=55, zorder=9, edgecolors='none', marker='o')
        if len(std_green) > 2:
            ax.scatter(sp_tan_esg*100, ep_tan_esg*100,
                       color=C_TAN, s=70, zorder=9, edgecolors='none', marker='o')
        ax.scatter(0, rf*100, color=C_TICK, s=38, zorder=8, edgecolors='none', marker='o')

        for i in range(n):
            ax.scatter(vols[i]*100, mu[i]*100,
                       color=C_DOT if active_mask[i] else '#3A3A3C',
                       s=34, zorder=6, edgecolors='none', alpha=0.9)
            ax.annotate(names[i], (vols[i]*100, mu[i]*100),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=C_ANN, va='bottom')

        ax.scatter(sp*100, ep*100, color=C_OPT, s=120, zorder=10, edgecolors='none', marker='o',
                   label=f'Optimal SR={sr:.2f}')
        ax.scatter(sp*100, ep*100, color='none', s=200, zorder=10,
                   edgecolors=C_OPT, linewidths=1.2, marker='o', alpha=0.4)
        ax.annotate(f'  SR = {sr:.2f}', (sp*100, ep*100),
                    textcoords="offset points", xytext=(8, -4),
                    fontsize=7.5, color=C_OPT, fontweight='600')

        ax.set_xlabel("Volatility (%)", fontsize=9, color=C_ANN, labelpad=7)
        ax.set_ylabel("Expected Return (%)", fontsize=9, color=C_ANN, labelpad=7)
        ax.set_xlim(left=0)

        handles = [
            plt.Line2D([0],[0], color=C_ESG, lw=2, label='ESG frontier'),
            plt.Line2D([0],[0], color=C_ALL, lw=1.5, alpha=0.75, label='Unconstrained'),
            plt.scatter([],[], color=C_OPT, s=50, label='Optimal', edgecolors='none'),
            plt.scatter([],[], color=C_TAN, s=45, label='ESG tangency', edgecolors='none'),
        ]
        legend = ax.legend(handles=handles, fontsize=7.5, framealpha=0,
                           edgecolor='none', labelcolor=C_ANN, loc='upper left')

        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Chart 2: ESG-SR Frontier ──────────────────────────────────────────────
    with c2:
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
        _style_ax(ax2, fig2, "ESG-Sharpe Frontier")

        if sw_esg:
            ax2.plot(sw_esg, sw_sr_vals, color=C_ESG, lw=2.2, zorder=4)
            ax2.fill_between(sw_esg, sw_sr_vals,
                             min(sw_sr_vals) - 0.02,
                             alpha=0.06, color=C_ESG, zorder=2)

        for i in range(len(mu_a)):
            sr_i = (mu_a[i] - rf) / vols_a[i]
            ax2.scatter(esg_a[i], sr_i, color=C_DOT, s=34, zorder=5,
                        edgecolors='none', alpha=0.9)
            ax2.annotate(names_a[i], (esg_a[i], sr_i),
                         textcoords="offset points", xytext=(5, 3),
                         fontsize=7, color=C_ANN)

        ax2.scatter(esg_tan_using, sr_tan_esg, color=C_TAN, s=75, zorder=9, edgecolors='none')
        ax2.annotate('ESG Tangency', (esg_tan_using, sr_tan_esg),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=7.5, color=C_TAN, fontweight='600')

        ax2.scatter(esg_tan_ignoring, sr_tan_all, color=C_ALL, s=60, zorder=8, edgecolors='none')
        ax2.annotate('Unconstrained', (esg_tan_ignoring, sr_tan_all),
                     textcoords="offset points", xytext=(6, -16),
                     fontsize=7.5, color=C_ANN)

        ax2.scatter(esg_bar, sr, color=C_OPT, s=110, zorder=10, edgecolors='none')
        ax2.scatter(esg_bar, sr, color='none', s=190, zorder=10,
                    edgecolors=C_OPT, linewidths=1.2, alpha=0.4)
        ax2.annotate(f'  SR = {sr:.2f}', (esg_bar, sr),
                     textcoords="offset points", xytext=(7, 3),
                     fontsize=7.5, color=C_OPT, fontweight='600')

        ax2.set_xlabel("ESG Score (0–10)", fontsize=9, color=C_ANN, labelpad=7)
        ax2.set_ylabel("Sharpe Ratio",     fontsize=9, color=C_ANN, labelpad=7)
        fig2.tight_layout(pad=1.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── Allocation Charts ─────────────────────────────────────────────────────
    st.markdown("#### Portfolio Allocation")
    pc, bc = st.columns(2)
    nz = [(names[i], w_opt[i], esg_scores[i]) for i in range(n) if w_opt[i] > 0.005]

    if nz:
        plabels = [x[0] for x in nz]
        pvals   = [x[1] for x in nz]
        pesg    = [x[2] for x in nz]

        # Green gradient palette — Apple system greens
        greens = ['#1C4A2A','#1E5C30','#266E38','#2E8042','#34914C',
                  '#3DA258','#45B364','#4EC470','#56D57C','#5EE688']

        with pc:
            f3, a3 = plt.subplots(figsize=(5, 4.5))
            f3.patch.set_facecolor(BG)
            a3.set_facecolor(BG)
            wedges, texts, autotexts = a3.pie(
                pvals, labels=plabels, autopct='%1.1f%%',
                colors=greens[:len(pvals)], startangle=140,
                textprops={'fontsize': 7.5, 'color': '#8E8E93'},
                wedgeprops={'edgecolor': BG, 'linewidth': 2.0})
            for at in autotexts:
                at.set_color('#EBEBF5')
                at.set_fontsize(7.5)
                at.set_fontweight('600')
            a3.set_title('Weight Allocation', fontsize=11.5, fontweight='600',
                         color=C_TITLE, pad=14)
            f3.tight_layout()
            st.pyplot(f3)
            plt.close()

        with bc:
            f4, a4 = plt.subplots(figsize=(5, 4.5))
            f4.patch.set_facecolor(BG)
            a4.set_facecolor(PLOT_BG)
            bcols = ['#e8e8ec' if s >= 7 else '#8e8e93' if s >= 5 else '#48484A' for s in pesg]
            bars  = a4.barh(plabels, [v*100 for v in pvals], color=bcols,
                            edgecolor='none', height=0.55)
            for bar, ev in zip(bars, pesg):
                a4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        f'ESG {ev:.1f}', va='center', fontsize=7, color=C_ANN)
            a4.set_xlabel("Weight (%)", fontsize=9, color=C_ANN, labelpad=6)
            a4.set_title('Weights & ESG', fontsize=11.5, fontweight='600',
                         color=C_TITLE, pad=14)
            a4.tick_params(colors=C_TICK, labelsize=8.5, length=0)
            for sp_ in a4.spines.values(): sp_.set_color(C_GRID); sp_.set_linewidth(0.5)
            a4.spines['top'].set_visible(False)
            a4.spines['right'].set_visible(False)
            a4.grid(True, alpha=1, color=C_GRID, axis='x', linestyle='-', linewidth=0.5)
            a4.set_axisbelow(True)
            f4.tight_layout()
            st.pyplot(f4)
            plt.close()

    # ── Sensitivity Analysis ──────────────────────────────────────────────────
    with st.expander("Sensitivity Analysis — ESG Preference (λ)"):
        lam_vals  = np.linspace(0, 5, 20)
        sens_rows = []
        for lv in lam_vals:
            ww = find_optimal(mu_a, cov_a, esg_a, rf, gamma, lv)
            ep2, sp2, sr2, esg2 = port_stats(ww, mu_a, cov_a, esg_a, rf)
            sens_rows.append({"λ": round(float(lv),2), "E[R](%)": round(ep2*100,2),
                              "σ(%)": round(sp2*100,2), "Sharpe": round(sr2,3), "ESG": round(esg2,2)})
        sens_df = pd.DataFrame(sens_rows)

        f5, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        f5.patch.set_facecolor(BG)

        for ax_, col_, c_, yl_, tl_ in [
            (axes[0], "Sharpe", C_ESG,   "Sharpe Ratio", "Sharpe vs λ"),
            (axes[1], "ESG",    C_TAN,   "ESG Score",    "ESG Score vs λ"),
        ]:
            ax_.set_facecolor(PLOT_BG)
            ax_.plot(sens_df["λ"], sens_df[col_], color=c_, lw=2)
            ax_.fill_between(sens_df["λ"], sens_df[col_], sens_df[col_].min(),
                             alpha=0.07, color=c_)
            ax_.set_title(tl_, fontsize=10.5, color=C_TITLE, fontweight='600')
            ax_.set_xlabel('λ', fontsize=9, color=C_ANN)
            ax_.set_ylabel(yl_, fontsize=9, color=C_ANN)
            ax_.tick_params(colors=C_TICK, labelsize=8)
            for sp_ in ax_.spines.values(): sp_.set_color(C_GRID); sp_.set_linewidth(0.5)
            ax_.spines['top'].set_visible(False)
            ax_.spines['right'].set_visible(False)
            ax_.grid(True, alpha=1, color=C_GRID, linestyle='-', linewidth=0.5)

        axes[2].set_facecolor(PLOT_BG)
        axes[2].plot(sens_df["λ"], sens_df["E[R](%)"], color=C_ESG, lw=2, label='E[R]')
        axes[2].plot(sens_df["λ"], sens_df["σ(%)"],    color=C_ALL, lw=2, linestyle='--', label='σ')
        axes[2].set_title('Return & Risk vs λ', fontsize=10.5, color=C_TITLE, fontweight='600')
        axes[2].set_xlabel('λ', fontsize=9, color=C_ANN)
        axes[2].set_ylabel('%', fontsize=9, color=C_ANN)
        leg = axes[2].legend(fontsize=8, facecolor=PLOT_BG, edgecolor=C_GRID, labelcolor=C_ANN)
        axes[2].tick_params(colors=C_TICK, labelsize=8)
        for sp_ in axes[2].spines.values(): sp_.set_color(C_GRID); sp_.set_linewidth(0.5)
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        axes[2].grid(True, alpha=1, color=C_GRID, linestyle='-', linewidth=0.5)

        f5.tight_layout(pad=1.8)
        st.pyplot(f5)
        plt.close()
        st.dataframe(sens_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        '<div class="info-box"><strong>Methodology:</strong> '
        'Utility U = E[Rp] − (γ/2)σ²p + λs̄, maximised via SLSQP (no short-selling). '
        '<strong>White frontier</strong>: MV frontier restricted to ESG-screened assets. '
        '<strong>Grey frontier</strong>: unconstrained MV frontier across all assets. '
        'The constrained frontier lies to the right of the unconstrained — same return, higher risk. '
        'ESG data: LSEG ESGCombinedScore CSV, most recent year per ticker, scaled 0–10.</div>',
        unsafe_allow_html=True)

    # Store portfolio data for chatbot
    st.session_state["chat_data"] = {
        "names": names, "mu": mu, "vols": vols, "esg_scores": esg_scores,
        "w_opt": w_opt, "ep": ep, "sp": sp, "sr": sr, "esg_bar": esg_bar,
        "gamma": gamma, "lam": lam, "rf": rf,
        "ep_tan_all": ep_tan_all, "sp_tan_all": sp_tan_all, "sr_tan_all": sr_tan_all,
        "ep_tan_esg": ep_tan_esg, "sp_tan_esg": sp_tan_esg, "sr_tan_esg": sr_tan_esg,
        "active_mask": active_mask, "esg_thresh": esg_thresh, "cov": cov, "n": n,
    }
    st.session_state["chat_history"] = []

else:
    st.markdown(
        '<div class="warn-box">Configure the asset universe above and click '
        '<strong>🌿  Optimise Portfolio</strong> to generate results.</div>',
        unsafe_allow_html=True)
    with st.expander("How does the model work?"):
        st.markdown(r"""
**Utility Function**
$$U = E[R_p] - \frac{\gamma}{2}\sigma_p^2 + \lambda \bar{s}$$
**Frontier Construction**
Two mean-variance frontiers are built by minimising portfolio standard deviation for each target return level:
- **White curve**: ESG-constrained — only assets passing the minimum ESG threshold. Lies to the *right* (higher σ for same E[R]).
- **Grey curve**: Unconstrained — uses all assets. Standard Markowitz frontier.
Both frontiers show their Capital Market Line (dashed), tangency portfolio, and the risk-free asset.

**ESG Data Source**
LSEG ESGCombinedScore CSV. The `valuescore` column (0–1) is scaled ×10 to give a 0–10 display score. Most recent available year per ticker.
""")


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO EXPLAINER CHATBOT — iMessage design
# ══════════════════════════════════════════════════════════════════════════════
if "chat_data" in st.session_state:
    st.markdown("---")
    st.markdown('<div class="section-header">Portfolio Explainer</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ── Chat window — iMessage shell ──────────────────────────────────────────
    st.markdown('''
<div class="chat-shell">
  <div class="chat-topbar">
    <div class="chat-topbar-left">
      <div class="chat-avatar">GP</div>
      <div>
        <div class="chat-name">Portfolio Analyst</div>
        <div class="chat-status-text">Powered by your live portfolio data</div>
      </div>
    </div>
    <div class="live-badge">
      <span class="live-dot"></span>&nbsp;Live
    </div>
  </div>''', unsafe_allow_html=True)

    # Messages
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    if not st.session_state["chat_history"]:
        st.markdown('''
<div class="chat-empty">
  <span class="chat-empty-icon">🌿</span>
  <div class="chat-empty-title">Ask me about your portfolio</div>
  <div class="chat-empty-sub">
    Weights, ESG cost, Sharpe ratio, the frontier,<br>
    utility function — or any asset by name.
  </div>
</div>''', unsafe_allow_html=True)

    for msg in st.session_state["chat_history"]:
        content = msg["content"].replace("\n", "<br>")
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-row-user"><div class="bubble-u">{content}</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="msg-row-bot">'
                f'<div class="bot-mini-avatar">GP</div>'
                f'<div class="bubble-b">{content}</div>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close .chat-messages

    # ── Suggestion chips ──────────────────────────────────────────────────────
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
    st.markdown('</div>', unsafe_allow_html=True)  # close .chat-chips

    # ── Compose bar ───────────────────────────────────────────────────────────
    st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        inp_c, btn_c = st.columns([6, 1])
        user_input = inp_c.text_input(
            "msg",
            placeholder="Ask anything about your portfolio…",
            label_visibility="collapsed")
        sent = btn_c.form_submit_button("Send", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # close .chat-shell

    if sent and user_input.strip():
        reply = answer_question(user_input.strip())
        st.session_state["chat_history"].append({"role": "user", "content": user_input.strip()})
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.get("chat_history"):
        clr, _ = st.columns([1, 5])
        with clr:
            if st.button("Clear chat", key="chat_clear", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER — Apple minimal
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('''
<div class="site-footer">
  <span>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="vertical-align:-2px;margin-right:6px">
      <path d="M12 3.5C7.8 3.5 4.5 7 4.5 11.2C4.5 15.1 7.5 18 11.5 18.3C11.5 18.3 11.5 14.5 14.8 11.8C18 9.1 20.5 8.5 20.5 8.5C20.5 8.5 18.8 3.5 12 3.5Z" fill="rgba(48,209,88,0.7)"/>
    </svg>
    GreenPort &copy; 2026
    <span class="footer-dot"></span>
    ECN316 Sustainable Finance
    <span class="footer-dot"></span>
    LSEG ESGCombinedScore Data
  </span>
  <span>
    Mean-Variance Optimisation &nbsp;&middot;&nbsp; SLSQP &nbsp;&middot;&nbsp; Streamlit
  </span>
</div>
''', unsafe_allow_html=True)
