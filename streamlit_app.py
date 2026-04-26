""" Streamlit UI for P2-ETF-FDA-SHAPE. """
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import config
from push_results import load_latest_result
from us_calendar import next_trading_day

st.set_page_config(page_title="FDA‑SHAPE Engine", layout="wide")

st.markdown("""
<style>
    .hero { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%);
            border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .metric-card { background: white; padding: 20px; border-radius: 12px;
                   border: 1px solid #E6E6F2; text-align: center; }
    .metric-label { color: #8C91A1; font-size: 12px; text-transform: uppercase; }
    .metric-value { font-size: 24px; font-weight: 700; color: #0E1117; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>FDA‑SHAPE — Functional Data Analysis ETF Engine</h1>', unsafe_allow_html=True)
st.markdown('<p>B‑Spline Smoothing · fPCA · Shape‑Based Prediction · Daily / Global / Adaptive</p>', unsafe_allow_html=True)

tab_fi, tab_eq, tab_comb = st.tabs(["FI/Commodities", "Equity Sectors", "Combined Universe"])
results = load_latest_result()

def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def format_pct(value):
    v = safe_float(value)
    if v is None or np.isnan(v):
        return "—"
    return f"{v*100:.1f}%"

def format_num(value, decimals=2):
    v = safe_float(value)
    if v is None or np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"

def display_metrics(metrics):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-label">ANN RETURN</div><div class="metric-value">{}</div></div>'.format(format_pct(metrics.get("ann_return"))), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-label">ANN VOL</div><div class="metric-value">{}</div></div>'.format(format_pct(metrics.get("ann_vol"))), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-label">SHARPE</div><div class="metric-value">{}</div></div>'.format(format_num(metrics.get("sharpe"))), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-label">MAX DD</div><div class="metric-value">{}</div></div>'.format(format_pct(metrics.get("max_dd"))), unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-card"><div class="metric-label">HIT RATE</div><div class="metric-value">{}</div></div>'.format(format_pct(metrics.get("hit_rate"))), unsafe_allow_html=True)

def display_predicted_returns_table(all_pred_returns: dict):
    if not all_pred_returns:
        return
    sorted_items = sorted(all_pred_returns.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sorted_items, columns=["ETF", "Predicted Return"])
    df["Predicted Return"] = df["Predicted Return"].apply(format_pct)
    st.markdown("#### All Predicted Returns (Current Window)")
    st.dataframe(df, use_container_width=True, hide_index=True)

def display_card(data, mode="global"):
    if not data or not data.get("ticker"):
        st.info("⏳ Waiting for training output...")
        return
    ticker = data.get("ticker")
    pred_return = data.get("pred_return")
    metrics = data.get("metrics", {})
    next_day = next_trading_day(datetime.utcnow())
    gen_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    st.markdown(f'<div class="hero"><div class="hero-ticker">{ticker}</div>', unsafe_allow_html=True)
    if pred_return is not None:
        st.markdown(f'<div style="font-size: 1.5rem;">Predicted Return: {format_pct(pred_return)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="opacity: 0.8;">Signal for {next_day.strftime("%Y-%m-%d")} · Generated {gen_time}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="opacity: 0.8;">Source: {mode} Training</div>', unsafe_allow_html=True)
    if mode == "Global" or mode == "Daily":
        window = data.get("optimal_window", "—")
        st.markdown(f'<div style="opacity: 0.8;">Optimal Window: {window} days</div>', unsafe_allow_html=True)
    else:
        window = data.get("adaptive_window", "—")
        cp_date = data.get("change_point_date", "—")
        st.markdown(f'<div style="opacity: 0.8;">Adaptive Window: {window} days</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="opacity: 0.8;">Change Point: {cp_date}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="opacity: 0.8;">Test: {data.get("test_start", "")} → {data.get("test_end", "")} ({metrics.get("n_days", "—")} days)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    display_metrics(metrics)
    all_preds = data.get("all_pred_returns")
    if all_preds:
        st.markdown('---')
        display_predicted_returns_table(all_preds)

# Render tabs
for tab, key in [(tab_fi, "fi"), (tab_eq, "equity"), (tab_comb, "combined")]:
    with tab:
        st.subheader(key.capitalize())
        daily, global_, adaptive = st.tabs(["📅 Daily (504d)", "🌍 Global", "🔄 Adaptive"])
        with daily:
            display_card(results.get(key, {}).get("daily", {}), "Daily")
        with global_:
            display_card(results.get(key, {}).get("global", {}), "Global")
        with adaptive:
            display_card(results.get(key, {}).get("adaptive", {}), "Adaptive")
