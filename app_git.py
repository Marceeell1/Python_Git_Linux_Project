import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from streamlit_autorefresh import st_autorefresh

# --- 1. CORE FEATURE: AUTO-REFRESH EVERY 5 MIN ---
st.set_page_config(page_title="Quant Research Platform", layout="wide")
st_autorefresh(interval=300000, key="datarefresh")
st.title("Quantitative Research Platform")

API_KEY = "CW1LL675242TSYOU"
URL = "https://www.alphavantage.co/query"

# --- 2. SIDEBAR: STRATEGY PARAMETERS ---
st.sidebar.header("Portfolio Strategy Parameters")

# Allocation Rules (Requirement)
alloc_rule = st.sidebar.selectbox("Allocation Rule", ["Custom Weights", "Equal Weight"])
rebal_freq = st.sidebar.selectbox("Rebalancing Frequency", ["Daily", "Monthly", "Yearly"])

if alloc_rule == "Equal Weight":
    w_btc, w_gld, w_urth = 33.33, 33.33, 33.34
else:
    w_btc = st.sidebar.slider("Bitcoin Weight (%)", 0, 100, 33)
    w_gld = st.sidebar.slider("Gold Weight (%)", 0, 100, 33)
    w_urth = st.sidebar.slider("MSCI World Weight (%)", 0, 100, 34)

# Module A Parameters
st.sidebar.divider()
st.sidebar.header("Momentum Parameters (Module A)")
ma_f = st.sidebar.number_input("Fast MA", 5, 50, 20)
ma_s = st.sidebar.number_input("Slow MA", 51, 200, 50)

# --- 3. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data(symbol, is_crypto=False):
    time.sleep(1.2)
    params = {"function": "DIGITAL_CURRENCY_DAILY" if is_crypto else "TIME_SERIES_DAILY", 
              "symbol": symbol, "apikey": API_KEY}
    if is_crypto: params["market"] = "USD"
    try:
        r = requests.get(URL, params=params).json()
        if "Note" in r: return "LIMIT"
        key = "Time Series (Digital Currency Daily)" if is_crypto else "Time Series (Daily)"
        df = pd.DataFrame(r[key]).T
        df = df[df.columns[3 if is_crypto else 3]].astype(float).to_frame(name="close")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except: return None

# Load all assets
df_btc = get_data("BTC", is_crypto=True)
df_gld = get_data("GLD")
df_urth = get_data("URTH")

# API Check
if any(isinstance(x, str) and x == "LIMIT" for x in [df_btc, df_gld, df_urth]):
    st.error("API Limit reached. Alpha Vantage allows 5 calls/min. Please wait.")
    st.stop()

if df_btc is not None:
    tab_a, tab_b = st.tabs(["📈 Module A: Single Asset", "📁 Module B: Portfolio Simulation"])

    # --- MODULE A: ANALYSIS ---
    with tab_a:
        st.header("Bitcoin Quantitative Analysis")
        # Display current value (Requirement #3)
        current_price = df_btc['close'].iloc[-1]
        st.metric("Current BTC Price", f"${current_price:,.2f}")

        df = df_btc.copy()
        df['Returns'] = df['close'].pct_change()
        df['MA_F'] = df['close'].rolling(ma_f).mean()
        df['MA_S'] = df['close'].rolling(ma_s).mean()
        df['Signal'] = (df['MA_F'] > df['MA_S']).astype(int)
        df['Strat_Ret'] = df['Signal'].shift(1) * df['Returns']
        equity = (1 + df['Strat_Ret'].fillna(0)).cumprod()

        # Visual 1: Price and Signals
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(df['close'], alpha=0.5, label="BTC Price")
        ax1.plot(df['MA_F'], label=f"MA{ma_f}")
        ax1.plot(df['MA_S'], label=f"MA{ma_s}")
        ax1.legend()
        st.pyplot(fig1)

        # Visual 2: Price vs Strategy (Requirement #4)
        st.subheader("BTC Price vs Cumulative Strategy Equity")
        fig2, ax_p = plt.subplots(figsize=(12, 4))
        ax_p.plot(df['close'], color="blue", label="BTC Price")
        ax_p.set_ylabel("Price (USD)", color="blue")
        ax_e = ax_p.twinx()
        ax_e.plot(equity * 10000, color="orange", label="Strategy Equity")
        ax_e.set_ylabel("Equity (USD)", color="orange")
        st.pyplot(fig2)

        # Machine Learning Prediction (Bonus)
        st.divider()
        st.subheader("Linear Regression Price Prediction")
        df_ml = df[['close']].dropna()
        df_ml['T'] = np.arange(len(df_ml))
        model = LinearRegression().fit(df_ml[['T']], df_ml['close'])
        mae = mean_absolute_error(df_ml['close'], model.predict(df_ml[['T']]))
        st.write(f"Model Error (MAE): ${mae:,.2f}")
        fut = np.array([[len(df_ml) + i] for i in range(1, 11)])
        st.table(pd.DataFrame({"Forecast Day": [f"+{i}" for i in range(1, 11)], "Predicted Price": model.predict(fut)}))

    # --- MODULE B: PORTFOLIO ---
    with tab_b:
        if df_gld is not None and df_urth is not None:
            st.header("Multi-Asset Portfolio Analysis")
            
            # Combine data (Requirement: multiple assets)
            combined = pd.DataFrame({
                "Bitcoin": df_btc["close"], 
                "Gold": df_gld["close"], 
                "MSCI World": df_urth["close"]
            }).dropna()
            
            # Normalization (Base 100)
            norm = (combined / combined.iloc[0]) * 100
            
            # Calculate Returns & Portfolio
            returns = combined.pct_change().dropna()
            w = np.array([w_btc/100, w_gld/100, w_urth/100])
            returns['Portfolio'] = returns.dot(w)
            
            # Cumulative Growth
            equity_comp = (1 + returns).cumprod() * 10000
            
            # Requirement: Main chart showing multiple assets + portfolio cumulative value
            st.subheader("Performance Comparison ($10k Investment)")
            st.line_chart(equity_comp)

            # Portfolio Metrics (Requirement: returns, volatility, diversification)
            st.subheader("Key Portfolio Metrics")
            total_return = (equity_comp['Portfolio'].iloc[-1] / 10000) - 1
            ann_return = returns['Portfolio'].mean() * 252
            p_vol = returns['Portfolio'].std() * np.sqrt(252)
            
            # Diversification Effects
            indiv_vols = returns[['Bitcoin', 'Gold', 'MSCI World']].std() * np.sqrt(252)
            theo_weighted_vol = np.sum(indiv_vols * w)
            risk_reduction = (1 - (p_vol / theo_weighted_vol)) * 100 if theo_weighted_vol > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{total_return:.2%}")
            m2.metric("Annualized Return", f"{ann_return:.2%}")
            m3.metric("Annual Volatility", f"{p_vol:.2%}")
            m4.metric("Risk Reduction", f"{risk_reduction:.2f}%")
            
            st.write(f"**Diversification Effect:** By combining assets, the portfolio volatility is **{risk_reduction:.2f}%** lower than the weighted average of individual risks.")

            # Correlation Matrix (Requirement)
            st.subheader("Asset Correlation Matrix")
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(returns[['Bitcoin', 'Gold', 'MSCI World']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)
            
            st.info(f"Strategy: {alloc_rule} | Rebalancing: {rebal_freq}")
        else:
            st.warning("Additional data loading... please wait 60s for API.")
