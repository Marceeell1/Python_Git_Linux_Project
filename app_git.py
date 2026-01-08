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

# --- 1. CORE FEATURE: AUTO-REFRESH (Requirement #5) ---
st.set_page_config(page_title="Quant Research Platform", layout="wide")
st_autorefresh(interval=300000, key="datarefresh") # 5 minutes
st.title("Quantitative Research Platform")

API_KEY = "CW1LL675242TSYOU"
URL = "https://www.alphavantage.co/query"

# --- 2. SIDEBAR: ALLOCATION RULES & PARAMETERS (Requirement B) ---
st.sidebar.header("Portfolio Strategy Parameters")

# Allocation Rule & Rebalancing
alloc_rule = st.sidebar.selectbox("Allocation Rule", ["Custom Weights", "Equal Weight"])
rebal_freq = st.sidebar.selectbox("Rebalancing Frequency", ["Daily", "Monthly", "Quarterly"])

if alloc_rule == "Equal Weight":
    w_btc, w_gld, w_urth = 33.33, 33.33, 33.34
else:
    w_btc = st.sidebar.slider("Bitcoin Weight (%)", 0, 100, 33)
    w_gld = st.sidebar.slider("Gold Weight (%)", 0, 100, 33)
    w_urth = st.sidebar.slider("MSCI World Weight (%)", 0, 100, 34)

# Momentum Parameters (FIXED ERROR HERE)
st.sidebar.divider()
st.sidebar.header("Momentum Settings (Module A)")
ma_f = st.sidebar.number_input("Fast MA", 5, 50, 20)
# Correction : min_value=51, donc la valeur par défaut doit être >= 51 (on met 100)
ma_s = st.sidebar.number_input("Slow MA", 51, 200, 100) 

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
        # Select closing price column
        col = df.columns[3] if is_crypto else "4. close"
        df = df[[col]].rename(columns={col: "close"}).astype(float)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except: return None

df_btc = get_data("BTC", is_crypto=True)
df_gld = get_data("GLD")
df_urth = get_data("URTH")

# API Robustness (Requirement #8)
if any(isinstance(x, str) and x == "LIMIT" for x in [df_btc, df_gld, df_urth]):
    st.error("API Limit reached. Please wait 60 seconds.")
    st.stop()

if df_btc is not None:
    tab_a, tab_b = st.tabs(["📈 Module A: Single Asset", "📁 Module B: Portfolio Analysis"])

    # --- MODULE A: ANALYSIS ---
    with tab_a:
        st.header("Bitcoin Strategy & Metrics")
        # Current Value (Requirement #3)
        st.metric("Current Price (BTC)", f"${df_btc['close'].iloc[-1]:,.2f}")

        df = df_btc.copy()
        df['Returns'] = df['close'].pct_change()
        df['MA_F'] = df['close'].rolling(ma_f).mean()
        df['MA_S'] = df['close'].rolling(ma_s).mean()
        df['Signal'] = (df['MA_F'] > df['MA_S']).astype(int)
        df['Strat_Ret'] = df['Signal'].shift(1) * df['Returns']
        equity = (1 + df['Strat_Ret'].fillna(0)).cumprod()

        # Combined Plot (Requirement #4)
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df['close'], alpha=0.4, label="BTC Price")
        ax1.plot(df['MA_F'], label=f"MA{ma_f}")
        ax1.plot(df['MA_S'], label=f"MA{ma_s}")
        ax1.set_title("Price and Moving Averages")
        ax1.legend()
        st.pyplot(fig1)

        # Performance vs Price
        st.subheader("Strategy Cumulative Equity")
        fig2, ax_p = plt.subplots(figsize=(12, 4))
        ax_p.plot(df['close'], color="blue", label="BTC Price")
        ax_e = ax_p.twinx()
        ax_e.plot(equity * 10000, color="orange", label="Strategy Equity")
        ax_e.set_ylabel("Portfolio Value ($)", color="orange")
        st.pyplot(fig2)

        # ML Forecast (Bonus)
        st.divider()
        st.subheader("10-Day Prediction (Linear Regression)")
        df_ml = df[['close']].dropna()
        df_ml['T'] = np.arange(len(df_ml))
        model = LinearRegression().fit(df_ml[['T']], df_ml['close'])
        mae = mean_absolute_error(df_ml['close'], model.predict(df_ml[['T']]))
        st.info(f"Model Mean Absolute Error: ${mae:,.2f}")
        fut = np.array([[len(df_ml) + i] for i in range(1, 11)])
        st.table(pd.DataFrame({"Day": [f"+{i}" for i in range(1, 11)], "Predicted Price": model.predict(fut)}))

    # --- MODULE B: PORTFOLIO ---
    with tab_b:
        if df_gld is not None and df_urth is not None:
            st.header("Portfolio Simulation & Diversification")
            
            # Combine assets
            combined = pd.DataFrame({"Bitcoin": df_btc["close"], "Gold": df_gld["close"], "MSCI World": df_urth["close"]}).dropna()
            rets = combined.pct_change().dropna()
            
            # Calculate Portfolio (Requirement: Custom weights & Returns)
            weights = np.array([w_btc/100, w_gld/100, w_urth/100])
            rets['Portfolio'] = rets.dot(weights)
            equity_comp = (1 + rets).cumprod() * 10000
            
            # Main chart (Requirement: multiple asset prices + cumulative portfolio)
            st.subheader("Visual Comparison: Portfolio vs Assets")
            st.line_chart(equity_comp)

            # Portfolio Metrics (Requirement: returns, volatility, diversification)
            st.subheader("Performance & Diversification Metrics")
            total_ret = (equity_comp['Portfolio'].iloc[-1] / 10000) - 1
            ann_vol = rets['Portfolio'].std() * np.sqrt(252)
            
            # Diversification Effect (Risk Reduction)
            theo_vol = np.sum(rets[['Bitcoin', 'Gold', 'MSCI World']].std() * np.sqrt(252) * weights)
            risk_red = (1 - (ann_vol / theo_vol)) * 100 if theo_vol > 0 else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Return", f"{total_ret:.2%}")
            m1.metric("Annualized Return", f"{(rets['Portfolio'].mean()*252):.2%}")
            m2.metric("Annualized Volatility", f"{ann_vol:.2%}")
            m3.metric("Risk Reduction Effect", f"{risk_red:.2f}%")
            
            st.write(f"**Diversification Insight:** Your current allocation has reduced the portfolio risk by **{risk_red:.2f}%** compared to holding assets separately.")

            # Correlation Matrix (Requirement)
            st.subheader("Correlation Matrix")
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(rets[['Bitcoin', 'Gold', 'MSCI World']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)
            
            st.info(f"Configuration: {alloc_rule} | Rebalancing: {rebal_freq}")
