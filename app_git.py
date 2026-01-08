import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Page Configuration
st.set_page_config(page_title="Quantitative Research Platform", layout="wide")
st.title("Quantitative Research Platform (Full Integrated Version)")

API_KEY = "CW1LL675242TSYOU"
URL = "https://www.alphavantage.co/query"

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Strategy Parameters")

st.sidebar.subheader("Module A: Momentum Settings")
ma_fast = st.sidebar.slider("Fast Moving Average (Days)", 5, 50, 20)
ma_slow = st.sidebar.slider("Slow Moving Average (Days)", 51, 200, 50)

st.sidebar.subheader("Module B: Portfolio Weights")
w_btc = st.sidebar.slider("Bitcoin Weight (%)", 0, 100, 33)
w_gld = st.sidebar.slider("Gold Weight (%)", 0, 100, 33)
w_urth = st.sidebar.slider("MSCI World Weight (%)", 0, 100, 34)

total_weight = w_btc + w_gld + w_urth
if total_weight != 100:
    st.sidebar.error(f"Total Weight: {total_weight}% (Adjust to 100%)")

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data(symbol, is_crypto=False):
    time.sleep(1.1)
    params = {
        "function": "DIGITAL_CURRENCY_DAILY" if is_crypto else "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY
    }
    if is_crypto:
        params["market"] = "USD"
    
    try:
        r = requests.get(URL, params=params).json()
        if "Note" in r:
            return "LIMIT"
        key = "Time Series (Digital Currency Daily)" if is_crypto else "Time Series (Daily)"
        if key in r:
            df = pd.DataFrame(r[key]).T
            col = "4b. close (USD)" if "4b. close (USD)" in df.columns else "4. close"
            df = df[[col]].rename(columns={col: "close"}).astype(float)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
    except:
        return None
    return None

# Load Data
with st.spinner("Loading data..."):
    df_btc = get_data("BTC", is_crypto=True)
    df_gld = get_data("GLD")
    df_urth = get_data("URTH")

# API Limit Check
is_limit = any(isinstance(x, str) and x == "LIMIT" for x in [df_btc, df_gld, df_urth])
if is_limit:
    st.error("API Limit reached (5 requests/min). Please wait 60 seconds.")
    st.stop()

if df_btc is not None:
    tab_a, tab_b = st.tabs(["Module A: Bitcoin Analysis", "Module B: Portfolio Analysis"])

    # --- MODULE A ---
    with tab_a:
        st.header(f"Bitcoin Strategy Analysis (MA{ma_fast}/{ma_slow})")
        df_a = df_btc.copy()
        
        # Calculations
        df_a['Returns'] = df_a['close'].pct_change()
        df_a['MA_F'] = df_a['close'].rolling(window=ma_fast).mean()
        df_a['MA_S'] = df_a['close'].rolling(window=ma_slow).mean()
        df_a['Signal'] = (df_a['MA_F'] > df_a['MA_S']).astype(int)
        df_a['Position'] = df_a['Signal'].shift(1).fillna(0)
        df_a['Strat_Ret'] = df_a['Position'] * df_a['Returns']
        
        # Performance Metrics
        bh_perf = (df_a['close'].iloc[-1] / df_a['close'].iloc[0]) - 1
        st_perf = (1 + df_a['Strat_Ret'].fillna(0)).prod() - 1
        sharpe = (df_a['Strat_Ret'].mean() / df_a['Strat_Ret'].std()) * np.sqrt(252)
        equity = (1 + df_a['Strat_Ret'].fillna(0)).cumprod()
        max_dd = ((equity - equity.cummax()) / equity.cummax()).min()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Buy & Hold", f"{bh_perf:.2%}")
        col2.metric("Strategy", f"{st_perf:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col4.metric("Max Drawdown", f"{max_dd:.2%}")

        # Plot 1: Price + MAs + Buy/Sell Markers
        st.subheader("1. Momentum Signals (Buy/Sell)")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df_a.index, df_a['close'], label="Price", alpha=0.3)
        ax1.plot(df_a.index, df_a['MA_F'], label=f"MA{ma_fast}")
        ax1.plot(df_a.index, df_a['MA_S'], label=f"MA{ma_slow}")
        buy = df_a[(df_a['Signal'] == 1) & (df_a['Signal'].shift(1) == 0)]
        sell = df_a[(df_a['Signal'] == 0) & (df_a['Signal'].shift(1) == 1)]
        ax1.scatter(buy.index, buy['close'], marker="^", color="green", s=100, label="Buy")
        ax1.scatter(sell.index, sell['close'], marker="v", color="red", s=100, label="Sell")
        ax1.legend()
        st.pyplot(fig1)

        # Plot 2: Performance Comparison
        st.subheader("2. Cumulative Performance ($10k base)")
        comp_df = pd.DataFrame({
            "Buy & Hold": (df_a['close'] / df_a['close'].iloc[0]) * 10000,
            "Strategy": equity * 10000
        })
        st.line_chart(comp_df)

        # Plot 3: Dual Axis (Price vs Equity)
        st.subheader("3. BTC Price vs Strategy Equity (Dual Axis)")
        fig3, ax_p = plt.subplots(figsize=(12, 5))
        ax_p.plot(df_a.index, df_a['close'], color="blue", label="BTC Price")
        ax_p.set_ylabel("BTC Price (USD)", color="blue")
        ax_e = ax_p.twinx()
        ax_e.plot(df_a.index, equity * 10000, color="orange", label="Strategy Equity")
        ax_e.set_ylabel("Strategy Value (USD)", color="orange")
        plt.title("BTC Price (Left) vs Strategy Value (Right)")
        st.pyplot(fig3)

        # ML Section
        st.divider()
        st.header("Linear Regression & Forecasting")
        df_ml = df_a[['close']].dropna()
        df_ml['T'] = np.arange(len(df_ml))
        model = LinearRegression().fit(df_ml[['T']], df_ml['close'])
        df_ml['Pred'] = model.predict(df_ml[['T']])
        
        # Plot 4: ML Prediction
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        ax4.plot(df_ml.index, df_ml['close'], label="Actual")
        ax4.plot(df_ml.index, df_ml['Pred'], label="Predicted", linestyle="--")
        ax4.legend()
        st.pyplot(fig4)
        
        mae = mean_absolute_error(df_ml['close'], df_ml['Pred'])
        st.write(f"Mean Absolute Error (MAE): **${mae:,.2f}**")
        
        # Forecast Table
        fut = np.array([[len(df_ml) + i] for i in range(1, 11)])
        forecast = pd.DataFrame({"Day": [f"Day +{i}" for i in range(1, 11)], "Predicted Price": model.predict(fut)})
        st.table(forecast)

    # --- MODULE B ---
    with tab_b:
        if df_gld is not None and df_urth is not None:
            st.header("Portfolio Diversification")
            combined = pd.DataFrame({"Bitcoin": df_btc["close"], "Gold": df_gld["close"], "MSCI World": df_urth["close"]}).dropna()
            rets = combined.pct_change().dropna()
            w = np.array([w_btc/100, w_gld/100, w_urth/100])
            rets['Portfolio'] = rets.dot(w)
            
            st.subheader("Asset Comparison (Base 100)")
            st.line_chart((1 + rets).cumprod() * 100)
            
            vol = rets['Portfolio'].std() * np.sqrt(252)
            theo_v = np.sum(rets[['Bitcoin', 'Gold', 'MSCI World']].std() * np.sqrt(252) * w)
            red = (1 - (vol / theo_v)) * 100 if theo_v > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Annual Volatility", f"{vol:.2%}")
            c2.metric("Risk Reduction", f"{red:.2f}%")
            c3.metric("Daily Variance", f"{rets['Portfolio'].var():.6f}")

            st.subheader("Correlation Heatmap")
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(rets[['Bitcoin', 'Gold', 'MSCI World']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)
