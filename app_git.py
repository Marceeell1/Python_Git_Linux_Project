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
st.title("Quantitative Research Platform (Professional Version)")

API_KEY = "CW1LL675242TSYOU"
URL = "https://www.alphavantage.co/query"

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Strategy Parameters")

# Parameters for Module A
st.sidebar.subheader("Module A: Momentum Settings")
ma_fast = st.sidebar.slider("Fast Moving Average (Days)", 5, 50, 20)
ma_slow = st.sidebar.slider("Slow Moving Average (Days)", 51, 200, 50)

# Parameters for Module B
st.sidebar.subheader("Module B: Portfolio Weights")
st.sidebar.info("Total must equal 100%")
w_btc = st.sidebar.slider("Bitcoin Weight (%)", 0, 100, 33)
w_gld = st.sidebar.slider("Gold Weight (%)", 0, 100, 33)
w_urth = st.sidebar.slider("MSCI World Weight (%)", 0, 100, 34)

total_weight = w_btc + w_gld + w_urth
if total_weight != 100:
    st.sidebar.error(f"Total Weight: {total_weight}% (Adjust to 100%)")

# --- DATA FETCHING FUNCTION ---
@st.cache_data(ttl=3600)
def get_data(symbol, is_crypto=False):
    time.sleep(1) # Security delay
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

# Initial Load
with st.spinner("Fetching market data..."):
    df_btc = get_data("BTC", is_crypto=True)
    df_gld = get_data("GLD")
    df_urth = get_data("URTH")

# Check for API limits
if df_btc == "LIMIT" or df_gld == "LIMIT" or df_urth == "LIMIT":
    st.error("API Limit reached. Alpha Vantage free tier allows 5 requests per minute. Please refresh in 60 seconds.")
    st.stop()

if df_btc is not None:
    tab_a, tab_b = st.tabs(["Module A: Single Asset Analysis", "Module B: Portfolio Analysis"])

    # --- MODULE A: BITCOIN ANALYSIS ---
    with tab_a:
        st.header(f"Bitcoin Momentum Strategy (MA{ma_fast}/MA{ma_slow})")
        df_a = df_btc.copy()
        
        # Calculations
        df_a['Returns'] = df_a['close'].pct_change()
        df_a['MA_Fast'] = df_a['close'].rolling(window=ma_fast).mean()
        df_a['MA_Slow'] = df_a['close'].rolling(window=ma_slow).mean()
        df_a['Signal'] = (df_a['MA_Fast'] > df_a['MA_Slow']).astype(int)
        df_a['Position'] = df_a['Signal'].shift(1).fillna(0)
        df_a['Strategy_Returns'] = (df_a['Position'] * df_a['Returns']).fillna(0)
        
        # Metrics
        bh_ret = (df_a['close'].iloc[-1] / df_a['close'].iloc[0]) - 1
        st_ret = (1 + df_a['Strategy_Returns']).prod() - 1
        sharpe_a = (df_a['Strategy_Returns'].mean() / df_a['Strategy_Returns'].std()) * np.sqrt(252)
        
        equity_a = (1 + df_a['Strategy_Returns']).cumprod()
        dd_a = (equity_a - equity_a.cummax()) / equity_a.cummax()
        max_dd_a = dd_a.min()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Buy & Hold Return", f"{bh_ret:.2%}")
        m2.metric("Strategy Return", f"{st_ret:.2%}")
        m3.metric("Sharpe Ratio", f"{sharpe_a:.2f}")
        m4.metric("Max Drawdown", f"{max_dd_a:.2%}")

        # Chart 1: Signals
        fig_sig, ax_sig = plt.subplots(figsize=(12, 5))
        ax_sig.plot(df_a.index, df_a['close'], label="Price", alpha=0.4)
        ax_sig.plot(df_a.index, df_a['MA_Fast'], label=f"MA{ma_fast}")
        ax_sig.plot(df_a.index, df_a['MA_Slow'], label=f"MA{ma_slow}")
        
        buys = df_a[(df_a['Signal'] == 1) & (df_a['Signal'].shift(1) == 0)]
        sells = df_a[(df_a['Signal'] == 0) & (df_a['Signal'].shift(1) == 1)]
        ax_sig.scatter(buys.index, buys['close'], marker="^", color="green", s=100, label="Buy Signal")
        ax_sig.scatter(sells.index, sells['close'], marker="v", color="red", s=100, label="Sell Signal")
        ax_sig.set_title("Price with Moving Average Crossovers")
        ax_sig.legend()
        st.pyplot(fig_sig)

        # Machine Learning Prediction
        st.divider()
        st.subheader("Predictive Analytics: Linear Regression")
        df_ml = df_a[['close']].dropna()
        df_ml['T'] = np.arange(len(df_ml))
        model = LinearRegression().fit(df_ml[['T']], df_ml['close'])
        df_ml['Pred'] = model.predict(df_ml[['T']])
        mae = mean_absolute_error(df_ml['close'], df_ml['Pred'])
        
        st.write(f"Model Mean Absolute Error: **${mae:,.2f}**")
        
        # Forecast 10 Days
        fut_idx = np.array([[len(df_ml) + i] for i in range(1, 11)])
        preds = model.predict(fut_idx)
        st.table(pd.DataFrame({"Forecast Day": [f"Day +{i+1}" for i in range(10)], "Predicted Price": preds}))

    # --- MODULE B: PORTFOLIO ANALYSIS ---
    with tab_b:
        if df_gld is not None and df_urth is not None:
            st.header("Multi-Asset Diversification Strategy")
            
            combined = pd.DataFrame({
                "Bitcoin": df_btc["close"],
                "Gold": df_gld["close"],
                "MSCI World": df_urth["close"]
            }).dropna()
            
            # Use weights from sidebar
            u_weights = np.array([w_btc/100, w_gld/100, w_urth/100])
            
            returns = combined.pct_change().dropna()
            returns['Portfolio'] = returns.dot(u_weights)
            
            # Growth Comparison
            equity_df = (1 + returns).cumprod() * 10000
            st.subheader("Growth of $10,000 Investment")
            st.line_chart(equity_df)
            
            # Risk Analysis
            ann_vol = returns['Portfolio'].std() * np.sqrt(252)
            indiv_vols = returns[['Bitcoin', 'Gold', 'MSCI World']].std() * np.sqrt(252)
            theoretical_risk = np.sum(indiv_vols * u_weights)
            risk_red = (1 - (ann_vol / theoretical_risk)) * 100 if theoretical_risk != 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Portfolio Volatility", f"{ann_vol:.2%}")
            c2.metric("Risk Reduction", f"{risk_red:.2f}%")
            c3.metric("Daily Variance", f"{returns['Portfolio'].var():.6f}")

            # Correlation
            st.subheader("Asset Correlation Matrix")
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(returns[['Bitcoin', 'Gold', 'MSCI World']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.warning("Data for some assets is missing. API cooling down...")

else:
    st.error("Fatal: Could not connect to API.")
