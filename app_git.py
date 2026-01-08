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

# --- SIDEBAR PARAMETERS (For User Interactivity) ---
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
    st.sidebar.error(f"Total Weight: {total_weight}% (Please adjust to 100%)")

# --- DATA FETCHING FUNCTION WITH CACHE ---
@st.cache_data(ttl=3600)
def get_data(symbol, is_crypto=False):
    time.sleep(1.1) # Security delay to respect API limits
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

# Initial Data Loading
with st.spinner("Accessing financial data..."):
    df_btc = get_data("BTC", is_crypto=True)
    df_gld = get_data("GLD")
    df_urth = get_data("URTH")

# --- CORRECTED ERROR CHECKING (Fixes the ValueError) ---
# We check if any return value is the string "LIMIT"
is_limit = any(isinstance(x, str) and x == "LIMIT" for x in [df_btc, df_gld, df_urth])

if is_limit:
    st.error("API Limit reached. The Alpha Vantage free tier allows 5 requests per minute. Please wait 60 seconds.")
    st.stop()

if df_btc is not None:
    tab_a, tab_b = st.tabs(["Module A: Single Asset Analysis", "Module B: Portfolio Analysis"])

    # --- MODULE A: BITCOIN ANALYSIS ---
    with tab_a:
        st.header(f"Bitcoin Strategy: MA{ma_fast} vs MA{ma_slow}")
        df_a = df_btc.copy()
        
        # Performance Calculations
        df_a['Returns'] = df_a['close'].pct_change()
        df_a['MA_Fast'] = df_a['close'].rolling(window=ma_fast).mean()
        df_a['MA_Slow'] = df_a['close'].rolling(window=ma_slow).mean()
        df_a['Signal'] = (df_a['MA_Fast'] > df_a['MA_Slow']).astype(int)
        df_a['Position'] = df_a['Signal'].shift(1).fillna(0)
        df_a['Strategy_Returns'] = (df_a['Position'] * df_a['Returns']).fillna(0)
        
        # Cumulative performance
        df_a['Equity_Strategy'] = (1 + df_a['Strategy_Returns']).cumprod() * 10000
        df_a['Equity_BH'] = (df_a['close'] / df_a['close'].iloc[0]) * 10000

        # Metrics display
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sharpe Ratio", f"{(df_a['Strategy_Returns'].mean() / df_a['Strategy_Returns'].std() * np.sqrt(252)):.2f}")
        m2.metric("Max Drawdown", f"{((df_a['Equity_Strategy'] - df_a['Equity_Strategy'].cummax()) / df_a['Equity_Strategy'].cummax()).min():.2%}")
        m3.metric("Final Portfolio Value", f"${df_a['Equity_Strategy'].iloc[-1]:,.0f}")
        m4.metric("Buy & Hold Value", f"${df_a['Equity_BH'].iloc[-1]:,.0f}")

        # Signal Chart
        fig_sig, ax_sig = plt.subplots(figsize=(12, 5))
        ax_sig.plot(df_a.index, df_a['close'], label="Price", alpha=0.3)
        ax_sig.plot(df_a.index, df_a['MA_Fast'], label=f"MA{ma_fast}")
        ax_sig.plot(df_a.index, df_a['MA_Slow'], label=f"MA{ma_slow}")
        st.pyplot(fig_sig)

        # Machine Learning Bonus
        st.divider()
        st.subheader("Predictive Modeling (Linear Regression)")
        df_ml = df_a[['close']].dropna()
        df_ml['T'] = np.arange(len(df_ml))
        model = LinearRegression().fit(df_ml[['T']], df_ml['close'])
        mae = mean_absolute_error(df_ml['close'], model.predict(df_ml[['T']]))
        st.info(f"Mean Absolute Error (MAE): ${mae:,.2f}")
        
        # 10-day forecast
        fut = np.array([[len(df_ml) + i] for i in range(1, 11)])
        st.table(pd.DataFrame({"Forecast Day": [f"Day +{i}" for i in range(1, 11)], "Price": model.predict(fut)}))

    # --- MODULE B: PORTFOLIO ANALYSIS ---
    with tab_b:
        if df_gld is not None and df_urth is not None:
            st.header("Multi-Asset Portfolio Simulation")
            
            combined = pd.DataFrame({
                "Bitcoin": df_btc["close"],
                "Gold": df_gld["close"],
                "MSCI World": df_urth["close"]
            }).dropna()
            
            # Apply user weights from sidebar
            weights = np.array([w_btc/100, w_gld/100, w_urth/100])
            returns = combined.pct_change().dropna()
            returns['Portfolio'] = returns.dot(weights)
            
            # Growth Chart
            st.subheader("Performance Comparison (Base $10,000)")
            st.line_chart((1 + returns).cumprod() * 10000)
            
            # Diversification Metrics
            vol_p = returns['Portfolio'].std() * np.sqrt(252)
            theoretical_v = np.sum(returns[['Bitcoin', 'Gold', 'MSCI World']].std() * np.sqrt(252) * weights)
            reduction = (1 - (vol_p / theoretical_v)) * 100 if theoretical_v > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Annualized Volatility", f"{vol_p:.2%}")
            c2.metric("Risk Reduction", f"{reduction:.2f}%")
            c3.metric("Daily Variance", f"{returns['Portfolio'].var():.6f}")

            # Correlation matrix
            st.subheader("Asset Correlation Matrix")
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(returns[['Bitcoin', 'Gold', 'MSCI World']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.warning("Additional assets are loading or unavailable. Check API limit.")

else:
    st.error("Unable to load Bitcoin data. Please check your internet connection and API Key.")
