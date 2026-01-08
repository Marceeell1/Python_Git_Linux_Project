import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="Quantitative Research Platform", layout="wide")
st.title("📊 Quantitative Research Platform (API Version)")

API_KEY = "CW1LL675242TSYOU"
URL = "https://www.alphavantage.co/query"

# Caching to avoid hitting the 5 requests/minute limit
@st.cache_data(ttl=3600)
def get_data(symbol, is_crypto=False):
    # Added sleep to prevent rapid-fire requests that trigger API blocks
    time.sleep(1)
    
    params = {
        "function": "DIGITAL_CURRENCY_DAILY" if is_crypto else "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY
    }
    if is_crypto:
        params["market"] = "USD"
    
    try:
        r = requests.get(URL, params=params).json()
        
        # API Rate Limit check
        if "Note" in r:
            st.error(f"API Limit reached for {symbol}. Please wait 1 minute.")
            return None
            
        key = "Time Series (Digital Currency Daily)" if is_crypto else "Time Series (Daily)"
        if key in r:
            df = pd.DataFrame(r[key]).T
            col = "4b. close (USD)" if "4b. close (USD)" in df.columns else "4. close"
            df = df[[col]].rename(columns={col: "close"}).astype(float)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
    except Exception as e:
        st.error(f"Connection error for {symbol}: {e}")
        return None
    return None

# Data Loading
df_btc = get_data("BTC", is_crypto=True)
df_gld = get_data("GLD")
df_urth = get_data("URTH")

if df_btc is not None:
    tab_a, tab_b = st.tabs(["📈 Module A: Bitcoin Strategy", "📁 Module B: Portfolio Analysis"])

    # Module A: Bitcoin Quantitative Analysis
    with tab_a:
        st.header("Bitcoin Price & Momentum Strategy")
        df_a = df_btc.copy()
        df_a['Daily_Return'] = df_a['close'].pct_change()
        df_a['SMA20'] = df_a['close'].rolling(window=20).mean()
        df_a['SMA50'] = df_a['close'].rolling(window=50).mean()
        
        # Trading Signal
        df_a['Signal'] = np.where(df_a['SMA20'] > df_a['SMA50'], 1, 0)
        df_a['Momentum_Return'] = df_a['Signal'].shift(1) * df_a['Daily_Return']
        
        # Cumulative Performance (starting with $10,000)
        df_a['Cum_Buy_Hold'] = (1 + df_a['Daily_Return']).cumprod() * 10000
        df_a['Cum_Momentum'] = (1 + df_a['Momentum_Return']).cumprod() * 10000

        st.subheader("Cumulative Returns Comparison (USD)")
        st.line_chart(df_a[['Cum_Buy_Hold', 'Cum_Momentum']])

    # Module B: Multi-Asset Portfolio Analysis
    with tab_b:
        if df_gld is not None and df_urth is not None:
            st.header("Diversification Analysis")
            combined = pd.DataFrame({
                "Bitcoin": df_btc["close"],
                "Gold": df_gld["close"],
                "MSCI World": df_urth["close"]
            }).dropna()
            
            returns_b = combined.pct_change().dropna()
            
            # Simple Portfolio (Equal Weights)
            weights = [0.33, 0.33, 0.34]
            returns_b['Portfolio'] = returns_b.dot(weights)
            portfolio_values = (1 + returns_b).cumprod() * 10000
            
            st.subheader("Asset Performance vs Portfolio")
            st.line_chart(portfolio_values[['Portfolio', 'Bitcoin', 'MSCI World']])
            
            # Correlation Heatmap
            st.subheader("Asset Correlation Matrix")
            fig_b, ax_b = plt.subplots()
            sns.heatmap(returns_b[['Bitcoin', 'Gold', 'MSCI World']].corr(), annot=True, cmap='coolwarm', ax=ax_b)
            st.pyplot(fig_b)
        else:
            st.warning("Data for Gold or MSCI World is unavailable. Please refresh in 1 minute.")
else:
    st.error("Unable to load initial data. Check your Alpha Vantage API key.")
