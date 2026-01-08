import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# Requirement #5: Auto-refresh every 5 minutes (300,000 ms)
from streamlit_autorefresh import st_autorefresh

# --- CONFIGURATION ---
st.set_page_config(page_title="Quantitative Research Platform", layout="wide")
st_autorefresh(interval=300000, key="datarefresh")
st.title("Quantitative Research Platform (Full Analytical Version)")

API_KEY = "CW1LL675242TSYOU"
URL = "https://www.alphavantage.co/query"

# --- SIDEBAR ---
st.sidebar.header("Strategy Settings")
st.sidebar.subheader("Module A: Momentum")
ma_fast = st.sidebar.slider("Fast MA", 5, 50, 20)
ma_slow = st.sidebar.slider("Slow MA", 51, 200, 50)

st.sidebar.subheader("Module B: Portfolio Weights")
w_btc = st.sidebar.slider("Bitcoin %", 0, 100, 33)
w_gld = st.sidebar.slider("Gold %", 0, 100, 33)
w_urth = st.sidebar.slider("MSCI World %", 0, 100, 34)

if w_btc + w_gld + w_urth != 100:
    st.sidebar.error(f"Total: {w_btc + w_gld + w_urth}% - Please adjust to 100%")

# --- DATA FUNCTION ---
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
        col = "4b. close (USD)" if "4b. close (USD)" in df.columns else "4. close"
        df = df[[col]].rename(columns={col: "close"}).astype(float)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except: return None

# Load Data
df_btc = get_data("BTC", is_crypto=True)
df_gld = get_data("GLD")
df_urth = get_data("URTH")

# Corrected API Limit check
is_limit = any(isinstance(x, str) and x == "LIMIT" for x in [df_btc, df_gld, df_urth])
if is_limit:
    st.error("API Limit reached. Refreshing in 5 minutes or wait 60s manually.")
    st.stop()

if df_btc is not None:
    tab_a, tab_b = st.tabs(["Module A: Analysis", "Module B: Portfolio"])

    # MODULE A
    with tab_a:
        st.header(f"Bitcoin Strategy (MA{ma_fast}/MA{ma_slow})")
        df = df_btc.copy()
        df['Returns'] = df['close'].pct_change()
        df['MA_F'], df['MA_S'] = df['close'].rolling(ma_fast).mean(), df['close'].rolling(ma_slow).mean()
        df['Signal'] = (df['MA_F'] > df['MA_S']).astype(int)
        df['Strat_Ret'] = df['Signal'].shift(1) * df['Returns']
        equity = (1 + df['Strat_Ret'].fillna(0)).cumprod()

        # Metrics & Charts
        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy Return", f"{((1 + df['Strat_Ret'].fillna(0)).prod() - 1):.2%}")
        c2.metric("Sharpe Ratio", f"{(df['Strat_Ret'].mean()/df['Strat_Ret'].std()*np.sqrt(252)):.2f}")
        c3.metric("Max Drawdown", f"{((equity - equity.cummax())/equity.cummax()).min():.2%}")

        st.subheader("1. Momentum Signals")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(df['close'], alpha=0.3)
        ax1.plot(df['MA_F'], label="Fast MA")
        ax1.plot(df['MA_S'], label="Slow MA")
        st.pyplot(fig1)

        st.subheader("2. Price vs Strategy Equity")
        fig2, ax_p = plt.subplots(figsize=(12, 4))
        ax_p.plot(df['close'], color="blue", label="BTC Price")
        ax_e = ax_p.twinx()
        ax_e.plot(equity * 10000, color="orange", label="Equity")
        st.pyplot(fig2)

        # ML
        st.divider()
        st.subheader("Linear Regression Prediction")
        df_ml = df[['close']].dropna()
        df_ml['T'] = np.arange(len(df_ml))
        model = LinearRegression().fit(df_ml[['T']], df_ml['close'])
        st.info(f"MAE: ${mean_absolute_error(df_ml['close'], model.predict(df_ml[['T']])):,.2f}")
        fut = np.array([[len(df_ml) + i] for i in range(1, 11)])
        st.table(pd.DataFrame({"Day": [f"+{i}" for i in range(1,11)], "Price": model.predict(fut)}))

    # MODULE B
    with tab_b:
        if df_gld is not None and df_urth is not None:
            st.header("Portfolio Diversification")
            combined = pd.DataFrame({"BTC": df_btc["close"], "Gold": df_gld["close"], "MSCI": df_urth["close"]}).dropna()
            rets = combined.pct_change().dropna()
            w = np.array([w_btc/100, w_gld/100, w_urth/100])
            rets['Portfolio'] = rets.dot(w)
            st.line_chart((1 + rets).cumprod() * 100)
            
            v_p = rets['Portfolio'].std() * np.sqrt(252)
            theo_v = np.sum(rets[['BTC', 'Gold', 'MSCI']].std() * np.sqrt(252) * w)
            st.metric("Risk Reduction", f"{(1 - (v_p/theo_v))*100:.2f}%")
            
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(rets[['BTC', 'Gold', 'MSCI']].corr(), annot=True, cmap='coolwarm')
            st.pyplot(fig_corr)
