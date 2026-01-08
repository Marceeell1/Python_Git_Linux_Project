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
st.title("Quantitative Research Platform (Full Version)")

API_KEY = "CW1LL675242TSYOU"
URL = "https://www.alphavantage.co/query"

@st.cache_data(ttl=3600)
def get_data(symbol, is_crypto=False):
    time.sleep(1) # Security delay for API limits
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
            st.error(f"API Limit reached for {symbol}. Please wait 1 minute.")
            return None
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

# Load all data at once
df_btc = get_data("BTC", is_crypto=True)
df_gld = get_data("GLD")
df_urth = get_data("URTH")

if df_btc is not None:
    tab_a, tab_b = st.tabs(["Module A: Bitcoin Strategy", "Module B: Portfolio Analysis"])

    # --- MODULE A: BITCOIN QUANTITATIVE ANALYSIS ---
    with tab_a:
        st.header("Bitcoin Strategy and Performance Metrics")
        
        df_a = df_btc.copy()
        # Calculations
        df_a['Returns'] = df_a['close'].pct_change()
        df_a['MA20'] = df_a['close'].rolling(window=20).mean()
        df_a['MA50'] = df_a['close'].rolling(window=50).mean()
        df_a['Signal'] = (df_a['MA20'] > df_a['MA50']).astype(int)
        df_a['Position'] = df_a['Signal'].shift(1).fillna(0)
        df_a['Strategy_Returns'] = df_a['Position'] * df_a['Returns']
        
        # Performance Metrics
        bh_return = (df_a['close'].iloc[-1] / df_a['close'].iloc[0] - 1)
        strat_return = (1 + df_a['Strategy_Returns'].fillna(0)).prod() - 1
        sharpe = (df_a['Strategy_Returns'].mean() / df_a['Strategy_Returns'].std()) * np.sqrt(252)
        
        equity = (1 + df_a['Strategy_Returns'].fillna(0)).cumprod()
        max_dd = -((equity - equity.cummax()) / equity.cummax()).min()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Buy & Hold Return", f"{bh_return:.2%}")
        col2.metric("Strategy Return", f"{strat_return:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col4.metric("Max Drawdown", f"{max_dd:.2%}")

        # Plot Signals
        st.subheader("Momentum Strategy Signals (MA20 vs MA50)")
        fig_sig, ax_sig = plt.subplots(figsize=(12, 5))
        ax_sig.plot(df_a.index, df_a['close'], label='Price', alpha=0.5)
        ax_sig.plot(df_a.index, df_a['MA20'], label='MA20')
        ax_sig.plot(df_a.index, df_a['MA50'], label='MA50')
        
        buy = df_a[(df_a['Signal'] == 1) & (df_a['Signal'].shift(1) == 0)]
        sell = df_a[(df_a['Signal'] == 0) & (df_a['Signal'].shift(1) == 1)]
        ax_sig.scatter(buy.index, buy['close'], marker='^', color='green', s=100, label='Buy')
        ax_sig.scatter(sell.index, sell['close'], marker='v', color='red', s=100, label='Sell')
        ax_sig.legend()
        st.pyplot(fig_sig)

        # Machine Learning Section
        st.divider()
        st.header("Linear Regression Price Prediction")
        
        df_ml = df_a[['close']].dropna()
        df_ml['Time_Index'] = np.arange(len(df_ml))
        X = df_ml[['Time_Index']]
        y = df_ml['close']
        
        model = LinearRegression()
        model.fit(X, y)
        df_ml['Predicted'] = model.predict(X)
        mae = mean_absolute_error(df_ml['close'], df_ml['Predicted'])
        
        st.write(f"Model Mean Absolute Error (MAE): ${mae:.2f}")
        
        fig_ml, ax_ml = plt.subplots(figsize=(10, 4))
        ax_ml.plot(df_ml.index, df_ml['close'], label='Actual Price')
        ax_ml.plot(df_ml.index, df_ml['Predicted'], label='Predicted (Regression)', linestyle='--')
        ax_ml.legend()
        st.pyplot(fig_ml)

        # Future 10 Days
        last_idx = df_ml['Time_Index'].iloc[-1]
        future_idx = np.arange(last_idx + 1, last_idx + 11).reshape(-1, 1)
        preds = model.predict(future_idx)
        
        st.subheader("Forecast for the next 10 days")
        future_df = pd.DataFrame({"Day": [f"Day +{i+1}" for i in range(10)], "Predicted Price": preds})
        st.table(future_df)

    # --- MODULE B: PORTFOLIO ANALYSIS ---
    with tab_b:
        if df_gld is not None and df_urth is not None:
            st.header("Multi-Asset Portfolio Comparison")
            
            combined = pd.DataFrame({
                "Bitcoin": df_btc["close"],
                "Gold": df_gld["close"],
                "MSCI World": df_urth["close"]
            }).dropna()
            
            returns = combined.pct_change().dropna()
            weights = np.array([0.33, 0.33, 0.34])
            returns['Portfolio'] = returns.dot(weights)
            
            # Growth of $10,000
            equity_df = (1 + returns).cumprod() * 10000
            st.subheader("Performance Comparison (Base $10,000)")
            st.line_chart(equity_df)
            
            # Metrics
            ann_vol = returns['Portfolio'].std() * np.sqrt(252)
            sharpe_p = (returns['Portfolio'].mean() / returns['Portfolio'].std()) * np.sqrt(252)
            
            # Risk Reduction calculation from Notebook
            indiv_vols = returns[['Bitcoin', 'Gold', 'MSCI World']].std() * np.sqrt(252)
            theoretical_risk = np.sum(indiv_vols * weights)
            risk_reduction = (1 - (ann_vol / theoretical_risk)) * 100

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Portfolio Sharpe", f"{sharpe_p:.2f}")
            m_col2.metric("Annualized Volatility", f"{ann_vol:.2%}")
            m_col3.metric("Risk Reduction", f"{risk_reduction:.2f}%")

            st.subheader("Correlation Heatmap")
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(returns[['Bitcoin', 'Gold', 'MSCI World']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.warning("Please wait 1 minute for API refresh to load Gold and MSCI World.")

else:
    st.error("Fatal Error: Could not load Bitcoin data. Verify your API Key.")
