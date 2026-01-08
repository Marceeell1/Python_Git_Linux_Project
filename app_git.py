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
st.title("Quantitative Research Platform (Full Analytical Version)")

API_KEY = "CW1LL675242TSYOU"
URL = "https://www.alphavantage.co/query"

# Caching mechanism for API stability
@st.cache_data(ttl=3600)
def get_data(symbol, is_crypto=False):
    time.sleep(1) # Delay to respect API limits
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

# Load Data
df_btc = get_data("BTC", is_crypto=True)
df_gld = get_data("GLD")
df_urth = get_data("URTH")

if df_btc is not None:
    tab_a, tab_b = st.tabs(["Module A: Analysis of Bitcoin", "Module B: Portfolio Analysis"])

    # --- MODULE A: FULL ANALYSIS FROM NOTEBOOK ---
    with tab_a:
        st.header("Quantitative Analysis of Bitcoin (BTC-USD)")
        df = df_btc.copy()
        
        # 1. Strategy Calculations (MA20/MA50)
        df["MA20"] = df["close"].rolling(window=20).mean()
        df["MA50"] = df["close"].rolling(window=50).mean()
        df["signal"] = (df["MA20"] > df["MA50"]).astype(int)
        df["position"] = df["signal"].shift(1).fillna(0)
        df["returns"] = df["close"].pct_change()
        df["strategy_returns"] = df["position"] * df["returns"]
        df["strategy_returns"] = df["strategy_returns"].fillna(0)
        
        # 2. Key Performance Metrics
        initial_price = df["close"].iloc[0]
        final_price = df["close"].iloc[-1]
        buy_hold_return = (final_price - initial_price) / initial_price
        
        strategy_perf = (1 + df["strategy_returns"]).prod() - 1
        
        sharpe = (df["strategy_returns"].mean() / df["strategy_returns"].std()) * np.sqrt(252)
        
        df["equity"] = (1 + df["strategy_returns"]).cumprod()
        rolling_max = df["equity"].cummax()
        drawdown = (df["equity"] - rolling_max) / rolling_max
        max_dd = drawdown.min()

        st.subheader("Strategy Metrics Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Buy & Hold Return", f"{buy_hold_return:.2%}")
        m2.metric("Momentum Return", f"{strategy_perf:.2%}")
        m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        m4.metric("Max Drawdown", f"{max_dd:.2%}")

        # 3. Momentum Strategy Visualization (Signals)
        st.subheader("Momentum Strategy: Signals (MA20/MA50)")
        fig_sig, ax_sig = plt.subplots(figsize=(14, 6))
        ax_sig.plot(df.index, df["close"], label="BTC Price", alpha=0.5, linewidth=1.3)
        ax_sig.plot(df.index, df["MA20"], label="MA20", alpha=0.8)
        ax_sig.plot(df.index, df["MA50"], label="MA50", alpha=0.8)
        
        buy_signals = df[(df["signal"] == 1) & (df["position"] == 0)]
        sell_signals = df[(df["signal"] == 0) & (df["position"] == 1)]
        ax_sig.scatter(buy_signals.index, buy_signals["close"], marker="^", color="green", s=100, label="Buy")
        ax_sig.scatter(sell_signals.index, sell_signals["close"], marker="v", color="red", s=100, label="Sell")
        ax_sig.legend()
        ax_sig.grid(True, alpha=0.3)
        st.pyplot(fig_sig)

        # 4. Price vs Strategy Equity Visualization
        st.subheader("BTC Price vs Strategy Cumulative Value")
        initial_capital = 10000
        df["equity_momentum"] = initial_capital * df["equity"]
        
        fig_eq, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(df.index, df["close"], color="blue", label="BTC Price (Close)")
        ax1.set_ylabel("BTC Price (USD)", color="blue")
        ax2 = ax1.twinx()
        ax2.plot(df.index, df["equity_momentum"], color="orange", label="Strategy Equity")
        ax2.set_ylabel("Portfolio Value (USD)", color="orange")
        plt.title("BTC Price vs Momentum Strategy Equity")
        fig_eq.tight_layout()
        st.pyplot(fig_eq)

        # 5. Linear Regression Model
        st.divider()
        st.header("Predictive Modeling: Linear Regression")
        df_model = df.copy()
        df_model["time_index"] = np.arange(len(df_model))
        
        X = df_model[["time_index"]]
        y = df_model["close"]
        
        model = LinearRegression()
        model.fit(X, y)
        df_model["predicted_price"] = model.predict(X)
        
        mae = mean_absolute_error(df_model["close"], df_model["predicted_price"])
        st.write(f"Model Mean Absolute Error (MAE): **${mae:,.2f} USD**")

        fig_ml, ax_ml = plt.subplots(figsize=(12, 5))
        ax_ml.plot(df_model.index, df_model["close"], label="Actual Price")
        ax_ml.plot(df_model.index, df_model["predicted_price"], label="Predicted Price", linestyle="--")
        ax_ml.set_title("Actual vs Predicted Price - Linear Regression")
        ax_ml.legend()
        st.pyplot(fig_ml)

        # 6. Future Forecast (10 Days)
        st.subheader("10-Day Price Forecast")
        last_time = df_model["time_index"].iloc[-1]
        future_times = np.arange(last_time + 1, last_time + 11).reshape(-1, 1)
        future_predictions = model.predict(future_times)
        
        forecast_data = pd.DataFrame({
            "Day": [f"Day +{i+1}" for i in range(10)],
            "Forecasted Price (USD)": [f"${p:,.2f}" for p in future_predictions]
        })
        st.table(forecast_data)

    # --- MODULE B: PORTFOLIO DIVERSIFICATION ---
    with tab_b:
        if df_gld is not None and df_urth is not None:
            st.header("Diversification Analysis (BTC, Gold, MSCI World)")
            
            combined = pd.DataFrame({
                "Bitcoin": df_btc["close"],
                "Gold": df_gld["close"],
                "MSCI World": df_urth["close"]
            }).dropna()
            
            returns_b = combined.pct_change().dropna()
            weights = np.array([0.33, 0.33, 0.34])
            returns_b['Portfolio'] = returns_b.dot(weights)
            
            # Comparison Plot
            st.subheader("Performance Comparison (Base 100)")
            comp_norm = (combined / combined.iloc[0]) * 100
            # Adding Portfolio to norm for chart
            comp_norm['Portfolio'] = (1 + returns_b['Portfolio']).cumprod() * 100
            st.line_chart(comp_norm)
            
            # Metrics
            ann_vol = returns_b['Portfolio'].std() * np.sqrt(252)
            indiv_vols = returns_b[['Bitcoin', 'Gold', 'MSCI World']].std() * np.sqrt(252)
            theoretical_risk = np.sum(indiv_vols * weights)
            risk_reduction = (1 - (ann_vol / theoretical_risk)) * 100
            
            c1, c2 = st.columns(2)
            c1.metric("Portfolio Annualized Volatility", f"{ann_vol:.2%}")
            c2.metric("Risk Reduction via Diversification", f"{risk_reduction:.2f}%")
            
            # Correlation Matrix
            st.subheader("Asset Correlation Heatmap")
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(returns_b[['Bitcoin', 'Gold', 'MSCI World']].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.warning("Please wait 1 minute for API reset to access Gold and MSCI World data.")
