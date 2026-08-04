import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
from sklearn.linear_model import LinearRegression
import io

# =====================================================================
# SYSTEM CONFIGURATION & INTERFACE LAYOUT
# =====================================================================
st.set_page_config(page_title="AI Financial & Crypto Dashboard", layout="wide")

st.title("📊 AI Portfolio Optimizer & Risk Dashboard")
st.write("Predict trends, calculate optimal weights, and test risk controls for Stocks and Cryptocurrencies.")

# ---------------------------------------------------------------------
# SIDEBAR CONTROLS (Your Visual Input Control Panel)
# ---------------------------------------------------------------------
st.sidebar.header("🔧 System Settings")

ticker_input = st.sidebar.text_input(
    "Asset Tickers (Comma Separated)", 
    "AAPL, MSFT, BTC-USD, ETH-USD"
)
tickers = [t.strip().upper() for t in ticker_input.split(",")]

start_date = st.sidebar.date_input("Backtest Start Date", pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("Backtest End Date", pd.to_datetime("2026-07-15"))

st.sidebar.subheader("🛡️ Risk & Fee Controls")
stop_loss = st.sidebar.slider("Stop-Loss Trigger (%)", -5.0, -0.5, -2.0, step=0.5) / 100
take_profit = st.sidebar.slider("Take-Profit Trigger (%)", 1.0, 10.0, 4.0, step=0.5) / 100
fee_rate = st.sidebar.number_input("Broker/Exchange Fee per Trade (%)", 0.0, 1.0, 0.1, step=0.05) / 100

run_pipeline = st.sidebar.button("🚀 Run Complete System Analysis")

# =====================================================================
# CORE SYSTEM ENGINE EXECUTION Pipeline
# =====================================================================
if run_pipeline:
    if len(tickers) < 2:
        st.error("Please enter at least 2 tickers to optimize a portfolio.")
    else:
        with st.spinner("📥 Fetching historical market datasets from live streams..."):
            try:
                df_raw = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
                
                if isinstance(df_raw.columns, pd.MultiIndex):
                    df_raw.columns = df_raw.columns.get_level_values(0)
                
                df_raw = df_raw.dropna()
                returns = df_raw.pct_change().dropna()
                
                st.success(f"Successfully processed {len(df_raw)} historical trading windows!")
                
                # -----------------------------------------------------
                # MODULE 1: MACHINE LEARNING & SENTIMENT FORECASTS
                # -----------------------------------------------------
                st.subheader("🤖 1. Machine Learning & News Sentiment Forecasts")
                col1, col2 = st.columns()
                
                predicted_moves = {}
                ml_summary_data = []
                
                for ticker in tickers:
                    if ticker not in df_raw.columns:
                        continue
                    
                    ticker_df = pd.DataFrame(df_raw[ticker]).rename(columns={ticker: 'Close'})
                    ticker_df['Lag_1'] = ticker_df['Close'].shift(1)
                    ticker_df['MA_5'] = ticker_df['Close'].rolling(5).mean()
                    
                    np.random.seed(42) 
                    ticker_df['News_Sentiment'] = np.random.uniform(-0.4, 0.7, size=len(ticker_df))
                    
                    ticker_df['Target'] = ticker_df['Close'].shift(-1)
                    ticker_df = ticker_df.dropna()
                    
                    X = ticker_df[['Lag_1', 'MA_5', 'News_Sentiment']]
                    y = ticker_df['Target']
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    latest_sentiment = np.random.uniform(-0.2, 0.8) 
                    latest_features = [[ticker_df['Close'].iloc[-1], ticker_df['Close'].rolling(5).mean().iloc[-1], latest_sentiment]]
                    prediction = model.predict(latest_features)
                    
                    last_close = ticker_df['Close'].iloc[-1]
                    expected_move = (prediction - last_close) / last_close
                    predicted_moves[ticker] = expected_move
                    
                    direction = "📈 UP" if expected_move > 0 else "📉 DOWN"
                    ml_summary_data.append({
                        "Asset Ticker": ticker,
                        "Current Price": f"${last_close:,.2f}",
                        "Simulated News Sentiment": f"{latest_sentiment:+.2f}",
                        "Predicted Next Move": f"{expected_move:+.2%}",
                        "Market Direction Signal": direction
                    })
                
                with col1:
                    st.dataframe(pd.DataFrame(ml_summary_data), use_container_width=True)
                with col2:
                    st.info("💡 **How it works:** The AI looks at yesterday's close price, the 5-day trend momentum, and alternative text news sentiment scores to forecast the upcoming direction.")

                # -----------------------------------------------------
                # MODULE 2: MATHEMATICAL PORTFOLIO OPTIMIZER
                # -----------------------------------------------------
                st.subheader("⚖️ 2. Mathematically Optimized Weight Allocations")
                
                active_tickers = list(predicted_moves.keys())
                num_assets = len(active_tickers)
                cov_matrix = returns[active_tickers].cov()
                
                def portfolio_volatility(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for asset in range(num_assets))
                initial_guess = num_assets * [1. / num_assets]
                
                optimized = sco.minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
                optimal_weights = optimized['x']
                
                weight_data = []
                for ticker, weight in zip(active_tickers, optimal_weights):
                    final_weight = weight if predicted_moves[ticker] > 0 else 0.0
                    weight_data.append({"Asset Ticker": ticker, "Optimal Allocation Weight": final_weight})
                
                df_weights = pd.DataFrame(weight_data)
                total_w = df_weights["Optimal Allocation Weight"].sum()
                if total_w > 0:
                    df_weights["Optimal Allocation Weight"] = df_weights["Optimal Allocation Weight"] / total_w
                else:
                    df_weights["Optimal Allocation Weight"] = 1.0 / len(df_weights) 
                
                c1, c2 = st.columns()
                with c1:
                    st.write("**Calculated Optimal Distributions:**")
                    df_weights_styled = df_weights.copy()
                    df_weights_styled["Optimal Allocation Weight"] = df_weights_styled["Optimal Allocation Weight"].map(lambda x: f"{x:.2%}")
                    st.table(df_weights_styled)
                with c2:
                    st.bar_chart(df_weights.set_index("Asset Ticker"))

                # -----------------------------------------------------
                # MODULE 3: STRATEGY BACKTESTER WITH ACCOUNT FRICTION
                # -----------------------------------------------------
                st.subheader("📉 3. Historical Backtest Simulation Performance")
                
                final_weights_arr = df_weights["Optimal Allocation Weight"].values
                portfolio_returns = (returns[active_tickers] * final_weights_arr).sum(axis=1)
                
                backtest_df = pd.DataFrame({'Market_Raw': portfolio_returns})
                backtest_df['SMA_10'] = backtest_df['Market_Raw'].rolling(10).mean()
                
                signals = []
                fees_paid = []
                net_returns = []
                current_pos = 0.0
                
                for i in range(len(backtest_df)):
                    raw_ret = backtest_df['Market_Raw'].iloc[i]
                    sma_val = backtest_df['SMA_10'].iloc[i]
                    target_pos = 1.0 if raw_ret > (sma_val if not pd.isna(sma_val) else 0) else 0.0
                    
                    if current_pos == 1.0:
                        if raw_ret <= stop_loss:
                            target_pos = 0.0  
                        elif raw_ret >= take_profit:
                            target_pos = 0.0  
                    
                    fee = fee_rate if target_pos != current_pos else 0.0
                    day_net = (raw_ret * current_pos) - fee
                    
                    signals.append(target_pos)
                    fees_paid.append(fee)
                    net_returns.append(day_net)
                    current_pos = target_pos
                    
                backtest_df['Net_Strategy'] = net_returns
                backtest_df['Cum_Market'] = (1 + backtest_df['Market_Raw']).cumprod()
                backtest_df['Cum_Strategy'] = (1 + backtest_df['Net_Strategy']).cumprod()
                
                chart_data = backtest_df[['Cum_Market', 'Cum_Strategy']].rename(
                    columns={'Cum_Market': 'Standard Buy & Hold Portfolio', 'Cum_Strategy': 'AI Tactical Portfolio (Net)'}
                )
                st.line_chart(chart_data)
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Benchmark Return", f"{(backtest_df['Cum_Market'].iloc[-1]-1):.2%}")
