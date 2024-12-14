import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.graph_objects as go
from textblob import TextBlob
import requests
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Set INR conversion rate (approximate, can be updated via an API for real-time rates)
USD_TO_INR = 1.0

# Google News API setup
NEWS_API_KEY = '56a72f0b22ae42dfadb3f5283e48caeb'  # Replace with your actual API key

# Streamlit App Header
st.set_page_config(page_title="Stock Market Predictor", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to bottom, #f0f2f6, #d9e6f2);
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        border-right: 1px solid #ccc;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto', sans-serif;
        color: #2c3e50;
    }
    .news-card {
        background: #ffffff;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        padding: 15px;
    }
    .news-title {
        font-size: 16px;
        font-weight: bold;
        color: #1a73e8;
    }
    .news-description {
        font-size: 14px;
        color: #333;
        margin-top: 5px;
    }
    .news-link {
        font-size: 13px;
        color: #0066cc;
        text-decoration: none;
    }
    .news-link:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

st.title('📈 **Stock Market Predictor**')

# Sidebar for user inputs
st.sidebar.header("Configuration")
stock = st.sidebar.text_input('Enter Stock Symbol (e.g., INFY.NS, RELIANCE.NS)', 'INFY.NS')

# Allow start_date to be any past date, up to today
start_date = st.sidebar.date_input('Start Date', value=datetime(2024, 7, 1), min_value=datetime(2000, 1, 1), max_value=datetime.today())
end_date = st.sidebar.date_input('End Date', value=datetime.today(), min_value=start_date, max_value=datetime.today())

# Fetch stock data based on user input dates
data = yf.download(stock, start=start_date, end=end_date)

# Check if data is empty
if data.empty:
    # st.error("⚠️ No data available for the selected stock and date range. Please try another combination.")
    st.stop()

# Convert Close price to INR
data['Close_INR'] = data['Close'] * USD_TO_INR
data.reset_index(inplace=True)

# Display stock data in a table
st.subheader('📊 **Stock Data (in INR)**')
st.dataframe(data[['Date', 'Close_INR']])  # Use st.dataframe for proper header display

# Moving Averages Calculation
ma_50 = data['Close_INR'].rolling(50).mean()
ma_100 = data['Close_INR'].rolling(100).mean()
ma_200 = data['Close_INR'].rolling(200).mean()

# Moving Averages Visualization with Plotly
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Close_INR'], mode='lines', name='Close Price', line=dict(color='green')))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma_50, mode='lines', name='50-day MA', line=dict(color='red', dash='dash')))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma_100, mode='lines', name='100-day MA', line=dict(color='blue', dash='dash')))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma_200, mode='lines', name='200-day MA', line=dict(color='purple', dash='dash')))

fig_ma.update_layout(
    title="Close Price and Moving Averages (in INR)",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    legend_title="Legend",
    template="plotly_white",
    height=600,
    width=1000
)

st.plotly_chart(fig_ma)


# Buy/Sell Signals Feature
def compute_rsi(data, window=14):
    delta = data['Close_INR'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data)

def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close_INR'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close_INR'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

data['MACD'], data['MACD_Signal'] = compute_macd(data)

def compute_bollinger_bands(data, window=20):
    sma = data['Close_INR'].rolling(window).mean()
    stddev = data['Close_INR'].rolling(window).std()
    upper_band = sma + (2 * stddev)
    lower_band = sma - (2 * stddev)
    return upper_band, lower_band

data['BB_Upper'], data['BB_Lower'] = compute_bollinger_bands(data)

def generate_signals(data):
    buy_signals = []
    sell_signals = []
    for i in range(len(data)):
        if data['RSI'][i] < 30 and data['Close_INR'][i] < data['BB_Lower'][i]:
            buy_signals.append(data['Close_INR'][i])
            sell_signals.append(np.nan)
        elif data['RSI'][i] > 70 and data['Close_INR'][i] > data['BB_Upper'][i]:
            sell_signals.append(data['Close_INR'][i])
            buy_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
    return buy_signals, sell_signals

data['Buy_Signal'], data['Sell_Signal'] = generate_signals(data)

# Plot Buy/Sell Signals
fig_signals = go.Figure()
fig_signals.add_trace(go.Scatter(x=data['Date'], y=data['Close_INR'], mode='lines', name='Close Price', line=dict(color='green')))
fig_signals.add_trace(go.Scatter(x=data['Date'], y=data['Buy_Signal'], mode='markers', name='Buy Signal', marker=dict(color='blue', size=10, symbol='triangle-up')))
fig_signals.add_trace(go.Scatter(x=data['Date'], y=data['Sell_Signal'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')))

fig_signals.update_layout(
    title="Buy/Sell Signals with Stock Prices (in INR)",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    legend_title="Legend",
    template="plotly_white",
    height=600,
    width=1000
)

# Buy/Sell Signals Summary
st.subheader("Buy/Sell Signals Summary")
buy_dates = data[data['Buy_Signal'].notnull()]['Date'].dt.strftime('%Y-%m-%d')
st.plotly_chart(fig_signals)

sell_dates = data[data['Sell_Signal'].notnull()]['Date'].dt.strftime('%Y-%m-%d')

summary_table = pd.DataFrame({
    'Buy Dates': buy_dates,
    'Sell Dates': sell_dates
}).fillna('---')

st.write(summary_table)

# Prepare Data for Predictions
min_required_data = 20  # Minimum data points required to proceed
if len(data) < min_required_data:
    st.error(f"⚠️ Not enough data points for analysis (minimum {min_required_data} required). Please select a larger date range.")
    st.stop()

time_steps = min(100, len(data) - 1)  # Adjust time_steps dynamically for small datasets
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close_INR']])
X, y = [], []

for i in range(time_steps, len(data_scaled)):
    X.append(data_scaled[i - time_steps:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
r2 = r2_score(y_test_rescaled, predictions_rescaled)
# st.sidebar.markdown(f"# Model Metrics")
# st.sidebar.markdown(f"**MSE:** {mse:.2f}")
# st.sidebar.markdown(f"**R² Score:** {r2:.2f}")

# Predictions Graph
fig_predicted = go.Figure()
fig_predicted.add_trace(go.Scatter(x=list(range(len(predictions_rescaled))), y=predictions_rescaled.flatten(), mode='lines', name='Predicted Price', line=dict(color='blue')))
fig_predicted.update_layout(
    title="Predicted Stock Prices (in INR)",
    xaxis_title="Time",
    yaxis_title="Price (₹)",
    legend_title="Legend",
    template="plotly_white",
    height=600,
    width=1000
)
st.plotly_chart(fig_predicted)

# Actual Prices Graph
fig_actual = go.Figure()
fig_actual.add_trace(go.Scatter(x=list(range(len(y_test_rescaled))), y=y_test_rescaled.flatten(), mode='lines', name='Actual Price', line=dict(color='green')))
fig_actual.update_layout(
    title="Actual Stock Prices (in INR)",
    xaxis_title="Time",
    yaxis_title="Price (₹)",
    legend_title="Legend",
    template="plotly_white",
    height=600,
    width=1000
)
st.plotly_chart(fig_actual)

def calculate_profit_loss(data):
    current_price = data['Close_INR'].iloc[-1]
    max_future_price = data['Close_INR'].max()
    min_future_price = data['Close_INR'].min()

    max_profit = max_future_price - current_price
    min_loss = current_price - min_future_price

    return max_profit, min_loss

max_profit, min_loss = calculate_profit_loss(data)

# Display the results
st.subheader("💰 **Maximum Profit and Minimum Loss Analysis**")
current_price = data['Close_INR'].iloc[-1]
st.write(f"**Current Price:** ₹{current_price:.2f}")
st.write(f"**Maximum Profit:** ₹{max_profit:.2f} (if the stock reaches ₹{data['Close_INR'].max():.2f})")
st.write(f"**Minimum Loss:** ₹{min_loss:.2f} (if the stock drops to ₹{data['Close_INR'].min():.2f})")


# NEWS SECTION
def fetch_full_name(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        return stock.info.get('longName')
    except Exception as e:
        st.error(f"Could not fetch company name for {stock_symbol}: {e}")
        return stock_symbol

def fetch_stock_news(stock_symbol):
    full_name = fetch_full_name(stock_symbol)
    if not full_name:
        full_name = stock_symbol

    query = f'"{full_name}" stock OR "{full_name}" share price'
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}'
    response = requests.get(url)

    if response.status_code != 200:
        st.error("⚠️ Failed to fetch news articles. Check your API key or network connection.")
        return []

    return response.json().get('articles', [])

def analyze_news_sentiment(news_data):
    sentiment_score = 0
    positive_count = 0
    neutral_count = 0
    negative_count = 0

    for article in news_data:
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''
        text = title + ' ' + description
        analysis = TextBlob(text)
        sentiment_score += analysis.sentiment.polarity

        if analysis.sentiment.polarity > 0:
            positive_count += 1
        elif analysis.sentiment.polarity == 0:
            neutral_count += 1
        else:
            negative_count += 1

    sentiment_score = sentiment_score / len(news_data) if news_data else 0
    return sentiment_score, positive_count, neutral_count, negative_count

def map_sentiment_score(sentiment_score):
    normalized_score = (sentiment_score + 1) * 50
    if normalized_score < 40:
        category = "Strong Sell"
    elif normalized_score < 60:
        category = "Neutral"
    else:
        category = "Strong Buy"
    return normalized_score, category

def display_stock_news_and_sentiment(stock_symbol):
    full_name = fetch_full_name(stock_symbol)
    st.subheader(f"📰 **News for {full_name}**")
    st.markdown("Latest news articles for the selected stock:")

    news_data = fetch_stock_news(stock_symbol)
    sentiment_score, pos_count, neu_count, neg_count = analyze_news_sentiment(news_data)
    normalized_score, category = map_sentiment_score(sentiment_score)

    st.sidebar.markdown(f"# Sentiment Analysis")
    st.sidebar.markdown(f"**Sentiment Score:** {normalized_score:.2f}")
    st.sidebar.markdown(f"**Sentiment Category:** {category}")
    st.sidebar.markdown(f"**Positive Articles:** {pos_count}")
    st.sidebar.markdown(f"**Neutral Articles:** {neu_count}")
    st.sidebar.markdown(f"**Negative Articles:** {neg_count}")

    if news_data:
        for article in news_data:  # Display top 5 news articles
            st.markdown(f"""
            <div class="news-card">
                <div class="news-title">{article['title']}</div>
                <div class="news-description">{article.get('description', 'No description available.')}</div>
                <a class="news-link" href="{article['url']}" target="_blank">Read more...</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write(f"No news articles found for {full_name}.")

display_stock_news_and_sentiment(stock)
