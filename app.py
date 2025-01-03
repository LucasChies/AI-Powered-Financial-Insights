from flask import Flask, render_template, request, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import feedparser
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import base64
import requests
from prophet import Prophet
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")  # Use variable for security

# Set up OpenAI API key
openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")  # Use environment variable
openai.api_base = "https://lchie-m48gp5rl-francecentral.openai.azure.com/"
openai.api_version = "2024-02-01"

# Configure logging
logging.basicConfig(level=logging.INFO)

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

@lru_cache(maxsize=100)  # Caching stock data
def get_stock_data(ticker):
    try:
        data = yf.download(ticker, period="10d")
        if data.empty:
            return None
        return data
    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        return None

# Function to generate AI insights
def get_ai_insight(prompt):
    try:
        response = openai.Completion.create(
            engine="gpt-35-turbo",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating AI insight: {e}"

# Updated get_crypto_data function to include AI insights
def get_crypto_data_with_insight(crypto_symbol):
    try:
        # Fetch crypto data
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY"),  # Store your API key in an environment variable
        }
        params = {"symbol": crypto_symbol.upper(), "convert": "USD"}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        if crypto_symbol.upper() in data["data"]:
            crypto_data = data["data"][crypto_symbol.upper()]

            # Generate AI insights
            insight_prompt = f"Analyze the cryptocurrency {crypto_symbol} based on the current market data:\n{crypto_data}. Provide insights and recommendations for potential investors."
            ai_insight = get_ai_insight(insight_prompt)

            return crypto_data, ai_insight
        else:
            return None, None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching cryptocurrency data: {e}")
        return None, None
    
# Crypto Analyze    
def analyze_crypto_data(crypto_data):
    try:
        market_cap = crypto_data["quote"]["USD"]["market_cap"]
        volume_24h = crypto_data["quote"]["USD"]["volume_24h"]
        price = crypto_data["quote"]["USD"]["price"]
        change_24h = crypto_data["quote"]["USD"]["percent_change_24h"]

        volume_market_cap_ratio = volume_24h / market_cap if market_cap > 0 else None

        analysis = {
            "Market Cap (USD)": market_cap,
            "Volume 24h (USD)": volume_24h,
            "Price (USD)": price,
            "Change 24h (%)": change_24h,
            "Volume/Market Cap Ratio": round(volume_market_cap_ratio, 2) if volume_market_cap_ratio else "N/A",
        }

        return analysis
    except KeyError as e:
        return f"Error in analyzing crypto data: {e}"

# Plotly crypto price
def plot_crypto_price(price_data, crypto_symbol):
    try:
        fig = px.line(
            price_data,
            x="Date",
            y="Price (USD)",
            title=f"{crypto_symbol} Price Chart",
            labels={"Price (USD)": "Price (USD)"}
        )
        return fig.to_html()
    except Exception as e:
        return f"Error generating crypto price chart: {e}"
    
# Crypto News from Cryptopanic API
def get_crypto_news(crypto_symbol, api_key, limit=5):
    """
    Fetch cryptocurrency news using the CryptoPanic API.

    Args:
        crypto_symbol (str): Cryptocurrency symbol (e.g., BTC, ETH).
        api_key (str): Your CryptoPanic API key.
        limit (int): Number of news items to fetch.

    Returns:
        list: A list of news items, each as a dictionary with title, URL, and published time.
    """
    try:
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": api_key,
            "currencies": crypto_symbol.lower(),  # API requires lowercase
            "filter": "trending",
            "public": "true"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()

        # Extract relevant news items
        news_items = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "published": item.get("published_at")
            }
            for item in data.get("results", [])[:limit]
        ]

        return news_items
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching crypto news: {e}")
        return []

# Risk management: Value at Risk (VaR)
def calculate_var(data, confidence_level=0.95):
    try:
        returns = data['Close'].pct_change().dropna()
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return round(var * 100, 2)  # As a percentage
    except Exception as e:
        return f"Error in VaR calculation: {e}"

# Risk management: Maximum Drawdown
def calculate_max_drawdown(data):
    try:
        running_max = data['Close'].cummax()
        drawdown = (data['Close'] - running_max) / running_max
        max_drawdown = drawdown.min()
        return round(max_drawdown * 100, 2)  # As a percentage
    except Exception as e:
        return f"Error in drawdown calculation: {e}"
    
# Forecast future stock prices
def forecast_stock_prices(data, periods=10):
    try:
        # Prepare data for Prophet
        df = data.reset_index()[['Date', 'Close']]
        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

        # Initialize and fit Prophet model
        model = Prophet()
        model.fit(df)

        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Extract relevant columns for forecast
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    except Exception as e:
        return f"Error in forecasting: {e}"
    
# Plotly to visualize forecast results
def plot_forecast(forecast, data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Historical'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
    fig.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
    return fig.to_html()
    
# Plot stock data
def plot_stock(data, ticker):
    try:
        fig = px.line(
            data.reset_index(),
            x='Date',
            y='Close',
            title=f"{ticker} Stock Prices - Last 10 Days",
            labels={"Close": "Close Price"}
        )
        return fig.to_html()
    except Exception as e:
        return f"Error generating plot: {e}"
    
# Plot comparison data
def plot_comparison(data1, data2, data3, ticker1, ticker2, ticker3):
    try:
        data1['Date'] = data1.index
        data1['Ticker'] = ticker1
        data2['Date'] = data2.index
        data2['Ticker'] = ticker2
        data3['Date'] = data3.index
        data3['Ticker'] = ticker3

        combined_data = pd.concat([data1, data2, data3])

        fig = px.line(
            combined_data,
            x='Date',
            y='Close',
            color='Ticker',
            title=f"Comparison of {ticker1} and {ticker2} and {ticker3}",
            labels={"Close": "Close Price"}
        )
        return fig.to_html()
    except Exception as e:
        return f"Error generating comparison plot: {e}"

# Heatmap
def generate_correlation_heatmap(data1, data2, data3, ticker1, ticker2, ticker3):
    try:
        # Prepare data for correlation
        merged_data = pd.DataFrame({
            ticker1: data1['Close'],
            ticker2: data2['Close'],
            ticker3: data3['Close']
        })
        correlation = merged_data.corr()

        # Plot heatmap
        plt.figure(figsize=(5, 4))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")

        # Save the plot as an image and encode it
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return f"<img src='data:image/png;base64,{encoded_image}'/>"
    except Exception as e:
        return f"Error generating heatmap: {e}"
    
# Generate correlation heatmap including AI insight integration
def generate_correlation_heatmap(data1, data2, data3, ticker1, ticker2, ticker3):
    try:
        # Prepare data for correlation
        merged_data = pd.DataFrame({
            ticker1: data1['Close'],
            ticker2: data2['Close'],
            ticker3: data3['Close']
        })
        correlation = merged_data.corr()

        # Plot heatmap
        plt.figure(figsize=(5, 4))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")

        # Save the plot as an image and encode it
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Generate AI insights
        insight_prompt = (
            f"Analyze the following stock correlation matrix:\n\n{correlation.to_string()}\n\n"
            "Provide key insights on how these stocks are related and what this might imply for portfolio diversification or risk."
        )
        ai_insight = get_ai_insight(insight_prompt)

        # Combine heatmap and insight into the output
        return f"<img src='data:image/png;base64,{encoded_image}'/><p>{ai_insight}</p>"
    except Exception as e:
        return f"Error generating heatmap: {e}"

    
# Function to fetch and parse RSS news feed from Yahoo Finance
def get_stock_news(ticker):
    try:
        rss_url = f"https://finance.yahoo.com/rss/quote/{ticker}"
        feed = feedparser.parse(rss_url)
        news = []
        for entry in feed.entries[:5]:
            news_item = {
                "title": entry.get("title", "No Title Available"),
                "link": entry.get("link", "#"),
                "published": entry.get("published", "No Date Available"),
                "summary": entry.get("summary", "No Summary Available")
            }
            news.append(news_item)
        return news
    except Exception as e:
        return f"Error retrieving news: {e}"

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    error = None
    if form.validate_on_submit():
        if form.username.data == 'admin' and form.password.data == 'password123':
            session['logged_in'] = True
            return redirect(url_for('index'))
        error = "Invalid username or password."
    return render_template('login.html', form=form, error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/crypto', methods=['GET', 'POST'])
def crypto_analysis():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        crypto_symbol = request.form.get('crypto_symbol')
        if not crypto_symbol:
            return render_template('crypto_form.html', error="Please enter a cryptocurrency symbol.")
        crypto_symbol = crypto_symbol.upper()

        # Fetch news using CryptoPanic API
        api_key = os.getenv("CRYPTOPANIC_API_KEY")  # Store your CryptoPanic API key in environment variables
        news_items = get_crypto_news(crypto_symbol, api_key)

        crypto_data, ai_insight = get_crypto_data_with_insight(crypto_symbol)
        if not crypto_data:
            return render_template('crypto_result.html', error="No data found for the given cryptocurrency.")

        crypto_analysis = analyze_crypto_data(crypto_data)
        price_data = pd.DataFrame({
            "Date": [pd.Timestamp.now()],
            "Price (USD)": [crypto_data["quote"]["USD"]["price"]]
        })
        price_chart = plot_crypto_price(price_data, crypto_symbol)

        return render_template(
            'crypto_result.html',
            crypto=crypto_data,
            analysis=crypto_analysis,
            insight=ai_insight,
            price_chart=price_chart,
            news_items=news_items
        )
    return render_template('crypto_form.html')

@app.route('/single', methods=['GET', 'POST'])
def single_stock():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        data = get_stock_data(ticker)

        # Check if the input is for crypto
        if data is None:  # If no stock data, try fetching crypto data
            crypto_data, ai_insight = get_crypto_data_with_insight(ticker)
            if crypto_data:
                return render_template('crypto_result.html', crypto=crypto_data, insight=ai_insight)
            return render_template('single_result.html', error="No data found for the given ticker or cryptocurrency.")

        # Stock-specific logic
        prompt = (
            f"Analyze the stock {ticker} based on the following historical data:\n\n"
            f"{data.tail(10).to_string()}\n\n"
            "Provide insights on the performance and recommendations for potential investors."
            "Consider the following factors in your analysis:\n"
            "- Recent price trends (uptrend, downtrend, or sideways movement).\n"
            "- Volatility and risk level (e.g., Value at Risk, maximum drawdown).\n"
            "- Key performance indicators (e.g., recent gains or losses).\n"
            "- Sector or industry trends that may impact the stock.\n"
            "- Comparison with similar stocks or indices in the market.\n"
            "- Company fundamentals (if available).\n\n"
            "Based on these factors, provide a detailed recommendation:\n"
            "- Is this stock a good investment at the current price? Why or why not?\n"
            "- If possible, include potential risks and rewards.\n"
            "- Suggest a target price range or specific scenarios under which it would be better to invest or avoid this stock.\n\n"
            "Your response should provide actionable insights for a potential investor."

        )
        ai_insight = get_ai_insight(prompt)
        var = calculate_var(data)
        max_drawdown = calculate_max_drawdown(data)
        plot = plot_stock(data, ticker)

        # Forecast future stock prices
        forecast = forecast_stock_prices(data)

        return render_template(
            'single_result.html',
            plot=plot,
            insight=ai_insight,
            var=var,
            max_drawdown=max_drawdown,
            forecast=forecast.to_html(index=False, justify="center"),
            news=get_stock_news(ticker)
        )
    return render_template('single_form.html')

@app.route('/compare', methods=['GET', 'POST'])
def compare_stocks():
    if request.method == 'POST':
        ticker1 = request.form['ticker1'].upper()
        ticker2 = request.form['ticker2'].upper()
        ticker3 = request.form['ticker3'].upper()

        data1 = get_stock_data(ticker1)
        data2 = get_stock_data(ticker2)
        data3 = get_stock_data(ticker3)

        if isinstance(data1, str):
            return render_template('comparison_result.html', error=f"Error retrieving data for {ticker1}: {data1}")
        if isinstance(data2, str):
            return render_template('comparison_result.html', error=f"Error retrieving data for {ticker2}: {data2}")
        if isinstance(data3, str):
            return render_template('comparison_result.html', error=f"Error retrieving data for {ticker3}: {data3}")

        plot = plot_comparison(data1, data2, data3, ticker1, ticker2, ticker3)
        heatmap = generate_correlation_heatmap(data1, data2, data3, ticker1, ticker2, ticker3)
        max_drawdown1 = calculate_max_drawdown(data1)
        max_drawdown2 = calculate_max_drawdown(data2)
        max_drawdown3 = calculate_max_drawdown(data3)
        var1 = calculate_var(data1)
        var2 = calculate_var(data2)
        var3 = calculate_var(data3)
        news1=get_stock_news(ticker1)
        news2=get_stock_news(ticker2)
        news3=get_stock_news(ticker3)
        insight = get_ai_insight(
            f"Compare the performance of two or tree stocks, {ticker1} and {ticker2} and {ticker3}, over the last 10 days.\n\n "
            f"Here is the data:\n\n{ticker1}:\n{data1.to_string()}\n\n{ticker2}:\n{data2.to_string()}\n\n{ticker3}:\n{data3.to_string()}\n\n"
            "Consider the following factors in your analysis:\n"
            "- Recent price trends (uptrend, downtrend, or sideways movement).\n"
            "- Volatility and risk level (e.g., Value at Risk, maximum drawdown).\n"
            "- Key performance indicators (e.g., recent gains or losses).\n"
            "- Sector or industry trends that may impact the stock.\n"
            "- Comparison with similar stocks or indices in the market.\n"
            "- Company fundamentals (if available).\n\n"
            "Based on these factors, provide a detailed recommendation:\n"
            "- Is this stock a good investment at the current price? Why or why not?\n"
            "- If possible, include potential risks and rewards.\n"
            "- Suggest a target price range or specific scenarios under which it would be better to invest or avoid this stock.\n\n"
            "Your response should provide actionable insights for a potential investor."
        )
        return render_template('comparison_result.html', plot=plot, var1=var1, var2=var2, var3=var3, max_drawdown1=max_drawdown1, max_drawdown2=max_drawdown2, max_drawdown3=max_drawdown3, heatmap=heatmap, insight=insight, news1=news1, news2=news2, news3=news3)
    return render_template('comparison_form.html')


""" 
if __name__ == '__main__':
    app.run(debug=True) """

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 for local testing
    serve(app, host="0.0.0.0", port=port)