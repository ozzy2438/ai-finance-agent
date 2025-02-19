import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from alpha_vantage.timeseries import TimeSeries
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import psycopg2
from psycopg2.extras import Json

# API Keys
NEWS_API_KEY = "db0ae9deccf8404aa2f54f5480d22cf3"
ALPHA_VANTAGE_API_KEY = "VWJO231IS0J5PED1"
DEEPSEEK_API_KEY = "sk-be1470aa10894ed481db064a400eaeb2"

# Initialize API clients
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY)

# PostgreSQL bağlantı bilgileri
DB_CONFIG = {
    'dbname': 'ai_financial_advisor',
    'user': 'postgres',
    'password': 'Allah248012',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    """Veritabanı bağlantısı oluştur"""
    return psycopg2.connect(**DB_CONFIG)

def save_chat_history(user_id, symbol, query, analysis, metrics, prediction, sentiment):
    """Chat geçmişini veritabanına kaydet"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Prediction değerini güvenli bir şekilde işle
        prediction_value = 0
        if prediction is not None:
            if isinstance(prediction, np.ndarray):
                if len(prediction) > 0:  # Array boş değilse
                    prediction_value = float(prediction[-1])  # Son değeri al
            else:
                prediction_value = float(prediction)  # Tek değer ise direkt dönüştür
        
        cur.execute("""
            INSERT INTO finance_chat.chat_history 
            (user_id, stock_symbol, query_text, analysis_result, 
             technical_signals, fundamental_metrics, price_prediction, news_sentiment)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (user_id, symbol, query, analysis, 
              Json(metrics), Json({'prediction': prediction_value}), 
              prediction_value, sentiment))
        
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Veritabanı hatası: {str(e)}")


def get_chat_history(user_id):
    """Kullanıcının chat geçmişini getir"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT stock_symbol, query_text, analysis_result, created_at 
            FROM finance_chat.chat_history 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, (user_id,))
        
        history = cur.fetchall()
        cur.close()
        conn.close()
        return history
    except Exception as e:
        st.error(f"Veritabanı hatası: {str(e)}")
        return []

# Page configuration
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: rgb(13, 17, 23);
    color: rgb(201, 209, 217);
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 24px;
    border-radius: 5px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #45a049;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.analysis-container {
    background-color: rgba(40, 44, 52, 0.95);
    padding: 25px;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
    margin: 15px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.typewriter {
    color: #e6e6e6;
    font-family: 'Roboto Mono', monospace;
    line-height: 1.6;
}
.stMarkdown div {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', 'source-code-pro', monospace;
}
pre {
    background-color: rgb(22, 27, 34) !important;
    border-radius: 6px !important;
    padding: 16px !important;
    font-size: 14px !important;
}
code {
    color: rgb(201, 209, 217) !important;
}
</style>
""", unsafe_allow_html=True)

# Custom CSS for typewriter effect
st.markdown("""
<style>
@keyframes typing {
    from { width: 0 }
    to { width: 100% }
}

.typewriter {
    overflow: hidden;
    border-right: .15em solid #4CAF50;
    white-space: pre-wrap;
    letter-spacing: .10em;
    animation: 
        typing 3.5s steps(40, end),
        blink-caret .75s step-end infinite;
}

.analysis-container {
    background-color: rgba(25,25,25,0.9);
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #4CAF50;
    margin: 10px 0;
}

@keyframes blink-caret {
    from, to { border-color: transparent }
    50% { border-color: #4CAF50; }
}
</style>
""", unsafe_allow_html=True)

def calculate_financial_metrics(data):
    """Calculate key financial metrics"""
    metrics = {}
    
    # Verileri düzleştir ve pandas Series'e dönüştür
    close_series = pd.Series(data['Close'].values.flatten(), index=data.index)
    volume_series = pd.Series(data['Volume'].values.flatten(), index=data.index)
    
    # Trend Indicators
    data['SMA20'] = ta.trend.sma_indicator(close_series, window=20)
    data['SMA50'] = ta.trend.sma_indicator(close_series, window=50)
    data['SMA200'] = ta.trend.sma_indicator(close_series, window=200)
    
    # Momentum Indicators
    data['RSI'] = ta.momentum.rsi(close_series)
    data['MACD'] = ta.trend.macd_diff(close_series)
    
    # Volatility Indicators
    bb_indicator = ta.volatility.BollingerBands(close_series)
    data['BB_upper'] = bb_indicator.bollinger_hband()
    data['BB_middle'] = bb_indicator.bollinger_mavg()
    data['BB_lower'] = bb_indicator.bollinger_lband()
    
    # Volume Indicators
    data['OBV'] = ta.volume.on_balance_volume(close_series, volume_series)
    
    # Calculate current signals
    current_price = close_series.iloc[-1]
    metrics['trend'] = 'Bullish' if current_price > data['SMA50'].iloc[-1] else 'Bearish'
    metrics['momentum'] = 'Bullish' if data['RSI'].iloc[-1] > 50 else 'Bearish'
    metrics['volatility'] = 'High' if (data['BB_upper'].iloc[-1] - data['BB_lower'].iloc[-1])/data['BB_middle'].iloc[-1] > 0.05 else 'Low'
    
    return data, metrics

def predict_price(data, days=30):
    """Simple price prediction using linear regression"""
    df = data.copy()
    df['Prediction'] = df['Close'].shift(-days)
    
    X = np.array(range(len(df)))
    X = X.reshape(-1, 1)
    y = df['Close'].values
    y = y.reshape(-1, 1)
    
    train_x = X[:-days]
    train_y = y[:-days]
    
    lr = LinearRegression()
    lr.fit(train_x, train_y)
    
    prediction_days = np.array(range(len(df), len(df) + days))
    prediction_days = prediction_days.reshape(-1, 1)
    
    price_prediction = lr.predict(prediction_days)
    
    # Tahmin değerlerini düzleştir
    price_prediction = price_prediction.flatten()
    
    return price_prediction

def analyze_stock(symbol):
    """Comprehensive stock analysis"""
    # Get historical data
    data = yf.download(symbol, period="1y")
    
    if data.empty:
        return None, None, None
    
    # Calculate metrics
    data, metrics = calculate_financial_metrics(data)
    
    # Get company info
    stock = yf.Ticker(symbol)
    info = stock.info
    
    # Calculate additional metrics
    try:
        metrics['pe_ratio'] = info.get('forwardPE', 0)
        metrics['market_cap'] = info.get('marketCap', 0)
        metrics['revenue_growth'] = info.get('revenueGrowth', 0)
        metrics['profit_margins'] = info.get('profitMargins', 0)
        metrics['debt_to_equity'] = info.get('debtToEquity', 0)
    except:
        pass
    
    # Price prediction
    prediction = predict_price(data)
    
    # Current price ve daily change hesaplama
    current_price = float(data['Close'].iloc[-1])
    daily_change = float(((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100)
    
    # Metrics'e ekle
    metrics['current_price'] = current_price
    metrics['daily_change'] = daily_change
    
    return data, metrics, prediction

def get_ai_insights(metrics, symbol, current_price, predicted_price, news_sentiment):
    """Get advanced AI insights using DeepSeek Reasoner"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "deepseek-reasoner",
                "messages": [
                    {
                        "role": "system", 
                        "content": """You are a professional financial analyst providing detailed stock analysis. 
                        Focus on specific technical levels, risk factors, and actionable recommendations. 
                        Be concise but thorough, and use bullet points where appropriate."""
                    },
                    {"role": "user", "content": analysis_prompt}
                ],
                "temperature": 0.3,  # Daha tutarlı yanıtlar için düşük sıcaklık
                "max_tokens": 2000,  # Daha uzun yanıtlar için token sayısını artır
                "top_p": 0.9
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"AI Insight Error: {str(e)}")
        return None

def analyze_news_sentiment(symbol):
    """Hisse senedi için haber duyarlılık analizi"""
    try:
        # Basit bir duyarlılık değeri döndür
        # Gerçek uygulamada haber API'si kullanılabilir
        return "NEUTRAL"
    except Exception as e:
        st.error(f"Haber analizi hatası: {str(e)}")
        return "UNKNOWN"

def generate_analysis_text(symbol, metrics, prediction, news_articles):
    """Generate AI analysis text"""
    current_price = metrics['current_price']
    predicted_price = float(prediction[-1]) if prediction is not None and len(prediction) > 0 else 0
    price_change = ((predicted_price - current_price) / current_price) * 100
    
    # DeepSeek API'den detaylı analiz al
    try:
        analysis_prompt = f"""
        Analyze {symbol} stock with the following metrics and provide a comprehensive analysis:
        
        Technical Data:
        - Current Price: ${current_price:.2f}
        - Predicted Price: ${predicted_price:.2f} (in 30 days)
        - Price Change: {price_change:.2f}%
        - Trend: {metrics['trend']}
        - Momentum: {metrics['momentum']}
        - Volatility: {metrics['volatility']}
        - RSI: {metrics.get('RSI', 0):.0f}
        
        Fundamental Data:
        - P/E Ratio: {metrics.get('pe_ratio', 0):.2f}
        - Profit Margin: {metrics.get('profit_margins', 0)*100:.1f}%
        - Debt/Equity: {metrics.get('debt_to_equity', 0):.2f}
        
        Please provide a detailed analysis covering:
        1. Market context and current valuation
        2. Technical analysis with specific support/resistance levels
        3. Risk factors and potential catalysts
        4. Short-term and long-term outlook
        5. Specific trading recommendations with entry/exit points
        6. Portfolio allocation suggestions
        """

        ai_analysis = get_ai_insights(metrics, symbol, current_price, predicted_price, analyze_news_sentiment(symbol))
    except:
        ai_analysis = "AI analysis temporarily unavailable."

    # Temel analiz metnini oluştur
    analysis = f"""### AI Analysis for {symbol}

Based on the comprehensive analysis of financial indicators and market conditions:

Technical Signals:
• Trend: {metrics['trend']}
• Momentum: {metrics['momentum']}
• Volatility: {metrics['volatility']}

Fundamental Metrics:
• P/E Ratio: {metrics.get('pe_ratio', 0):.2f}
• Profit Margin: {metrics.get('profit_margins', 0)*100:.1f}%
• Debt to Equity: {metrics.get('debt_to_equity', 0):.2f}

Price Prediction:
The AI model predicts a price of ${predicted_price:.2f} in 30 days, representing a {price_change:.2f}% change.

News Sentiment:
Current market sentiment based on recent news: {analyze_news_sentiment(symbol)}

Detailed AI Analysis:
{ai_analysis}

Quick Recommendation:
{
    "🟢 Strong Buy - Multiple indicators suggest significant upside potential." if price_change > 10 and metrics['trend'] == 'Bullish'
    else "🟡 Buy - Positive momentum with moderate upside potential." if price_change > 5 and metrics['momentum'] == 'Bullish'
    else "🔴 Strong Sell - Multiple indicators suggest significant downside risk." if price_change < -10 and metrics['trend'] == 'Bearish'
    else "🟠 Sell - Negative momentum with moderate downside risk." if price_change < -5 and metrics['momentum'] == 'Bearish'
    else "⚪ Hold - Mixed signals suggest maintaining current position."
}"""
    
    # Streamlit'e özel markdown formatında göster
    st.markdown(
        f"""
        <div style="background-color: rgb(13, 17, 23); 
                    color: rgb(201, 209, 217); 
                    padding: 16px; 
                    border-radius: 6px; 
                    font-family: 'Consolas', monospace;
                    white-space: pre-wrap;
                    line-height: 1.2;
                    font-size: 12px;">
        {analysis}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    return analysis

# Main app
st.title("🤖 AI Financial Advisor")

# User input
query = st.text_input("Ask me anything about stocks (e.g., 'Should I buy AAPL stock now?', 'Analyze TSLA performance')")

if query:
    # Extract stock symbol from query
    common_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    symbol = None
    
    for s in common_symbols:
        if s in query:
            symbol = s
            break
    
    if not symbol:
        st.warning("Please include a valid stock symbol in your question.")
    else:
        with st.spinner(f"Analyzing {symbol}..."):
            # Get stock data and analysis
            data, metrics, prediction = analyze_stock(symbol)
            
            if data is not None:
                # Display current stock price and basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        f"{symbol} Price", 
                        f"${metrics['current_price']:.2f}", 
                        f"{metrics['daily_change']:.2f}%"
                    )
                
                # Display main price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Price",
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA20'],
                    name="SMA20",
                    line=dict(color='#ffd700', width=1.0)
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA50'],
                    name="SMA50",
                    line=dict(color='#00bfff', width=1.0)
                ))
                
                fig.update_layout(
                    title=f"{symbol} Stock Price and Moving Averages",
                    yaxis_title="Price",
                    template="plotly_dark",
                    plot_bgcolor='rgba(25,25,25,1)',
                    paper_bgcolor='rgba(25,25,25,1)',
                    font=dict(color='white'),
                    xaxis=dict(
                        gridcolor='rgba(128,128,128,0.1)',
                        zerolinecolor='rgba(128,128,128,0.2)'
                    ),
                    yaxis=dict(
                        gridcolor='rgba(128,128,128,0.1)',
                        zerolinecolor='rgba(128,128,128,0.2)'
                    ),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display technical indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI Chart
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        name="RSI"
                    ))
                    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                    rsi_fig.update_layout(
                        title="Relative Strength Index (RSI)",
                        template="plotly_dark",
                        plot_bgcolor='rgba(25,25,25,1)',
                        paper_bgcolor='rgba(25,25,25,1)',
                        font=dict(color='white'),
                        xaxis=dict(
                            gridcolor='rgba(128,128,128,0.1)',
                            zerolinecolor='rgba(128,128,128,0.2)'
                        ),
                        yaxis=dict(
                            gridcolor='rgba(128,128,128,0.1)',
                            zerolinecolor='rgba(128,128,128,0.2)'
                        ),
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                
                with col2:
                    # MACD Chart
                    macd_fig = go.Figure()
                    macd_fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['MACD'],
                        name="MACD"
                    ))
                    macd_fig.update_layout(
                        title="MACD",
                        template="plotly_dark",
                        plot_bgcolor='rgba(25,25,25,1)',
                        paper_bgcolor='rgba(25,25,25,1)',
                        font=dict(color='white'),
                        xaxis=dict(
                            gridcolor='rgba(128,128,128,0.1)',
                            zerolinecolor='rgba(128,128,128,0.2)'
                        ),
                        yaxis=dict(
                            gridcolor='rgba(128,128,128,0.1)',
                            zerolinecolor='rgba(128,128,128,0.2)'
                        ),
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                
                # Display AI analysis and news
                try:
                    news = newsapi.get_everything(
                        q=symbol,
                        language='en',
                        sort_by='publishedAt',
                        page_size=5
                    )
                    
                    # Display AI analysis with typewriter effect
                    analysis_text = generate_analysis_text(symbol, metrics, prediction, news['articles'])
                    st.markdown(f'<div class="analysis-container"><div class="typewriter">{analysis_text}</div></div>', unsafe_allow_html=True)
                    
                    # Display news articles
                    st.subheader("Recent News")
                    for article in news['articles']:
                        st.markdown(f'<div class="analysis-container">', unsafe_allow_html=True)
                        st.write(f"**{article['title']}**")
                        st.write(article['description'])
                        st.write(f"Source: {article['source']['name']} | Published: {article['publishedAt']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error fetching news: {str(e)}")
                    analysis_text = generate_analysis_text(symbol, metrics, prediction, [])
                    st.markdown(f'<div class="analysis-container"><div class="typewriter">{analysis_text}</div></div>', unsafe_allow_html=True)
            else:
                st.error(f"Could not fetch data for {symbol}")

# Sidebar with additional features
with st.sidebar:
    st.title("Settings")
    st.markdown("### Analysis Parameters")
    time_period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
    prediction_days = st.slider("Prediction Days", 7, 90, 30)
    
    st.markdown("### Indicators")
    show_sma = st.checkbox("Show Moving Averages", True)
    show_bollinger = st.checkbox("Show Bollinger Bands", False)
    show_volume = st.checkbox("Show Volume", True)

    st.title("Chat Geçmişi")
    user_id = "default_user"  # Gerçek uygulamada kullanıcı kimliği kullanılmalı
    chat_history = get_chat_history(user_id)
    
    for symbol, query, analysis, created_at in chat_history:
        with st.expander(f"{symbol} - {created_at.strftime('%Y-%m-%d %H:%M')}"):
            st.write(f"Soru: {query}")
            st.write(f"Analiz: {analysis[:200]}...")

# Analiz sonuçlarını kaydet
if query and symbol:
    try:
        data, metrics, prediction = analyze_stock(symbol)
        if data is not None:
            news_sentiment = analyze_news_sentiment(symbol)
            analysis_text = generate_analysis_text(symbol, metrics, prediction, [])
            
            # Tahmin verisini düzleştir
            if isinstance(prediction, np.ndarray):
                prediction = prediction.flatten()
            
            save_chat_history(
                user_id=user_id,
                symbol=symbol,
                query=query,
                analysis=analysis_text,
                metrics=metrics,
                prediction=prediction,
                sentiment=news_sentiment
            )
    except Exception as e:
        st.error(f"Analiz kaydedilirken hata oluştu: {str(e)}")

# Teknik analiz bölümünde
technical_metrics = {
    'sma': data['SMA20'].tolist() if 'SMA20' in locals() else [],
    'rsi': data['RSI'].tolist() if 'RSI' in locals() else [],
    'macd': data['MACD'].tolist() if 'MACD' in locals() else [],
    'bollinger': {
        'upper': data['BB_upper'].tolist() if 'BB_upper' in locals() else [],
        'lower': data['BB_lower'].tolist() if 'BB_lower' in locals() else [],
        'middle': data['BB_middle'].tolist() if 'BB_middle' in locals() else []
    }
}

if __name__ == "__main__":
    pass 