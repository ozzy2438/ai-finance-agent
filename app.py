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
        
        cur.execute("""
            INSERT INTO finance_chat.chat_history 
            (user_id, stock_symbol, query_text, analysis_result, 
             technical_signals, fundamental_metrics, price_prediction, news_sentiment)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (user_id, symbol, query, analysis, 
              Json(metrics), Json({'prediction': float(prediction)}), 
              float(prediction), sentiment))
        
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
    background-color: #1a1a1a;
    color: white;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
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
    
    # Trend Indicators
    data['SMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['SMA200'] = ta.trend.sma_indicator(data['Close'], window=200)
    
    # Momentum Indicators
    data['RSI'] = ta.momentum.rsi(data['Close'])
    data['MACD'] = ta.trend.macd_diff(data['Close'])
    
    # Volatility Indicators
    bb_indicator = ta.volatility.BollingerBands(data['Close'])
    data['BB_upper'] = bb_indicator.bollinger_hband()
    data['BB_middle'] = bb_indicator.bollinger_mavg()
    data['BB_lower'] = bb_indicator.bollinger_lband()
    
    # Volume Indicators
    data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
    
    # Calculate current signals
    current_price = data['Close'].iloc[-1]
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
                    {"role": "system", "content": "You are a professional financial analyst who provides detailed stock analysis and investment recommendations in Turkish."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"AI Insight Error: {str(e)}")
    
    # Fallback analysis when API fails
    price_change = ((predicted_price - current_price) / current_price) * 100
    pe_ratio = metrics.get('pe_ratio', 0)
    profit_margins = metrics.get('profit_margins', 0)
    debt_to_equity = metrics.get('debt_to_equity', 0)
    
    # Calculate confidence level
    confidence = 50  # Base confidence
    if metrics['trend'] == metrics['momentum']:  # If trend and momentum agree
        confidence += 20
    if news_sentiment != "Neutral":  # If there's clear sentiment
        confidence += 10
    if abs(price_change) > 10:  # If strong price movement expected
        confidence += 20
    confidence = min(confidence, 100)  # Cap at 100%
    
    # Determine overall signal
    signal = "Neutral"
    if metrics['trend'] == 'Bullish' and metrics['momentum'] == 'Bullish' and price_change > 0:
        signal = "Bullish"
    elif metrics['trend'] == 'Bearish' and metrics['momentum'] == 'Bearish' and price_change < 0:
        signal = "Bearish"
    
    # Generate valuation context
    valuation_status = "aşırı değerli"
    if pe_ratio < 20:
        valuation_status = "makul değerli"
    elif pe_ratio < 40:
        valuation_status = "yüksek değerli"
    
    analysis = f"""
Genel Sinyal: {signal}, güven seviyesi %{confidence}.

Piyasa Bağlamı: {symbol} hisseleri şu anda {valuation_status} görünüyor. Hisse senedi son dönemde {price_change:.1f}% değişim gösterdi. Teknik göstergeler {metrics['trend'].lower()} bir eğilim, {metrics['momentum'].lower()} bir momentum ve {metrics['volatility'].lower()} volatilite gösteriyor.

Detaylı Analiz:
- Teknik Analiz: RSI göstergesi {'aşırı alım' if metrics.get('RSI', 50) > 70 else 'aşırı satım' if metrics.get('RSI', 50) < 30 else 'nötr'} bölgesinde. MACD {'pozitif' if metrics.get('MACD', 0) > 0 else 'negatif'} sinyal veriyor.
- Temel Analiz: F/K oranı {pe_ratio:.2f}, kar marjı %{profit_margins*100:.1f}, borç/özkaynak oranı {debt_to_equity:.2f}
- Piyasa Duyarlılığı: Haberler {news_sentiment.lower()} yönde sinyal veriyor
- Risk Değerlendirmesi: {'Yüksek volatilite riski mevcut' if metrics['volatility'] == 'High' else 'Volatilite riski düşük'}, {'Değerleme riski yüksek' if pe_ratio > 50 else 'Değerleme makul seviyelerde'}

Tavsiye: """

    # Generate recommendation based on portfolio impact
    if signal == "Bullish" and confidence > 70:
        analysis += """Portföyünüzün %20'sinden fazlasını oluşturuyorsa, kar realizasyonu düşünülebilir. Uzun vadeli yatırımcıysanız ve volatiliteyi tolere edebiliyorsanız, mevcut pozisyonu korumak mantıklı olabilir. Risk iştahınız düşükse veya likidite ihtiyacınız varsa, mevcut değerleme ve düşüş sinyalleri göz önüne alındığında kısmi satış düşünülebilir."""
    elif signal == "Bearish" and confidence > 70:
        analysis += """Portföyünüzün %20'sinden fazlasını oluşturuyorsa, pozisyonu azaltmak mantıklı olabilir. Mevcut değerleme ve teknik göstergeler satış baskısına işaret ediyor. Ancak uzun vadeli büyüme potansiyeline inanıyorsanız, düşük seviyeleri alım fırsatı olarak değerlendirebilirsiniz."""
    else:
        analysis += """Mevcut pozisyonu korumak ve piyasa koşullarını yakından takip etmek öneriliyor. Yeni pozisyon açmak için daha net sinyaller beklenebilir. Portföy çeşitlendirmesi önemli, tek bir hissede yoğunlaşmaktan kaçının."""
    
    return analysis

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
    current_price = yf.download(symbol, period="1d")['Close'].iloc[-1]
    predicted_price = prediction[-1][0]
    price_change = ((predicted_price - current_price) / current_price) * 100
    
    # Analyze news sentiment
    news_sentiment = analyze_news_sentiment(symbol)
    
    # Get AI insights
    ai_insights = get_ai_insights(metrics, symbol, current_price, predicted_price, news_sentiment)
    
    analysis = f"""
    ### AI Analysis for {symbol}

    Based on the comprehensive analysis of financial indicators and market conditions:

    #### Technical Signals:
    - Trend: {metrics['trend']}
    - Momentum: {metrics['momentum']}
    - Volatility: {metrics['volatility']}

    #### Fundamental Metrics:
    - P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
    - Profit Margin: {metrics.get('profit_margins', 'N/A')}
    - Debt to Equity: {metrics.get('debt_to_equity', 'N/A')}

    #### Price Prediction:
    The AI model predicts a price of ${predicted_price:.2f} in 30 days, representing a {price_change:.2f}% change.

    #### News Sentiment:
    Current market sentiment based on recent news: {news_sentiment}

    #### Detailed AI Analysis:
    {ai_insights if ai_insights else 'AI analysis temporarily unavailable'}

    #### Quick Recommendation:
    """
    
    if price_change > 10 and metrics['trend'] == 'Bullish':
        analysis += "🟢 Strong Buy - Multiple indicators suggest significant upside potential."
    elif price_change > 5 and metrics['momentum'] == 'Bullish':
        analysis += "🟡 Buy - Positive momentum with moderate upside potential."
    elif price_change < -10 and metrics['trend'] == 'Bearish':
        analysis += "🔴 Strong Sell - Multiple indicators suggest significant downside risk."
    elif price_change < -5 and metrics['momentum'] == 'Bearish':
        analysis += "🟠 Sell - Negative momentum with moderate downside risk."
    else:
        analysis += "⚪ Hold - Mixed signals suggest maintaining current position."
    
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
                current_price = data['Close'].iloc[-1]
                daily_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{symbol} Price", f"${current_price:.2f}", f"{daily_change:.2f}%")
                
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
                    line=dict(color='#ffd700', width=1.5)
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['SMA50'],
                    name="SMA50",
                    line=dict(color='#00bfff', width=1.5)
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
if query:
    # ... existing analysis code ...
    news_sentiment = analyze_news_sentiment(symbol)
    save_chat_history(
        user_id=user_id,
        symbol=symbol,
        query=query,
        analysis=analysis_text,
        metrics=metrics,
        prediction=prediction[-1][0] if prediction is not None else 0,
        sentiment=news_sentiment
    )

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