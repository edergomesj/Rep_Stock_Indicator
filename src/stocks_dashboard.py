#Importing libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import ta.trend
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

#Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Constants
TIME_PERIODS = ['1d','1wk','1mo','1y','max']
INTERVAL_MAPPING = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}
DEFAULT_SYMBOLS = ['AAPL','GOOGL','AMZN','MSFT']

@dataclass
class ChartConfig:
    ticker: str
    time_period: str
    chart_type: str
    indicators: List[str]

    def validate(self) -> bool:
        return (
            isinstance(self.ticker, str) and
            self.time_period in TIME_PERIODS and
            self.chart_type in ['Candlestick', 'Line']
        )


# Part1: Define functions for pulling, processing, and creating technical indicators

#Fetch stock data based on the ticker, period, and interval
@st.cache_data(ttl=300) #cache for 5 minutes
def fetch_stock_data(ticker:str, period: str, interval: str):
    try:
        logger.info(f"Fetching data for ticker: {ticker} with period: {period}")
        end_date = datetime.now()
        if period == '1wk':
            start_date = end_date - timedelta(days=7)
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        else:
            data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data available for ticker '{ticker}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}"
        return pd.Dataframe()
    

#Process data to ensure it is timezone-aware and has the correct format
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        if data.empty:
            return data

        if data.index.tzinfo is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert('US/Eastern')
        data.reset_index(inplace=True)
        data.rename(columns={'date':'Datetime'}, inplace=True)
        data.fillna(method='ffill', inplace=True)
        return data

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        st.error("Error processing data")
        return pd.DataFrame()

#Calculate basic metrics from the stock data
def calculate_metrics(data: pd.DataFrame) -> Tuple[float, float, float, float, float, float, float]:
    try:
        if data.empty:
            raise ValueError("No data available for calculations.")

        last_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[0]
        change = last_close - prev_close
        pct_change = (change / prev_close) *100
        high = float(data['High'].max())
        low = float(data['Low'].min())
        volume = float(data['Volume'].sum())

        return last_close, prev_close, change, pct_change, high, low, volume

    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        st.error("Error calculating metrics")
        return 0.0,0.0,0.0,0.0,0.0,0.0,0.0

# Add simple moving average (SMA) and exponencial moving average (EMA) indicators
def add_technical_indicators(data: pd.Dataframe) -> pd.DataFrame:
    try:
        if data.empty:
            return data

    #Fill any missing values before calculating indicators
    data['Close'] = data['Close'].fillna(method='ffill')
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)

    #Fill any Nan values created by the indicators
    data.fillna(method='bfill', inplace=True)
    return data
    except Exception as e:
        logger.error(f"Error adding technical indicators: {str(e)}")
        st.error("Error calculating technical indicators")
        return data        


# Part 2A: Creating the dashboard App layout

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title("Real Time Stock Dashboard")

# 2A: Sidebar parameters
#Sidebar for users input parameters

st.sidebar.header('Chart Parameters')
ticker = st.sidebar.text_input('Ticker', 'ADBE')
time_period = st.sidebar.selectbox('Time Period', ['1d','1wk','1mo','1y','max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20','EMA 20'])

#Mapping of time periods to data intervals

interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

# Parte 2B: Main Content Area

#Update the dashboard based on user input
if st.sidebar.button('Update'):
    if data.empty:
        st.error(f"No data available for ticker '{ticker}' and time period '{time_period}'. Please try a different input.")
    else:    
        data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
        data = process_data(data)
        data = add_technical_indicators(data)

        last_close, prev_close, change, pct_change, high, low, volume = calculate_metrics(data)

        #display main metrics
        st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} USD", delta=f"{change:.2f} ({pct_change:.2f}%)")

        col1, col2, col3 = st.columns(3)
        col1.metric("High", f"{high:.2f} USD")
        col2.metric("Low", f"{low:.2f} USD")
        col3.metric("Volume", f"{volume:,}")

        #Plot the stock price chart
        fig = go.figure()
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(x=data['Datetime'],open=data['Open'], high=data['High'], low=data['Low'],close=data["Close"]))
        else:
            fig = px.line(data, x='Datetime',y='Close')

        #Add technical indicators to the chart
        for indicator in indicators:
            if indicator == 'SMA 20':
                fig.add_trace(go.Scatter(x=data['Datetime'], i=data['SMA_20'], name='SMA 20'))
            elif indicator == 'EMA 20':
                fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))
        
        #Format graph
        fig.update_layout(title=f'{ticker} {time_period.upper()} Chart',xaxis_title='Time',yaxis_title='Price (USD)',height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Display historical data and technical indicators
        st.subheader('Historical Data')
        st.dataframe(data[['Datetime','Open','High','Low','Close','Volume']])

        st.subheader('Technical Indicators')
        st.dataframe(data[['Datetime','SMA_20','EMA_20']])

# Part 2C: Sidebar prices
#Sidebar section for real-time Stock Prices of selected symbols
st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ['AAPL','GOOGL','AMZN','MSFT']
for symbol in stock_symbols:
    real_time_data = fetch_stock_data(symbol,'1d','1m')
    if not real_time_data.empty:
        real_time_data = process_data(real_time_data)
        last_price = real_time_data['Close'].iloc[-1]
        change = last_price - real_time_data['Open'].iloc[0]
        pct_change = (change / real_time_data['Open'].iloc[0])*100
        last_price.fillna(0, inplace=True)
        change.fillna(0, inplace=True)
        pct_change.fillna(0, inplace=True)
        st.sidebar.metric(f"{symbol}",last_price,pct_change)
        
# Sidebar information section
st.sidebar.subheader('About')
st.sidebar.info('This dashboard provides real-time stock data and technical indicators for the selected ticker and time periods.')