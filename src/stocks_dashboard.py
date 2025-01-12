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

def create_stock_chart(data: pd.DataFrame, config: ChartConfig) -> go.Figure:
#create stock price chart with indicators

    try:
        fig = go.Figure()
        if config.chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(
                x=data['Datetime'],
                open=data['Open'],
                high=data['High'], 
                low=data['Low'],
                close=data["Close"],
                name= 'OHLC'
            ))
        else:
            fig.add_trace(go.scatter(
                x=data['Datetime'],
                y=data['Close'],
                mode='lines',
                name='Close'
            ))

        #add technical indicators
        if 'SMA_20' in config.indicators:
            fig.add_trace(go.scatter(x=data['Datetime'],
            y=data['SMA_20'], 
            name='SMA 20', 
            line=dict(dash='dash')
        )) 

        if 'EMA_20' in config.indicators:
            fig.add_trace(go.scatter(x=data['Datetime'],
            y=data['EMA_20'], 
            name='EMA 20', 
            line=dict(dash='dot')
        ))

        #update layout
        fig.update_layout(
            title = f'{config.ticker} {config.time_period.upper()} Chart',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=600,
            template='plotly_white',
            hovermode='x unified',
            showlegend=True
        )   

        return fig

    except Exception as e: 
        logger.error(f"Error creating stock chart: {str(e)}")
        st.error("Error creating stock chart")
        return go.figure()

# Part 2: Main Application

def main():