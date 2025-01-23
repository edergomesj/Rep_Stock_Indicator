# Real-Time Stock Dashboard

A Streamlit-based dashboard that provides real-time stock data visualization and technical analysis tools. This application allows users to track stock prices, view technical indicators, and analyze historical data through an interactive web interface.

## Features

- **Real-time Stock Data**: Live price updates for selected stocks
- **Interactive Charts**: Both candlestick and line charts available
- **Technical Indicators**: 
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
- **Multiple Timeframes**: Support for different time periods (1 day to max available)
- **Key Metrics Display**: Price, volume, highs, and lows
- **Historical Data Tables**: Detailed view of historical prices and indicators

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-dashboard.git
cd stock-dashboard
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- yfinance
- ta (Technical Analysis library)
- pytz

## Usage

1. Run the application:
```bash
streamlit run stocks_dashboard.py
```

2. Open your web browser and navigate to the displayed localhost address (typically http://localhost:8501)

3. Use the sidebar to:
   - Enter a stock ticker symbol
   - Select time period
   - Choose chart type (Candlestick/Line)
   - Add technical indicators

4. Click the "Update" button to refresh the data and visualization

## Application Structure

- `stocks_dashboard.py`: Main application file containing all functionality
- Key components:
  - Data fetching and processing
  - Technical indicator calculations
  - Chart creation
  - Real-time price updates
  - Interactive UI elements

## Configuration

Default settings can be modified in the constants section:
- `TIME_PERIODS`: Available time periods for analysis
- `INTERVAL_MAPPING`: Data granularity for each time period
- `DEFAULT_SYMBOLS`: Stock symbols shown in real-time sidebar

## Screenshots

[Add screenshots of your dashboard here]

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/stock-dashboard](https://github.com/yourusername/stock-dashboard)

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web application framework
- [yfinance](https://github.com/ranaroussi/yfinance) for providing stock market data
- [TA-Lib](https://ta-lib.org/) for technical analysis indicators