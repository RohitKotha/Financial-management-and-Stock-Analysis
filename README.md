# 🎯 AI Financial Intelligence Platform

## Overview

An advanced AI-driven Deep Learning system that integrates personal financial management and stock market intelligence within a unified platform. This comprehensive solution leverages state-of-the-art deep learning models to empower users with real-time insights, optimize investment strategies, and enable data-driven financial decisions.

## 🚀 Key Features

### 💳 Personal Finance Management
- **User Authentication**: Individual user accounts with isolated private data
- **Expense Tracking & Categorization**: AI-powered automatic categorization using NLP and Machine Learning
- **Budget Planning & Alerts**: Smart budget management with real-time alerts and recommendations
- **Financial Predictions**: LSTM networks predict future cash flows and expenses
- **Financial Goals Tracking**: Set and monitor progress toward financial objectives
- **Personalized Advice**: AI-driven insights and recommendations based on spending behavior

### 📈 Stock Market Intelligence
- **Real-time Stock Analysis**: Live market data and comprehensive stock information
- **Price Predictions**: LSTM and Transformer networks forecast stock prices
- **Technical Indicators**: RSI, MACD, Moving Averages, Bollinger Bands
- **Trading Signals**: AI-generated buy/sell recommendations
- **Portfolio Optimization**: Modern Portfolio Theory for optimal asset allocation

### 🎯 Portfolio Management
- **Risk Analysis**: VaR, Beta, volatility metrics, and diversification scores
- **Portfolio Optimization**: Maximize Sharpe ratio while managing risk
- **Rebalancing Recommendations**: Data-driven portfolio adjustment suggestions
- **Performance Tracking**: Real-time P&L monitoring and performance metrics

### 🤖 AI-Powered Features
- **Deep Learning Models**: LSTM networks for time series predictions
- **NLP for Categorization**: Intelligent transaction classification
- **Predictive Analytics**: Forecast expenses, cash flow, and stock prices
- **Behavioral Analysis**: Understand spending patterns and habits
- **Personalized Recommendations**: Tailored financial advice

## 📦 Project Structure

```
finalyear project/
│
├── app.py                      # Main Streamlit application
├── core/                       # Shared core utilities
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── database.py             # Database management (SQLite)
│   └── utils.py                # Utility functions
│
├── modules/                    # Domain and ML modules
│   ├── __init__.py
│   ├── expense_categorizer.py  # ML-based expense categorization
│   ├── budget_manager.py       # Budget planning and alerts
│   ├── financial_lstm.py       # LSTM for financial predictions
│   ├── stock_prediction.py     # Stock market prediction models
│   ├── portfolio_optimizer.py  # Portfolio optimization algorithms
│   └── financial_advisor.py    # AI financial advisor
│
├── requirements.txt            # Python dependencies
├── financial_data.db          # SQLite database (auto-created)
└── models/                    # Trained models directory (auto-created)
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Navigate to the project directory**:
   ```powershell
   cd "c:\finalyear project"
   ```

2. **Create a virtual environment (recommended)**:
   ```powershell
   py -3.11 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   .\venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. **Run the application**:
   ```powershell
   .\venv\Scripts\python.exe -m streamlit run app.py
   ```

5. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

## 🎮 Usage Guide

### Getting Started

1. **Dashboard**: View comprehensive financial overview
2. **Add Transactions**: Record income and expenses with automatic categorization
3. **Set Budgets**: Define spending limits for different categories
4. **Train Models**: Train ML models with your data for better predictions
5. **Analyze Stocks**: Research and predict stock market trends
6. **Manage Portfolio**: Track and optimize your investment portfolio
7. **Get AI Advice**: Receive personalized financial recommendations

### Sample Data

To explore the platform with sample data, click the **"Load Sample Data"** button on the Dashboard page.

## 🧠 Machine Learning Models

### 1. Expense Categorizer
- **Algorithm**: Multinomial Naive Bayes with TF-IDF
- **Purpose**: Automatically categorize transactions
- **Features**: NLP-based text analysis of transaction descriptions

### 2. Financial LSTM
- **Architecture**: Multi-layer LSTM network
- **Purpose**: Predict future expenses and cash flow
- **Input**: Historical transaction data (30+ days)
- **Output**: Daily expense predictions for 7-90 days

### 3. Stock Prediction LSTM
- **Architecture**: Deep LSTM with dropout layers
- **Purpose**: Forecast stock prices
- **Input**: Historical stock data (60-day sequences)
- **Output**: 30-day price predictions

### 4. Portfolio Optimizer
- **Method**: Modern Portfolio Theory (MPT)
- **Algorithm**: Sharpe Ratio maximization via SLSQP optimization
- **Features**: Risk-return optimization, diversification analysis

## 📊 Technical Indicators

The platform implements several technical analysis indicators:

- **Moving Averages**: 20, 50, 200-day MAs
- **RSI** (Relative Strength Index): Momentum indicator
- **MACD** (Moving Average Convergence Divergence): Trend indicator
- **Bollinger Bands**: Volatility indicator
- **Trading Signals**: Automated buy/sell recommendations

## 🔒 Data Privacy

- All data is stored locally in SQLite database
- No external data transmission
- User data remains on local machine
- Models are trained locally

## 🎯 Key Technologies

- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow/Keras
- **Machine Learning**: Scikit-learn
- **NLP**: NLTK, TextBlob
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Stock Data**: yfinance
- **Optimization**: SciPy
- **Database**: SQLite

## 📈 Performance Metrics

The system provides comprehensive metrics:

- **Budget Utilization**: Real-time spending vs. budget
- **Savings Rate**: Income vs. expenses ratio
- **Portfolio Returns**: P&L tracking
- **Risk Metrics**: VaR, Beta, Volatility
- **Prediction Accuracy**: Model performance indicators

## 🔧 Configuration

Key settings can be modified in `config.py`:

- Model parameters (LSTM units, epochs, batch size)
- Budget thresholds
- Risk levels
- Prediction horizons
- Category definitions

## 🚧 Future Enhancements

- [ ] Transformer models for enhanced predictions
- [ ] Multi-currency support
- [ ] Mobile application
- [ ] Real-time notifications
- [ ] Advanced tax optimization
- [ ] Cryptocurrency tracking
- [ ] Integration with banking APIs
- [ ] Social features for benchmarking

## 📝 Notes

- **Initial Training**: Models require 30+ days of data for optimal accuracy
- **Stock Data**: Fetched from Yahoo Finance (requires internet connection)
- **Database**: Automatically created on first run
- **Models**: Saved and loaded automatically for persistence

## 🐛 Troubleshooting

### Common Issues

1. **Module Import Errors**:
   ```powershell
   pip install --upgrade -r requirements.txt
   ```

2. **Database Locked**:
   - Close other instances of the application
   - Delete `financial_data.db` and restart

3. **Stock Data Not Loading**:
   - Check internet connection
   - Verify stock symbol is valid
   - Try a different time period

4. **Model Training Fails**:
   - Ensure sufficient data (30+ transactions)
   - Reduce epochs or batch size
   - Check available RAM

## 👨‍💻 Development

### Running Tests
```powershell
# Add tests as needed
python -m pytest tests/
```

### Code Style
Follow PEP 8 guidelines for Python code.

## 📄 License

This project is for educational and personal use.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows project structure
- Documentation is updated
- Models are tested
- UI remains user-friendly

## 📧 Support

For issues or questions:
- Check the troubleshooting section
- Review code documentation
- Test with sample data first

---

**Built with ❤️ using AI and Deep Learning**

*Empowering Smart Financial Decisions Through Artificial Intelligence*
