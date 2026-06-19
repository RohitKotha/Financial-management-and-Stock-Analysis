"""
Configuration file for the AI Financial Management System
"""

import os

# Database Configuration
DATABASE_PATH = "financial_data.db"

# Model Paths
MODELS_DIR = "models"
EXPENSE_CATEGORIZER_PATH = os.path.join(MODELS_DIR, "expense_categorizer.pkl")
LSTM_FINANCIAL_PATH = os.path.join(MODELS_DIR, "lstm_financial_model.h5")
LSTM_STOCK_PATH = os.path.join(MODELS_DIR, "lstm_stock_model.h5")
TRANSFORMER_PATH = os.path.join(MODELS_DIR, "transformer_model")

# Categories
EXPENSE_CATEGORIES = [
    "Food & Dining",
    "Transportation",
    "Shopping",
    "Entertainment",
    "Bills & Utilities",
    "Healthcare",
    "Education",
    "Travel",
    "Insurance",
    "Investments",
    "Other"
]

INCOME_CATEGORIES = [
    "Salary",
    "Freelance",
    "Investments",
    "Business",
    "Other"
]

# Model Parameters
LSTM_SEQUENCE_LENGTH = 30
LSTM_HIDDEN_UNITS = 128
LSTM_DROPOUT = 0.2
EPOCHS = 50
BATCH_SIZE = 32

# Stock Market Parameters
DEFAULT_STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
PREDICTION_DAYS = 30

# Budget Alert Thresholds
BUDGET_WARNING_THRESHOLD = 0.8  # 80%
BUDGET_CRITICAL_THRESHOLD = 0.95  # 95%

# Portfolio Risk Levels
RISK_LEVELS = {
    "Conservative": {"stocks": 0.4, "bonds": 0.6},
    "Moderate": {"stocks": 0.6, "bonds": 0.4},
    "Aggressive": {"stocks": 0.8, "bonds": 0.2}
}

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
