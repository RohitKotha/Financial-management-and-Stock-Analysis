"""
Utility functions for data processing and analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from textblob import TextBlob
from . import config

def preprocess_text(text):
    """Preprocess text for NLP tasks"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def extract_keywords(description):
    """Extract keywords from transaction description"""
    blob = TextBlob(description.lower())
    return blob.noun_phrases

def categorize_transaction_keywords(description):
    """Simple keyword-based categorization"""
    description = description.lower()
    
    keywords_map = {
        "Food & Dining": ["restaurant", "cafe", "food", "lunch", "dinner", "breakfast", 
                          "pizza", "burger", "starbucks", "mcdonald", "grocery"],
        "Transportation": ["uber", "lyft", "taxi", "gas", "fuel", "parking", "metro", 
                          "train", "bus", "flight", "airline"],
        "Shopping": ["amazon", "walmart", "target", "ebay", "store", "mall", "shop"],
        "Entertainment": ["movie", "netflix", "spotify", "game", "concert", "theater"],
        "Bills & Utilities": ["electricity", "water", "internet", "phone", "rent", 
                              "mortgage", "insurance", "utility"],
        "Healthcare": ["hospital", "doctor", "pharmacy", "medical", "health", "clinic"],
        "Education": ["school", "university", "college", "course", "book", "tuition"],
        "Travel": ["hotel", "airbnb", "booking", "travel", "vacation", "trip"],
    }
    
    for category, keywords in keywords_map.items():
        if any(keyword in description for keyword in keywords):
            return category
    
    return "Other"

def calculate_spending_by_category(transactions_df):
    """Calculate spending by category"""
    if transactions_df.empty:
        return pd.DataFrame()
    
    expenses = transactions_df[transactions_df['type'] == 'expense']
    spending = expenses.groupby('category')['amount'].sum().reset_index()
    spending = spending.sort_values('amount', ascending=False)
    return spending

def calculate_monthly_summary(transactions_df):
    """Calculate monthly income, expenses, and savings"""
    if transactions_df.empty:
        return {"income": 0, "expenses": 0, "savings": 0}
    
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    current_month = datetime.now().strftime('%Y-%m')
    
    monthly_data = transactions_df[
        transactions_df['date'].dt.strftime('%Y-%m') == current_month
    ]
    
    income = monthly_data[monthly_data['type'] == 'income']['amount'].sum()
    expenses = monthly_data[monthly_data['type'] == 'expense']['amount'].sum()
    savings = income - expenses
    
    return {
        "income": float(income),
        "expenses": float(expenses),
        "savings": float(savings)
    }

def get_spending_trends(transactions_df, months=6):
    """Get spending trends over the last N months"""
    if transactions_df.empty:
        return pd.DataFrame()
    
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    expenses = transactions_df[transactions_df['type'] == 'expense'].copy()
    
    expenses['month'] = expenses['date'].dt.to_period('M')
    monthly_spending = expenses.groupby('month')['amount'].sum().reset_index()
    monthly_spending['month'] = monthly_spending['month'].astype(str)
    
    return monthly_spending.tail(months)

def calculate_budget_status(budgets_df, transactions_df, month):
    """Calculate budget utilization status"""
    if budgets_df.empty:
        return pd.DataFrame()
    
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    monthly_expenses = transactions_df[
        (transactions_df['date'].dt.strftime('%Y-%m') == month) &
        (transactions_df['type'] == 'expense')
    ]
    
    spent_by_category = monthly_expenses.groupby('category')['amount'].sum()
    
    budget_status = []
    for _, budget in budgets_df.iterrows():
        category = budget['category']
        budget_amount = budget['amount']
        spent = spent_by_category.get(category, 0)
        remaining = budget_amount - spent
        utilization = (spent / budget_amount * 100) if budget_amount > 0 else 0
        
        budget_status.append({
            'category': category,
            'budget': budget_amount,
            'spent': spent,
            'remaining': remaining,
            'utilization': utilization
        })
    
    return pd.DataFrame(budget_status)

def prepare_time_series_data(data, sequence_length):
    """Prepare time series data for LSTM"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def normalize_data(data):
    """Normalize data for neural networks"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data, mean, std
    return (data - mean) / std, mean, std

def denormalize_data(normalized_data, mean, std):
    """Denormalize data"""
    if std == 0:
        return normalized_data
    return normalized_data * std + mean

def calculate_portfolio_metrics(portfolio_df, current_prices):
    """Calculate portfolio performance metrics"""
    if portfolio_df.empty:
        return {}
    
    total_investment = (portfolio_df['quantity'] * portfolio_df['purchase_price']).sum()
    
    current_value = 0
    for _, holding in portfolio_df.iterrows():
        symbol = holding['symbol']
        quantity = holding['quantity']
        if symbol in current_prices:
            current_value += quantity * current_prices[symbol]
        else:
            current_value += quantity * holding['purchase_price']
    
    profit_loss = current_value - total_investment
    profit_loss_pct = (profit_loss / total_investment * 100) if total_investment > 0 else 0
    
    return {
        'total_investment': total_investment,
        'current_value': current_value,
        'profit_loss': profit_loss,
        'profit_loss_pct': profit_loss_pct
    }

def generate_financial_insights(transactions_df, budgets_df):
    """Generate AI-driven financial insights"""
    insights = []
    
    if not transactions_df.empty:
        # Analyze spending patterns
        expenses = transactions_df[transactions_df['type'] == 'expense']
        
        if len(expenses) > 0:
            top_category = expenses.groupby('category')['amount'].sum().idxmax()
            top_amount = expenses.groupby('category')['amount'].sum().max()
            insights.append(f"Your highest spending category is {top_category} with ₹{top_amount:.2f}")
            
            # Check for unusual spending
            recent_expenses = expenses.tail(10)
            avg_expense = expenses['amount'].mean()
            large_expenses = recent_expenses[recent_expenses['amount'] > avg_expense * 2]
            
            if len(large_expenses) > 0:
                insights.append(f"You have {len(large_expenses)} unusually large expenses recently")
    
    # Budget insights
    if not budgets_df.empty and not transactions_df.empty:
        current_month = datetime.now().strftime('%Y-%m')
        budget_status = calculate_budget_status(budgets_df, transactions_df, current_month)
        
        if not budget_status.empty:
            over_budget = budget_status[budget_status['utilization'] > 100]
            if len(over_budget) > 0:
                insights.append(f"You're over budget in {len(over_budget)} categories!")
            
            near_limit = budget_status[
                (budget_status['utilization'] > 80) & (budget_status['utilization'] <= 100)
            ]
            if len(near_limit) > 0:
                insights.append(f"You're approaching budget limits in {len(near_limit)} categories")
    
    if not insights:
        insights.append("Your finances look healthy! Keep up the good work.")
    
    return insights
