"""
AI Financial Management & Stock Intelligence Platform
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

# Import custom modules
from core import config
from core.database import Database
from modules.expense_categorizer import ExpenseCategorizer
from modules.budget_manager import BudgetManager, FinancialGoalTracker
from modules.financial_lstm import FinancialLSTM
from modules.stock_prediction import StockLSTM, StockAnalyzer, get_stock_recommendations
from modules.portfolio_optimizer import PortfolioOptimizer, RiskProfiler
from modules.financial_advisor import FinancialAdvisor
from core.utils import (
    calculate_spending_by_category, calculate_monthly_summary,
    get_spending_trends, generate_financial_insights,
    calculate_portfolio_metrics
)

# Page configuration
st.set_page_config(
    page_title="Financial Management and stock Analysis",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .auth-title {
        text-align: center;
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1rem;
    }
    .auth-form-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .auth-form-subtitle {
        color: #6b7280;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = Database()
    st.session_state.categorizer = ExpenseCategorizer()
    st.session_state.budget_manager = BudgetManager(st.session_state.db)
    st.session_state.goal_tracker = FinancialGoalTracker(st.session_state.db)
    st.session_state.financial_lstm = FinancialLSTM()
    st.session_state.stock_lstm = StockLSTM()
    st.session_state.portfolio_optimizer = PortfolioOptimizer()
    st.session_state.advisor = FinancialAdvisor(st.session_state.db)
    
    # Try to load trained models
    st.session_state.categorizer.load_model()
    st.session_state.financial_lstm.load_model()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'current_user' not in st.session_state:
    st.session_state.current_user = None


def render_auth_screen():
    st.markdown('<h1 class="main-header">Financial Management and stock Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="auth-title">Access your account to manage expenses and analyze stocks</p>', unsafe_allow_html=True)

    _, center_col, _ = st.columns([1.2, 2.2, 1.2])

    with center_col:
        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            st.markdown('<div class="auth-form-title">Welcome Back</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-form-subtitle">Enter your credentials to continue</div>', unsafe_allow_html=True)
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                login_btn = st.form_submit_button("Login", type="primary")

            if login_btn:
                success, message, user = st.session_state.db.authenticate_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.current_user = user
                    st.session_state.db.set_current_user(user['id'])
                    st.success(f"Welcome, {user['username']}!")
                    st.rerun()
                else:
                    st.error(message)

        with register_tab:
            st.markdown('<div class="auth-form-title">Create Account</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-form-subtitle">Set up your login details</div>', unsafe_allow_html=True)
            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Choose username")
                new_email = st.text_input("Email", placeholder="Enter email")
                new_password = st.text_input("Password", type="password", placeholder="Create password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                register_btn = st.form_submit_button("Create Account")

            if register_btn:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = st.session_state.db.create_user(new_username, new_password, new_email)
                    if success:
                        login_success, _, user = st.session_state.db.authenticate_user(new_username, new_password)
                        if login_success:
                            st.session_state.authenticated = True
                            st.session_state.current_user = user
                            st.session_state.db.set_current_user(user['id'])
                            st.success("Account created and logged in successfully")
                            st.rerun()
                    else:
                        st.error(message)


if st.session_state.authenticated and st.session_state.current_user:
    st.session_state.db.set_current_user(st.session_state.current_user['id'])
else:
    render_auth_screen()
    st.stop()

# Main header
st.markdown('<h1 class="main-header">Financial Management and stock Analysis</h1>', unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.caption(f"Logged in as: {st.session_state.current_user['username']}")
    if st.button("Logout"):
        st.session_state.db.clear_current_user()
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.rerun()

    st.divider()
    st.title("Navigation")
    
    page = st.radio(
        "Select Module",
        [
            "Dashboard",
            "Expense Tracking",
            "Budget Planning",
            "Stock Market Analysis",
            "Portfolio Management",
            "AI Financial Advisor",
            "Settings & Training"
        ]
    )
    
    st.divider()
    
    # Quick stats in sidebar
    st.subheader("Quick Stats")
    transactions = st.session_state.db.get_transactions()
    if not transactions.empty:
        summary = calculate_monthly_summary(transactions)
        st.metric("Monthly Income", f"₹{summary['income']:,.2f}")
        st.metric("Monthly Expenses", f"₹{summary['expenses']:,.2f}")
        st.metric("Net Savings", f"₹{summary['savings']:,.2f}", 
                 delta=f"{(summary['savings']/summary['income']*100):.1f}%" if summary['income'] > 0 else "0%")

# ============ DASHBOARD PAGE ============
if page == "Dashboard":
    st.header("Financial Overview Dashboard")
    
    # Get data
    transactions = st.session_state.db.get_transactions()
    
    if transactions.empty:
        st.info("Welcome! Start by adding your first transaction in the Expense Tracking module.")
    else:
        # Monthly summary
        summary = calculate_monthly_summary(transactions)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Income", f"₹{summary['income']:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Expenses", f"₹{summary['expenses']:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Net Savings", f"₹{summary['savings']:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            savings_rate = (summary['savings'] / summary['income'] * 100) if summary['income'] > 0 else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Savings Rate", f"{savings_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spending by Category")
            spending = calculate_spending_by_category(transactions)
            if not spending.empty:
                fig = px.pie(spending, values='amount', names='category', 
                           title='Expense Distribution',
                           color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Spending Trend")
            trends = get_spending_trends(transactions, months=6)
            if not trends.empty:
                fig = px.line(trends, x='month', y='amount',
                            title='Monthly Spending Trend',
                            markers=True)
                fig.update_traces(line_color='#667eea', line_width=3)
                st.plotly_chart(fig, width='stretch')
        
        # AI Insights
        st.subheader("AI-Powered Insights")
        budgets = st.session_state.db.get_budgets(datetime.now().strftime('%Y-%m'))
        insights = generate_financial_insights(transactions, budgets)
        
        cols = st.columns(min(len(insights), 3))
        for idx, insight in enumerate(insights[:3]):
            with cols[idx % 3]:
                st.info(insight)
        
        # Budget alerts
        alerts = st.session_state.budget_manager.check_alerts()
        if alerts:
            st.subheader("Budget Alerts")
            for alert in alerts:
                if alert['level'] == 'critical':
                    st.error(alert['message'])
                else:
                    st.warning(alert['message'])

# ============ EXPENSE TRACKING PAGE ============
elif page == "Expense Tracking":
    st.header("Expense Tracking & Categorization")
    
    tab1, tab2 = st.tabs(["Add Transaction", "View Transactions"])
    
    with tab1:
        st.subheader("Add New Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trans_type = st.selectbox("Transaction Type", ["Expense", "Income"])
            date = st.date_input("Date", datetime.now())
            description = st.text_input("Description", placeholder="e.g., Starbucks coffee")
        
        with col2:
            amount = st.number_input("Amount (INR)", min_value=0.01, value=10.0, step=0.01)
            
            if trans_type == "Expense":
                # Auto-suggest category
                suggested_category = st.session_state.categorizer.predict(description) if description else "Other"
                category = st.selectbox("Category", config.EXPENSE_CATEGORIES, 
                                      index=config.EXPENSE_CATEGORIES.index(suggested_category))
            else:
                category = st.selectbox("Category", config.INCOME_CATEGORIES)
        
        if st.button("Add Transaction", type="primary"):
            st.session_state.db.add_transaction(
                date.strftime('%Y-%m-%d'),
                description,
                amount,
                category,
                trans_type.lower()
            )
            st.success(f"{trans_type} of ₹{amount:.2f} added successfully!")
            st.rerun()
    
    with tab2:
        st.subheader("Recent Transactions")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox("Type", ["All", "Expense", "Income"])
        with col2:
            start_date = st.date_input("From", datetime.now() - timedelta(days=30))
        with col3:
            end_date = st.date_input("To", datetime.now())
        
        # Get filtered transactions
        trans_filter = None if filter_type == "All" else filter_type.lower()
        transactions = st.session_state.db.get_transactions(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            trans_filter
        )
        
        if not transactions.empty:
            st.dataframe(
                transactions[['date', 'description', 'amount', 'category', 'type']],
                width='stretch',
                hide_index=True
            )
            
            # Download option
            csv = transactions.to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv,
                "transactions.csv",
                "text/csv"
            )
        else:
            st.info("No transactions found for the selected period.")

# ============ BUDGET PLANNING PAGE ============
elif page == "Budget Planning":
    st.header("Budget Planning & Alerts")
    
    tab1, tab2, tab3 = st.tabs(["Set Budget", "Budget Status", "Financial Goals"])
    
    with tab1:
        st.subheader("Set Monthly Budget")
        
        current_month = datetime.now().strftime('%Y-%m')
        
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category", config.EXPENSE_CATEGORIES)
        with col2:
            budget_amount = st.number_input("Budget Amount (INR)", min_value=0.0, value=500.0, step=50.0)
        
        if st.button("Set Budget", type="primary"):
            st.session_state.budget_manager.set_budget(category, budget_amount, current_month)
            st.success(f"Budget of ₹{budget_amount:.2f} set for {category}")
            st.rerun()
        
        # Show recommendations
        st.divider()
        st.subheader("Budget Recommendations")
        recommendations = st.session_state.budget_manager.get_budget_recommendations()
        
        if recommendations:
            for rec in recommendations[:5]:
                st.info(f"**{rec['category']}**: ₹{rec['recommended_amount']:.2f} - {rec['reason']}")
    
    with tab2:
        st.subheader("Current Budget Status")
        
        budget_status = st.session_state.budget_manager.get_current_budget_status()
        
        if not budget_status.empty:
            for _, row in budget_status.iterrows():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{row['category']}**")
                    progress = min(row['utilization'] / 100, 1.0)
                    
                    # Color based on utilization
                    if row['utilization'] >= 95:
                        color = 'red'
                    elif row['utilization'] >= 80:
                        color = 'orange'
                    else:
                        color = 'green'
                    
                    st.progress(progress)
                    st.caption(f"₹{row['spent']:.2f} of ₹{row['budget']:.2f} ({row['utilization']:.1f}%)")
                
                with col2:
                    if row['remaining'] >= 0:
                        st.metric("Remaining", f"₹{row['remaining']:.2f}")
                    else:
                        st.metric("Over", f"₹{abs(row['remaining']):.2f}")
        else:
            st.info("No budgets set for this month. Set your first budget above!")
    
    with tab3:
        st.subheader("Financial Goals")
        
        # Add goal
        with st.expander("Add New Goal"):
            col1, col2 = st.columns(2)
            with col1:
                goal_name = st.text_input("Goal Name", placeholder="e.g., Emergency Fund")
                target_amount = st.number_input("Target Amount (INR)", min_value=0.0, value=10000.0)
            with col2:
                target_date = st.date_input("Target Date", datetime.now() + timedelta(days=365))
                current_amount = st.number_input("Current Amount (INR)", min_value=0.0, value=0.0)
            
            if st.button("Add Goal"):
                st.session_state.goal_tracker.add_goal(
                    goal_name, target_amount, target_date.strftime('%Y-%m-%d'), current_amount
                )
                st.success(f"Goal '{goal_name}' added!")
                st.rerun()
        
        # Show goals
        goals = st.session_state.goal_tracker.get_goals_progress()
        
        if not goals.empty:
            for _, goal in goals.iterrows():
                with st.container():
                    st.write(f"### {goal['goal_name']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Progress", f"{goal['progress_pct']:.1f}%")
                    with col2:
                        st.metric("Remaining", f"₹{goal['remaining']:.2f}")
                    with col3:
                        st.metric("Days Left", f"{goal['days_remaining']}")
                    
                    st.progress(min(goal['progress_pct'] / 100, 1.0))
                    
                    if goal['monthly_savings_needed'] > 0:
                        st.caption(f"Save ₹{goal['monthly_savings_needed']:.2f}/month to reach your goal")
                    
                    st.divider()

# ============ STOCK MARKET PAGE ============
elif page == "Stock Market Analysis":
    st.header("Stock Market Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Stock Analysis", "Recommendations", "Technical Indicators"])
    
    with tab1:
        st.subheader("Real-Time Stock Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            symbol = st.text_input("Stock Symbol", value="AAPL", placeholder="e.g., AAPL, GOOGL")
        
        with col2:
            period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"])
        
        if symbol:
            stock_data = StockAnalyzer.get_stock_data(symbol, period)
            
            if stock_data is not None and not stock_data.empty:
                # Stock info
                info = StockAnalyzer.get_stock_info(symbol)
                
                if info:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("Market Cap", f"${info['market_cap']/1e9:.2f}B")
                    with col3:
                        st.metric("P/E Ratio", f"{info['pe_ratio']:.2f}")
                    with col4:
                        st.metric("Beta", f"{info['beta']:.2f}")
                
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name='Price'
                ))
                
                fig.update_layout(
                    title=f'{symbol} Stock Price',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Trading signals
                st.subheader("AI Trading Signals")
                signals = StockAnalyzer.generate_trading_signals(stock_data)
                
                if signals:
                    for signal in signals:
                        if signal['type'] == 'BUY':
                            st.success(f"BUY - **{signal['indicator']}**: {signal['message']}")
                        else:
                            st.error(f"SELL - **{signal['indicator']}**: {signal['message']}")
                else:
                    st.info("No strong signals at the moment. Hold position.")
    
    with tab2:
        st.subheader("LSTM-Based Buy/Sell/Hold Recommendations")

        symbol = st.text_input("Stock Symbol for Recommendation", value="TSLA")
        forecast_days = st.slider("Forecast Horizon (business days)", 5, 60, 30)

        if st.button("Generate LSTM Recommendation", type="primary"):
            with st.spinner(f"Training LSTM and forecasting {symbol}..."):
                stock_data = StockAnalyzer.get_stock_data(symbol, period='6mo')

                if stock_data is None or stock_data.empty:
                    st.error(f"Unable to fetch data for {symbol}. Please check the symbol and try again.")
                else:
                    train_success, train_msg = st.session_state.stock_lstm.train(symbol, period='2y')

                    if not train_success:
                        st.error(f"LSTM training failed: {train_msg}")
                    else:
                        predictions, pred_msg = st.session_state.stock_lstm.predict_future(symbol, days=forecast_days)

                        if predictions is None or predictions.empty:
                            st.error(f"LSTM prediction failed: {pred_msg}")
                        else:
                            info = StockAnalyzer.get_stock_info(symbol)
                            news_items = StockAnalyzer.get_company_news(symbol, limit=8)
                            news_trend = StockAnalyzer.analyze_news_trend(news_items)
                            current_price = stock_data['Close'].iloc[-1]
                            predicted_price = predictions['predicted_price'].iloc[-1]
                            expected_change_pct = ((predicted_price - current_price) / current_price) * 100

                            # Base score from LSTM projected movement.
                            lstm_score = max(-10.0, min(10.0, expected_change_pct))
                            # News score uses headline sentiment trend in the range roughly [-6, +6].
                            news_score = max(-6.0, min(6.0, news_trend.get('score', 0.0) * 6.0))
                            combined_score = lstm_score + news_score

                            # Map combined LSTM + news score to a recommendation band.
                            if combined_score >= 8:
                                recommendation = "STRONG BUY"
                            elif combined_score >= 3:
                                recommendation = "BUY"
                            elif combined_score <= -8:
                                recommendation = "STRONG SELL"
                            elif combined_score <= -3:
                                recommendation = "SELL"
                            else:
                                recommendation = "HOLD"

                            trend_label = news_trend.get('trend', 'NEUTRAL')
                            reason = (
                                f"LSTM expected move: {expected_change_pct:+.2f}%. "
                                f"News trend: {trend_label}. "
                                f"Combined score: {combined_score:+.2f}."
                            )

                            confidence = min(95.0, 55.0 + (abs(combined_score) * 2.5))

                            st.session_state['recommendation'] = {
                                'symbol': symbol,
                                'recommendation': recommendation,
                                'confidence': confidence,
                                'reason': reason,
                                'current_price': current_price,
                                'predicted_price': predicted_price,
                                'expected_change_pct': expected_change_pct,
                                'forecast_days': forecast_days,
                                'predictions': predictions,
                                'historical': stock_data.tail(60),
                                'info': info,
                                'news_items': news_items,
                                'news_trend': news_trend,
                                'lstm_score': lstm_score,
                                'news_score': news_score,
                                'combined_score': combined_score
                            }

                            st.success(f"LSTM recommendation generated for {symbol}!")

        # Display recommendation
        if 'recommendation' in st.session_state:
            rec = st.session_state['recommendation']

            # Main recommendation card
            if rec['recommendation'] in ['STRONG BUY', 'BUY']:
                card_color = '#28a745'
            elif rec['recommendation'] in ['STRONG SELL', 'SELL']:
                card_color = '#dc3545'
            else:
                card_color = '#ffc107'

            st.markdown(f"""
            <div style='background: {card_color}; padding: 2rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0;'>
                <h1 style='margin: 0; font-size: 3rem;'>{rec['recommendation']}</h1>
                <h3 style='margin: 0.5rem 0;'>{rec['symbol']}</h3>
                <p style='margin: 0; font-size: 1.2rem;'>Confidence: {rec['confidence']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${rec['current_price']:.2f}")
            with col2:
                st.metric("Predicted Price", f"${rec['predicted_price']:.2f}")
            with col3:
                st.metric("Expected Change", f"{rec['expected_change_pct']:+.2f}%")

            if 'combined_score' in rec:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("LSTM Score", f"{rec['lstm_score']:+.2f}")
                with col2:
                    st.metric("News Score", f"{rec['news_score']:+.2f}")
                with col3:
                    st.metric("Combined Score", f"{rec['combined_score']:+.2f}")

            if rec.get('news_trend'):
                st.subheader("Company News Trend")
                news_trend = rec['news_trend']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Trend", news_trend.get('trend', 'N/A'))
                with col2:
                    st.metric("Positive", news_trend.get('positive_count', 0))
                with col3:
                    st.metric("Negative", news_trend.get('negative_count', 0))
                with col4:
                    st.metric("Neutral", news_trend.get('neutral_count', 0))

                st.write(news_trend.get('summary', 'News trend not available.'))

                if rec.get('news_items'):
                    st.write("Latest Headlines")
                    for item in rec['news_items'][:5]:
                        title = item.get('title') or 'Untitled'
                        link = item.get('link')
                        publisher = item.get('publisher', 'Unknown')
                        published_at = item.get('published_at')
                        date_text = published_at.strftime('%Y-%m-%d %H:%M') if published_at else 'N/A'

                        if link:
                            st.markdown(f"- [{title}]({link}) ({publisher}, {date_text})")
                        else:
                            st.write(f"- {title} ({publisher}, {date_text})")
                else:
                    st.info("No recent company news available for this symbol.")

            # Historical + forecast chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rec['historical'].index,
                y=rec['historical']['Close'],
                mode='lines',
                name='Recent Close Price',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=rec['predictions']['date'],
                y=rec['predictions']['predicted_price'],
                mode='lines+markers',
                name='LSTM Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig.update_layout(
                title=f"{rec['symbol']} LSTM Forecast ({rec['forecast_days']} Business Days)",
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                height=450
            )
            st.plotly_chart(fig, width='stretch')

            st.subheader("Analysis Summary")
            st.info(rec['reason'])
            st.caption("This recommendation combines LSTM forecasted movement and company news trend.")

            if rec['info']:
                st.subheader("Stock Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Sector**: {rec['info'].get('sector', 'N/A')}")
                    st.write(f"**Industry**: {rec['info'].get('industry', 'N/A')}")
                    st.write(f"**Market Cap**: ${rec['info'].get('market_cap', 0)/1e9:.2f}B")
                with col2:
                    st.write(f"**P/E Ratio**: {rec['info'].get('pe_ratio', 0):.2f}")
                    st.write(f"**Beta**: {rec['info'].get('beta', 0):.2f}")
                    st.write(f"**Dividend Yield**: {rec['info'].get('dividend_yield', 0)*100:.2f}%")

            st.subheader("Recommended Actions")
            if rec['recommendation'] in ['STRONG BUY', 'BUY']:
                st.success("Consider buying shares or increasing your position.")
                st.write("- Review your portfolio allocation before investing")
                st.write("- Consider staggered entries across multiple days")
                st.write("- Set risk limits and a stop-loss")
            elif rec['recommendation'] in ['STRONG SELL', 'SELL']:
                st.warning("Consider selling shares or reducing your position.")
                st.write("- Review tax implications before selling")
                st.write("- Consider partial exits if conviction is low")
                st.write("- Monitor for trend reversal before re-entry")
            else:
                st.info("Maintain your current position and monitor.")
                st.write("- Wait for a stronger directional forecast")
                st.write("- Re-run forecast after major market events")
                st.write("- Combine with your risk profile and fundamentals")
    
    with tab3:
        st.subheader("Technical Indicators")
        
        symbol = st.text_input("Stock Symbol for Analysis", value="MSFT")
        
        if symbol:
            stock_data = StockAnalyzer.get_stock_data(symbol, period='6mo')
            
            if stock_data is not None and not stock_data.empty:
                indicators = StockAnalyzer.calculate_technical_indicators(stock_data)
                
                # Moving Averages Chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['Close'],
                                        name='Price', line=dict(color='black', width=2)))
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['MA_20'],
                                        name='MA 20', line=dict(color='blue', width=1)))
                fig.add_trace(go.Scatter(x=indicators.index, y=indicators['MA_50'],
                                        name='MA 50', line=dict(color='orange', width=1)))
                
                fig.update_layout(title=f'{symbol} - Moving Averages',
                                xaxis_title='Date', yaxis_title='Price ($)')
                
                st.plotly_chart(fig, width='stretch')
                
                # RSI Chart
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=indicators.index, y=indicators['RSI'],
                                                name='RSI', line=dict(color='purple', width=2)))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.update_layout(title='RSI Indicator', yaxis_title='RSI')
                    st.plotly_chart(fig_rsi, width='stretch')
                
                with col2:
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=indicators.index, y=indicators['MACD'],
                                                 name='MACD', line=dict(color='blue', width=2)))
                    fig_macd.add_trace(go.Scatter(x=indicators.index, y=indicators['Signal_Line'],
                                                 name='Signal', line=dict(color='red', width=2)))
                    fig_macd.update_layout(title='MACD', yaxis_title='Value')
                    st.plotly_chart(fig_macd, width='stretch')

# ============ PORTFOLIO MANAGEMENT PAGE ============
elif page == "Portfolio Management":
    st.header("Portfolio Management & Optimization")
    
    tab1, tab2, tab3 = st.tabs(["My Portfolio", "Optimization", "Risk Analysis"])
    
    with tab1:
        st.subheader("Current Portfolio Holdings")
        
        # Add stock to portfolio
        with st.expander("Add Stock to Portfolio"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                symbol = st.text_input("Symbol", placeholder="AAPL")
            with col2:
                quantity = st.number_input("Quantity", min_value=0.01, value=1.0)
            with col3:
                purchase_price = st.number_input("Purchase Price", min_value=0.01, value=100.0)
            with col4:
                purchase_date = st.date_input("Purchase Date", datetime.now())
            
            if st.button("Add to Portfolio"):
                st.session_state.db.add_portfolio_item(
                    symbol.upper(), quantity, purchase_price, purchase_date.strftime('%Y-%m-%d')
                )
                st.success(f"Added {quantity} shares of {symbol.upper()}")
                st.rerun()
        
        # Display portfolio
        portfolio = st.session_state.db.get_portfolio()
        
        if not portfolio.empty:
            # Get current prices
            symbols = portfolio['symbol'].unique()
            current_prices = {}
            
            for sym in symbols:
                try:
                    ticker = yf.Ticker(sym)
                    current_prices[sym] = ticker.history(period='1d')['Close'].iloc[-1]
                except:
                    current_prices[sym] = 0
            
            # Calculate metrics
            portfolio_data = []
            for _, holding in portfolio.iterrows():
                symbol = holding['symbol']
                quantity = holding['quantity']
                purchase_price = holding['purchase_price']
                current_price = current_prices.get(symbol, purchase_price)
                
                investment = quantity * purchase_price
                current_value = quantity * current_price
                profit_loss = current_value - investment
                profit_loss_pct = (profit_loss / investment * 100) if investment > 0 else 0
                
                portfolio_data.append({
                    'Symbol': symbol,
                    'Quantity': quantity,
                    'Purchase Price': f'${purchase_price:.2f}',
                    'Current Price': f'${current_price:.2f}',
                    'Investment': f'${investment:.2f}',
                    'Current Value': f'${current_value:.2f}',
                    'P/L': f'${profit_loss:.2f}',
                    'P/L %': f'{profit_loss_pct:+.2f}%'
                })
            
            st.dataframe(pd.DataFrame(portfolio_data), width='stretch', hide_index=True)
            
            # Total metrics
            metrics = calculate_portfolio_metrics(portfolio, current_prices)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Investment", f"${metrics['total_investment']:.2f}")
            with col2:
                st.metric("Current Value", f"${metrics['current_value']:.2f}")
            with col3:
                st.metric("Total P/L", f"${metrics['profit_loss']:.2f}")
            with col4:
                st.metric("Total Return", f"{metrics['profit_loss_pct']:.2f}%")
        else:
            st.info("Your portfolio is empty. Add your first stock above!")
    
    with tab2:
        st.subheader("Portfolio Optimization")
        
        portfolio = st.session_state.db.get_portfolio()
        
        if not portfolio.empty:
            symbols = portfolio['symbol'].unique().tolist()
            
            st.write("**Current Holdings:**", ", ".join(symbols))
            
            risk_level = st.select_slider(
                "Risk Tolerance",
                options=["Conservative", "Moderate", "Aggressive"]
            )
            
            if st.button("Optimize Portfolio", type="primary"):
                with st.spinner("Optimizing portfolio allocation..."):
                    result = st.session_state.portfolio_optimizer.optimize_portfolio(symbols, risk_level)
                    
                    if result:
                        st.success("Optimal allocation calculated!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Expected Return", f"{result['expected_return']*100:.2f}%")
                        with col2:
                            st.metric("Risk (Volatility)", f"{result['risk']*100:.2f}%")
                        with col3:
                            st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
                        
                        st.subheader("Optimal Allocation")
                        allocation = result['allocation']
                        
                        fig = px.pie(allocation, values='Weight', names='Symbol',
                                   title='Recommended Portfolio Allocation')
                        st.plotly_chart(fig, width='stretch')
                        
                        st.dataframe(allocation, width='stretch', hide_index=True)
                    else:
                        st.error("Unable to optimize portfolio. Need more historical data.")
        else:
            st.info("Add stocks to your portfolio to enable optimization.")
    
    with tab3:
        st.subheader("Risk Analysis")
        
        portfolio = st.session_state.db.get_portfolio()
        
        if not portfolio.empty:
            symbols = portfolio['symbol'].unique()
            current_prices = {}
            
            for sym in symbols:
                try:
                    ticker = yf.Ticker(sym)
                    current_prices[sym] = ticker.history(period='1d')['Close'].iloc[-1]
                except:
                    current_prices[sym] = portfolio[portfolio['symbol'] == sym]['purchase_price'].mean()
            
            risk_metrics = st.session_state.portfolio_optimizer.analyze_portfolio_risk(
                portfolio, current_prices
            )
            
            if risk_metrics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Portfolio Volatility", f"{risk_metrics['volatility']*100:.2f}%")
                    st.metric("Beta", f"{risk_metrics['beta']:.2f}")
                    
                    # Interpret beta
                    if risk_metrics['beta'] > 1:
                        st.caption("More volatile than market")
                    elif risk_metrics['beta'] < 1:
                        st.caption("Less volatile than market")
                    else:
                        st.caption("Same volatility as market")
                
                with col2:
                    st.metric("Value at Risk (95%)", f"{risk_metrics['var_95']*100:.2f}%")
                    st.metric("Diversification Score", f"{risk_metrics['diversification_score']:.1f}/100")
                    
                    # Interpret diversification
                    if risk_metrics['diversification_score'] > 70:
                        st.caption("Well diversified")
                    elif risk_metrics['diversification_score'] > 40:
                        st.caption("Moderate diversification")
                    else:
                        st.caption("Poor diversification")
        else:
            st.info("Add stocks to your portfolio to view risk analysis.")

# ============ AI FINANCIAL ADVISOR PAGE ============
elif page == "AI Financial Advisor":
    st.header("AI-Powered Financial Advisor")
    
    tab1, tab2, tab3 = st.tabs(["Personalized Advice", "Monthly Report", "Investment Tips"])
    
    with tab1:
        st.subheader("Personalized Financial Advice")
        
        if st.button("Generate New Advice", type="primary"):
            with st.spinner("Analyzing your finances..."):
                advice_list = st.session_state.advisor.generate_personalized_advice()
                st.session_state['advice'] = advice_list
        
        if 'advice' in st.session_state:
            advice_list = st.session_state['advice']
            
            if advice_list:
                for advice in advice_list:
                    priority_colors = {
                        'Critical': 'Critical',
                        'High': 'High',
                        'Medium': 'Medium',
                        'Low': 'Low'
                    }
                    
                    with st.expander(f"{priority_colors.get(advice['priority'], '')} {advice['title']}", expanded=True):
                        st.write(f"**Category:** {advice['category']}")
                        st.write(f"**Priority:** {advice['priority']}")
                        st.write(advice['message'])
                        st.info(f"**Action:** {advice['action']}")
            else:
                st.success("Great! No major issues detected. Your finances are healthy!")
    
    with tab2:
        st.subheader("Monthly Financial Report")
        
        report = st.session_state.advisor.generate_monthly_report()
        
        if report:
            st.write(f"### Report for {report['month']}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Income", f"₹{report['income']:.2f}")
            with col2:
                st.metric("Expenses", f"₹{report['expenses']:.2f}")
            with col3:
                st.metric("Savings", f"₹{report['savings']:.2f}")
            with col4:
                st.metric("Savings Rate", f"{report['savings_rate']:.1f}%")
            
            st.divider()
            
            # Expense breakdown
            st.subheader("Expense Breakdown")
            expense_data = pd.DataFrame(list(report['expense_by_category'].items()),
                                       columns=['Category', 'Amount'])
            expense_data = expense_data.sort_values('Amount', ascending=False)
            
            fig = px.bar(expense_data, x='Category', y='Amount',
                        title='Spending by Category',
                        color='Amount',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, width='stretch')
            
            # Month-over-month comparison
            if report['expense_change_pct'] != 0:
                if report['expense_change_pct'] > 0:
                    st.warning(f"Expenses increased by {report['expense_change_pct']:.1f}% compared to last month")
                else:
                    st.success(f"Expenses decreased by {abs(report['expense_change_pct']):.1f}% compared to last month")
        else:
            st.info("No data available for monthly report. Add transactions to generate report.")
    
    with tab3:
        st.subheader("Investment Recommendations")
        
        portfolio = st.session_state.db.get_portfolio()
        
        col1, col2 = st.columns(2)
        with col1:
            risk_tolerance = st.select_slider(
                "Risk Tolerance",
                options=["Conservative", "Moderate", "Aggressive"]
            )
        
        recommendations = st.session_state.advisor.get_investment_recommendations(
            portfolio, risk_tolerance
        )
        
        for rec in recommendations:
            with st.expander(f"{rec['recommendation']}", expanded=True):
                st.write(f"**Type:** {rec['type']}")
                
                details = rec['details']
                if isinstance(details, dict):
                    for key, value in details.items():
                        if isinstance(value, list):
                            st.write(f"**{key.replace('_', ' ').title()}:**")
                            for item in value:
                                st.write(f"  - {item}")
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.write(details)

# ============ SETTINGS PAGE ============
elif page == "Settings & Training":
    st.header("Settings & Model Training")
    
    tab1, tab2 = st.tabs(["Train Models", "System Info"])
    
    with tab1:
        st.subheader("Machine Learning Model Training")
        
        # Expense Categorizer Training
        st.write("### Expense Categorizer")
        
        if st.button("Train Categorizer"):
            with st.spinner("Training categorization model..."):
                transactions = st.session_state.db.get_transactions()
                user_expenses = transactions[transactions['type'] == 'expense'] if not transactions.empty else pd.DataFrame()

                if user_expenses.empty:
                    st.warning("No expense data found. Add expenses first to train the model.")
                else:
                    descriptions = user_expenses['description'].tolist()
                    categories = user_expenses['category'].tolist()

                    accuracy = st.session_state.categorizer.train(descriptions, categories)

                    if accuracy:
                        st.session_state.categorizer.save_model()
                        st.success(f"Model trained! Accuracy: {accuracy*100:.2f}%")
                    else:
                        st.warning("Insufficient data for ML training. Add at least 10 labeled expense entries.")
        
        st.divider()
        
        # Expense LSTM Training
        st.write("### Expense Prediction LSTM")
        
        transactions = st.session_state.db.get_transactions()
        
        if not transactions.empty and len(transactions) >= 30:
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.number_input("Training Epochs", min_value=10, max_value=100, value=50)
            
            with col2:
                batch_size = st.number_input("Batch Size", min_value=8, max_value=64, value=32)
            
            if st.button("Train Expense LSTM"):
                with st.spinner("Training LSTM model..."):
                    success, message = st.session_state.financial_lstm.train(
                        transactions, epochs=epochs, batch_size=batch_size
                    )
                    
                    if success:
                        st.session_state.financial_lstm.save_model()
                        st.success(f"{message}")
                    else:
                        st.error(message)
        else:
            st.info("Need at least 30 days of transaction data to train LSTM model")
    
    with tab2:
        st.subheader("System Information")
        
        transactions = st.session_state.db.get_transactions()
        portfolio = st.session_state.db.get_portfolio()
        budgets = st.session_state.db.get_budgets()
        goals = st.session_state.db.get_financial_goals()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Data Statistics")
            st.metric("Total Transactions", len(transactions))
            st.metric("Portfolio Holdings", len(portfolio))
            st.metric("Active Budgets", len(budgets))
            st.metric("Financial Goals", len(goals))
        
        with col2:
            st.write("### Model Status")
            
            cat_status = "Trained" if st.session_state.categorizer.is_trained else "Not Trained"
            lstm_status = "Trained" if st.session_state.financial_lstm.is_trained else "Not Trained"
            
            st.write(f"**Expense Categorizer:** {cat_status}")
            st.write(f"**Financial LSTM:** {lstm_status}")
        
        st.divider()
        
        # Data Export
        st.write("### Data Export")
        
        if st.button("Export All Data"):
            if not transactions.empty:
                csv = transactions.to_csv(index=False)
                st.download_button(
                    "Download Transactions CSV",
                    csv,
                    "all_transactions.csv",
                    "text/csv"
                )
