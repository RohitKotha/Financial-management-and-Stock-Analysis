"""
AI-Powered Personalized Financial Advisor
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core import config

class FinancialAdvisor:
    def __init__(self, database):
        self.db = database
        
    def analyze_spending_behavior(self, transactions_df):
        """Analyze user's spending patterns and behavior"""
        if transactions_df.empty:
            return {}
        
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        expenses = transactions_df[transactions_df['type'] == 'expense']
        
        if expenses.empty:
            return {}
        
        analysis = {}
        
        # Total and average spending
        analysis['total_spending'] = expenses['amount'].sum()
        analysis['avg_transaction'] = expenses['amount'].mean()
        analysis['median_transaction'] = expenses['amount'].median()
        
        # Category breakdown
        category_spending = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
        analysis['top_categories'] = category_spending.head(5).to_dict()
        
        # Spending frequency
        analysis['transaction_frequency'] = len(expenses)
        analysis['avg_daily_spending'] = expenses['amount'].sum() / max(len(expenses), 1)
        
        # Weekend vs weekday spending
        expenses['weekday'] = expenses['date'].dt.dayofweek
        weekend_spending = expenses[expenses['weekday'] >= 5]['amount'].sum()
        weekday_spending = expenses[expenses['weekday'] < 5]['amount'].sum()
        analysis['weekend_spending_ratio'] = (
            weekend_spending / (weekend_spending + weekday_spending) 
            if (weekend_spending + weekday_spending) > 0 else 0
        )
        
        # Identify spending spikes
        daily_spending = expenses.groupby(expenses['date'].dt.date)['amount'].sum()
        analysis['max_daily_spending'] = daily_spending.max()
        analysis['spending_volatility'] = daily_spending.std()
        
        return analysis
    
    def generate_personalized_advice(self):
        """Generate comprehensive personalized financial advice"""
        transactions_df = self.db.get_transactions()
        budgets_df = self.db.get_budgets(datetime.now().strftime('%Y-%m'))
        goals_df = self.db.get_financial_goals()
        
        advice = []
        
        # Analyze spending behavior
        if not transactions_df.empty:
            behavior = self.analyze_spending_behavior(transactions_df)
            
            # Advice based on spending patterns
            if behavior.get('weekend_spending_ratio', 0) > 0.4:
                advice.append({
                    'category': 'Spending Behavior',
                    'priority': 'Medium',
                    'title': 'Weekend Spending Pattern Detected',
                    'message': 'You tend to spend more on weekends. Consider planning weekend activities with a budget in mind.',
                    'action': 'Set a weekend spending limit'
                })
            
            # High spending volatility
            if behavior.get('spending_volatility', 0) > behavior.get('avg_daily_spending', 0):
                advice.append({
                    'category': 'Spending Behavior',
                    'priority': 'Medium',
                    'title': 'Inconsistent Spending Pattern',
                    'message': 'Your daily spending varies significantly. Try to maintain more consistent spending habits.',
                    'action': 'Create a daily spending plan'
                })
            
            # Category-specific advice
            top_categories = behavior.get('top_categories', {})
            if top_categories:
                top_category = list(top_categories.keys())[0]
                top_amount = list(top_categories.values())[0]
                
                advice.append({
                    'category': 'Category Focus',
                    'priority': 'High',
                    'title': f'High Spending in {top_category}',
                    'message': f'You\'ve spent ₹{top_amount:.2f} on {top_category}. This is your largest expense category.',
                    'action': f'Review {top_category} expenses for savings opportunities'
                })
        
        # Budget-related advice
        if not budgets_df.empty and not transactions_df.empty:
            from core.utils import calculate_budget_status
            budget_status = calculate_budget_status(
                budgets_df, 
                transactions_df, 
                datetime.now().strftime('%Y-%m')
            )
            
            if not budget_status.empty:
                over_budget = budget_status[budget_status['utilization'] > 100]
                
                for _, row in over_budget.iterrows():
                    advice.append({
                        'category': 'Budget Alert',
                        'priority': 'Critical',
                        'title': f'Over Budget in {row["category"]}',
                        'message': f'You\'re ₹{abs(row["remaining"]):.2f} over budget in {row["category"]}.',
                        'action': 'Reduce spending or adjust budget'
                    })
        
        # Financial goals advice
        if not goals_df.empty:
            goals_df['target_date'] = pd.to_datetime(goals_df['target_date'])
            goals_df['days_remaining'] = (goals_df['target_date'] - datetime.now()).dt.days
            
            for _, goal in goals_df.iterrows():
                progress = (goal['current_amount'] / goal['target_amount'] * 100)
                
                if progress < 50 and goal['days_remaining'] < 180:
                    advice.append({
                        'category': 'Financial Goals',
                        'priority': 'High',
                        'title': f'Goal at Risk: {goal["goal_name"]}',
                        'message': f'You\'re at {progress:.1f}% progress with only {goal["days_remaining"]} days left.',
                        'action': f'Increase monthly contribution to ₹{(goal["target_amount"] - goal["current_amount"]) / max(goal["days_remaining"] / 30, 1):.2f}'
                    })
        
        # Income vs expenses analysis
        if not transactions_df.empty:
            current_month = datetime.now().strftime('%Y-%m')
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
            monthly_data = transactions_df[
                transactions_df['date'].dt.strftime('%Y-%m') == current_month
            ]
            
            income = monthly_data[monthly_data['type'] == 'income']['amount'].sum()
            expenses = monthly_data[monthly_data['type'] == 'expense']['amount'].sum()
            
            savings_rate = ((income - expenses) / income * 100) if income > 0 else 0
            
            if savings_rate < 20:
                advice.append({
                    'category': 'Savings',
                    'priority': 'High',
                    'title': 'Low Savings Rate',
                    'message': f'Your savings rate is {savings_rate:.1f}%. Aim for at least 20% of income.',
                    'action': 'Increase income or reduce expenses to save more'
                })
            elif savings_rate > 50:
                advice.append({
                    'category': 'Savings',
                    'priority': 'Low',
                    'title': 'Excellent Savings Rate!',
                    'message': f'Your savings rate of {savings_rate:.1f}% is outstanding! Keep up the great work.',
                    'action': 'Consider investing surplus savings'
                })
        
        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        advice.sort(key=lambda x: priority_order.get(x['priority'], 99))
        
        return advice
    
    def get_investment_recommendations(self, portfolio_df, risk_tolerance='Moderate'):
        """Generate investment recommendations"""
        recommendations = []
        
        # Asset allocation advice
        allocation_advice = {
            'Conservative': {
                'stocks': '30-40%',
                'bonds': '50-60%',
                'cash': '10-15%',
                'message': 'Focus on capital preservation with steady income'
            },
            'Moderate': {
                'stocks': '50-60%',
                'bonds': '30-40%',
                'cash': '5-10%',
                'message': 'Balance growth and stability'
            },
            'Aggressive': {
                'stocks': '70-85%',
                'bonds': '10-20%',
                'cash': '5-10%',
                'message': 'Maximize growth potential'
            }
        }
        
        advice = allocation_advice.get(risk_tolerance, allocation_advice['Moderate'])
        
        recommendations.append({
            'type': 'Asset Allocation',
            'recommendation': f'Recommended allocation for {risk_tolerance} risk profile',
            'details': advice
        })
        
        # Diversification advice
        if not portfolio_df.empty:
            unique_stocks = portfolio_df['symbol'].nunique()
            
            if unique_stocks < 5:
                recommendations.append({
                    'type': 'Diversification',
                    'recommendation': 'Increase Portfolio Diversification',
                    'details': {
                        'current': f'{unique_stocks} stocks',
                        'suggested': '8-15 stocks across different sectors',
                        'reason': 'Reduce risk through diversification'
                    }
                })
            elif unique_stocks > 30:
                recommendations.append({
                    'type': 'Diversification',
                    'recommendation': 'Consider Portfolio Consolidation',
                    'details': {
                        'current': f'{unique_stocks} stocks',
                        'suggested': '15-25 high-quality stocks',
                        'reason': 'Easier to manage and monitor'
                    }
                })
        
        # Sector diversification
        recommendations.append({
            'type': 'Sector Allocation',
            'recommendation': 'Diversify Across Sectors',
            'details': {
                'suggested_sectors': [
                    'Technology (20-25%)',
                    'Healthcare (15-20%)',
                    'Financial Services (10-15%)',
                    'Consumer Goods (10-15%)',
                    'Industrial (10-15%)',
                    'Other (15-20%)'
                ]
            }
        })
        
        return recommendations
    
    def generate_monthly_report(self):
        """Generate comprehensive monthly financial report"""
        transactions_df = self.db.get_transactions()
        
        if transactions_df.empty:
            return None
        
        current_month = datetime.now().strftime('%Y-%m')
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        monthly_data = transactions_df[
            transactions_df['date'].dt.strftime('%Y-%m') == current_month
        ]
        
        # Calculate key metrics
        income = monthly_data[monthly_data['type'] == 'income']['amount'].sum()
        expenses = monthly_data[monthly_data['type'] == 'expense']['amount'].sum()
        savings = income - expenses
        savings_rate = (savings / income * 100) if income > 0 else 0
        
        # Category breakdown
        expense_by_category = monthly_data[
            monthly_data['type'] == 'expense'
        ].groupby('category')['amount'].sum().sort_values(ascending=False)
        
        # Compare to previous month
        prev_month = (datetime.now() - timedelta(days=30)).strftime('%Y-%m')
        prev_data = transactions_df[
            transactions_df['date'].dt.strftime('%Y-%m') == prev_month
        ]
        
        prev_expenses = prev_data[prev_data['type'] == 'expense']['amount'].sum()
        expense_change = ((expenses - prev_expenses) / prev_expenses * 100) if prev_expenses > 0 else 0
        
        report = {
            'month': current_month,
            'income': income,
            'expenses': expenses,
            'savings': savings,
            'savings_rate': savings_rate,
            'expense_by_category': expense_by_category.to_dict(),
            'expense_change_pct': expense_change,
            'transaction_count': len(monthly_data),
            'avg_transaction_size': monthly_data['amount'].mean()
        }
        
        return report
    
    def get_tax_optimization_tips(self, income, investments):
        """Provide tax optimization suggestions"""
        tips = []
        
        # Tax-advantaged accounts
        tips.append({
            'category': 'Tax Strategy',
            'tip': 'Maximize Retirement Contributions',
            'details': 'Consider maxing out 401(k) and IRA contributions to reduce taxable income',
            'potential_savings': 'Up to ₹7,500/year in tax savings'
        })
        
        # Capital gains management
        if investments > 0:
            tips.append({
                'category': 'Investment Tax',
                'tip': 'Tax Loss Harvesting',
                'details': 'Offset capital gains by selling losing positions before year-end',
                'potential_savings': 'Varies based on gains'
            })
        
        # HSA contributions
        tips.append({
            'category': 'Healthcare',
            'tip': 'Health Savings Account (HSA)',
            'details': 'Triple tax advantage - deductible contributions, tax-free growth, tax-free withdrawals for medical',
            'potential_savings': 'Up to ₹1,000+/year'
        })
        
        return tips
