"""
Budget Planning and Alert System
"""

import pandas as pd
from datetime import datetime
from core import config
from core.utils import calculate_budget_status

class BudgetManager:
    def __init__(self, database):
        self.db = database
        
    def set_budget(self, category, amount, month=None):
        """Set budget for a category"""
        if month is None:
            month = datetime.now().strftime('%Y-%m')
        
        self.db.set_budget(category, amount, month)
        
    def get_current_budget_status(self):
        """Get current month's budget status"""
        current_month = datetime.now().strftime('%Y-%m')
        budgets_df = self.db.get_budgets(current_month)
        
        if budgets_df.empty:
            return pd.DataFrame()
        
        transactions_df = self.db.get_transactions()
        budget_status = calculate_budget_status(budgets_df, transactions_df, current_month)
        
        return budget_status
    
    def check_alerts(self):
        """Check for budget alerts"""
        budget_status = self.get_current_budget_status()
        
        alerts = []
        
        if budget_status.empty:
            return alerts
        
        for _, row in budget_status.iterrows():
            category = row['category']
            utilization = row['utilization']
            spent = row['spent']
            budget = row['budget']
            
            if utilization >= config.BUDGET_CRITICAL_THRESHOLD * 100:
                alerts.append({
                    'level': 'critical',
                    'category': category,
                    'message': f"CRITICAL: You've spent ₹{spent:.2f} out of ₹{budget:.2f} ({utilization:.1f}%) in {category}!",
                    'utilization': utilization
                })
            elif utilization >= config.BUDGET_WARNING_THRESHOLD * 100:
                alerts.append({
                    'level': 'warning',
                    'category': category,
                    'message': f"WARNING: You've spent ₹{spent:.2f} out of ₹{budget:.2f} ({utilization:.1f}%) in {category}.",
                    'utilization': utilization
                })
        
        return alerts
    
    def get_budget_recommendations(self):
        """Generate budget recommendations based on spending patterns"""
        transactions_df = self.db.get_transactions()
        
        if transactions_df.empty:
            return []
        
        recommendations = []
        
        # Analyze past 3 months spending
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        expenses = transactions_df[transactions_df['type'] == 'expense']
        
        if not expenses.empty:
            # Calculate average spending by category
            avg_spending = expenses.groupby('category')['amount'].mean()
            
            for category, avg in avg_spending.items():
                recommended_budget = avg * 1.1  # Add 10% buffer
                recommendations.append({
                    'category': category,
                    'recommended_amount': recommended_budget,
                    'reason': f"Based on average spending of ₹{avg:.2f}"
                })
        
        return recommendations
    
    def forecast_monthly_expenses(self):
        """Forecast monthly expenses based on current trends"""
        transactions_df = self.db.get_transactions()
        
        if transactions_df.empty:
            return 0
        
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        current_month = datetime.now().strftime('%Y-%m')
        
        # Get current month expenses
        monthly_expenses = transactions_df[
            (transactions_df['date'].dt.strftime('%Y-%m') == current_month) &
            (transactions_df['type'] == 'expense')
        ]
        
        if monthly_expenses.empty:
            return 0
        
        # Calculate daily average and project for the whole month
        days_passed = datetime.now().day
        total_spent = monthly_expenses['amount'].sum()
        daily_average = total_spent / days_passed
        
        # Estimate total month (assuming 30 days)
        forecasted_total = daily_average * 30
        
        return forecasted_total
    
    def get_savings_recommendations(self):
        """Generate personalized savings recommendations"""
        budget_status = self.get_current_budget_status()
        transactions_df = self.db.get_transactions()
        
        if budget_status.empty or transactions_df.empty:
            return []
        
        recommendations = []
        
        # Find categories with highest overspending
        over_budget = budget_status[budget_status['utilization'] > 100].sort_values(
            'utilization', ascending=False
        )
        
        if not over_budget.empty:
            for _, row in over_budget.head(3).iterrows():
                overspent = row['spent'] - row['budget']
                recommendations.append({
                    'type': 'reduce_spending',
                    'category': row['category'],
                    'message': f"Consider reducing {row['category']} spending by ₹{overspent:.2f} to stay within budget",
                    'potential_savings': overspent
                })
        
        # Analyze transaction patterns
        expenses = transactions_df[transactions_df['type'] == 'expense']
        
        if not expenses.empty:
            # Find frequently occurring small expenses that add up
            frequent_expenses = expenses[expenses['amount'] < 20].groupby('description').agg({
                'amount': ['count', 'sum']
            }).reset_index()
            
            frequent_expenses.columns = ['description', 'count', 'total']
            frequent_expenses = frequent_expenses[frequent_expenses['count'] >= 5].sort_values(
                'total', ascending=False
            )
            
            if not frequent_expenses.empty:
                for _, row in frequent_expenses.head(2).iterrows():
                    recommendations.append({
                        'type': 'reduce_frequency',
                        'category': 'General',
                        'message': f"You've spent ₹{row['total']:.2f} on '{row['description']}' ({int(row['count'])} times). Consider reducing frequency.",
                        'potential_savings': row['total'] * 0.3  # 30% reduction
                    })
        
        return recommendations

class FinancialGoalTracker:
    def __init__(self, database):
        self.db = database
    
    def add_goal(self, goal_name, target_amount, target_date, current_amount=0):
        """Add a new financial goal"""
        self.db.add_financial_goal(goal_name, target_amount, target_date, current_amount)
    
    def get_goals_progress(self):
        """Get progress on all financial goals"""
        goals_df = self.db.get_financial_goals()
        
        if goals_df.empty:
            return pd.DataFrame()
        
        goals_df['progress_pct'] = (goals_df['current_amount'] / goals_df['target_amount'] * 100)
        goals_df['remaining'] = goals_df['target_amount'] - goals_df['current_amount']
        
        # Calculate days remaining
        goals_df['target_date'] = pd.to_datetime(goals_df['target_date'])
        goals_df['days_remaining'] = (goals_df['target_date'] - datetime.now()).dt.days
        
        # Calculate required monthly savings
        goals_df['monthly_savings_needed'] = goals_df.apply(
            lambda row: row['remaining'] / max(row['days_remaining'] / 30, 1) if row['days_remaining'] > 0 else 0,
            axis=1
        )
        
        return goals_df
    
    def update_goal_progress(self, goal_id, amount_to_add):
        """Update progress on a goal"""
        goals_df = self.db.get_financial_goals()
        current_amount = goals_df[goals_df['id'] == goal_id]['current_amount'].values[0]
        new_amount = current_amount + amount_to_add
        self.db.update_goal_progress(goal_id, new_amount)
