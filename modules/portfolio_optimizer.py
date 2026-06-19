"""
Portfolio Optimization and Risk Analysis Module
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
from core import config

class PortfolioOptimizer:
    def __init__(self):
        self.portfolio = None
        self.expected_returns = None
        self.cov_matrix = None
        
    def get_portfolio_data(self, symbols, period='1y'):
        """Fetch historical data for portfolio stocks"""
        try:
            data = yf.download(symbols, period=period, progress=False)['Close']
            
            if isinstance(data, pd.Series):
                data = data.to_frame()
            
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def calculate_returns(self, prices):
        """Calculate daily returns"""
        returns = prices.pct_change().dropna()
        return returns
    
    def calculate_expected_returns(self, returns):
        """Calculate expected annual returns"""
        return returns.mean() * 252  # Annualized
    
    def calculate_covariance_matrix(self, returns):
        """Calculate covariance matrix"""
        return returns.cov() * 252  # Annualized
    
    def portfolio_performance(self, weights, returns, cov_matrix):
        """Calculate portfolio return and risk"""
        portfolio_return = np.sum(returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def negative_sharpe(self, weights, returns, cov_matrix):
        """Negative Sharpe ratio for optimization"""
        return -self.portfolio_performance(weights, returns, cov_matrix)[2]
    
    def optimize_portfolio(self, symbols, risk_level='Moderate'):
        """Optimize portfolio allocation"""
        try:
            # Fetch data
            prices = self.get_portfolio_data(symbols, period='2y')
            
            if prices is None or prices.empty:
                return None
            
            # Calculate returns and statistics
            returns = self.calculate_returns(prices)
            expected_returns = self.calculate_expected_returns(returns)
            cov_matrix = self.calculate_covariance_matrix(returns)
            
            num_assets = len(symbols)
            
            # Constraints
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(num_assets))
            
            # Initial guess
            init_guess = num_assets * [1. / num_assets]
            
            # Optimize for maximum Sharpe ratio
            result = minimize(
                self.negative_sharpe,
                init_guess,
                args=(expected_returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            optimal_weights = result.x
            
            # Calculate performance
            ret, risk, sharpe = self.portfolio_performance(
                optimal_weights, expected_returns, cov_matrix
            )
            
            # Create allocation dataframe
            allocation = pd.DataFrame({
                'Symbol': symbols,
                'Weight': optimal_weights,
                'Expected_Return': expected_returns.values
            })
            
            allocation = allocation.sort_values('Weight', ascending=False)
            
            return {
                'allocation': allocation,
                'expected_return': ret,
                'risk': risk,
                'sharpe_ratio': sharpe
            }
            
        except Exception as e:
            print(f"Optimization error: {e}")
            return None
    
    def calculate_var(self, returns, confidence=0.95):
        """Calculate Value at Risk"""
        var = np.percentile(returns, (1 - confidence) * 100)
        return var
    
    def calculate_cvar(self, returns, confidence=0.95):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()
        return cvar
    
    def analyze_portfolio_risk(self, portfolio_df, current_prices):
        """Comprehensive portfolio risk analysis"""
        if portfolio_df.empty:
            return None
        
        symbols = portfolio_df['symbol'].unique().tolist()
        
        # Get historical data
        prices = self.get_portfolio_data(symbols, period='1y')
        
        if prices is None or prices.empty:
            return None
        
        returns = self.calculate_returns(prices)
        
        # Calculate portfolio weights
        total_value = 0
        for symbol in symbols:
            holdings = portfolio_df[portfolio_df['symbol'] == symbol]
            quantity = holdings['quantity'].sum()
            price = current_prices.get(symbol, holdings['purchase_price'].mean())
            total_value += quantity * price
        
        weights = []
        for symbol in symbols:
            holdings = portfolio_df[portfolio_df['symbol'] == symbol]
            quantity = holdings['quantity'].sum()
            price = current_prices.get(symbol, holdings['purchase_price'].mean())
            weight = (quantity * price) / total_value if total_value > 0 else 0
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Risk metrics
        var_95 = self.calculate_var(portfolio_returns, 0.95)
        cvar_95 = self.calculate_cvar(portfolio_returns, 0.95)
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Beta calculation (relative to S&P 500)
        try:
            spy = yf.download('SPY', period='1y', progress=False)['Close']
            spy_returns = spy.pct_change().dropna()
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(spy_returns.index)
            port_ret_aligned = portfolio_returns.loc[common_dates]
            spy_ret_aligned = spy_returns.loc[common_dates]
            
            covariance = np.cov(port_ret_aligned, spy_ret_aligned)[0][1]
            market_variance = np.var(spy_ret_aligned)
            beta = covariance / market_variance if market_variance > 0 else 1.0
        except:
            beta = 1.0
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'beta': beta,
            'diversification_score': self.calculate_diversification_score(weights)
        }
    
    def calculate_diversification_score(self, weights):
        """Calculate portfolio diversification score (0-100)"""
        # Herfindahl index
        herfindahl = np.sum(weights ** 2)
        
        # Normalize to 0-100 scale
        # Perfect diversification (equal weights) = 100
        # Single stock = 0
        n = len(weights)
        min_herfindahl = 1.0  # Single asset
        max_herfindahl = 1.0 / n  # Equal distribution
        
        if min_herfindahl == max_herfindahl:
            return 100
        
        score = 100 * (1 - (herfindahl - max_herfindahl) / (min_herfindahl - max_herfindahl))
        return max(0, min(100, score))
    
    def rebalance_portfolio(self, current_portfolio, target_allocation):
        """Generate rebalancing recommendations"""
        recommendations = []
        
        for _, target in target_allocation.iterrows():
            symbol = target['Symbol']
            target_weight = target['Weight']
            
            current = current_portfolio[current_portfolio['symbol'] == symbol]
            
            if not current.empty:
                current_weight = current['weight'].values[0]
                difference = target_weight - current_weight
                
                if abs(difference) > 0.05:  # 5% threshold
                    action = 'BUY' if difference > 0 else 'SELL'
                    recommendations.append({
                        'symbol': symbol,
                        'action': action,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'difference': difference
                    })
        
        return recommendations

class RiskProfiler:
    """Assess investor risk profile and provide recommendations"""
    
    @staticmethod
    def calculate_risk_score(age, income, investment_horizon, risk_tolerance):
        """Calculate risk score based on investor profile"""
        # Age factor (younger = higher risk capacity)
        age_score = max(0, min(100, (65 - age) / 45 * 100))
        
        # Income factor (higher income = higher risk capacity)
        income_score = min(100, income / 100000 * 50)
        
        # Time horizon factor (longer = higher risk capacity)
        horizon_score = min(100, investment_horizon / 30 * 100)
        
        # Risk tolerance (subjective)
        tolerance_map = {'Low': 25, 'Moderate': 50, 'High': 75, 'Very High': 100}
        tolerance_score = tolerance_map.get(risk_tolerance, 50)
        
        # Weighted average
        risk_score = (
            age_score * 0.25 +
            income_score * 0.2 +
            horizon_score * 0.3 +
            tolerance_score * 0.25
        )
        
        return risk_score
    
    @staticmethod
    def recommend_asset_allocation(risk_score):
        """Recommend asset allocation based on risk score"""
        if risk_score >= 70:
            return {
                'stocks': 80,
                'bonds': 15,
                'cash': 5,
                'profile': 'Aggressive Growth'
            }
        elif risk_score >= 50:
            return {
                'stocks': 60,
                'bonds': 30,
                'cash': 10,
                'profile': 'Moderate Growth'
            }
        elif risk_score >= 30:
            return {
                'stocks': 40,
                'bonds': 50,
                'cash': 10,
                'profile': 'Conservative Growth'
            }
        else:
            return {
                'stocks': 20,
                'bonds': 60,
                'cash': 20,
                'profile': 'Capital Preservation'
            }
