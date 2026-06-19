"""
Stock Market Prediction Module using LSTM and Transformer Networks
"""

import numpy as np
import pandas as pd
import yfinance as yf
import os
import warnings

# Reduce TensorFlow startup noise in console output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

warnings.filterwarnings("ignore", message=".*reset_default_graph.*")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.get_logger().setLevel("ERROR")
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from core import config

class StockLSTM:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.sequence_length = 60  # Use 60 days of history
        
    def build_model(self, input_shape):
        """Build LSTM model for stock prediction"""
        if not HAS_TENSORFLOW:
            return None
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_stock_data(self, stock_data):
        """Prepare stock data for LSTM"""
        if len(stock_data) < self.sequence_length + 1:
            return None, None
        
        # Use closing prices
        prices = stock_data['Close'].values.reshape(-1, 1)
        
        # Normalize
        prices_scaled = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(self.sequence_length, len(prices_scaled)):
            X.append(prices_scaled[i-self.sequence_length:i, 0])
            y.append(prices_scaled[i, 0])
        
        return np.array(X), np.array(y)
    
    def train(self, stock_symbol, period='2y'):
        """Train model on stock data"""
        try:
            # Download stock data
            stock_data = yf.download(stock_symbol, period=period, progress=False)
            
            if stock_data.empty:
                return False, f"No data available for {stock_symbol}"
            
            X, y = self.prepare_stock_data(stock_data)
            
            if X is None or len(X) < 50:
                return False, "Insufficient data for training"
            
            # Reshape for LSTM
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Build model
            self.model = self.build_model((X.shape[1], 1))
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.is_trained = True
            return True, "Model trained successfully"
            
        except Exception as e:
            return False, f"Error training model: {str(e)}"
    
    def predict_future(self, stock_symbol, days=30):
        """Predict stock prices for next N days"""
        try:
            # Get recent data
            stock_data = yf.download(stock_symbol, period='3mo', progress=False)
            
            if stock_data.empty:
                return None, f"No data available for {stock_symbol}"
            
            # Get last sequence
            last_sequence = stock_data['Close'].values[-self.sequence_length:]
            last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
            
            predictions = []
            current_sequence = last_sequence_scaled.flatten()
            
            for _ in range(days):
                # Prepare input
                input_seq = current_sequence.reshape(1, self.sequence_length, 1)
                
                # Predict
                next_pred = self.model.predict(input_seq, verbose=0)[0, 0]
                predictions.append(next_pred)
                
                # Update sequence
                current_sequence = np.append(current_sequence[1:], next_pred)
            
            # Denormalize
            predictions = self.scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            ).flatten()
            
            # Create dates
            last_date = stock_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=days,
                freq='B'  # Business days
            )
            
            predictions_df = pd.DataFrame({
                'date': future_dates[:len(predictions)],
                'predicted_price': predictions[:len(future_dates)]
            })
            
            return predictions_df, "Predictions generated successfully"
            
        except Exception as e:
            return None, f"Error making predictions: {str(e)}"

class StockAnalyzer:
    """Analyze stock data and provide insights"""
    
    @staticmethod
    def get_stock_data(symbol, period='1y'):
        """Fetch stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            return None
    
    @staticmethod
    def calculate_technical_indicators(stock_data):
        """Calculate technical indicators"""
        df = stock_data.copy()
        
        # Moving Averages
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df
    
    @staticmethod
    def generate_trading_signals(stock_data):
        """Generate buy/sell signals based on technical indicators"""
        df = StockAnalyzer.calculate_technical_indicators(stock_data)
        
        if df.empty:
            return []
        
        signals = []
        latest = df.iloc[-1]
        
        # RSI signals
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            if latest['RSI'] < 30:
                signals.append({
                    'type': 'BUY',
                    'indicator': 'RSI',
                    'message': f"RSI is {latest['RSI']:.2f} (Oversold - potential buy signal)"
                })
            elif latest['RSI'] > 70:
                signals.append({
                    'type': 'SELL',
                    'indicator': 'RSI',
                    'message': f"RSI is {latest['RSI']:.2f} (Overbought - potential sell signal)"
                })
        
        # Moving Average Crossover
        if all(col in df.columns for col in ['MA_20', 'MA_50']):
            if not pd.isna(latest['MA_20']) and not pd.isna(latest['MA_50']):
                prev = df.iloc[-2]
                if prev['MA_20'] < prev['MA_50'] and latest['MA_20'] > latest['MA_50']:
                    signals.append({
                        'type': 'BUY',
                        'indicator': 'MA Crossover',
                        'message': "20-day MA crossed above 50-day MA (Bullish signal)"
                    })
                elif prev['MA_20'] > prev['MA_50'] and latest['MA_20'] < latest['MA_50']:
                    signals.append({
                        'type': 'SELL',
                        'indicator': 'MA Crossover',
                        'message': "20-day MA crossed below 50-day MA (Bearish signal)"
                    })
        
        # MACD signals
        if all(col in df.columns for col in ['MACD', 'Signal_Line']):
            if not pd.isna(latest['MACD']) and not pd.isna(latest['Signal_Line']):
                if latest['MACD'] > latest['Signal_Line']:
                    signals.append({
                        'type': 'BUY',
                        'indicator': 'MACD',
                        'message': "MACD is above signal line (Bullish momentum)"
                    })
                else:
                    signals.append({
                        'type': 'SELL',
                        'indicator': 'MACD',
                        'message': "MACD is below signal line (Bearish momentum)"
                    })
        
        return signals
    
    @staticmethod
    def get_stock_info(symbol):
        """Get stock information"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52w_high': info.get('fiftyTwoWeekHigh', 0),
                '52w_low': info.get('fiftyTwoWeekLow', 0)
            }
        except:
            return None

    @staticmethod
    def get_company_news(symbol, limit=10):
        """Fetch recent company news for trend analysis"""
        try:
            ticker = yf.Ticker(symbol)
            news_items = []

            # yfinance can expose either .news or .get_news() depending on version.
            raw_news = []
            try:
                raw_news = ticker.news or []
            except Exception:
                raw_news = []

            if not raw_news:
                try:
                    raw_news = ticker.get_news() or []
                except Exception:
                    raw_news = []

            for item in raw_news[:limit]:
                title = item.get('title') or item.get('content', {}).get('title') or ""
                link = item.get('link') or item.get('content', {}).get('canonicalUrl', {}).get('url') or ""
                publisher = item.get('publisher') or item.get('content', {}).get('provider', {}).get('displayName') or "Unknown"
                published_at = None

                published_ts = item.get('providerPublishTime') or item.get('pubDate')
                if published_ts:
                    try:
                        published_at = datetime.fromtimestamp(int(published_ts))
                    except Exception:
                        published_at = None

                if title:
                    news_items.append({
                        'title': title,
                        'link': link,
                        'publisher': publisher,
                        'published_at': published_at
                    })

            return news_items
        except Exception:
            return []

    @staticmethod
    def analyze_news_trend(news_items):
        """Estimate bullish/bearish trend from news headlines"""
        if not news_items:
            return {
                'trend': 'NEUTRAL',
                'score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'summary': 'No recent company headlines available to infer trend.'
            }

        positive_keywords = {
            'beat', 'surge', 'rise', 'rally', 'growth', 'strong', 'upgrade',
            'profit', 'record', 'expand', 'bullish', 'outperform', 'gains'
        }
        negative_keywords = {
            'miss', 'drop', 'fall', 'decline', 'downgrade', 'loss', 'weak',
            'lawsuit', 'investigation', 'bearish', 'underperform', 'cuts', 'plunge'
        }

        scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for item in news_items:
            text = (item.get('title') or '').lower()
            pos_hits = sum(1 for kw in positive_keywords if kw in text)
            neg_hits = sum(1 for kw in negative_keywords if kw in text)
            raw_score = pos_hits - neg_hits

            if raw_score > 0:
                positive_count += 1
            elif raw_score < 0:
                negative_count += 1
            else:
                neutral_count += 1

            if raw_score > 0:
                scores.append(min(raw_score, 3) / 3.0)
            elif raw_score < 0:
                scores.append(max(raw_score, -3) / 3.0)
            else:
                scores.append(0.0)

        avg_score = float(np.mean(scores)) if scores else 0.0

        if avg_score >= 0.2:
            trend = 'BULLISH'
            summary = 'Recent headlines are mostly positive for the company.'
        elif avg_score <= -0.2:
            trend = 'BEARISH'
            summary = 'Recent headlines are mostly negative for the company.'
        else:
            trend = 'NEUTRAL'
            summary = 'Recent headlines are mixed with no strong directional bias.'

        return {
            'trend': trend,
            'score': avg_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'summary': summary
        }

def get_stock_recommendations(portfolio_symbols):
    """Get AI-powered stock recommendations"""
    recommendations = []
    
    for symbol in portfolio_symbols:
        try:
            stock_data = StockAnalyzer.get_stock_data(symbol, period='6mo')
            
            if stock_data is not None and not stock_data.empty:
                # Calculate returns
                recent_return = (
                    (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0]) / 
                    stock_data['Close'].iloc[0] * 100
                )
                
                # Get signals
                signals = StockAnalyzer.generate_trading_signals(stock_data)
                
                buy_signals = len([s for s in signals if s['type'] == 'BUY'])
                sell_signals = len([s for s in signals if s['type'] == 'SELL'])
                
                if buy_signals > sell_signals:
                    recommendation = 'BUY'
                    confidence = buy_signals / (buy_signals + sell_signals) * 100
                elif sell_signals > buy_signals:
                    recommendation = 'SELL'
                    confidence = sell_signals / (buy_signals + sell_signals) * 100
                else:
                    recommendation = 'HOLD'
                    confidence = 50
                
                recommendations.append({
                    'symbol': symbol,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'recent_return': recent_return,
                    'signals': signals
                })
        except:
            continue
    
    return recommendations
