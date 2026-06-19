"""
LSTM Model for Financial Predictions (Cash Flow and Expense Forecasting)
"""

import numpy as np
import pandas as pd
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
    tf = None
    keras = None
    layers = None
    HAS_TENSORFLOW = False
from sklearn.preprocessing import MinMaxScaler
from core import config
from core.utils import prepare_time_series_data

class FinancialLSTM:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.sequence_length = config.LSTM_SEQUENCE_LENGTH
        
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        if not HAS_TENSORFLOW:
            return None
        model = keras.Sequential([
            layers.LSTM(config.LSTM_HIDDEN_UNITS, return_sequences=True, 
                       input_shape=input_shape),
            layers.Dropout(config.LSTM_DROPOUT),
            layers.LSTM(config.LSTM_HIDDEN_UNITS // 2, return_sequences=False),
            layers.Dropout(config.LSTM_DROPOUT),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, transactions_df):
        """Prepare transaction data for LSTM training"""
        if transactions_df.empty:
            return None, None
        
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        transactions_df = transactions_df.sort_values('date')
        
        # Aggregate daily expenses
        daily_expenses = transactions_df[
            transactions_df['type'] == 'expense'
        ].groupby('date')['amount'].sum().reset_index()
        
        # Fill missing dates with 0
        date_range = pd.date_range(
            start=daily_expenses['date'].min(),
            end=daily_expenses['date'].max(),
            freq='D'
        )
        
        daily_expenses = daily_expenses.set_index('date').reindex(date_range, fill_value=0)
        daily_expenses = daily_expenses.reset_index()
        daily_expenses.columns = ['date', 'amount']
        
        if len(daily_expenses) < self.sequence_length + 1:
            return None, None
        
        # Normalize data
        expenses_scaled = self.scaler.fit_transform(
            daily_expenses['amount'].values.reshape(-1, 1)
        )
        
        # Create sequences
        X, y = prepare_time_series_data(expenses_scaled, self.sequence_length)
        
        return X, y
    
    def train(self, transactions_df, epochs=None, batch_size=None):
        """Train the LSTM model"""
        if not HAS_TENSORFLOW:
            return False, "TensorFlow is not installed. LSTM training is unavailable."

        if epochs is None:
            epochs = config.EPOCHS
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        X, y = self.prepare_data(transactions_df)
        
        if X is None or len(X) < 10:
            return False, "Insufficient data for training"
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
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
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.is_trained = True
        
        val_loss = history.history['val_loss'][-1]
        return True, f"Model trained successfully. Validation loss: {val_loss:.4f}"
    
    def predict_next_days(self, transactions_df, days=30):
        """Predict expenses for the next N days"""
        if not self.is_trained:
            return None, "Model not trained"
        
        if transactions_df.empty:
            return None, "No transaction data"
        
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        transactions_df = transactions_df.sort_values('date')
        
        # Get recent data
        daily_expenses = transactions_df[
            transactions_df['type'] == 'expense'
        ].groupby('date')['amount'].sum().reset_index()
        
        if len(daily_expenses) < self.sequence_length:
            return None, "Insufficient historical data"
        
        # Prepare last sequence
        recent_expenses = daily_expenses.tail(self.sequence_length)['amount'].values
        recent_scaled = self.scaler.transform(recent_expenses.reshape(-1, 1))
        
        predictions = []
        current_sequence = recent_scaled.copy()
        
        for _ in range(days):
            # Reshape for prediction
            input_seq = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next day
            next_pred = self.model.predict(input_seq, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred
        
        # Denormalize predictions
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        # Create prediction dataframe
        last_date = daily_expenses['date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted_expense': predictions
        })
        
        return predictions_df, "Predictions generated successfully"
    
    def save_model(self, path=config.LSTM_FINANCIAL_PATH):
        """Save trained model"""
        if self.is_trained:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            
            # Save scaler
            scaler_path = path.replace('.h5', '_scaler.pkl')
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def load_model(self, path=config.LSTM_FINANCIAL_PATH):
        """Load trained model"""
        if not HAS_TENSORFLOW:
            return False

        if os.path.exists(path):
            try:
                # Load without compile metadata to avoid cross-version metric deserialization issues.
                self.model = keras.models.load_model(path, compile=False)
            except Exception:
                self.model = None
                self.is_trained = False
                return False

            self.is_trained = True
            
            # Load scaler
            scaler_path = path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            return True
        return False

def calculate_cash_flow_forecast(transactions_df, days=30):
    """Calculate simple cash flow forecast without ML"""
    if transactions_df.empty:
        return None
    
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    
    # Calculate average daily income and expenses
    recent_data = transactions_df[
        transactions_df['date'] >= (pd.Timestamp.now() - pd.Timedelta(days=90))
    ]
    
    avg_daily_income = recent_data[
        recent_data['type'] == 'income'
    ]['amount'].sum() / 90
    
    avg_daily_expense = recent_data[
        recent_data['type'] == 'expense'
    ]['amount'].sum() / 90
    
    # Project forward
    future_dates = pd.date_range(
        start=pd.Timestamp.now(),
        periods=days,
        freq='D'
    )
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'projected_income': avg_daily_income,
        'projected_expense': avg_daily_expense,
        'net_cash_flow': avg_daily_income - avg_daily_expense
    })
    
    # Calculate cumulative
    forecast_df['cumulative_cash_flow'] = forecast_df['net_cash_flow'].cumsum()
    
    return forecast_df
