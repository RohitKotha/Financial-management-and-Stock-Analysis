"""
Expense Categorization Module using Machine Learning and NLP
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
import os
from core import config
from core.utils import preprocess_text, categorize_transaction_keywords

class ExpenseCategorizer:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.is_trained = False
        
    def create_model(self):
        """Create ML pipeline for expense categorization"""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
    def train(self, descriptions, categories):
        """Train the categorization model"""
        if len(descriptions) < 10:
            # Not enough data to train, use keyword-based approach
            return False
        
        # Preprocess descriptions
        descriptions = [preprocess_text(desc) for desc in descriptions]
        
        # Create and train model
        self.create_model()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            descriptions, categories, test_size=0.2, random_state=42
        )
        
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate accuracy
        accuracy = self.pipeline.score(X_test, y_test)
        return accuracy
    
    def predict(self, description):
        """Predict category for a transaction description"""
        if not self.is_trained:
            # Fall back to keyword-based categorization
            return categorize_transaction_keywords(description)
        
        description = preprocess_text(description)
        prediction = self.pipeline.predict([description])[0]
        return prediction
    
    def predict_batch(self, descriptions):
        """Predict categories for multiple descriptions"""
        if not self.is_trained:
            return [categorize_transaction_keywords(desc) for desc in descriptions]
        
        descriptions = [preprocess_text(desc) for desc in descriptions]
        predictions = self.pipeline.predict(descriptions)
        return predictions
    
    def save_model(self, path=config.EXPENSE_CATEGORIZER_PATH):
        """Save trained model to disk"""
        if self.is_trained:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self.pipeline, f)
    
    def load_model(self, path=config.EXPENSE_CATEGORIZER_PATH):
        """Load trained model from disk"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.pipeline = pickle.load(f)
                self.is_trained = True
            return True
        return False

def auto_categorize_transactions(transactions_df, categorizer):
    """Automatically categorize transactions"""
    if transactions_df.empty:
        return transactions_df
    
    # Get transactions without categories or with 'Other' category
    uncategorized = transactions_df[
        (transactions_df['category'].isna()) | (transactions_df['category'] == 'Other')
    ]
    
    if not uncategorized.empty:
        descriptions = uncategorized['description'].tolist()
        predicted_categories = categorizer.predict_batch(descriptions)
        
        transactions_df.loc[uncategorized.index, 'category'] = predicted_categories
    
    return transactions_df
