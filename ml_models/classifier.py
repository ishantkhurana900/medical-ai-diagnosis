"""
Machine Learning classifier for medical symptom analysis
Uses scikit-learn for training and prediction
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from collections import defaultdict
import os

class SymptomClassifier:
    def __init__(self):
        """Initialize the symptom classifier"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )

        # Use ensemble of classifiers
        self.classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def load_training_data(self, csv_path='data/training_data.csv'):
        """Load and prepare training data"""
        try:
            df = pd.read_csv(csv_path)
            return df
        except FileNotFoundError:
            print(f"Training data file {csv_path} not found")
            return None

    def prepare_features(self, df):
        """Prepare features for training"""
        # Extract text features using TF-IDF
        X_text = self.vectorizer.fit_transform(df['text'])

        # Prepare labels
        y = self.label_encoder.fit_transform(df['condition'])

        return X_text, y

    def train(self, training_data_path='data/training_data.csv'):
        """Train the classifier on medical data"""
        print("Loading training data...")
        df = self.load_training_data(training_data_path)

        if df is None:
            print("Cannot train without data")
            return False

        print(f"Training with {len(df)} examples")

        # Prepare features
        X, y = self.prepare_features(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train each classifier
        for name, classifier in self.classifiers.items():
            print(f"Training {name}...")
            classifier.fit(X_train, y_train)

            # Evaluate
            train_score = classifier.score(X_train, y_train)
            test_score = classifier.score(X_test, y_test)

            print(f"{name} - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")

        self.is_trained = True
        print("Training completed!")

        # Save models
        self.save_models()

        return True

    def predict(self, text):
        """Predict condition from symptom text"""
        if not self.is_trained:
            print("Model not trained yet")
            return None

        # Vectorize input text
        X = self.vectorizer.transform([text])

        # Get predictions from all classifiers
        predictions = {}
        probabilities = {}

        for name, classifier in self.classifiers.items():
            pred = classifier.predict(X)[0]
            pred_proba = classifier.predict_proba(X)[0]

            # Convert back to condition name
            condition = self.label_encoder.inverse_transform([pred])[0]

            predictions[name] = condition
            probabilities[name] = pred_proba.max()

        # Ensemble prediction (majority vote with confidence weighting)
        weighted_predictions = defaultdict(float)
        for name, condition in predictions.items():
            weighted_predictions[condition] += probabilities[name]

        # Get top prediction
        best_condition = max(weighted_predictions, key=weighted_predictions.get)
        confidence = weighted_predictions[best_condition] / len(self.classifiers)

        return {
            'condition': best_condition,
            'confidence': confidence,
            'all_predictions': predictions,
            'probabilities': probabilities
        }

    def get_top_conditions(self, text, top_k=3):
        """Get top K condition predictions"""
        if not self.is_trained:
            return []

        X = self.vectorizer.transform([text])

        # Get probabilities from the best performing classifier (random forest)
        proba = self.classifiers['random_forest'].predict_proba(X)[0]

        # Get top K predictions
        top_indices = np.argsort(proba)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            condition = self.label_encoder.inverse_transform([idx])[0]
            confidence = proba[idx]

            if confidence > 0.1:  # Only include predictions with reasonable confidence
                results.append({
                    'condition': condition,
                    'confidence': confidence
                })

        return results

    def save_models(self, base_path='models'):
        """Save trained models and vectorizer"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)

            joblib.dump(self.vectorizer, f'{base_path}_vectorizer.pkl')
            joblib.dump(self.label_encoder, f'{base_path}_label_encoder.pkl')

            for name, classifier in self.classifiers.items():
                joblib.dump(classifier, f'{base_path}_{name}.pkl')

            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self, base_path='models'):
        """Load pre-trained models"""
        try:
            self.vectorizer = joblib.load(f'{base_path}_vectorizer.pkl')
            self.label_encoder = joblib.load(f'{base_path}_label_encoder.pkl')

            for name in self.classifiers.keys():
                self.classifiers[name] = joblib.load(f'{base_path}_{name}.pkl')

            self.is_trained = True
            print("Models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
