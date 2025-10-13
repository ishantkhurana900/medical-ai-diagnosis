"""
Main prediction engine that combines text preprocessing and ML classification
Provides the primary interface for symptom analysis and condition prediction
"""

import json
import pandas as pd
from datetime import datetime
from .preprocessor import TextPreprocessor
from .classifier import SymptomClassifier
import os

class SymptomPredictor:
    def __init__(self):
        """Initialize the complete prediction system"""
        self.preprocessor = TextPreprocessor()
        self.classifier = SymptomClassifier()
        self.conditions_db = self.load_conditions_database()
        self.model_trained = False

    def load_conditions_database(self):
        """Load medical conditions database"""
        try:
            df = pd.read_csv('data/conditions.csv')
            conditions = {}

            for _, row in df.iterrows():
                try:
                    recommendations = eval(row['recommendations']) if isinstance(row['recommendations'], str) else row['recommendations']
                except:
                    recommendations = ['Rest', 'Stay hydrated', 'Monitor symptoms']

                conditions[row['name']] = {
                    'id': row['id'],
                    'name': row['name'],
                    'severity': row['severity'],
                    'description': row['description'],
                    'recommendations': recommendations,
                    'see_doctor': row['see_doctor']
                }

            return conditions
        except Exception as e:
            print(f"Error loading conditions database: {e}")
            return {}

    def train_model(self):
        """Train the ML model with available data"""
        print("Training medical AI model...")

        try:
            success = self.classifier.train('data/training_data.csv')
            if success:
                self.model_trained = True
                print("Model training completed successfully!")
            else:
                print("Model training failed - using rule-based fallback")
                self.model_trained = False

        except Exception as e:
            print(f"Error during training: {e}")
            print("Using rule-based prediction system")
            self.model_trained = False

        return self.model_trained

    def predict_conditions(self, symptoms_text):
        """Main prediction method - analyzes symptoms and returns possible conditions"""

        # Preprocess the input
        features = self.preprocessor.preprocess_for_ml(symptoms_text)

        # Extract symptoms
        extracted_symptoms = features['symptoms']
        severity = features['severity']

        # Get predictions
        if self.model_trained:
            # Use trained ML model
            predictions = self.classifier.get_top_conditions(symptoms_text, top_k=3)
        else:
            # Use rule-based fallback system
            predictions = self.rule_based_prediction(extracted_symptoms)

        # Enhance predictions with detailed information
        detailed_conditions = []
        for pred in predictions:
            condition_name = pred['condition']
            if condition_name in self.conditions_db:
                condition_info = self.conditions_db[condition_name].copy()
                condition_info['confidence'] = pred['confidence']
                condition_info['confidence_level'] = self.get_confidence_level(pred['confidence'])
                detailed_conditions.append(condition_info)

        # Assess overall severity
        overall_severity = self.assess_overall_severity(extracted_symptoms, severity, detailed_conditions)

        # Generate recommendations
        recommendations = self.generate_recommendations(detailed_conditions, overall_severity)

        # Check for emergency symptoms
        emergency_warning = self.check_emergency_symptoms(symptoms_text)

        return {
            'input_symptoms': extracted_symptoms,
            'severity': overall_severity,
            'conditions': detailed_conditions,
            'recommendations': recommendations,
            'emergency_warning': emergency_warning,
            'disclaimer': 'This analysis is for educational purposes only. Please consult a healthcare professional for proper medical advice.',
            'timestamp': datetime.now().isoformat()
        }

    def rule_based_prediction(self, symptoms):
        """Fallback rule-based prediction when ML model is not available"""

        # Simple rule-based matching
        condition_matches = {}

        # Define symptom-condition mappings
        symptom_conditions = {
            'headache': [('Common Cold', 0.6), ('Migraine Headache', 0.8), ('Tension Headache', 0.9)],
            'fever': [('Common Cold', 0.7), ('Influenza (Flu)', 0.9), ('Food Poisoning', 0.6)],
            'cough': [('Common Cold', 0.9), ('Influenza (Flu)', 0.75), ('Allergic Reaction', 0.6)],
            'sore throat': [('Common Cold', 0.8), ('Influenza (Flu)', 0.6)],
            'runny nose': [('Common Cold', 0.95), ('Allergic Reaction', 0.7)],
            'fatigue': [('Common Cold', 0.7), ('Influenza (Flu)', 0.85), ('Anxiety Disorder', 0.7)],
            'nausea': [('Migraine Headache', 0.7), ('Gastritis', 0.8), ('Food Poisoning', 0.9)],
            'vomiting': [('Food Poisoning', 0.85), ('Gastritis', 0.7)],
            'diarrhea': [('Food Poisoning', 0.9), ('Gastritis', 0.6)],
            'stomach pain': [('Gastritis', 0.9), ('Food Poisoning', 0.8)],
            'back pain': [('Tension Headache', 0.5)],
            'chest pain': [('Anxiety Disorder', 0.6)],
            'shortness of breath': [('Anxiety Disorder', 0.7)],
            'dizziness': [('Migraine Headache', 0.6), ('Anxiety Disorder', 0.6)],
            'muscle aches': [('Influenza (Flu)', 0.8), ('Tension Headache', 0.6)],
            'joint pain': [('Influenza (Flu)', 0.7)],
            'skin rash': [('Allergic Reaction', 0.85)],
            'loss of appetite': [('Gastritis', 0.7), ('Influenza (Flu)', 0.6)],
            'difficulty sleeping': [('Anxiety Disorder', 0.8)],
            'anxiety': [('Anxiety Disorder', 0.95)]
        }

        # Calculate condition scores
        for symptom in symptoms:
            if symptom in symptom_conditions:
                for condition, weight in symptom_conditions[symptom]:
                    if condition not in condition_matches:
                        condition_matches[condition] = 0
                    condition_matches[condition] += weight

        # Normalize scores and get top conditions
        if condition_matches:
            max_score = max(condition_matches.values())
            normalized_matches = [
                {
                    'condition': condition,
                    'confidence': min(score / max_score * 0.8, 0.9)  # Cap confidence for rule-based
                }
                for condition, score in condition_matches.items()
            ]

            # Sort by confidence and return top 3
            normalized_matches.sort(key=lambda x: x['confidence'], reverse=True)
            return normalized_matches[:3]

        return [{'condition': 'General Consultation Needed', 'confidence': 0.5}]

    def get_confidence_level(self, confidence):
        """Convert numeric confidence to descriptive level"""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"

    def assess_overall_severity(self, symptoms, text_severity, conditions):
        """Assess overall severity based on symptoms and conditions"""

        # Check for high-severity symptoms
        high_severity_symptoms = ['chest pain', 'shortness of breath', 'severe headache']
        has_high_severity = any(symptom in high_severity_symptoms for symptom in symptoms)

        # Check condition severities
        condition_severities = [c.get('severity', 'Low') for c in conditions]
        has_high_severity_condition = 'High' in condition_severities or 'Medium' in condition_severities

        if has_high_severity or text_severity == 'High':
            return 'High'
        elif has_high_severity_condition or text_severity == 'Medium':
            return 'Medium'
        else:
            return 'Low'

    def generate_recommendations(self, conditions, severity):
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # General recommendations based on severity
        if severity == 'High':
            recommendations.append("⚠️ Seek immediate medical attention")
            recommendations.append("💊 Consider visiting emergency room or urgent care")
        elif severity == 'Medium':
            recommendations.append("💊 Consider consulting a healthcare professional")
            recommendations.append("📞 Monitor symptoms and seek help if they worsen")
        else:
            recommendations.append("🏠 Rest and home care may be sufficient")
            recommendations.append("📋 Monitor symptoms and consult doctor if concerned")

        # Add specific recommendations from conditions
        specific_recommendations = set()
        for condition in conditions:
            if 'recommendations' in condition and condition['recommendations']:
                for rec in condition['recommendations'][:2]:  # Limit to 2 per condition
                    specific_recommendations.add(f"• {rec}")

        recommendations.extend(list(specific_recommendations)[:3])  # Limit to 3 specific recommendations

        return recommendations

    def check_emergency_symptoms(self, text):
        """Check for emergency symptoms that require immediate attention"""
        emergency_keywords = [
            'chest pain', 'difficulty breathing', 'hard to breathe', 'cant breathe',
            'severe bleeding', 'loss of consciousness', 'unconscious',
            'severe headache', 'worst headache', 'suicidal', 'want to die'
        ]

        text_lower = text.lower()
        for keyword in emergency_keywords:
            if keyword in text_lower:
                return {
                    'warning': True,
                    'message': f"⚠️ EMERGENCY: You mentioned '{keyword}' - seek immediate medical attention!",
                    'action': "Call emergency services (911) or go to the nearest emergency room immediately"
                }

        return {'warning': False}

    def get_system_status(self):
        """Get current system status"""
        return {
            'model_trained': self.model_trained,
            'conditions_loaded': len(self.conditions_db) > 0,
            'preprocessor_ready': self.preprocessor is not None,
            'total_conditions': len(self.conditions_db)
        }
