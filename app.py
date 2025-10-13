"""
Simplified Medical AI Flask App - Guaranteed to Work
"""

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class SimplePredictor:
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = None
        self.model = None
        self.is_trained = False
        self.load_models()

    def load_models(self):
        """Load trained models if they exist"""
        try:
            self.vectorizer = joblib.load('models/vectorizer.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.model = joblib.load('models/best_model.pkl')
            self.is_trained = True
            print("✅ AI models loaded successfully")
        except:
            print("⚠️ No trained models found. Run: python ultra_simple_trainer.py")

    def predict_conditions(self, symptoms_text):
        """Predict medical conditions"""
        if not self.is_trained:
            return {
                'input_symptoms': [symptoms_text],
                'severity': 'Unknown',
                'conditions': [],
                'recommendations': ['Please train the AI first: python ultra_simple_trainer.py'],
                'emergency_warning': {'warning': False},
                'disclaimer': 'AI model needs training',
                'timestamp': datetime.now().isoformat()
            }

        try:
            # Make prediction
            X = self.vectorizer.transform([symptoms_text])
            probabilities = self.model.predict_proba(X)[0]

            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]

            conditions = []
            for idx in top_indices:
                condition = self.label_encoder.inverse_transform([idx])[0]
                confidence = probabilities[idx]

                if confidence > 0.1:
                    conditions.append({
                        'name': condition,
                        'confidence': confidence,
                        'confidence_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low',
                        'severity': 'Medium',
                        'description': f'AI suggests possible {condition}',
                        'recommendations': ['Rest', 'Stay hydrated', 'Monitor symptoms', 'Consult doctor if worsens'],
                        'see_doctor': 'If symptoms persist or worsen'
                    })

            # Assess severity
            danger_words = ['severe', 'extreme', 'chest pain', 'difficulty breathing', 'can\'t breathe']
            severity = 'High' if any(word in symptoms_text.lower() for word in danger_words) else 'Medium'

            # Emergency check
            emergency_keywords = ['chest pain', 'difficulty breathing', 'severe headache', 'can\'t breathe']
            emergency_warning = {
                'warning': any(keyword in symptoms_text.lower() for keyword in emergency_keywords)
            }
            if emergency_warning['warning']:
                emergency_warning['message'] = '⚠️ EMERGENCY: Seek immediate medical attention!'
                emergency_warning['action'] = 'Call 911 or go to emergency room immediately'

            return {
                'input_symptoms': symptoms_text.split(),
                'severity': severity,
                'conditions': conditions,
                'recommendations': [
                    'Consult healthcare provider for proper diagnosis',
                    'Monitor symptoms closely', 
                    'Rest and stay hydrated',
                    'Seek immediate help if symptoms worsen'
                ],
                'emergency_warning': emergency_warning,
                'disclaimer': 'This AI analysis is for educational purposes only. Always consult healthcare professionals.',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'input_symptoms': [symptoms_text],
                'severity': 'Unknown',
                'conditions': [],
                'recommendations': [f'Error in AI prediction: {e}'],
                'emergency_warning': {'warning': False},
                'disclaimer': 'Prediction error occurred',
                'timestamp': datetime.now().isoformat()
            }

# Initialize predictor
predictor = SimplePredictor()

# Database model
class DiagnosisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_input = db.Column(db.Text, nullable=False)
    predicted_conditions = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if request.method == 'POST':
        symptoms_text = request.form.get('symptoms', '')
        if symptoms_text:
            results = predictor.predict_conditions(symptoms_text)

            # Save to database
            try:
                diagnosis = DiagnosisHistory(
                    user_input=symptoms_text,
                    predicted_conditions=json.dumps(results['conditions'])
                )
                db.session.add(diagnosis)
                db.session.commit()
            except:
                pass

            return render_template('results.html', symptoms=symptoms_text, results=results)
        else:
            return render_template('diagnosis.html', error='Please describe your symptoms')

    return render_template('diagnosis.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/status')
def api_status():
    return jsonify({
        'model_trained': predictor.is_trained,
        'status': 'Ready' if predictor.is_trained else 'Need to train model'
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✅ Database initialized")

    if predictor.is_trained:
        print("🤖 AI Model ready with trained medical knowledge!")
    else:
        print("⚠️ AI Model not trained. Run: python ultra_simple_trainer.py")

    print("🏥 Medical AI Assistant starting...")
    print("📱 Open http://127.0.0.1:5000 in your browser")

    app.run(debug=True, host='0.0.0.0', port=5000)
