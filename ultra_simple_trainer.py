"""
Ultra Simple Medical AI Trainer - Minimal Dependencies
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os

class UltraSimpleMedicalTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=150, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.label_encoder = LabelEncoder()
        os.makedirs('models', exist_ok=True)

    def create_medical_training_data(self):
        """Create comprehensive medical training data"""

        # Load existing data if available
        training_examples = []
        try:
            existing_df = pd.read_csv('data/training_data.csv')
            for _, row in existing_df.iterrows():
                training_examples.append({
                    'text': row['text'],
                    'condition': row['condition'],
                    'probability': row['probability']
                })
            print(f"✅ Loaded {len(existing_df)} existing examples")
        except:
            print("ℹ️  Creating training data from medical knowledge")

        # Comprehensive medical knowledge base
        medical_conditions = {
            'Common Cold': [
                'runny nose', 'nasal congestion', 'sneezing', 'sore throat', 
                'cough', 'mild headache', 'fatigue', 'low grade fever'
            ],
            'Influenza (Flu)': [
                'high fever', 'muscle aches', 'severe fatigue', 'headache', 
                'dry cough', 'chills', 'body aches', 'weakness'
            ],
            'Migraine Headache': [
                'severe headache', 'throbbing pain', 'nausea', 'vomiting', 
                'light sensitivity', 'sound sensitivity', 'visual disturbances'
            ],
            'Tension Headache': [
                'mild headache', 'pressure sensation', 'band-like pain', 
                'muscle tension', 'stress headache'
            ],
            'Gastritis': [
                'stomach pain', 'abdominal pain', 'nausea', 'bloating', 
                'heartburn', 'loss of appetite', 'feeling full quickly'
            ],
            'Food Poisoning': [
                'nausea', 'vomiting', 'diarrhea', 'stomach cramps', 
                'abdominal pain', 'fever', 'dehydration'
            ],
            'Anxiety Disorder': [
                'excessive worry', 'nervousness', 'restlessness', 'fatigue', 
                'difficulty concentrating', 'muscle tension', 'sleep problems'
            ],
            'Allergic Reaction': [
                'skin rash', 'itching', 'hives', 'swelling', 'runny nose', 
                'sneezing', 'watery eyes'
            ],
            'Asthma': [
                'wheezing', 'shortness of breath', 'chest tightness', 
                'persistent cough', 'difficulty breathing'
            ],
            'Hypertension': [
                'headache', 'dizziness', 'blurred vision', 'chest pain', 
                'shortness of breath', 'nausea'
            ],
            'Diabetes': [
                'excessive thirst', 'frequent urination', 'fatigue', 
                'blurred vision', 'slow healing wounds', 'increased hunger'
            ],
            'Pneumonia': [
                'chest pain', 'cough with phlegm', 'fever', 'chills', 
                'shortness of breath', 'fatigue', 'difficulty breathing'
            ],
            'Bronchitis': [
                'persistent cough', 'mucus production', 'fatigue', 
                'shortness of breath', 'chest discomfort', 'low fever'
            ],
            'Arthritis': [
                'joint pain', 'stiffness', 'swelling', 'reduced range of motion', 
                'morning stiffness', 'joint tenderness'
            ],
            'Back Pain': [
                'lower back pain', 'muscle spasms', 'stiffness', 
                'limited mobility', 'radiating pain', 'muscle tension'
            ]
        }

        # Generate training examples
        templates = [
            "I have {symptom}",
            "experiencing {symptom}",
            "feeling {symptom}",
            "suffering from {symptom}",
            "been having {symptom}",
            "symptoms include {symptom}"
        ]

        for condition, symptoms in medical_conditions.items():
            for symptom in symptoms:
                for template in templates:
                    text = template.format(symptom=symptom)
                    training_examples.append({
                        'text': text,
                        'condition': condition,
                        'probability': 0.85
                    })

            # Create combination examples
            if len(symptoms) >= 2:
                for i in range(min(15, len(symptoms))):
                    selected = np.random.choice(symptoms, np.random.randint(2, min(4, len(symptoms)+1)), replace=False)
                    combined_text = "I have " + " and ".join(selected)
                    training_examples.append({
                        'text': combined_text,
                        'condition': condition,
                        'probability': 0.9
                    })

        print(f"✅ Generated {len(training_examples)} training examples")
        return training_examples

    def train_models(self):
        """Train the medical AI models"""

        print("🔄 Creating medical training data...")
        training_data = self.create_medical_training_data()

        if len(training_data) < 100:
            print("❌ Not enough training data")
            return False

        # Prepare data
        df = pd.DataFrame(training_data)
        df = df.drop_duplicates(subset=['text', 'condition'])

        print(f"🧠 Training on {len(df)} examples...")

        # Vectorize text
        X = self.vectorizer.fit_transform(df['text'])
        y = self.label_encoder.fit_transform(df['condition'])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train models
        best_score = 0
        best_model = None

        for name, model in self.models.items():
            print(f"⚡ Training {name}...")
            model.fit(X_train, y_train)

            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            if test_score > best_score:
                best_score = test_score
                best_model = model

            print(f"  ✅ {name}: Train={train_score:.1%}, Test={test_score:.1%}")

        # Save models
        print("💾 Saving models...")
        joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(best_model, 'models/best_model.pkl')

        print(f"🎉 Training complete! Best accuracy: {best_score:.1%}")
        return True

    def test_model(self):
        """Test the trained model"""

        try:
            # Load models
            vectorizer = joblib.load('models/vectorizer.pkl')
            label_encoder = joblib.load('models/label_encoder.pkl')
            model = joblib.load('models/best_model.pkl')

            # Test cases
            test_cases = [
                "I have severe headache with nausea",
                "experiencing runny nose and sore throat",
                "stomach pain with vomiting",
                "feeling anxious and restless",
                "chest pain and shortness of breath"
            ]

            print("\n🧪 Testing trained model:")
            for test in test_cases:
                X = vectorizer.transform([test])
                proba = model.predict_proba(X)[0]
                top_idx = np.argmax(proba)
                condition = label_encoder.inverse_transform([top_idx])[0]
                confidence = proba[top_idx]

                print(f"\n'{test}':")
                print(f"  → {condition}: {confidence:.1%} confidence")

            return True

        except Exception as e:
            print(f"❌ Error testing model: {e}")
            return False

if __name__ == "__main__":
    print("🏥 Ultra Simple Medical AI Trainer")
    print("=" * 50)

    trainer = UltraSimpleMedicalTrainer()

    # Train models
    success = trainer.train_models()

    if success:
        # Test models
        trainer.test_model()
        print("\n✅ Medical AI is ready!")
        print("\n🚀 Next steps:")
        print("1. Run: python app.py")
        print("2. Open: http://127.0.0.1:5000")
        print("3. Test your medical AI!")
    else:
        print("❌ Training failed")
