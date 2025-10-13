"""
Text preprocessing module for medical symptom analysis
Handles cleaning, tokenization, and feature extraction from user input
"""

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import json
import os

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with medical-specific configurations"""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Medical-specific stop words to remove
        self.medical_stop_words = {
            'i', 'have', 'am', 'feel', 'feeling', 'experiencing', 'got', 'get',
            'really', 'very', 'quite', 'pretty', 'somewhat', 'little', 'bit'
        }

        # Load medical knowledge base
        self.knowledge_base = {
            'severity_keywords': {
                'high': ['severe', 'extreme', 'unbearable', 'worst', 'excruciating', 'intense'],
                'medium': ['moderate', 'significant', 'noticeable', 'uncomfortable', 'bad'],
                'low': ['mild', 'slight', 'minor', 'little bit', 'somewhat', 'light']
            },
            'emergency_symptoms': [
                'chest pain', 'difficulty breathing', 'hard to breathe', 'cant breathe',
                'severe bleeding', 'loss of consciousness', 'unconscious',
                'severe headache', 'worst headache', 'suicidal', 'want to die'
            ]
        }

        # Symptom synonyms for better matching
        self.symptom_synonyms = {
            'headache': ['head pain', 'head ache', 'pain in head', 'head hurts', 'migraine'],
            'fever': ['high temperature', 'hot', 'feverish', 'burning up', 'temperature'],
            'cough': ['coughing', 'dry cough', 'wet cough', 'hacking cough'],
            'sore throat': ['throat pain', 'throat hurts', 'scratchy throat'],
            'runny nose': ['nasal congestion', 'stuffy nose', 'nose running', 'congestion'],
            'fatigue': ['tired', 'exhausted', 'weak', 'no energy', 'weakness'],
            'nausea': ['nauseous', 'sick to stomach', 'queasy'],
            'vomiting': ['throwing up', 'puking', 'vomit'],
            'diarrhea': ['loose stools', 'watery stool', 'frequent bowel movements'],
            'stomach pain': ['abdominal pain', 'belly ache', 'stomach hurts', 'stomach ache'],
            'back pain': ['backache', 'back hurts', 'lower back pain'],
            'chest pain': ['chest hurts', 'tight chest', 'chest pressure'],
            'shortness of breath': ['hard to breathe', 'breathing difficulty', 'out of breath'],
            'dizziness': ['dizzy', 'lightheaded', 'feeling faint'],
            'muscle aches': ['muscle pain', 'sore muscles', 'muscle soreness'],
            'joint pain': ['joints hurt', 'achy joints', 'joint aches'],
            'skin rash': ['rash', 'skin irritation', 'red spots'],
            'loss of appetite': ['not hungry', 'no appetite', 'dont want to eat'],
            'difficulty sleeping': ['cant sleep', 'insomnia', 'trouble sleeping'],
            'anxiety': ['anxious', 'worried', 'nervous', 'panic']
        }

    def clean_text(self, text):
        """Clean and normalize input text"""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove punctuation except for medical-relevant ones
        text = re.sub(r'[^\w\s\-\']', '', text)

        return text.strip()

    def tokenize_and_lemmatize(self, text):
        """Tokenize text and lemmatize words"""
        tokens = word_tokenize(text)

        # Filter out stop words and lemmatize
        filtered_tokens = []
        for token in tokens:
            if (token not in self.stop_words and 
                token not in self.medical_stop_words and 
                len(token) > 1):
                lemmatized = self.lemmatizer.lemmatize(token)
                filtered_tokens.append(lemmatized)

        return filtered_tokens

    def extract_symptoms(self, text):
        """Extract symptoms from text using keyword matching"""
        text = self.clean_text(text)
        extracted_symptoms = []

        # Check for direct symptom matches
        for symptom, synonyms in self.symptom_synonyms.items():
            # Check main symptom
            if symptom in text:
                extracted_symptoms.append(symptom)
                continue

            # Check synonyms
            for synonym in synonyms:
                if synonym in text:
                    extracted_symptoms.append(symptom)
                    break

        return list(set(extracted_symptoms))  # Remove duplicates

    def assess_severity(self, text):
        """Assess severity level from text using keywords"""
        text = self.clean_text(text)

        severity_scores = {'high': 0, 'medium': 0, 'low': 0}

        for level, keywords in self.knowledge_base['severity_keywords'].items():
            for keyword in keywords:
                if keyword in text:
                    severity_scores[level] += 1

        # Determine overall severity
        if severity_scores['high'] > 0:
            return 'High'
        elif severity_scores['medium'] > 0:
            return 'Medium'
        elif severity_scores['low'] > 0:
            return 'Low'
        else:
            return 'Medium'  # Default to medium if no severity indicators found

    def preprocess_for_ml(self, text):
        """Preprocess text for machine learning model input"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)

        # Create features
        features = {
            'text': ' '.join(tokens),
            'symptoms': self.extract_symptoms(text),
            'severity': self.assess_severity(text),
            'word_count': len(tokens),
            'char_count': len(cleaned)
        }

        return features
