from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import re
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class MentalHealthPredictor:
    def __init__(self, model_prefix="mental_health_model"):
        try:
            # Load the trained model components
            self.vectorizer = joblib.load(f"{model_prefix}_vectorizer.pkl")
            self.classifier = joblib.load(f"{model_prefix}_classifier.pkl")
            
            # Load mappings
            with open(f"{model_prefix}_mappings.json", 'r') as f:
                mappings = json.load(f)
                self.emotion_mapping = mappings['emotion_mapping']
                self.stress_mapping = mappings['stress_mapping']
            
            print("Model loaded successfully!")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure to run train_model.py first to generate the model files.")
            self.model_loaded = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def predict_stress_level(self, text):
        """Predict stress level for a given text"""
        if not self.model_loaded:
            return {
                'error': 'Model not loaded. Please run train_model.py first.',
                'stress_level': 'unknown',
                'confidence': 0.0
            }
        
        try:
            # Preprocess the text
            clean_text = self.preprocess_text(text)
            
            if not clean_text:
                return {
                    'stress_level': 'low',
                    'confidence': 0.5,
                    'message': 'Empty or invalid text provided'
                }
            
            # Vectorize the text
            text_vectorized = self.vectorizer.transform([clean_text])
            
            # Predict
            prediction = self.classifier.predict(text_vectorized)[0]
            probabilities = self.classifier.predict_proba(text_vectorized)[0]
            
            # Get confidence score
            confidence = max(probabilities)
            
            # Get all class probabilities
            class_probabilities = dict(zip(self.classifier.classes_, probabilities))
            
            return {
                'stress_level': prediction,
                'confidence': float(confidence),
                'probabilities': class_probabilities,
                'processed_text': clean_text
            }
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'stress_level': 'unknown',
                'confidence': 0.0
            }

# Initialize the predictor
predictor = MentalHealthPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict_stress():
    """Predict stress level for given text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please send JSON with "text" field.'
            }), 400
        
        text = data['text']
        
        # Get prediction
        result = predictor.predict_stress_level(text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_stress_batch():
    """Predict stress levels for multiple texts"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'No texts provided. Please send JSON with "texts" array.'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'Texts must be an array.'
            }), 400
        
        # Get predictions for all texts
        results = []
        for text in texts:
            result = predictor.predict_stress_level(text)
            results.append({
                'text': text,
                'prediction': result
            })
        
        return jsonify({
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if not predictor.model_loaded:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'model_loaded': True,
        'vectorizer_features': predictor.vectorizer.get_feature_names_out().shape[0],
        'classifier_classes': predictor.classifier.classes_.tolist(),
        'emotion_mapping': predictor.emotion_mapping,
        'stress_mapping': predictor.stress_mapping
    })

if __name__ == '__main__':
    print("=== Mental Health Chatbot API Server ===")
    print("Starting server...")
    print("Available endpoints:")
    print("- GET  /health - Health check")
    print("- POST /predict - Predict stress level for single text")
    print("- POST /predict_batch - Predict stress levels for multiple texts")
    print("- GET  /model_info - Get model information")
    print()
    print("Example usage:")
    print('curl -X POST http://localhost:5000/predict \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"text": "I am feeling very anxious today"}\'')
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000) 