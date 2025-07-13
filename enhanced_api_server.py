from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import re
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class EnhancedMentalHealthPredictor:
    def __init__(self, model_prefix="enhanced_mental_health_model", fallback_prefix="mental_health_model"):
        self.model_loaded = False
        self.enhanced_model_loaded = False
        self.fallback_model_loaded = False
        
        # Try to load enhanced model first
        try:
            self.load_enhanced_model(model_prefix)
        except Exception as e:
            print(f"Enhanced model not available: {e}")
        
        # Try to load fallback model
        try:
            self.load_fallback_model(fallback_prefix)
        except Exception as e:
            print(f"Fallback model not available: {e}")
        
        if not self.model_loaded:
            print("❌ No models loaded. Please run training first.")
    
    def load_enhanced_model(self, model_prefix):
        """Load the enhanced trained model"""
        try:
            # Load the trained model components
            self.enhanced_vectorizer = joblib.load(f"{model_prefix}_vectorizer.pkl")
            self.enhanced_classifier = joblib.load(f"{model_prefix}_classifier.pkl")
            
            # Load mappings
            with open(f"{model_prefix}_mappings.json", 'r') as f:
                mappings = json.load(f)
                self.enhanced_emotion_mapping = mappings['emotion_mapping']
                self.enhanced_stress_mapping = mappings['stress_mapping']
                self.local_dataset_path = mappings.get('local_dataset_path', 'Not specified')
            
            print("✅ Enhanced model loaded successfully!")
            print(f"Local dataset path: {self.local_dataset_path}")
            self.enhanced_model_loaded = True
            self.model_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading enhanced model: {e}")
            self.enhanced_model_loaded = False
    
    def load_fallback_model(self, model_prefix):
        """Load the fallback trained model"""
        try:
            # Load the trained model components
            self.fallback_vectorizer = joblib.load(f"{model_prefix}_vectorizer.pkl")
            self.fallback_classifier = joblib.load(f"{model_prefix}_classifier.pkl")
            
            # Load mappings
            with open(f"{model_prefix}_mappings.json", 'r') as f:
                mappings = json.load(f)
                self.fallback_emotion_mapping = mappings['emotion_mapping']
                self.fallback_stress_mapping = mappings['stress_mapping']
            
            print("✅ Fallback model loaded successfully!")
            self.fallback_model_loaded = True
            self.model_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading fallback model: {e}")
            self.fallback_model_loaded = False
    
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
    
    def predict_stress_level(self, text, use_enhanced=True):
        """Predict stress level for a given text"""
        if not self.model_loaded:
            return {
                'error': 'No models loaded. Please run training first.',
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
            
            # Try enhanced model first if requested and available
            if use_enhanced and self.enhanced_model_loaded:
                return self._predict_with_model(
                    clean_text, 
                    self.enhanced_vectorizer, 
                    self.enhanced_classifier, 
                    'enhanced'
                )
            elif self.fallback_model_loaded:
                return self._predict_with_model(
                    clean_text, 
                    self.fallback_vectorizer, 
                    self.fallback_classifier, 
                    'fallback'
                )
            else:
                return {
                    'error': 'No models available for prediction',
                    'stress_level': 'unknown',
                    'confidence': 0.0
                }
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'stress_level': 'unknown',
                'confidence': 0.0
            }
    
    def _predict_with_model(self, clean_text, vectorizer, classifier, model_type):
        """Make prediction with a specific model"""
        # Vectorize the text
        text_vectorized = vectorizer.transform([clean_text])
        
        # Predict
        prediction = classifier.predict(text_vectorized)[0]
        probabilities = classifier.predict_proba(text_vectorized)[0]
        
        # Get confidence score
        confidence = max(probabilities)
        
        # Get all class probabilities
        class_probabilities = dict(zip(classifier.classes_, probabilities))
        
        return {
            'stress_level': prediction,
            'confidence': float(confidence),
            'probabilities': class_probabilities,
            'processed_text': clean_text,
            'model_type': model_type
        }

# Initialize the predictor
predictor = EnhancedMentalHealthPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model_loaded,
        'enhanced_model_loaded': predictor.enhanced_model_loaded,
        'fallback_model_loaded': predictor.fallback_model_loaded
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
        use_enhanced = data.get('use_enhanced', True)  # Default to enhanced model
        
        # Get prediction
        result = predictor.predict_stress_level(text, use_enhanced)
        
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
        use_enhanced = data.get('use_enhanced', True)
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'Texts must be an array.'
            }), 400
        
        # Get predictions for all texts
        results = []
        for text in texts:
            result = predictor.predict_stress_level(text, use_enhanced)
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
    """Get information about the loaded models"""
    if not predictor.model_loaded:
        return jsonify({
            'error': 'No models loaded'
        }), 500
    
    info = {
        'enhanced_model_loaded': predictor.enhanced_model_loaded,
        'fallback_model_loaded': predictor.fallback_model_loaded
    }
    
    if predictor.enhanced_model_loaded:
        info['enhanced_model'] = {
            'vectorizer_features': predictor.enhanced_vectorizer.get_feature_names_out().shape[0],
            'classifier_classes': predictor.enhanced_classifier.classes_.tolist(),
            'emotion_mapping': predictor.enhanced_emotion_mapping,
            'stress_mapping': predictor.enhanced_stress_mapping,
            'local_dataset_path': getattr(predictor, 'local_dataset_path', 'Not specified')
        }
    
    if predictor.fallback_model_loaded:
        info['fallback_model'] = {
            'vectorizer_features': predictor.fallback_vectorizer.get_feature_names_out().shape[0],
            'classifier_classes': predictor.fallback_classifier.classes_.tolist(),
            'emotion_mapping': predictor.fallback_emotion_mapping,
            'stress_mapping': predictor.fallback_stress_mapping
        }
    
    return jsonify(info)

@app.route('/compare_models', methods=['POST'])
def compare_models():
    """Compare predictions from both models"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please send JSON with "text" field.'
            }), 400
        
        text = data['text']
        
        # Get predictions from both models
        enhanced_result = predictor.predict_stress_level(text, use_enhanced=True)
        fallback_result = predictor.predict_stress_level(text, use_enhanced=False)
        
        return jsonify({
            'text': text,
            'enhanced_prediction': enhanced_result,
            'fallback_prediction': fallback_result,
            'models_available': {
                'enhanced': predictor.enhanced_model_loaded,
                'fallback': predictor.fallback_model_loaded
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("=== Enhanced Mental Health Chatbot API Server ===")
    print("Starting enhanced server...")
    print("Available endpoints:")
    print("- GET  /health - Health check")
    print("- POST /predict - Predict stress level (uses enhanced model by default)")
    print("- POST /predict_batch - Predict stress levels for multiple texts")
    print("- GET  /model_info - Get model information")
    print("- POST /compare_models - Compare predictions from both models")
    print()
    print("Example usage:")
    print('curl -X POST http://localhost:5001/predict \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"text": "I am feeling very anxious today", "use_enhanced": true}\'')
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5001) 