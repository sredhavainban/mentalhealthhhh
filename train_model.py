import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import re

class MentalHealthMLTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.emotion_mapping = {
            0: 'sadness',
            1: 'joy', 
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }
        self.stress_mapping = {
            'sadness': 'high',
            'anger': 'high', 
            'fear': 'high',
            'joy': 'low',
            'love': 'low',
            'surprise': 'medium'
        }
        
    def load_and_preprocess_data(self):
        """Load the emotion dataset and preprocess it"""
        print("Loading emotion dataset...")
        
        try:
            # Load the dataset
            dataset = load_dataset("dair-ai/emotion")
            
            # Convert to pandas DataFrame
            train_data = dataset['train'].to_pandas()
            test_data = dataset['test'].to_pandas()
            validation_data = dataset['validation'].to_pandas()
            
            # Combine all data for training
            all_data = pd.concat([train_data, test_data, validation_data], ignore_index=True)
            
            print(f"Dataset loaded successfully!")
            print(f"Total samples: {len(all_data)}")
            print(f"Emotion distribution:")
            print(all_data['label'].value_counts())
            
            return all_data
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def prepare_training_data(self, data):
        """Prepare data for training"""
        print("Preprocessing text data...")
        
        # Clean text data
        data['text_clean'] = data['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        data = data[data['text_clean'].str.len() > 0]
        
        # Map emotions to stress levels
        data['stress_level'] = data['label'].map(self.stress_mapping)
        
        print(f"Data prepared! Final samples: {len(data)}")
        print(f"Stress level distribution:")
        print(data['stress_level'].value_counts())
        
        return data
    
    def train_model(self, data):
        """Train the machine learning model"""
        print("Training model...")
        
        # Prepare features and labels
        X = data['text_clean']
        y = data['stress_level']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text data
        print("Vectorizing text data...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train the classifier
        print("Training Random Forest classifier...")
        self.classifier.fit(X_train_vectorized, y_train)
        
        # Evaluate the model
        y_pred = self.classifier.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model training completed!")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def save_model(self, filename_prefix="mental_health_model"):
        """Save the trained model and vectorizer"""
        print("Saving model...")
        
        # Save the vectorizer
        joblib.dump(self.vectorizer, f"{filename_prefix}_vectorizer.pkl")
        
        # Save the classifier
        joblib.dump(self.classifier, f"{filename_prefix}_classifier.pkl")
        
        # Save the mappings
        mappings = {
            'emotion_mapping': self.emotion_mapping,
            'stress_mapping': self.stress_mapping
        }
        
        with open(f"{filename_prefix}_mappings.json", 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print(f"Model saved successfully!")
        print(f"Files created:")
        print(f"- {filename_prefix}_vectorizer.pkl")
        print(f"- {filename_prefix}_classifier.pkl")
        print(f"- {filename_prefix}_mappings.json")
    
    def predict_stress_level(self, text):
        """Predict stress level for a given text"""
        # Preprocess the text
        clean_text = self.preprocess_text(text)
        
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([clean_text])
        
        # Predict
        prediction = self.classifier.predict(text_vectorized)[0]
        probabilities = self.classifier.predict_proba(text_vectorized)[0]
        
        return {
            'stress_level': prediction,
            'confidence': max(probabilities),
            'probabilities': dict(zip(self.classifier.classes_, probabilities))
        }
    
    def generate_training_data_for_js(self):
        """Generate JavaScript-compatible training data"""
        print("Generating JavaScript training data...")
        
        # Load and prepare data
        data = self.load_and_preprocess_data()
        if data is None:
            return
        
        data = self.prepare_training_data(data)
        
        # Create keyword lists for JavaScript
        high_stress_texts = data[data['stress_level'] == 'high']['text_clean'].tolist()
        medium_stress_texts = data[data['stress_level'] == 'medium']['text_clean'].tolist()
        low_stress_texts = data[data['stress_level'] == 'low']['text_clean'].tolist()
        
        # Extract common words for each stress level
        def extract_common_words(texts, top_n=20):
            all_words = []
            for text in texts:
                words = text.split()
                all_words.extend(words)
            
            word_freq = pd.Series(all_words).value_counts()
            return word_freq.head(top_n).index.tolist()
        
        high_keywords = extract_common_words(high_stress_texts)
        medium_keywords = extract_common_words(medium_stress_texts)
        low_keywords = extract_common_words(low_stress_texts)
        
        # Create JavaScript object
        js_data = {
            'high_stress_keywords': high_keywords,
            'medium_stress_keywords': medium_keywords,
            'low_stress_keywords': low_keywords,
            'training_samples': {
                'high': len(high_stress_texts),
                'medium': len(medium_stress_texts),
                'low': len(low_stress_texts)
            }
        }
        
        # Save to JSON file
        with open('training_data.js', 'w') as f:
            f.write('const TRAINING_DATA = ')
            json.dump(js_data, f, indent=2)
            f.write(';')
        
        print("JavaScript training data saved to 'training_data.js'")
        print(f"Keywords extracted:")
        print(f"- High stress: {len(high_keywords)} keywords")
        print(f"- Medium stress: {len(medium_keywords)} keywords")
        print(f"- Low stress: {len(low_keywords)} keywords")

def main():
    """Main training function"""
    print("=== Mental Health Chatbot ML Training ===")
    print()
    
    # Initialize trainer
    trainer = MentalHealthMLTrainer()
    
    # Load and prepare data
    data = trainer.load_and_preprocess_data()
    if data is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Prepare training data
    data = trainer.prepare_training_data(data)
    
    # Train the model
    accuracy = trainer.train_model(data)
    
    # Save the model
    trainer.save_model()
    
    # Generate JavaScript training data
    trainer.generate_training_data_for_js()
    
    print()
    print("=== Training Complete ===")
    print("You can now:")
    print("1. Use the trained model in Python with predict_stress_level()")
    print("2. Include 'training_data.js' in your HTML for enhanced keyword detection")
    print("3. The model files are ready for deployment")

if __name__ == "__main__":
    main() 