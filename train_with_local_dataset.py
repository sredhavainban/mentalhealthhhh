import pandas as pd
import numpy as np
import os
import json
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

class EnhancedMentalHealthTrainer:
    def __init__(self, local_dataset_path="C:/Users/Sre Dhava Inban/OneDrive/Desktop/mental health/dataset"):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.local_dataset_path = local_dataset_path
        
        # Emotion mapping for the dair-ai/emotion dataset
        self.emotion_mapping = {
            0: 'sadness',
            1: 'joy', 
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }
        
        # Stress level mapping
        self.stress_mapping = {
            'sadness': 'high',
            'anger': 'high', 
            'fear': 'high',
            'joy': 'low',
            'love': 'low',
            'surprise': 'medium'
        }
        
    def load_local_dataset(self):
        """Load and process the local dataset"""
        print(f"Loading local dataset from: {self.local_dataset_path}")
        
        if not os.path.exists(self.local_dataset_path):
            print(f"❌ Local dataset path not found: {self.local_dataset_path}")
            print(f"Please create the directory or update the path in the script.")
            return None
        
        try:
            # Try to load different file formats
            dataset_files = []
            for file in os.listdir(self.local_dataset_path):
                if file.endswith(('.csv', '.json', '.xlsx', '.txt')):
                    dataset_files.append(os.path.join(self.local_dataset_path, file))
            
            if not dataset_files:
                print("❌ No dataset files found in the specified directory")
                print(f"Supported formats: .csv, .json, .xlsx, .txt")
                print(f"Please add your dataset files to: {self.local_dataset_path}")
                return None
            
            print(f"Found dataset files: {dataset_files}")
            
            local_data = []
            
            for file_path in dataset_files:
                print(f"Processing file: {file_path}")
                
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif file_path.endswith('.json'):
                        df = pd.read_json(file_path)
                    elif file_path.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                    elif file_path.endswith('.txt'):
                        # Try to read as JSON first, then as plain text
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                df = pd.read_json(content)
                        except:
                            # Read as plain text with one entry per line
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                df = pd.DataFrame({'text': lines})
                    
                    # Standardize column names
                    df = self.standardize_columns(df)
                    
                    # Basic validation
                    if len(df) > 0:
                        local_data.append(df)
                        print(f"  ✅ Loaded {len(df)} samples from {os.path.basename(file_path)}")
                    else:
                        print(f"  ⚠️  No valid data in {os.path.basename(file_path)}")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {os.path.basename(file_path)}: {e}")
                    continue
            
            if local_data:
                combined_local_data = pd.concat(local_data, ignore_index=True)
                print(f"✅ Local dataset loaded successfully!")
                print(f"Local samples: {len(combined_local_data)}")
                print(f"Columns: {list(combined_local_data.columns)}")
                
                # Show sample data
                print(f"Sample data:")
                print(combined_local_data.head())
                
                return combined_local_data
            else:
                print("❌ No valid data found in local dataset")
                return None
                
        except Exception as e:
            print(f"❌ Error loading local dataset: {e}")
            return None
    
    def standardize_columns(self, df):
        """Standardize column names and structure"""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Map common column names to standard format
        column_mapping = {
            'message': 'text',
            'content': 'text',
            'input': 'text',
            'sentence': 'text',
            'utterance': 'text',
            'label': 'stress_level',
            'emotion': 'stress_level',
            'sentiment': 'stress_level',
            'category': 'stress_level',
            'class': 'stress_level'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # If no text column found, try to use the first string column
        if 'text' not in df.columns:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df = df.rename(columns={col: 'text'})
                    break
        
        # If no stress_level column found, create a default one
        if 'stress_level' not in df.columns:
            df['stress_level'] = 'medium'  # Default stress level
        
        return df
    
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
    
    def map_stress_levels(self, data):
        """Map stress levels to standardized format"""
        # Create a mapping for common stress level terms
        stress_mapping = {
            'high': 'high',
            'severe': 'high',
            'critical': 'high',
            'extreme': 'high',
            'very high': 'high',
            'medium': 'medium',
            'moderate': 'medium',
            'average': 'medium',
            'low': 'low',
            'mild': 'low',
            'minimal': 'low',
            'sad': 'high',
            'angry': 'high',
            'fearful': 'high',
            'anxious': 'high',
            'depressed': 'high',
            'happy': 'low',
            'joyful': 'low',
            'content': 'low',
            'calm': 'low'
        }
        
        # Handle NaN values in stress_level column
        data = data.copy()
        if 'stress_level' in data.columns:
            # Fill NaN values with 'medium' as default
            data['stress_level'] = data['stress_level'].fillna('medium')
            # Apply mapping
            data['stress_level'] = data['stress_level'].astype(str).str.lower().map(
                lambda x: stress_mapping.get(x, 'medium')
            )
        
        return data
    
    def combine_datasets(self, emotion_data, local_data):
        """Combine emotion dataset with local dataset"""
        print("Combining datasets...")
        
        # Prepare emotion dataset
        emotion_data['text_clean'] = emotion_data['text'].apply(self.preprocess_text)
        emotion_data['stress_level'] = emotion_data['label'].map(self.stress_mapping)
        emotion_data = emotion_data[['text_clean', 'stress_level']]
        
        # Prepare local dataset
        local_data['text_clean'] = local_data['text'].apply(self.preprocess_text)
        local_data = self.map_stress_levels(local_data)
        local_data = local_data[['text_clean', 'stress_level']]
        
        # Remove empty texts and NaN values
        emotion_data = emotion_data[emotion_data['text_clean'].str.len() > 0]
        emotion_data = emotion_data.dropna()  # Remove NaN values
        
        local_data = local_data[local_data['text_clean'].str.len() > 0]
        local_data = local_data.dropna()  # Remove NaN values
        
        # Combine datasets
        combined_data = pd.concat([emotion_data, local_data], ignore_index=True)
        
        # Final cleanup - remove any remaining NaN values
        combined_data = combined_data.dropna()
        
        print(f"Combined dataset statistics:")
        print(f"- Emotion dataset samples: {len(emotion_data)}")
        print(f"- Local dataset samples: {len(local_data)}")
        print(f"- Total combined samples: {len(combined_data)}")
        print(f"- Stress level distribution:")
        print(combined_data['stress_level'].value_counts())
        
        return combined_data
    
    def load_emotion_dataset(self):
        """Load the dair-ai/emotion dataset"""
        print("Loading emotion dataset...")
        
        try:
            dataset = load_dataset("dair-ai/emotion")
            
            # Convert to pandas DataFrame
            train_data = dataset['train'].to_pandas()
            test_data = dataset['test'].to_pandas()
            validation_data = dataset['validation'].to_pandas()
            
            # Combine all data
            all_data = pd.concat([train_data, test_data, validation_data], ignore_index=True)
            
            print(f"✅ Emotion dataset loaded successfully!")
            print(f"Emotion samples: {len(all_data)}")
            
            return all_data
            
        except Exception as e:
            print(f"❌ Error loading emotion dataset: {e}")
            return None
    
    def train_enhanced_model(self):
        """Train the enhanced model with both datasets"""
        print("=== Enhanced Mental Health ML Training ===")
        print()
        
        # Load emotion dataset
        emotion_data = self.load_emotion_dataset()
        if emotion_data is None:
            print("Failed to load emotion dataset. Exiting.")
            return False
        
        # Load local dataset
        local_data = self.load_local_dataset()
        if local_data is None:
            print("Failed to load local dataset. Using only emotion dataset.")
            local_data = pd.DataFrame(columns=['text', 'stress_level'])
        
        # Combine datasets
        combined_data = self.combine_datasets(emotion_data, local_data)
        
        if len(combined_data) == 0:
            print("❌ No valid data to train on.")
            return False
        
        # Train the model
        print("Training enhanced model...")
        
        # Prepare features and labels
        X = combined_data['text_clean']
        y = combined_data['stress_level']
        
        # Additional validation - ensure no NaN values
        print(f"Data validation:")
        print(f"- X shape: {X.shape}")
        print(f"- y shape: {y.shape}")
        print(f"- X has NaN: {X.isna().any()}")
        print(f"- y has NaN: {y.isna().any()}")
        
        # Remove any remaining NaN values
        valid_mask = ~(X.isna() | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"After cleaning:")
        print(f"- X shape: {X.shape}")
        print(f"- y shape: {y.shape}")
        
        if len(X) == 0:
            print("❌ No valid data after cleaning.")
            return False
        
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
        
        print(f"✅ Enhanced model training completed!")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return True
    
    def save_enhanced_model(self, filename_prefix="enhanced_mental_health_model"):
        """Save the enhanced trained model"""
        print("Saving enhanced model...")
        
        # Save the vectorizer
        joblib.dump(self.vectorizer, f"{filename_prefix}_vectorizer.pkl")
        
        # Save the classifier
        joblib.dump(self.classifier, f"{filename_prefix}_classifier.pkl")
        
        # Save the mappings
        mappings = {
            'emotion_mapping': self.emotion_mapping,
            'stress_mapping': self.stress_mapping,
            'local_dataset_path': self.local_dataset_path
        }
        
        with open(f"{filename_prefix}_mappings.json", 'w') as f:
            json.dump(mappings, f, indent=2)
        
        print(f"✅ Enhanced model saved successfully!")
        print(f"Files created:")
        print(f"- {filename_prefix}_vectorizer.pkl")
        print(f"- {filename_prefix}_classifier.pkl")
        print(f"- {filename_prefix}_mappings.json")
    
    def generate_enhanced_training_data_for_js(self):
        """Generate enhanced JavaScript training data"""
        print("Generating enhanced JavaScript training data...")
        
        # Load and prepare data
        emotion_data = self.load_emotion_dataset()
        local_data = self.load_local_dataset()
        
        if emotion_data is None:
            print("❌ Cannot generate training data without emotion dataset")
            return
        
        # Combine datasets
        if local_data is not None:
            combined_data = self.combine_datasets(emotion_data, local_data)
        else:
            emotion_data['text_clean'] = emotion_data['text'].apply(self.preprocess_text)
            emotion_data['stress_level'] = emotion_data['label'].map(self.stress_mapping)
            combined_data = emotion_data[['text_clean', 'stress_level']]
        
        # Create keyword lists for JavaScript
        high_stress_texts = combined_data[combined_data['stress_level'] == 'high']['text_clean'].tolist()
        medium_stress_texts = combined_data[combined_data['stress_level'] == 'medium']['text_clean'].tolist()
        low_stress_texts = combined_data[combined_data['stress_level'] == 'low']['text_clean'].tolist()
        
        # Extract common words for each stress level
        def extract_common_words(texts, top_n=30):
            all_words = []
            for text in texts:
                words = text.split()
                all_words.extend(words)
            
            word_freq = pd.Series(all_words).value_counts()
            return word_freq.head(top_n).index.tolist()
        
        high_keywords = extract_common_words(high_stress_texts)
        medium_keywords = extract_common_words(medium_stress_texts)
        low_keywords = extract_common_words(low_stress_texts)
        
        # Create enhanced JavaScript object
        js_data = {
            'high_stress_keywords': high_keywords,
            'medium_stress_keywords': medium_keywords,
            'low_stress_keywords': low_keywords,
            'training_samples': {
                'high': len(high_stress_texts),
                'medium': len(medium_stress_texts),
                'low': len(low_stress_texts)
            },
            'dataset_info': {
                'emotion_samples': len(emotion_data),
                'local_samples': len(local_data) if local_data is not None else 0,
                'total_samples': len(combined_data)
            }
        }
        
        # Save to JSON file
        with open('enhanced_training_data.js', 'w') as f:
            f.write('const ENHANCED_TRAINING_DATA = ')
            json.dump(js_data, f, indent=2)
            f.write(';')
        
        print("✅ Enhanced JavaScript training data saved to 'enhanced_training_data.js'")
        print(f"Enhanced keywords extracted:")
        print(f"- High stress: {len(high_keywords)} keywords")
        print(f"- Medium stress: {len(medium_keywords)} keywords")
        print(f"- Low stress: {len(low_keywords)} keywords")

def main():
    """Main enhanced training function"""
    print("=== Enhanced Mental Health Chatbot ML Training ===")
    print()
    
    # Initialize enhanced trainer
    trainer = EnhancedMentalHealthTrainer()
    
    # Train the enhanced model
    success = trainer.train_enhanced_model()
    
    if success:
        # Save the enhanced model
        trainer.save_enhanced_model()
        
        # Generate enhanced JavaScript training data
        trainer.generate_enhanced_training_data_for_js()
        
        print()
        print("=== Enhanced Training Complete ===")
        print("You can now:")
        print("1. Use the enhanced trained model for better predictions")
        print("2. Include 'enhanced_training_data.js' in your HTML")
        print("3. The model now includes your local dataset for better accuracy")
    else:
        print("❌ Enhanced training failed.")

if __name__ == "__main__":
    main() 