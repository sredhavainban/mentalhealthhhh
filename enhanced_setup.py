#!/usr/bin/env python3
"""
Enhanced Setup script for Mental Health Chatbot ML Training
Supports both basic training and enhanced training with local dataset
"""

import subprocess
import sys
import os
import argparse

def install_requirements():
    """Install required Python packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_basic_training():
    """Run the basic ML training script"""
    print("Starting basic ML training...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Basic training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during basic training: {e}")
        return False

def run_enhanced_training():
    """Run the enhanced ML training script with local dataset"""
    print("Starting enhanced ML training with local dataset...")
    try:
        subprocess.check_call([sys.executable, "train_with_local_dataset.py"])
        print("✅ Enhanced training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during enhanced training: {e}")
        return False

def check_basic_files():
    """Check if basic model files exist"""
    required_files = [
        "mental_health_model_vectorizer.pkl",
        "mental_health_model_classifier.pkl", 
        "mental_health_model_mappings.json",
        "training_data.js"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing basic model files: {missing_files}")
        return False
    else:
        print("✅ All basic model files generated successfully!")
        return True

def check_enhanced_files():
    """Check if enhanced model files exist"""
    required_files = [
        "enhanced_mental_health_model_vectorizer.pkl",
        "enhanced_mental_health_model_classifier.pkl", 
        "enhanced_mental_health_model_mappings.json",
        "enhanced_training_data.js"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing enhanced model files: {missing_files}")
        return False
    else:
        print("✅ All enhanced model files generated successfully!")
        return True

def update_html_for_enhanced():
    """Update HTML to include enhanced training data"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if enhanced training data is already included
        if 'enhanced_training_data.js' not in content:
            # Add enhanced training data script before script.js
            content = content.replace(
                '    <script src="training_data.js"></script>',
                '    <script src="training_data.js"></script>\n    <script src="enhanced_training_data.js"></script>'
            )
            
            with open('index.html', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Updated HTML to include enhanced training data")
        else:
            print("✅ HTML already includes enhanced training data")
        
        return True
    except Exception as e:
        print(f"❌ Error updating HTML: {e}")
        return False

def update_javascript_for_enhanced():
    """Update JavaScript to use enhanced training data"""
    try:
        with open('script.js', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update the constructor to use enhanced training data
        enhanced_constructor = '''    constructor() {
        this.stressLevel = 0;
        this.conversationHistory = [];
        
        // Use enhanced ML-trained keywords if available, otherwise fall back to basic training data
        this.stressKeywords = {
            high: ENHANCED_TRAINING_DATA ? ENHANCED_TRAINING_DATA.high_stress_keywords : 
                  (TRAINING_DATA ? TRAINING_DATA.high_stress_keywords : ['anxiety', 'panic', 'overwhelmed', 'terrified', 'desperate', 'hopeless', 'suicidal', 'can\'t breathe', 'heart racing', 'sweating', 'nausea']),
            medium: ENHANCED_TRAINING_DATA ? ENHANCED_TRAINING_DATA.medium_stress_keywords : 
                    (TRAINING_DATA ? TRAINING_DATA.medium_stress_keywords : ['stressed', 'worried', 'nervous', 'tense', 'frustrated', 'angry', 'sad', 'depressed', 'lonely', 'tired', 'exhausted', 'burnout']),
            low: ENHANCED_TRAINING_DATA ? ENHANCED_TRAINING_DATA.low_stress_keywords : 
                 (TRAINING_DATA ? TRAINING_DATA.low_stress_keywords : ['concerned', 'uneasy', 'bothered', 'annoyed', 'irritated', 'slightly', 'a bit', 'somewhat'])
        };'''
        
        # Replace the constructor
        import re
        pattern = r'constructor\(\) \{[\s\S]*?this\.stressKeywords = \{[\s\S]*?\};'
        content = re.sub(pattern, enhanced_constructor, content)
        
        with open('script.js', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Updated JavaScript to use enhanced training data")
        return True
    except Exception as e:
        print(f"❌ Error updating JavaScript: {e}")
        return False

def main():
    """Main enhanced setup function"""
    parser = argparse.ArgumentParser(description='Mental Health Chatbot ML Setup')
    parser.add_argument('--mode', choices=['basic', 'enhanced', 'both'], default='both',
                       help='Training mode: basic (emotion dataset only), enhanced (with local dataset), or both')
    parser.add_argument('--local-dataset-path', 
                       default="C:/Users/Sre Dhava Inban/OneDrive/Desktop/mental health/dataset",
                       help='Path to local dataset directory')
    
    args = parser.parse_args()
    
    print("=== Enhanced Mental Health Chatbot ML Setup ===")
    print(f"Mode: {args.mode}")
    print(f"Local dataset path: {args.local_dataset_path}")
    print()
    
    # Step 1: Install requirements
    if not install_requirements():
        print("Setup failed at requirements installation.")
        return False
    
    print()
    
    success = True
    
    # Step 2: Run training based on mode
    if args.mode in ['basic', 'both']:
        print("=== Basic Training ===")
        if not run_basic_training():
            print("Basic training failed.")
            success = False
        else:
            if not check_basic_files():
                print("Basic training failed - missing files.")
                success = False
        print()
    
    if args.mode in ['enhanced', 'both']:
        print("=== Enhanced Training ===")
        # Update the local dataset path in the enhanced training script
        try:
            with open('train_with_local_dataset.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update the default path
            content = content.replace(
                'local_dataset_path="C:/Users/Sre Dhava Inban/OneDrive/Desktop/mental health/dataset"',
                f'local_dataset_path="{args.local_dataset_path}"'
            )
            
            with open('train_with_local_dataset.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Updated local dataset path to: {args.local_dataset_path}")
        except Exception as e:
            print(f"❌ Error updating dataset path: {e}")
            success = False
        
        if not run_enhanced_training():
            print("Enhanced training failed.")
            success = False
        else:
            if not check_enhanced_files():
                print("Enhanced training failed - missing files.")
                success = False
            else:
                # Update HTML and JavaScript for enhanced training
                update_html_for_enhanced()
                update_javascript_for_enhanced()
        print()
    
    if success:
        print("=== Setup Complete! ===")
        print("You can now:")
        print("1. Open index.html in your browser to use the chatbot")
        print("2. Run 'python api_server.py' to start the ML API server")
        
        if args.mode in ['enhanced', 'both']:
            print("3. The chatbot now uses enhanced ML training with your local dataset")
            print("4. Enhanced keywords provide better stress detection accuracy")
        else:
            print("3. The chatbot uses basic ML training with emotion dataset")
        
        print("5. For API integration, set window.useMLAPI = true in browser console")
    else:
        print("❌ Setup failed.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 