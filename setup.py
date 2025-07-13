#!/usr/bin/env python3
"""
Setup script for Mental Health Chatbot ML Training
"""

import subprocess
import sys
import os

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

def run_training():
    """Run the ML training script"""
    print("Starting ML training...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during training: {e}")
        return False

def check_files():
    """Check if required files exist"""
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
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All model files generated successfully!")
        return True

def main():
    """Main setup function"""
    print("=== Mental Health Chatbot ML Setup ===")
    print()
    
    # Step 1: Install requirements
    if not install_requirements():
        print("Setup failed at requirements installation.")
        return False
    
    print()
    
    # Step 2: Run training
    if not run_training():
        print("Setup failed during training.")
        return False
    
    print()
    
    # Step 3: Check generated files
    if not check_files():
        print("Setup failed - missing model files.")
        return False
    
    print()
    print("=== Setup Complete! ===")
    print("You can now:")
    print("1. Open index.html in your browser to use the chatbot")
    print("2. Run 'python api_server.py' to start the ML API server")
    print("3. The chatbot will use ML-trained keywords for better stress detection")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 