#!/usr/bin/env python3
"""
Test script to verify all components of the Mental Health Chatbot are working
"""

import os
import sys
import importlib

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing module imports...")
    
    required_modules = [
        'pandas',
        'numpy', 
        'sklearn',
        'joblib',
        'datasets',
        'flask',
        'flask_cors',
        'openpyxl'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {failed_imports}")
        print("Run: python install_dependencies.py")
        return False
    else:
        print("\n✅ All modules imported successfully!")
        return True

def test_files():
    """Test if required files exist"""
    print("\nTesting file existence...")
    
    required_files = [
        'index.html',
        'styles.css', 
        'script.js',
        'train_model.py',
        'train_with_local_dataset.py',
        'api_server.py',
        'enhanced_api_server.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All required files found!")
        return True

def test_dataset_directory():
    """Test if dataset directory exists"""
    print("\nTesting dataset directory...")
    
    dataset_path = "C:/Users/Sre Dhava Inban/OneDrive/Desktop/mental health/dataset"
    
    if os.path.exists(dataset_path):
        print(f"✅ Dataset directory exists: {dataset_path}")
        
        # Check for dataset files
        files = os.listdir(dataset_path)
        dataset_files = [f for f in files if f.endswith(('.csv', '.json', '.xlsx', '.txt'))]
        
        if dataset_files:
            print(f"✅ Found dataset files: {dataset_files}")
            return True
        else:
            print("⚠️  No dataset files found")
            print("Run: python create_sample_dataset.py")
            return False
    else:
        print(f"❌ Dataset directory not found: {dataset_path}")
        print("Run: python create_sample_dataset.py")
        return False

def test_model_files():
    """Test if model files exist"""
    print("\nTesting model files...")
    
    model_files = [
        'mental_health_model_vectorizer.pkl',
        'mental_health_model_classifier.pkl',
        'mental_health_model_mappings.json',
        'enhanced_mental_health_model_vectorizer.pkl',
        'enhanced_mental_health_model_classifier.pkl', 
        'enhanced_mental_health_model_mappings.json'
    ]
    
    basic_models = []
    enhanced_models = []
    
    for file in model_files:
        if os.path.exists(file):
            if file.startswith('enhanced'):
                enhanced_models.append(file)
            else:
                basic_models.append(file)
    
    if basic_models:
        print(f"✅ Basic model files: {len(basic_models)}")
    else:
        print("❌ No basic model files found")
        print("Run: python setup.py")
    
    if enhanced_models:
        print(f"✅ Enhanced model files: {len(enhanced_models)}")
    else:
        print("❌ No enhanced model files found")
        print("Run: python enhanced_setup.py --mode enhanced")
    
    return len(basic_models) > 0 or len(enhanced_models) > 0

def main():
    """Run all tests"""
    print("=== Mental Health Chatbot Setup Test ===")
    print()
    
    tests = [
        test_imports,
        test_files,
        test_dataset_directory,
        test_model_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Open index.html in your browser")
        print("2. Run: python enhanced_api_server.py")
        print("3. Enable ML API in browser console:")
        print("   window.useMLAPI = true;")
        print("   window.mlAPIEndpoint = 'http://localhost:5001';")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main() 