#!/usr/bin/env python3
"""
Quick script to install missing dependencies for the Mental Health Chatbot
"""

import subprocess
import sys

def install_missing_deps():
    """Install missing dependencies"""
    print("Installing missing dependencies...")
    
    missing_deps = [
        "flask",
        "flask-cors", 
        "openpyxl"
    ]
    
    for dep in missing_deps:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {dep}: {e}")
            return False
    
    print()
    print("✅ All dependencies installed!")
    print("You can now run:")
    print("- python enhanced_api_server.py")
    print("- python api_server.py")
    return True

if __name__ == "__main__":
    install_missing_deps() 