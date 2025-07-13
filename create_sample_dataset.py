#!/usr/bin/env python3
"""
Create a sample dataset for testing the enhanced ML training
"""

import pandas as pd
import os
import json

def create_sample_dataset():
    """Create a sample dataset with mental health related text and stress levels"""
    
    # Sample data with different stress levels
    sample_data = [
        # High stress examples
        {"text": "I'm feeling extremely anxious and can't stop worrying about everything", "stress_level": "high"},
        {"text": "I'm so depressed and hopeless, I don't know how to go on", "stress_level": "high"},
        {"text": "I'm having panic attacks and can't breathe properly", "stress_level": "high"},
        {"text": "I'm so angry and frustrated with everything in my life", "stress_level": "high"},
        {"text": "I'm terrified of what might happen tomorrow", "stress_level": "high"},
        {"text": "I feel completely overwhelmed and can't handle anything", "stress_level": "high"},
        {"text": "I'm so sad and lonely, I just want to cry all the time", "stress_level": "high"},
        {"text": "I'm having suicidal thoughts and don't know what to do", "stress_level": "high"},
        {"text": "I'm so stressed about work that I can't sleep", "stress_level": "high"},
        {"text": "I'm feeling desperate and don't know where to turn", "stress_level": "high"},
        
        # Medium stress examples
        {"text": "I'm a bit worried about my upcoming presentation", "stress_level": "medium"},
        {"text": "I'm feeling somewhat stressed about my relationship", "stress_level": "medium"},
        {"text": "I'm concerned about my health but trying to stay positive", "stress_level": "medium"},
        {"text": "I'm a little nervous about the interview tomorrow", "stress_level": "medium"},
        {"text": "I'm feeling moderately anxious about the future", "stress_level": "medium"},
        {"text": "I'm somewhat bothered by recent events", "stress_level": "medium"},
        {"text": "I'm feeling a bit uneasy about my financial situation", "stress_level": "medium"},
        {"text": "I'm moderately concerned about my family's health", "stress_level": "medium"},
        {"text": "I'm feeling a bit tense about upcoming changes", "stress_level": "medium"},
        {"text": "I'm somewhat worried but trying to stay calm", "stress_level": "medium"},
        
        # Low stress examples
        {"text": "I'm feeling quite happy and content today", "stress_level": "low"},
        {"text": "I'm in a good mood and enjoying my day", "stress_level": "low"},
        {"text": "I'm feeling calm and relaxed", "stress_level": "low"},
        {"text": "I'm content with how things are going", "stress_level": "low"},
        {"text": "I'm feeling peaceful and at ease", "stress_level": "low"},
        {"text": "I'm in a positive mood and looking forward to the day", "stress_level": "low"},
        {"text": "I'm feeling grateful and thankful", "stress_level": "low"},
        {"text": "I'm feeling joyful and excited about life", "stress_level": "low"},
        {"text": "I'm feeling satisfied with my current situation", "stress_level": "low"},
        {"text": "I'm feeling optimistic about the future", "stress_level": "low"}
    ]
    
    # Create dataset directory
    dataset_path = "C:/Users/Sre Dhava Inban/OneDrive/Desktop/mental health/dataset"
    os.makedirs(dataset_path, exist_ok=True)
    
    # Create CSV file
    df_csv = pd.DataFrame(sample_data)
    csv_path = os.path.join(dataset_path, "mental_health_data.csv")
    df_csv.to_csv(csv_path, index=False)
    print(f"✅ Created CSV dataset: {csv_path}")
    print(f"   Samples: {len(df_csv)}")
    
    # Create JSON file
    json_path = os.path.join(dataset_path, "mental_health_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    print(f"✅ Created JSON dataset: {json_path}")
    
    # Create Excel file
    excel_path = os.path.join(dataset_path, "mental_health_data.xlsx")
    df_csv.to_excel(excel_path, index=False)
    print(f"✅ Created Excel dataset: {excel_path}")
    
    # Create text file (one entry per line)
    txt_path = os.path.join(dataset_path, "mental_health_data.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(f"{item['text']}\t{item['stress_level']}\n")
    print(f"✅ Created text dataset: {txt_path}")
    
    print()
    print("=== Sample Dataset Created ===")
    print(f"Dataset location: {dataset_path}")
    print("Files created:")
    print("- mental_health_data.csv")
    print("- mental_health_data.json") 
    print("- mental_health_data.xlsx")
    print("- mental_health_data.txt")
    print()
    print("You can now run:")
    print("python enhanced_setup.py --mode enhanced")
    print()
    print("Or add your own dataset files to the same directory!")

if __name__ == "__main__":
    create_sample_dataset() 