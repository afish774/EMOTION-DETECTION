from transformers import pipeline
import pandas as pd

def demo():
    print("Loading model...")
    classifier = pipeline(
        "text-classification", 
        model="bhadresh-savani/distilbert-base-uncased-emotion", 
        top_k=None
    )
    
    text = "I am absolutely thrilled that this project is working so well!"
    print(f"\nAnalyzing text: '{text}'\n")
    
    results = classifier(text)
    emotions = results[0]
    
    # Sort and display
    sorted_emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
    
    print(f"{'EMOTION':<12} | {'CONFIDENCE':<10}")
    print("-" * 25)
    for emotion in sorted_emotions:
        print(f"{emotion['label'].upper():<12} | {emotion['score']:.4f}")

if __name__ == "__main__":
    demo()
