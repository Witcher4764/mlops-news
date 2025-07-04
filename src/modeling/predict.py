from asyncio.log import logger
import os
import pandas as pd
from joblib import load
from typing import Union
from newspaper import Article
import re


import warnings
warnings.filterwarnings("ignore")

# ---------- LOAD ARTIFACTS ----------
def load_artifacts(model_dir: str = "models"):
    model_path = os.path.join(model_dir, "lightgbm_model.joblib")
    encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")

    model = load(model_path)
    label_encoder = load(encoder_path)
    vectorizer = load(vectorizer_path)

    return model, label_encoder, vectorizer

# ---------- PREDICT ----------
def predict(text: Union[str, list], model, label_encoder, vectorizer):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    X = vectorizer.transform(texts)
    preds = model.predict(X)
    labels = label_encoder.inverse_transform(preds)

    return labels

# ---------- PREDICT WITH PROBABILITIES ----------
def predict_with_probabilities(text: Union[str, list], model, label_encoder, vectorizer):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    X = vectorizer.transform(texts)
    probabilities = model.predict_proba(X)
    
    # Get all class names
    class_names = label_encoder.classes_
    
    results = []
    for i, text_input in enumerate(texts):
        text_probs = probabilities[i]
        # Create a dictionary of class name to probability
        prob_dict = {class_name: prob for class_name, prob in zip(class_names, text_probs)}
        # Sort by probability in descending order
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        results.append({
            'text': text_input,
            'probabilities': sorted_probs
        })
    
    return results


def get_article_text(url: str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        cleaned_text = re.sub(r'\n{2,}', '\n', text)
        return cleaned_text.strip()
    except Exception as e:
        print(f"Failed to fetch article: {e}")
        return ""
    

# ---------- MAIN ----------
if __name__ == "__main__":
    try:
        model, le, vectorizer = load_artifacts()
        
        # Sample text input
        url = input("Enter the URL of the article: ")
        sample_text = get_article_text(url)
        print("*" * 100)
        print(sample_text[:500])

        # Get predictions with probabilities for all classes
        results = predict_with_probabilities(sample_text, model, le, vectorizer)
        
        
        print("\nClass Probabilities (sorted by confidence):")
        print("-" * 50)
        
        for class_name, probability in results[0]['probabilities'].items():
            print(f"{class_name:20}: {probability:.4f} ({probability*100:.2f}%)")
        
        # Also show the top predicted class
        top_class = list(results[0]['probabilities'].keys())[0]
        print(f"\nTop predicted category: {top_class}")

    except Exception as e:
        print(f"Prediction pipeline failed: {e}")
