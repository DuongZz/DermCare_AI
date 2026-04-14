import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

MODEL_PATH = "./models/nlp"

def test_load():
    print(f"Loading from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("✅ Load Success!")
        
        text = "Tôi bị nổi mề đay"
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        print(f"✅ Predict Success: {outputs.logits.shape}")
    except Exception as e:
        print(f"❌ Load Failed: {e}")

if __name__ == "__main__":
    test_load()
