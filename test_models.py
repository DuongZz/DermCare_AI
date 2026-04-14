import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

models_to_try = [
    'models/gemini-flash-latest',
    'models/gemini-pro-latest',
    'models/gemini-1.5-flash',
    'models/gemini-1.5-pro',
    'models/gemini-2.0-flash-lite',
    'models/gemini-flash-lite-latest'
]

print("--- Testing Models ---")
for model_name in models_to_try:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("hello")
        print(f"✅ SUCCESS: {model_name}")
    except Exception as e:
        print(f"❌ FAILED: {model_name} - {str(e)[:100]}")
