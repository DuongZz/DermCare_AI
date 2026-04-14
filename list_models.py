import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

print("--- AVAILABLE MODELS ---")
with open('models_list.txt', 'w') as f:
    try:
        for m in genai.list_models():
            f.write(f"{m.name}\n")
            print(f"Added: {m.name}")
    except Exception as e:
        f.write(f"Error: {e}\n")
        print(f"Error: {e}")
print("Done writing to models_list.txt")
