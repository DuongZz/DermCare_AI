"""
Script nhanh để kiểm tra class names từ file best1.pt
Chạy: python inspect_model.py
"""
from dotenv import load_dotenv
import os
load_dotenv()

MODEL_URL = os.getenv("MODEL_URL")
MODEL_PATH = "./models/best1.pt"

def download_model():
    os.makedirs("./models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        if not MODEL_URL:
            print("❌ Lỗi: MODEL_URL chưa được cấu hình trong file .env")
            print("Vui lòng sao chép MODEL_URL từ Supabase vào file .env")
            return False
        
        print("Đang tải model từ Supabase...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Tải xong!")
    else:
        print("✅ Model đã có sẵn.")
    return True

if __name__ == "__main__":
    if download_model():
        from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print("\n📋 Danh sách class trong model:")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")
    print(f"\n✅ Tổng cộng {len(model.names)} classes")
    print(f"Task type: {model.task}")
