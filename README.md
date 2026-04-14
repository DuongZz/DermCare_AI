---
title: Dermcare AI
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# DermCare AI Server

FastAPI server xử lý AI chẩn đoán bệnh da liễu cho ứng dụng DermCare.

## Cấu trúc project

```
Dermcare-AI/
├── main.py                 # App FastAPI chính
├── requirements.txt        # Danh sách packages
├── .env                    # Biến môi trường (tạo từ .env.example)
├── start.bat               # Script chạy server (Windows)
└── app/
    ├── schemas.py          # Pydantic models (request/response types)
    ├── routes/
    │   ├── health.py       # GET /api/health
    │   └── diagnosis.py    # POST /api/diagnosis/analyze
    └── services/
        └── ai_service.py   # Logic xử lý YOLOv8 model
```

## Cài đặt & Chạy

1. **Cấu hình môi trường:**
   Tạo file `.env` từ `.env.example` và điền `MODEL_URL` (nếu cần).
   ```bash
   cp .env.example .env
   ```

2. **Khởi động server (Windows):**
   Bạn chỉ cần chạy file `start.bat`:
   ```powershell
   .\start.bat
   ```
   *Lưu ý: Script sẽ tự động kích hoạt môi trường ảo (venv) và chạy server tại cổng 8000.*

3. **Khởi động thủ công:**
   ```bash
   venv\Scripts\activate
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Endpoints

| Method | URL | Mô tả |
|--------|-----|--------|
| `GET` | `/` | Root info |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/diagnosis/analyze` | Upload ảnh để AI chẩn đoán |

## Docs tương tác

Sau khi chạy server, vào:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Cấu hình AI
Mở `.env` và điền `MODEL_URL` nếu bạn muốn thay đổi nguồn tải model:
```env
MODEL_URL=https://your-supabase-url...
```

> **Nếu chưa có file model**, server sẽ tự động tải từ Supabase tại lần chạy đầu tiên.
