from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from app.routes import diagnosis, health, knowledge

load_dotenv()

app = FastAPI(
    title="DermCare AI Server",
    description="FastAPI server xử lý AI chẩn đoán bệnh da liễu qua Ảnh (YOLOv8) và Văn bản (PhoBERT) cho ứng dụng DermCare",
    version="1.1.0",
)

# CORS - Cho phép Node.js backend và Next.js FE kết nối
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,http://localhost:4000,https://www.dermcare.io.vn,https://dermcare.io.vn,https://dermcare-be.onrender.com"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các routes
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(diagnosis.router, prefix="/api/diagnosis", tags=["Diagnosis"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["Knowledge Base"])


@app.get("/")
async def root():
    return {"message": "DermCare AI Server đang hoạt động 🚀", "version": "1.0.0"}
