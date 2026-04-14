# Sử dụng Python 3.10 slim để nhẹ nhàng và ổn định
FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và các gói AI
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file yêu cầu và cài đặt (Pip không cache để giảm dung lượng Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Tạo thư mục chứa model nếu chưa có
RUN mkdir -p models

# Hugging Face Spaces mặc định chạy trên cổng 7860
ENV PORT=7860
EXPOSE 7860

# Lệnh khởi chạy FastAPI qua Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
