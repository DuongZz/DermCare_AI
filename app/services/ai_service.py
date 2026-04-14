import os
import io
import urllib.request
import base64
import cv2
import numpy as np

from app.schemas import DiagnosisResponse

MODEL_URL = os.getenv("MODEL_URL")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/best1.pt")

from app.constants import CLASS_INFO, get_mock_diagnosis

_model = None


def _get_model():
    """Lazy load YOLOv8 model — chỉ tải 1 lần khi server khởi động"""
    global _model
    if _model is not None:
        return _model

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        if not MODEL_URL:
            print("[AI] ⚠️ Cảnh báo: MODEL_URL trống trong .env. Không thể tải model.")
            raise ValueError("MODEL_URL is missing in environment variables")
        
        print(f"[AI] 📥 Đang tải model từ Supabase...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[AI] ✅ Tải model xong!")

    from ultralytics import YOLO
    _model = YOLO(MODEL_PATH)
    print(f"[AI] ✅ Model sẵn sàng! Task: {_model.task} | Classes: {list(_model.names.values())}")
    return _model


async def analyze_skin_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> DiagnosisResponse:
    """
    Chạy YOLOv8 Segmentation inference trên ảnh da.
    Trả về class có confidence cao nhất.
    """
    try:
        model = _get_model()
    except Exception as e:
        print(f"[AI] ⚠️ Lỗi load model: {e}. Trả về mock.")
        return _mock_response()

    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Chạy inference (segment task dùng boxes để lấy confidence)
    results = model(image, verbose=False)

    if not results or len(results) == 0:
        return _mock_response()

    result = results[0]

    # Với Segment/Detect: lấy detection có confidence cao nhất
    if result.boxes is None or len(result.boxes) == 0:
        # Không phát hiện bệnh nào → clean skin
        return DiagnosisResponse(
            disease_name="Không phát hiện bệnh",
            specialization="Da liễu",
            confidence=0.95,
            description="Không phát hiện dấu hiệu bệnh da bất thường trong ảnh. Da có vẻ khỏe mạnh.",
            recommendations=["Tiếp tục duy trì thói quen chăm sóc da", "Dùng kem chống nắng hàng ngày"],
            should_see_doctor=False,
            severity="mild",
            details={"Không phát hiện bệnh": 0.95}
        )

    # Gom nhóm theo class và lấy confidence cao nhất mỗi class
    class_scores: dict[int, float] = {}
    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        if cls_id not in class_scores or conf > class_scores[cls_id]:
            class_scores[cls_id] = conf

    # Lấy class có confidence cao nhất
    best_cls_id = max(class_scores, key=lambda k: class_scores[k])
    confidence = class_scores[best_cls_id]
    class_name = model.names[best_cls_id]  # tên từ model (vd: "trungca")

    # Lấy thông tin chi tiết
    info = CLASS_INFO.get(class_name, {
        "display": class_name,
        "specialization": "Da liễu Bệnh lý",
        "severity": "moderate",
        "description": f"Phát hiện: {class_name}.",
        "recommendations": ["Đến gặp bác sĩ để được tư vấn chính xác"],
        "should_see_doctor": True,
    })

    # Vẽ bounding box lên ảnh (plot result)
    # result.plot() trả về numpy array (BGR)
    plotted_img = result.plot(labels=False, conf=False)
    
    # Convert BGR to RGB for consistent display if needed, but imencode expects BGR
    _, buffer = cv2.imencode('.jpg', plotted_img)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Convert class ids to system keys for details
    details_map = {}
    for cls_id, conf in class_scores.items():
        c_name = model.names[cls_id]
        details_map[c_name] = round(conf, 4)

    return DiagnosisResponse(
        disease_name=info["display"],
        specialization=info["specialization"],
        confidence=round(confidence, 3),
        description=info["description"],
        recommendations=info["recommendations"],
        should_see_doctor=info["should_see_doctor"],
        severity=info["severity"],
        processed_image=processed_image_base64,
        details=details_map
    )


def _mock_response() -> DiagnosisResponse:
    return get_mock_diagnosis()
