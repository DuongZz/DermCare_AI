import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pyvi import ViTokenizer
from app.schemas import DiagnosisResponse

# Load config from environment
MODEL_PATH = os.getenv("NLP_MODEL_PATH", "./models/nlp")

_model = None
_tokenizer = None

# Mapping từ nhãn của BERT sang key của CLASS_INFO trong ai_service.py
# BERT Labels: 0: bongnuoc, 1: muncoc, 2: noimeday, 3: tonthuonghacto, 4: vaynen
LABEL_MAPPING = {
    "bongnuoc": "bong_nuoc",
    "muncoc": "muncoc",
    "noimeday": "noimeday",
    "tonthuonghacto": "ton_thuong_hac_to",
    "vaynen": "vaynen"
}

def _load_model():
    """Tải model BERT và Tokenizer (Lazy load)"""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    try:
        print(f"[NLP] 📥 Đang nạp model BERT từ {MODEL_PATH}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()
        print(f"[NLP] ✅ Model BERT sẵn sàng!")
        return _model, _tokenizer
    except Exception as e:
        print(f"[NLP] ❌ Lỗi khi nạp model BERT: {e}")
        return None, None

async def analyze_text_description(text: str) -> DiagnosisResponse:
    """
    Sử dụng PhoBERT để phân loại văn bản mô tả bệnh.
    """
    from app.constants import CLASS_INFO, get_mock_diagnosis
    
    model, tokenizer = _load_model()
    if model is None or tokenizer is None:
        print("[NLP] ⚠️ Model chưa sẵn sàng, trả về mock.")
        return get_mock_diagnosis()

    # Tiền xử lý văn bản (Tách từ tiếng Việt bằng pyvi)
    text_segmented = ViTokenizer.tokenize(text)
    
    # Tokenize input
    inputs = tokenizer(text_segmented, return_tensors="pt", truncation=True, max_length=256, padding=True)
    
    # Chạy inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_class_id = torch.max(probabilities, dim=1)
        
    confidence = confidence.item()
    predicted_id = predicted_class_id.item()
    class_name_bert = model.config.id2label.get(predicted_id) or model.config.id2label.get(str(predicted_id))
    
    # Extract all probabilities for details
    details_map = {}
    for i, prob in enumerate(probabilities[0]):
        c_name = model.config.id2label.get(i) or model.config.id2label.get(str(i))
        sys_key = LABEL_MAPPING.get(c_name, c_name)
        details_map[sys_key] = round(prob.item(), 4)

    # Map sang key chuẩn của hệ thống
    system_key = LABEL_MAPPING.get(class_name_bert, class_name_bert)
    
    # Lấy thông tin chi tiết từ CLASS_INFO
    info = CLASS_INFO.get(system_key, {
        "display": system_key,
        "specialization": "Da liễu",
        "severity": "moderate",
        "description": f"Dựa trên mô tả: {text}",
        "recommendations": ["Vui lòng cung cấp thêm hình ảnh để chẩn đoán chính xác hơn"],
        "should_see_doctor": True,
    })

    return DiagnosisResponse(
        disease_name=info["display"],
        specialization=info["specialization"],
        confidence=round(confidence, 3),
        description=info["description"],
        recommendations=info["recommendations"],
        should_see_doctor=info["should_see_doctor"],
        severity=info["severity"],
        processed_image=None,
        details=details_map
    )
