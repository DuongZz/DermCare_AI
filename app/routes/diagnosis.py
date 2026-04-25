from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional
from app.schemas import DiagnosisResponse
from app.services.ai_service import analyze_skin_image
from app.services.nlp_service import analyze_text_description

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@router.post("/analyze", response_model=DiagnosisResponse)
async def analyze_skin(
    file: Optional[UploadFile] = File(default=None, description="Ảnh vùng da cần chẩn đoán"),
    description: Optional[str] = Form(default=None, description="Mô tả triệu chứng bằng văn bản")
):
    """
    **Chẩn đoán bệnh da liễu qua Ảnh, Văn bản hoặc cả hai.**

    - Nếu chỉ gửi **Ảnh**: Sử dụng YOLOv8 để phân tích.
    - Nếu chỉ gửi **Văn bản**: Sử dụng PhoBERT để phân tích.
    - Nếu gửi **cả hai**: Kết hợp (**Late Fusion**) xác suất từ cả hai mô hình (Image: 60%, Text: 40%).
    """
    if not file and not description:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp ít nhất ảnh hoặc mô tả văn bản.")

    img_result = None
    nlp_result = None

    # 1. Xử lý ảnh nếu có
    if file:
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail=f"Chỉ hỗ trợ ảnh định dạng: {', '.join(ALLOWED_TYPES)}")
        
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Kích thước ảnh không được vượt quá 10MB")
        
        try:
            img_result = await analyze_skin_image(contents, file.content_type)
        except Exception as e:
            print(f"[AI] Lỗi xử lý ảnh: {e}")
            raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh AI: {str(e)}")

    # 2. Xử lý văn bản nếu có
    if description:
        try:
            nlp_result = await analyze_text_description(description)
        except Exception as e:
            print(f"[NLP] Lỗi xử lý văn bản: {e}")
            # Nếu chỉ có mô tả mà mô tả lỗi thì báo lỗi luôn
            if not file:
                raise HTTPException(status_code=500, detail=f"Lỗi xử lý văn bản AI: {str(e)}")

    # 3. Tổng hợp kết quả (Hybrid Late Fusion Logic)
    if img_result and nlp_result:
        img_details = img_result.details or {}
        nlp_details = nlp_result.details or {}
        
        # Nếu một trong hai không có details (ví dụ YOLO không nhận diện được bệnh), rơi vào logic fallback
        if not img_details and not nlp_details:
            return img_result if img_result.confidence >= nlp_result.confidence else nlp_result
            
        cv_best_class = img_result.disease_name
        cv_best_score = img_result.confidence
        
        nlp_best_class = nlp_result.disease_name
        nlp_best_score = nlp_result.confidence
        
        if cv_best_class == nlp_best_class:
            # Hai model cùng đoán chung 1 bệnh -> Boost logic (+10% trust)
            final_score = min(0.99, max(cv_best_score, nlp_best_score) + 0.1)
            
            # Trả về kết quả (dùng img_result làm khung, vì nó có processed_image)
            img_result.confidence = round(final_score, 3)
            # Cập nhật lại details nếu cần
            if img_result.details:
                for raw_key, conf in list(img_result.details.items()):
                    # Tìm key mapping ra disease_name nay
                    from app.constants import CLASS_INFO
                    if CLASS_INFO.get(raw_key, {}).get("display") == cv_best_class:
                        img_result.details[raw_key] = round(final_score, 3)
                        
            return img_result
        else:
            # Hai model đoán khác nhau -> Lấy kết quả cao nhất
            if cv_best_score >= nlp_best_score:
                return img_result
            else:
                return nlp_result

    # Trường hợp chỉ có 1 trong 2 được truyền lên
    if img_result and not nlp_result:
        return img_result
        
    if nlp_result and not img_result:
        # User yêu cầu điều chỉnh lời khuyên nếu chỉ dùng nháp (NLP) < 50%
        if nlp_result.confidence > 0.50:
            nlp_result.description = f"Hệ thống nhận thấy bạn có dấu hiệu của bệnh {nlp_result.disease_name}. " + nlp_result.description
        elif nlp_result.confidence >= 0.40:
            nlp_result.description = f"Mô tả của bạn vẫn chưa rõ ràng, tôi nhận thấy dấu hiệu này có thể liên quan đến bệnh {nlp_result.disease_name}, nhưng để chính xác hơn, bạn hãy cung cấp các hình ảnh về vùng da bị bệnh để nhận kết quả cụ thể hơn. "
            nlp_result.recommendations.insert(0, "Cần tải lên hình ảnh vùng da bị tổn thương")
        else:
            nlp_result.description = "Câu hỏi hoặc mô tả của bạn có vẻ không chứa các thông tin triệu chứng da liễu rõ ràng. Vui lòng mô tả chi tiết hơn (ví dụ: biểu hiện ở đâu, có ngứa hay đau rát không, xuất hiện từ bao giờ...) hoặc đính kèm hình ảnh để DARA AI có thể hỗ trợ tốt nhất nhé!"
            nlp_result.recommendations = ["Mô tả chi tiết hơn về các triệu chứng", "Đính kèm hình ảnh vùng da bị bệnh"]
            
        return nlp_result
        
    return img_result or nlp_result
