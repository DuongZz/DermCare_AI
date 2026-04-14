from pydantic import BaseModel
from typing import Optional, List


class DiagnosisResponse(BaseModel):
    """Kết quả trả về sau khi AI phân tích ảnh da"""
    disease_name: str
    specialization: str
    confidence: float
    description: str
    recommendations: List[str]
    should_see_doctor: bool
    severity: str
    processed_image: Optional[str] = None
    details: Optional[dict] = None


class ChatMessage(BaseModel):
    """Tin nhắn cho AI chat"""
    content: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Phản hồi từ AI chat"""
    message: str
    diagnosis: Optional[DiagnosisResponse] = None
    conversation_id: Optional[str] = None
