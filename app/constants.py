from typing import List, Dict, Any
from app.schemas import DiagnosisResponse

CLASS_INFO = {
    "bong_nuoc": {
        "display": "Phồng nước",
        "specialization": "Nhiễm trùng da & Ký sinh trùng",
        "severity": "moderate",
        "description": "Phồng nước trên da do bỏng, ma sát, hoặc nhiễm virus (Herpes, Thủy đậu).",
        "recommendations": [
            "Không tự chọc vỡ bọng nước",
            "Giữ vùng da sạch và khô thoáng",
            "Đến gặp bác sĩ nếu có dấu hiệu nhiễm trùng (đỏ, mủ, sốt)",
        ],
        "should_see_doctor": True,
    },
    "muncoc": {
        "display": "Mụn cóc",
        "specialization": "Nhiễm trùng da & Ký sinh trùng",
        "severity": "mild",
        "description": "Mụn cóc do virus HPV gây ra, lây qua tiếp xúc trực tiếp.",
        "recommendations": [
            "Không tự cắt hoặc bóc mụn cóc",
            "Giữ tay sạch, tránh chạm vào mụn cóc người khác",
            "Có thể dùng kem Salicylic acid theo chỉ dẫn",
        ],
        "should_see_doctor": False,
    },
    "noimeday": {
        "display": "Nổi mề đay",
        "specialization": "Da liễu Bệnh lý & Miễn dịch",
        "severity": "moderate",
        "description": "Nổi mề đay (Urticaria) do phản ứng dị ứng gây sẩn phù đỏ ngứa.",
        "recommendations": [
            "Xác định và tránh xa tác nhân gây dị ứng",
            "Dùng thuốc kháng histamine theo chỉ định",
            "Đến cấp cứu ngay nếu khó thở hoặc phù mặt",
        ],
        "should_see_doctor": True,
    },
    "ton_thuong_hac_to": {
        "display": "Tổn thương hắc tố",
        "specialization": "U & Ung thư da",
        "severity": "severe",
        "description": "Tổn thương hắc tố bất thường có thể là dấu hiệu của Melanoma. Cần khám ngay.",
        "recommendations": [
            "ĐẾN KHÁM BÁC SĨ DA LIỄU NGAY LẬP TỨC",
            "Không tự ý điều trị tại nhà",
            "Dùng kem chống nắng SPF 50+ và tránh ánh nắng trực tiếp",
        ],
        "should_see_doctor": True,
    },
    "trungca": {
        "display": "Trứng cá (Mụn)",
        "specialization": "Da liễu Thẩm mỹ",
        "severity": "mild",
        "description": "Mụn trứng cá do tắc nghẽn lỗ chân lông, vi khuẩn và bã nhờn.",
        "recommendations": [
            "Rửa mặt 2 lần/ngày bằng sữa rửa mặt dịu nhẹ",
            "Không nặn mụn để tránh để lại sẹo",
            "Dùng sản phẩm non-comedogenic (không gây bít lỗ chân lông)",
        ],
        "should_see_doctor": False,
    },
    "vaynen": {
        "display": "Vảy nến",
        "specialization": "Da liễu Bệnh lý & Miễn dịch",
        "severity": "moderate",
        "description": "Vảy nến (Psoriasis) là bệnh da mãn tính gây các mảng đỏ phủ vảy trắng bạc.",
        "recommendations": [
            "Dưỡng ẩm da thường xuyên",
            "Tránh căng thẳng và chấn thương da",
            "Điều trị theo phác đồ của bác sĩ chuyên khoa",
        ],
        "should_see_doctor": True,
    },
}

def get_mock_diagnosis() -> DiagnosisResponse:
    return DiagnosisResponse(
        disease_name="Trứng cá (Mụn)",
        specialization="Da liễu Thẩm mỹ",
        confidence=0.80,
        description="[Demo] Model chưa sẵn sàng. Dữ liệu mẫu.",
        recommendations=["Rửa mặt sạch 2 lần/ngày", "Không nặn mụn"],
        should_see_doctor=False,
        severity="mild",
    )
