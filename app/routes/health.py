from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """Kiểm tra server có hoạt động không"""
    return {
        "status": "ok",
        "message": "DermCare AI Server đang hoạt động",
        "timestamp": datetime.utcnow().isoformat(),
    }
