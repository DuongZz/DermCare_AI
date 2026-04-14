from fastapi import APIRouter, HTTPException, Query
from app.services.rag_service import rag_service
from pydantic import BaseModel
from typing import List

router = APIRouter()

class KnowledgeRequest(BaseModel):
    question: str

class KnowledgeResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/query", response_model=KnowledgeResponse)
async def query_knowledge_base(request: KnowledgeRequest):
    """
    Truy vấn kiến thức về bệnh da liễu từ hệ thống RAG.
    """
    result = await rag_service.query(request.question)
    return result

@router.post("/rebuild-index")
async def rebuild_index():
    """
    Cập nhật lại vector index từ các file .txt mới.
    """
    try:
        rag_service.rebuild_index()
        return {"message": "Đã cập nhật lại vector index thành công."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
