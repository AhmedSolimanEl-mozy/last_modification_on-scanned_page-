"""
Pydantic models for API request and response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for /ask endpoint."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="السؤال بالعربية",
        examples=["ما هي الأصول في ٢٠٢٤؟"]
    )


class Citation(BaseModel):
    """Citation model with page and text."""
    page: int = Field(..., description="رقم الصفحة")
    text: str = Field(..., description="النص المقتبس")


class AnswerResponse(BaseModel):
    """Response model for /ask endpoint."""
    answer: str = Field(..., description="الإجابة بالعربية")
    citations: List[Citation] = Field(
        default_factory=list,
        description="المراجع من المستندات"
    )
    retrieval_info: Optional[dict] = Field(
        None,
        description="معلومات إضافية عن عملية البحث (للتطوير)"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="نوع الخطأ")
    message: str = Field(..., description="تفاصيل الخطأ")
    detail: Optional[str] = Field(None, description="تفاصيل فنية إضافية")
