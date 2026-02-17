"""
FastAPI Application - Phase 2
Arabic Financial RAG System Backend

Implements /ask endpoint with dual retrieval and LLM integration.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import QuestionRequest, AnswerResponse, ErrorResponse
from api.retrieval import DualRetriever
from api.llm_client import FinancialAnalystLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
retriever = None
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global retriever, llm
    
    # Startup
    logger.info("🚀 Starting Arabic Financial RAG API...")
    try:
        retriever = DualRetriever()
        llm = FinancialAnalystLLM()
        logger.info("✓ All components initialized successfully")
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if retriever:
        retriever.close()
    logger.info("✓ Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Arabic Financial RAG API",
    description="Backend API for Arabic financial document Q&A with citations",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Arabic Financial RAG API",
        "version": "2.0.0",
        "endpoints": {
            "ask": "/ask (POST)",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "retriever": "ready" if retriever else "not initialized",
        "llm": "ready" if llm else "not initialized"
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about Arabic financial documents.
    
    **Request Body:**
    ```json
    {
        "question": "ما هي الأصول في ٢٠٢٤؟"
    }
    ```
    
    **Response:**
    ```json
    {
        "answer": "إجمالي الأصول في ٣١ ديسمبر ٢٠٢٤ بلغ ٨٬١٣٧٬٣٩٤ مليون جنيه (صفحة ٣)",
        "citations": [
            {
                "page": 3,
                "text": "إجمالي الأصول..."
            }
        ]
    }
    ```
    """
    try:
        logger.info(f"Received question: {request.question}")
        
        # Step 1: Retrieve context
        context, retrieved_units = retriever.retrieve(request.question)
        
        if not retrieved_units:
            logger.warning("No relevant units found")
            return AnswerResponse(
                answer="عذراً، لم أجد معلومات ذات صلة في المستندات المتاحة.",
                citations=[],
                retrieval_info={
                    "units_retrieved": 0,
                    "context_length": 0
                }
            )
        
        # Step 2: Generate answer with LLM
        answer, citations = llm.generate_answer(
            question=request.question,
            context=context,
            retrieved_units=retrieved_units
        )
        
        logger.info(f"Generated answer with {len(citations)} citations")
        
        # Step 3: Return response
        return AnswerResponse(
            answer=answer,
            citations=citations,
            retrieval_info={
                "units_retrieved": len(retrieved_units),
                "context_length": len(context),
                "pages_referenced": list(set(u['page'] for u in retrieved_units))
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"خطأ في البيانات المدخلة: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="حدث خطأ في معالجة السؤال. الرجاء المحاولة مرة أخرى."
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Custom exception handler for general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "حدث خطأ داخلي. الرجاء المحاولة مرة أخرى.",
            "detail": str(exc) if app.debug else None
        }
    )


if __name__ == "__main__":
    import uvicorn
    from api.config import settings
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level="info"
    )
