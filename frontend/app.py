"""
Arabic Financial RAG System - Streamlit Frontend
Phase 3: Simple chat interface with citations

Features:
- RTL Arabic layout
- Chat message bubbles
- Citations panel
- FastAPI backend integration
"""

import streamlit as st
import httpx
from typing import Dict, List, Any
import time

# Page configuration
st.set_page_config(
    page_title="المحلل المالي | Financial Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for RTL and Arabic styling
st.markdown("""
<style>
    /* Import Arabic font */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    
    /* Global RTL support */
    .main {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif;
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        color: #1e3a5f;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Cairo', sans-serif;
    }
    
    .subtitle {
        text-align: center;
        color: #5f7d9a;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-family: 'Cairo', sans-serif;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        font-family: 'Cairo', sans-serif;
    }
    
    .assistant-message {
        background: #f8f9fa;
        color: #1e3a5f;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        font-family: 'Cairo', sans-serif;
        line-height: 1.8;
    }
    
    /* Citations */
    .citation-box {
        background: #fff9e6;
        border: 1px solid #f0e68c;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-family: 'Cairo', sans-serif;
    }
    
    .citation-title {
        font-weight: 600;
        color: #856404;
        margin-bottom: 0.5rem;
    }
    
    .citation-page {
        color: #856404;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .citation-text {
        color: #555;
        font-size: 0.9rem;
        margin-top: 0.3rem;
        direction: rtl;
    }
    
    /* Input box */
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif;
        font-size: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Cairo', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Error message */
    .error-message {
        background: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #c62828;
        margin: 1rem 0;
        font-family: 'Cairo', sans-serif;
    }
    
    /* Loading */
    .loading-text {
        text-align: center;
        color: #667eea;
        font-size: 1rem;
        font-family: 'Cairo', sans-serif;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"

def call_rag_api(question: str) -> Dict[str, Any]:
    """
    Call the FastAPI backend /ask endpoint.
    
    Args:
        question: User question in Arabic
        
    Returns:
        Response dictionary with answer and citations
    """
    try:
        response = httpx.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},
            timeout=30.0
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json()
            }
        else:
            return {
                "success": False,
                "error": f"خطأ في الخادم: {response.status_code}"
            }
    except httpx.ConnectError:
        return {
            "success": False,
            "error": "تعذر الاتصال بالخادم. تأكد من تشغيل API على المنفذ 8000."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"حدث خطأ: {str(e)}"
        }

def display_message(message: str, is_user: bool = False):
    """Display a chat message."""
    css_class = "user-message" if is_user else "assistant-message"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)

def display_citations(citations: List[Dict[str, Any]]):
    """Display citations panel."""
    if not citations:
        return
    
    st.markdown('<div class="citation-title">📚 المراجع:</div>', unsafe_allow_html=True)
    
    for citation in citations:
        page = citation['page']
        text = citation['text']
        
        st.markdown(f"""
        <div class="citation-box">
            <span class="citation-page">صفحة {page}</span>
            <div class="citation-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-title">📊 المحلل المالي</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">نظام ذكي للإجابة على الأسئلة المالية باللغة العربية</p>', unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            display_message(message["content"], is_user=True)
        else:
            display_message(message["content"], is_user=False)
            if "citations" in message and message["citations"]:
                display_citations(message["citations"])
    
    # Input section
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_question = st.text_input(
            "اكتب سؤالك هنا",
            key="question_input",
            placeholder="مثال: ما هي الأصول في ديسمبر ٢٠٢٤؟",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.button("إرسال", use_container_width=True)
    
    # Handle submission
    if submit_button and user_question.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })
        
        # Display user message immediately
        display_message(user_question, is_user=True)
        
        # Show loading
        with st.spinner():
            st.markdown('<p class="loading-text">⏳ جاري البحث وتحليل البيانات...</p>', unsafe_allow_html=True)
            
            # Call API
            result = call_rag_api(user_question)
        
        if result["success"]:
            data = result["data"]
            answer = data["answer"]
            citations = data.get("citations", [])
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })
            
            # Display answer and citations
            display_message(answer, is_user=False)
            if citations:
                display_citations(citations)
        else:
            # Display error
            error_msg = result["error"]
            st.markdown(f'<div class="error-message">⚠️ {error_msg}</div>', unsafe_allow_html=True)
        
        # Rerun to update UI
        st.rerun()
    
    # Footer info
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem; font-family: 'Cairo', sans-serif;">
        💡 يستخدم هذا النظام الذكاء الاصطناعي لتحليل المستندات المالية. جميع الإجابات مدعومة بمراجع من المستندات الأصلية.
    </div>
    """, unsafe_allow_html=True)
    
    # Clear button in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ الإعدادات")
        if st.button("🗑️ مسح المحادثة"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 معلومات")
        st.markdown(f"**عدد الرسائل:** {len(st.session_state.messages)}")
        st.markdown(f"**حالة API:** {'🟢 متصل' if True else '🔴 غير متصل'}")

if __name__ == "__main__":
    main()
