"""
LLM Client for Groq API Integration

Handles communication with Groq API using deepseek-r1-distill-llama-70b model.
Implements financial analyst persona with strict citation requirements.
"""

import logging
import re
from typing import List, Dict, Any, Tuple

from groq import Groq

from api.config import settings
from api.models import Citation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialAnalystLLM:
    """LLM client with financial analyst persona."""
    
    def __init__(self):
        """Initialize Groq client."""
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
        logger.info(f"✓ Initialized Groq client with {self.model}")
    
    def _build_system_prompt(self) -> str:
        """
Build system prompt for financial analyst persona.
        
        Returns:
            System prompt in Arabic
        """
        return """أنت محلل مالي محترف متخصص في تحليل القوائم المالية للبنوك والمؤسسات المالية المصرية.

قواعد صارمة يجب الالتزام بها:
١. استخدم فقط المعلومات الموجودة في السياق المقدم أدناه
٢. لا تخترع أو تتوقع أي أرقام أو بيانات غير موجودة
٣. إذا كانت المعلومة المطلوبة غير موجودة في السياق، قل بوضوح: "المعلومة غير موجودة في المستندات المتاحة"
٤. اذكر دائماً رقم الصفحة كمرجع عند الاستشهاد بالمعلومات
٥. استخدم لغة عربية رسمية ودقيقة
٦. كن موجزاً ومباشراً في الإجابة
٧. عند ذكر الأرقام، احتفظ بها كما هي في السياق (بالأرقام العربية أو الغربية)

تذكر: مصداقيتك تعتمد على دقة معلوماتك وذكر المراجع الصحيحة."""

    def _build_user_prompt(self, context: str, question: str) -> str:
        """
        Build user prompt with context and question.
        
        Args:
            context: Retrieved context from database
            question: User question
            
        Returns:
            Formatted user prompt
        """
        return f"""السياق من المستندات المالية:

{context}

────────────────────────────────────

السؤال: {question}

يرجى الإجابة على السؤال بناءً على السياق أعلاه فقط، مع ذكر رقم الصفحة كمرجع."""

    def generate_answer(
        self, 
        question: str, 
        context: str, 
        retrieved_units: List[Dict[str, Any]]
    ) -> Tuple[str, List[Citation]]:
        """
        Generate answer using Groq API.
        
        Args:
            question: User question
            context: Retrieved context
            retrieved_units: List of units used for context
            
        Returns:
            Tuple of (answer_text, citations_list)
        """
        try:
            # Build prompts
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(context, question)
            
            logger.info(f"Calling Groq API with {self.model}")
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.groq_temperature,
                max_tokens=settings.groq_max_tokens,
            )
            
            answer = response.choices[0].message.content
            logger.info("✓ Received response from Groq")
            
            # Extract citations
            citations = self._extract_citations(answer, retrieved_units)
            
            return answer, citations
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            # Return fallback error message in Arabic
            error_answer = f"عذراً، حدث خطأ في معالجة السؤال. الرجاء المحاولة مرة أخرى."
            return error_answer, []
    
    def _extract_citations(
        self, 
        answer: str, 
        retrieved_units: List[Dict[str, Any]]
    ) -> List[Citation]:
        """
        Extract citations from answer and retrieved units.
        
        Args:
            answer: LLM generated answer
            retrieved_units: Units used for context
            
        Returns:
            List of Citation objects
        """
        citations = []
        seen_pages = set()
        
        # Extract page numbers mentioned in answer (both Arabic and Western numerals)
        page_patterns = [
            r'صفحة\s*[٠-٩]+',  # Arabic: صفحة ٣
            r'صفحة\s*\d+',     # Mixed: صفحة 3
            r'\[صفحة\s*[٠-٩]+\]',  # Bracketed Arabic
            r'\[صفحة\s*\d+\]',     # Bracketed Western
        ]
        
        mentioned_pages = set()
        for pattern in page_patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                # Extract the number
                page_num_str = re.search(r'[٠-٩\d]+', match).group()
                # Convert Arabic numerals to Western
                page_num = self._arabic_to_western(page_num_str)
                mentioned_pages.add(page_num)
        
        # If no pages mentioned explicitly, use all pages from retrieved units
        if not mentioned_pages:
            mentioned_pages = set(unit['page'] for unit in retrieved_units)
        
        # Build citations from retrieved units for mentioned pages
        for page in sorted(mentioned_pages):
            if page in seen_pages:
                continue
            seen_pages.add(page)
            
            # Find units from this page
            page_units = [u for u in retrieved_units if u['page'] == page]
            if not page_units:
                continue
            
            # Take most relevant text (highest similarity or first with numeric data)
            best_unit = max(
                page_units, 
                key=lambda u: (
                    1.0 if u.get('numeric_data') else 0.0,
                    u.get('similarity', 0.0)
                )
            )
            
            citations.append(
                Citation(
                    page=page,
                    text=best_unit['raw_text'][:200]  # Limit citation text length
                )
            )
        
        logger.info(f"Extracted {len(citations)} citations")
        return citations
    
    def _arabic_to_western(self, arabic_num: str) -> int:
        """
        Convert Arabic numerals to Western numerals.
        
        Args:
            arabic_num: String with Arabic numerals (٠-٩) or Western (0-9)
            
        Returns:
            Integer value
        """
        arabic_map = {
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }
        
        western = ''.join(arabic_map.get(c, c) for c in arabic_num)
        return int(western)
