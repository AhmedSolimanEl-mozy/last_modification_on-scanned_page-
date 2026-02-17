"""
Gemini API client for document intelligence tasks.

Handles:
- Table understanding
- Noise filtering
- Sentence-row alignment validation
"""

import os
import time
from typing import List, Dict, Optional
import google.generativeai as genai


class GeminiClient:
    """Client for interacting with Gemini 1.5 Flash API."""
    
    def __init__(self):
        """Initialize Gemini client with API key from environment."""
        self.api_key = os.environ.get('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set it before running the pipeline."
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.request_count = 0
        self.total_tokens = 0
    
    def validate_alignment(
        self,
        sentence: str,
        row_data: Dict[str, str],
        context: Optional[str] = None
    ) -> Dict:
        """
        Validate if a sentence aligns with a table row's numeric data.
        
        Args:
            sentence: The sentence text
            row_data: Dictionary of column_name -> value
            context: Optional context (e.g., table headers)
        
        Returns:
            {
                'confidence': 'high' | 'medium' | 'low',
                'reasoning': str (optional explanation)
            }
        """
        prompt = self._build_alignment_prompt(sentence, row_data, context)
        
        try:
            response = self.model.generate_content(prompt)
            self.request_count += 1
            
            result_text = response.text.strip().lower()
            
            # Parse confidence level
            if 'high' in result_text:
                confidence = 'high'
            elif 'medium' in result_text:
                confidence = 'medium'
            elif 'low' in result_text:
                confidence = 'low'
            else:
                confidence = 'unknown'
            
            return {
                'confidence': confidence,
                'reasoning': response.text.strip()
            }
        
        except Exception as e:
            print(f"  [Gemini] Warning: API call failed: {e}")
            return {
                'confidence': 'unknown',
                'reasoning': f'API error: {str(e)}'
            }
    
    def validate_alignments_batch(
        self,
        alignments: List[Dict],
        batch_size: int = 10
    ) -> List[Dict]:
        """
        Validate multiple sentence-row alignments in batches.
        
        Args:
            alignments: List of {'sentence': str, 'row_data': dict}
            batch_size: Number of alignments per batch
        
        Returns:
            List of validation results
        """
        results = []
        
        for i in range(0, len(alignments), batch_size):
            batch = alignments[i:i + batch_size]
            
            print(f"  [Gemini] Validating batch {i//batch_size + 1} "
                  f"({len(batch)} alignments)...")
            
            for alignment in batch:
                result = self.validate_alignment(
                    alignment['sentence'],
                    alignment['row_data'],
                    alignment.get('context')
                )
                results.append(result)
                
                # Simple rate limiting (avoid overwhelming API)
                time.sleep(0.5)
        
        return results
    
    def identify_noise_regions(self, image_description: str) -> Dict:
        """
        Identify if a region contains noise (stamps, logos, signatures).
        
        Args:
            image_description: Description of the region
        
        Returns:
            {
                'is_noise': bool,
                'noise_type': 'stamp' | 'logo' | 'signature' | 'handwriting' | None
            }
        """
        prompt = f"""
You are analyzing a region from an Arabic financial document PDF.

Region description: {image_description}

Is this region noise that should be excluded from text extraction?
Noise includes: stamps, logos, signatures, handwritten notes.

Answer with JSON format:
{{
    "is_noise": true/false,
    "noise_type": "stamp" | "logo" | "signature" | "handwriting" | null
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            self.request_count += 1
            
            # Simple parsing (in production, use proper JSON parsing)
            result_text = response.text.strip().lower()
            
            is_noise = 'true' in result_text or 'noise' in result_text
            
            noise_type = None
            if 'stamp' in result_text:
                noise_type = 'stamp'
            elif 'logo' in result_text:
                noise_type = 'logo'
            elif 'signature' in result_text:
                noise_type = 'signature'
            elif 'handwriting' in result_text or 'handwritten' in result_text:
                noise_type = 'handwriting'
            
            return {
                'is_noise': is_noise,
                'noise_type': noise_type
            }
        
        except Exception as e:
            print(f"  [Gemini] Warning: API call failed: {e}")
            return {
                'is_noise': False,
                'noise_type': None
            }
    
    def _build_alignment_prompt(
        self,
        sentence: str,
        row_data: Dict[str, str],
        context: Optional[str] = None
    ) -> str:
        """Build prompt for alignment validation."""
        row_str = ', '.join(f"{k}: {v}" for k, v in row_data.items())
        
        prompt = f"""
You are validating sentence-to-table-row alignment for accurate citation in a financial document.

Sentence: "{sentence}"

Table Row Data: {row_str}
"""
        
        if context:
            prompt += f"\nContext: {context}"
        
        prompt += """

Question: Does this sentence describe or reference the numeric values in this table row?

Answer with ONLY one word: high, medium, or low

Confidence:"""
        
        return prompt
    
    def get_usage_stats(self) -> Dict:
        """Get API usage statistics."""
        return {
            'total_requests': self.request_count,
            'total_tokens': self.total_tokens
        }
