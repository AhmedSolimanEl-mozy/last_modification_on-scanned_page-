#!/usr/bin/env python3
"""
Test Script for Arabic Financial RAG API - Phase 2

Tests the /ask endpoint with various question types:
1. General semantic queries
2. Numeric queries with year filtering
3. Comparison queries
4. Missing data queries
5. Paragraph expansion validation
"""

import json
import time
from typing import List, Dict, Any
from datetime import datetime

import httpx

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "ما هي الاستثمارات المالية للبنك؟",
        "type": "semantic",
        "description": "General question about financial investments"
    },
    {
        "id": 2,
        "question": "كم بلغت الأصول في ٣١ ديسمبر ٢٠٢٤؟",
        "type": "numeric_year",
        "description": "Specific question about 2024 assets"
    },
    {
        "id": 3,
        "question": "قارن بين القروض في ٢٠٢٤ و ٢٠٢٣",
        "type": "comparison",
        "description": "Comparison between two years"
    },
    {
        "id": 4,
        "question": "ما هو رأس المال المدفوع؟",
        "type": "semantic",
        "description": "Question about paid-up capital"
    },
    {
        "id": 5,
        "question": "كم كانت ودائع العملاء في ٢٠٢٤؟",
        "type": "numeric_year",
        "description": "Customer deposits in 2024"
    },
    {
        "id": 6,
        "question": "ما هي الأرباح في ٢٠٢٥؟",
        "type": "missing_data",
        "description": "Question about data not in documents (should say not available)"
    },
    {
        "id": 7,
        "question": "من هم مراقبو الحسابات؟",
        "type": "semantic",
        "description": "Question about auditors"
    }
]


class APITester:
    """Test client for the RAG API."""
    
    def __init__(self, base_url: str):
        """Initialize tester with API base URL."""
        self.base_url = base_url
        self.results = []
    
    def check_api_health(self) -> bool:
        """Check if API is running."""
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=5.0)
            if response.status_code == 200:
                print("✓ API is healthy and ready")
                print(f"  Status: {response.json()}")
                return True
            else:
                print(f"✗ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to API: {e}")
            print(f"  Make sure the API is running at {self.base_url}")
            return False
    
    def test_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a single question.
        
        Args:
            question_data: Question configuration
            
        Returns:
            Test result dictionary
        """
        print(f"\n[Test {question_data['id']}] {question_data['description']}")
        print(f"  السؤال: {question_data['question']}")
        print(f"  Type: {question_data['type']}")
        
        try:
            start_time = time.time()
            
            response = httpx.post(
                f"{self.base_url}/ask",
                json={"question": question_data['question']},
                timeout=30.0
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"  ✓ Success ({elapsed_time:.2f}s)")
                print(f"  الإجابة: {data['answer'][:150]}...")
                print(f"  Citations: {len(data['citations'])} pages")
                
                if data['citations']:
                    for citation in data['citations'][:3]:  # Show first 3
                        print(f"    - صفحة {citation['page']}: {citation['text'][:80]}...")
                
                result = {
                    "question_id": question_data['id'],
                    "question": question_data['question'],
                    "type": question_data['type'],
                    "status": "success",
                    "response_time": elapsed_time,
                    "answer": data['answer'],
                    "citations": data['citations'],
                    "retrieval_info": data.get('retrieval_info', {})
                }
                
                # Validation checks
                validations = []
                
                # Check if answer is in Arabic
                if any('\u0600' <= c <= '\u06FF' for c in data['answer']):
                    validations.append("✓ Answer is in Arabic")
                else:
                    validations.append("✗ Answer is not in Arabic")
                
                # Check for citations
                if data['citations']:
                    validations.append(f"✓ Has {len(data['citations'])} citations")
                else:
                    validations.append("⚠ No citations provided")
                
                # Check for "not available" message if missing data
                if question_data['type'] == 'missing_data':
                    if 'غير موجودة' in data['answer'] or 'غير متوفرة' in data['answer']:
                        validations.append("✓ Correctly stated data not available")
                    else:
                        validations.append("⚠ Should state data not available")
                
                result['validations'] = validations
                
                for v in validations:
                    print(f"  {v}")
                
                return result
                
            else:
                print(f"  ✗ Failed: {response.status_code}")
                print(f"  Error: {response.text}")
                
                return {
                    "question_id": question_data['id'],
                    "question": question_data['question'],
                    "type": question_data['type'],
                    "status": "failed",
                    "error": response.text,
                    "response_time": elapsed_time
                }
                
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            return {
                "question_id": question_data['id'],
                "question": question_data['question'],
                "type": question_data['type'],
                "status": "error",
                "error": str(e),
                "response_time": 0
            }
    
    def run_all_tests(self, questions: List[Dict[str, Any]]):
        """Run all test questions."""
        print("=" * 80)
        print("Arabic Financial RAG API - Test Suite")
        print("=" * 80)
        
        # Health check first
        if not self.check_api_health():
            print("\nAborted: API is not available")
            return
        
        # Run tests
        for question_data in questions:
            result = self.test_question(question_data)
            self.results.append(result)
            time.sleep(1)  # Rate limiting
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r['status'] == 'success')
        failed = sum(1 for r in self.results if r['status'] == 'failed')
        errors = sum(1 for r in self.results if r['status'] == 'error')
        
        print(f"Total tests: {total}")
        print(f"✓ Success: {success}")
        print(f"✗ Failed: {failed}")
        print(f"⚠ Errors: {errors}")
        
        if success > 0:
            avg_time = sum(r.get('response_time', 0) for r in self.results if r['status'] == 'success') / success
            print(f"\nAverage response time: {avg_time:.2f}s")
        
        print()
    
    def save_results(self):
        """Save results to JSON and markdown files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_filename = f"api_test_results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "api_url": self.base_url,
                "total_tests": len(self.results),
                "results": self.results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Results saved to {json_filename}")
        
        # Save Markdown report
        md_filename = f"api_test_report_{timestamp}.md"
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write("# Arabic Financial RAG API - Test Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**API**: {self.base_url}\n\n")
            
            f.write("## Summary\n\n")
            total = len(self.results)
            success = sum(1 for r in self.results if r['status'] == 'success')
            f.write(f"- Total: {total}\n")
            f.write(f"- Success: {success}\n")
            f.write(f"- Failed: {total - success}\n\n")
            
            f.write("## Test Results\n\n")
            for result in self.results:
                f.write(f"### Test {result['question_id']}: {result['type']}\n\n")
                f.write(f"**السؤال**: {result['question']}\n\n")
                f.write(f"**Status**: {result['status']}\n\n")
                
                if result['status'] == 'success':
                    f.write(f"**الإجابة**: {result['answer']}\n\n")
                    
                    if result.get('citations'):
                        f.write("**المراجع**:\n")
                        for cit in result['citations']:
                            f.write(f"- صفحة {cit['page']}: {cit['text']}\n")
                        f.write("\n")
                    
                    if result.get('validations'):
                        f.write("**Validations**:\n")
                        for v in result['validations']:
                            f.write(f"- {v}\n")
                        f.write("\n")
                
                f.write("---\n\n")
        
        print(f"✓ Report saved to {md_filename}")


def main():
    """Main entry point."""
    tester = APITester(API_BASE_URL)
    tester.run_all_tests(TEST_QUESTIONS)


if __name__ == "__main__":
    main()
