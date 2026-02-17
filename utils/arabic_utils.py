"""
Arabic utility functions for text normalization and processing.
"""

import re
import unicodedata

# Arabic Unicode ranges
ARABIC_RANGE = range(0x0600, 0x06FF + 1)
ARABIC_PRESENTATION_FORMS_A = range(0xFB50, 0xFDFF + 1)
ARABIC_PRESENTATION_FORMS_B = range(0xFE70, 0xFEFF + 1)

# Diacritics (tashkeel) to remove
ARABIC_DIACRITICS = [
    '\u064B',  # Fathatan
    '\u064C',  # Dammatan
    '\u064D',  # Kasratan
    '\u064E',  # Fatha
    '\u064F',  # Damma
    '\u0650',  # Kasra
    '\u0651',  # Shadda
    '\u0652',  # Sukun
    '\u0653',  # Maddah
    '\u0654',  # Hamza above
    '\u0655',  # Hamza below
    '\u0656',  # Subscript alef
    '\u0657',  # Inverted damma
    '\u0658',  # Mark noon ghunna
    '\u0670',  # Superscript alef
]


def normalize_arabic(text: str, remove_diacritics: bool = True) -> str:
    """
    Apply light Arabic normalization.
    
    Args:
        text: Input Arabic text
        remove_diacritics: Whether to remove Arabic diacritics (tashkeel)
    
    Returns:
        Normalized Arabic text
    """
    if not text:
        return text
    
    normalized = text
    
    # 1. Remove diacritics (tashkeel)
    if remove_diacritics:
        for diacritic in ARABIC_DIACRITICS:
            normalized = normalized.replace(diacritic, '')
    
    # 2. Normalize Alef forms
    # أ (Alef with hamza above) → ا (bare Alef)
    # إ (Alef with hamza below) → ا
    # آ (Alef with madda) → ا
    normalized = normalized.replace('أ', 'ا')
    normalized = normalized.replace('إ', 'ا')
    normalized = normalized.replace('آ', 'ا')
    
    # 3. Normalize Ya
    # ى (Alef maksura) → ي (Ya)
    normalized = normalized.replace('ى', 'ي')
    
    # 4. Normalize Ta marbuta
    # ة (Ta marbuta) → ه (Ha)
    normalized = normalized.replace('ة', 'ه')
    
    # 5. Remove zero-width characters
    normalized = normalized.replace('\u200B', '')  # Zero-width space
    normalized = normalized.replace('\u200C', '')  # Zero-width non-joiner
    normalized = normalized.replace('\u200D', '')  # Zero-width joiner
    
    # 6. Normalize whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized


def is_arabic_text(text: str, threshold: float = 0.3) -> bool:
    """
    Check if text is predominantly Arabic.
    
    Args:
        text: Input text
        threshold: Minimum ratio of Arabic characters (0.0-1.0)
    
    Returns:
        True if text contains >= threshold ratio of Arabic characters
    """
    if not text:
        return False
    
    non_space_chars = [c for c in text if not c.isspace()]
    if not non_space_chars:
        return False
    
    arabic_count = sum(
        1 for c in non_space_chars
        if ord(c) in ARABIC_RANGE or
           ord(c) in ARABIC_PRESENTATION_FORMS_A or
           ord(c) in ARABIC_PRESENTATION_FORMS_B
    )
    
    ratio = arabic_count / len(non_space_chars)
    return ratio >= threshold


def extract_numbers(text: str) -> list:
    """
    Extract numeric values from text (both Arabic-Indic and Western digits).
    
    Returns:
        List of number strings found in text
    """
    # Arabic-Indic digits: ٠١٢٣٤٥٦٧٨٩
    # Western digits: 0123456789
    # Also match common separators: space, comma, dot
    
    arabic_indic_pattern = r'[٠-٩][٠-٩\s,\.]*[٠-٩]|[٠-٩]'
    western_pattern = r'[0-9][0-9\s,\.]*[0-9]|[0-9]'
    
    numbers = []
    
    # Find Arabic-Indic numbers
    for match in re.finditer(arabic_indic_pattern, text):
        numbers.append(match.group())
    
    # Find Western numbers
    for match in re.finditer(western_pattern, text):
        numbers.append(match.group())
    
    return numbers


def clean_extracted_text(text: str) -> str:
    """
    Clean extracted text by removing artifacts and normalizing spacing.
    
    Args:
        text: Raw extracted text
    
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    cleaned = text
    
    # Remove HTML tags if present
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def split_into_sentences(text: str) -> list:
    """
    Split Arabic text into sentences.
    
    Note: For financial documents, often one sentence per line.
    This function handles both newline-based and punctuation-based splitting.
    
    Args:
        text: Input text
    
    Returns:
        List of sentence strings
    """
    if not text:
        return []
    
    # First, try splitting by newlines (common in financial docs)
    lines = text.split('\n')
    sentences = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Further split by sentence-ending punctuation if needed
        # Arabic uses: ، (comma), . (period), ؟ (question mark), ! (exclamation)
        parts = re.split(r'[\.؟!]\s+', line)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    
    return sentences
