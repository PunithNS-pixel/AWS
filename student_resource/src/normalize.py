import re
import unicodedata

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Normalize Unicode characters (NFC form)
    text = unicodedata.normalize('NFC', text)
    
    # 3. Remove punctuation but KEEP letters, numbers, and combining marks (for Indic scripts)
    # \p{M} matches combining marks (vowels/viramas in Indic scripts)
    # \w matches letters and numbers
    cleaned_chars = []
    for char in text:
        category = unicodedata.category(char)
        # Keep letters (L), numbers (N), combining marks (M), and spaces (Z)
        if category.startswith(('L', 'N', 'M', 'Z')):
            cleaned_chars.append(char)
            
    text = "".join(cleaned_chars)
    
    # 4. Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- Test the function ---
if __name__ == "__main__":
    sample_text = "Example Business, Inc. (भारत)!"
    print(f"Original: {sample_text}")
    print(f"Normalized: {normalize_text(sample_text)}")