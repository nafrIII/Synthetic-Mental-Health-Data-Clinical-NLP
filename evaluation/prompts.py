sentenceCheck_prompt = """
You are an expert linguist evaluating synthetic text data quality.

Analyse each of the following texts and determine whether it is a COMPLETE, well-formed sentence or passage.

A text is considered INCOMPLETE if it:
- Ends abruptly mid-sentence or mid-word
- Starts with a fragment that doesn't form a complete thought
- Contains only metadata, tags, or instructions (not actual content)
- Is garbled or incoherent

A text is considered COMPLETE if it:
- Contains at least one full sentence with a subject and predicate
- Expresses a complete thought, even if informal (social media style is acceptable)
- May contain minor grammar issues but is still understandable

For each text, respond in this exact JSON format:
[
  {{"id": 1, "is_complete": true, "reason": "brief explanation"}},
  {{"id": 2, "is_complete": false, "reason": "brief explanation"}}
]

Here are the texts to evaluate:

{texts}
"""

PII_prompt = """
You are a privacy expert evaluating synthetic text data for personal information leakage.

Analyse each of the following texts and check whether it contains any real-looking personal identifiable information (PII):

1. NAMES: Full names that appear to be real people's names (not generic references like "my friend" or "someone")
2. PHONE NUMBERS: Any sequence that looks like a phone number (e.g., 555-1234, +1-800-123-4567, (123) 456-7890)
3. ADDRESSES: Physical addresses including street names, house numbers, city names with zip codes, etc.

IMPORTANT: Only flag information that looks like REAL, SPECIFIC personal data. Do NOT flag:
- Generic first names used in examples (e.g., "my friend John said...")
- Clearly fictional or placeholder names (e.g. "John Doe")
- City/country names used in general context (e.g., "I live in New York" is fine, but "123 Main St, New York, NY 10001" is PII)

For each text, respond in this exact JSON format:
[
  {{"id": 1, "has_pii": false, "pii_types": [], "pii_details": ""}},
  {{"id": 2, "has_pii": true, "pii_types": ["name", "phone"], "pii_details": "Found full name 'John Smith' and phone number '555-1234'"}}
]

Here are the texts to evaluate:

{texts}
"""