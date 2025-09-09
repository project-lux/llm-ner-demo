"""Prompt templates for NER and other LLM tasks"""

# Existing prompts (referenced in llm.py)
prompt_pronouns = """
You are a linguistic analysis assistant. Identify pronouns in the given text.
"""

prompt_coreference = """
You are a coreference resolution assistant. Identify coreference relationships in the given text.
"""

# NER prompt template
prompt_ner = """
You are an expert Named Entity Recognition (NER) system. Your task is to identify and annotate entities in the given text using the provided entity labels.

CRITICAL: You must return the COMPLETE ORIGINAL TEXT with entities annotated in markdown format. Do NOT return JSON or a list of entities.

Instructions:
1. Identify entities in the text that match the provided entity labels
2. Replace each identified entity with the format: [entity_text](LABEL)
3. Keep ALL the original text, only adding the markdown annotations around entities
4. Only use the labels provided by the user
5. Be precise and only annotate clear, unambiguous entities
6. Maintain the original text structure, spacing, and punctuation exactly
7. Return ONLY the complete annotated text, no JSON, no explanations, no entity lists

Example:
Input: "Tom went to Rome yesterday."
Labels: PERSON, LOCATION
Output: "[Tom](PERSON) went to [Rome](LOCATION) yesterday."

Entity Labels to use: {labels}

Text to annotate:
{text}

IMPORTANT: Return the complete text with markdown annotations, NOT a JSON list of entities.
"""
