# rag_evaluator.py
import re
def simple_retrieval_score(question, retrieved_chunks):
    """
    Basic retrieval quality:
    How much of the question's important words appear in the retrieved chunks?
    """
    question_keywords = set(question.lower().split())
    context_text = " ".join(retrieved_chunks).lower()

    # Only count non-trivial words (ignore very short words like "is", "the")
    filtered_keywords = {word for word in question_keywords if len(word) > 2}

    match_count = sum(1 for word in filtered_keywords if word in context_text)
    score = match_count / len(filtered_keywords) if filtered_keywords else 0.0

    return round(score, 4)


import re

def simple_faithfulness_score(answer, retrieved_chunks):
    """
    Improved faithfulness score:
    Checks if unique, meaningful keywords and numeric values from the answer appear in the retrieved chunks.
    """

    # Helper to normalize and clean text
    def normalize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9.]+", " ", text)  # keep letters, digits, and dots
        return text.strip()

    normalized_answer = normalize(answer)
    normalized_context = normalize(" ".join(retrieved_chunks))

    # Extract keywords (filter out short words)
    answer_words = set(word for word in normalized_answer.split() if len(word) > 2)

    # Extract floating point numbers (e.g. 3.52)
    answer_numbers = set(re.findall(r"\b\d+\.\d+\b", normalized_answer))

    # Union of all unique terms
    all_terms = answer_words.union(answer_numbers)

    # Count how many unique terms appear in the context
    matched_terms = {term for term in all_terms if term in normalized_context}

    score = len(matched_terms) / len(all_terms) if all_terms else 0.0
    return round(min(score, 1.0), 4)  
