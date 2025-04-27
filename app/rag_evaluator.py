# rag_evaluator.py

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

def simple_faithfulness_score(answer, retrieved_chunks):
    """
    Basic faithfulness quality:
    How much of the answer's key words appear in the retrieved context?
    """
    answer_keywords = set(answer.lower().split())
    context_text = " ".join(retrieved_chunks).lower()

    filtered_keywords = {word for word in answer_keywords if len(word) > 2}

    match_count = sum(1 for word in filtered_keywords if word in context_text)
    score = match_count / len(filtered_keywords) if filtered_keywords else 0.0

    return round(score, 4)
