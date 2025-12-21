from typing import List

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    true_positives = sum(1 for r in retrieved_k if r in relevant_set)
    return true_positives / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)

    true_positives = sum(1 for r in retrieved_k if r in relevant_set)
    return true_positives / len(relevant_set)
