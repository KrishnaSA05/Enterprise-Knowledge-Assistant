"""
evaluation/eval_data.py
-----------------------
Ground truth dataset — section names verified against actual FAISS index.

All relevant_sections values are confirmed to exist in the index.
Run `python test.py` to recheck if index is ever rebuilt.
"""

EVAL_QUESTIONS = [

    # ── Category 1: Standard RAG ─────────────────────────────────────────────

    {
        "question": "What benefits does the company offer to employees?",
        "relevant_sections": [
            "Employee Profit Sharing",
            "Health Insurance",
            "Life Insurance"
        ],
        "expected_keywords": ["insurance", "profit", "benefits"],
        "query_type": "retrieve"
    },
    {
        "question": "What is the company's code of conduct?",
        "relevant_sections": [
            "Not OK",
            "OK",
            "Some Definitions & Resources"
        ],
        "expected_keywords": ["conduct", "behavior", "acceptable"],
        "query_type": "retrieve"
    },
    {
        "question": "How does the company support career growth?",
        "relevant_sections": [
            "Mastery & Titles",
            "Pay & Promotions",
            "Individual Contributor Expectations"
        ],
        "expected_keywords": ["career", "growth", "promotion"],
        "query_type": "retrieve"
    },
    {
        "question": "What is the vacation policy at 37signals?",
        "relevant_sections": [
            "Paid Time Off",
            "Scheduling Time Off"
        ],
        "expected_keywords": ["vacation", "days", "time off"],
        "query_type": "retrieve"
    },
    {
        "question": "How does the company handle sick leave?",
        "relevant_sections": [
            "Paid Sick Time"
        ],
        "expected_keywords": ["sick", "leave", "days"],
        "query_type": "retrieve"
    },

    # ── Category 2: Vocabulary Mismatch ──────────────────────────────────────
    # "Not OK" section contains the moonlighting rule —
    # confirmed from evaluation logs (Q7 retrieved "Not OK" with score 6.20)

    {
        "question": "Are employees allowed to moonlight?",
        "relevant_sections": ["Not OK"],
        "expected_keywords": ["moonlight", "another company", "full time"],
        "query_type": "retrieve"
    },
    {
        "question": "Can we work in another company while working at 37signals?",
        "relevant_sections": ["Not OK"],
        "expected_keywords": ["moonlight", "another company", "full time"],
        "query_type": "retrieve"
    },
    {
        "question": "Does 37signals allow outside employment?",
        "relevant_sections": ["Not OK"],
        "expected_keywords": ["moonlight", "another company"],
        "query_type": "retrieve"
    },

    # ── Category 3: Multi-section ─────────────────────────────────────────────

    {
        "question": "What financial benefits does 37signals provide?",
        "relevant_sections": [
            "Employee Profit Sharing",
            "Retirement Plan",
            "Life Insurance"
        ],
        "expected_keywords": ["profit", "retirement", "insurance"],
        "query_type": "retrieve"
    },
    {
        "question": "What health and wellness support does the company offer?",
        "relevant_sections": [
            "Health Insurance",
            "Dental Insurance",
            "Vision Insurance"
        ],
        "expected_keywords": ["health", "dental", "vision"],
        "query_type": "retrieve"
    },

    # ── Category 4: Out-of-Scope ──────────────────────────────────────────────

    {
        "question": "What is the capital of France?",
        "relevant_sections": [],
        "expected_keywords": ["don't know", "not found", "documents"],
        "query_type": "retrieve"
    },
    {
        "question": "What is the current stock price of Basecamp?",
        "relevant_sections": [],
        "expected_keywords": ["don't know", "not found", "documents"],
        "query_type": "retrieve"
    },
    {
        "question": "What is Python programming language?",
        "relevant_sections": [],
        "expected_keywords": ["don't know", "not found", "documents"],
        "query_type": "retrieve"
    },

    # ── Category 5: Direct / Chitchat ─────────────────────────────────────────

    {
        "question": "Hi there!",
        "relevant_sections": [],
        "expected_keywords": ["help", "assist", "policy"],
        "query_type": "direct"
    },
    {
        "question": "Thanks for your help!",
        "relevant_sections": [],
        "expected_keywords": [],
        "query_type": "direct"
    },
]