from typing import List, Dict
import os
from dotenv import load_dotenv
from groq import Groq

from generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()
class Generator:
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant"
    ):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def build_context(self, chunks: List[Dict]) -> str:
        """
        Build grounded context with explicit source attribution.
        """
        context_blocks = []

        for chunk in chunks:
            section = chunk["metadata"].get("section", "Unknown")
            source = chunk["metadata"].get("source_file", "Unknown")
            text = chunk["metadata"].get("text", "")

            block = (
                f"[Source: {source} | Section: {section}]\n{text}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        context = self.build_context(retrieved_chunks)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,   # low = factual
            max_tokens=512
        )

        return response.choices[0].message.content.strip()
