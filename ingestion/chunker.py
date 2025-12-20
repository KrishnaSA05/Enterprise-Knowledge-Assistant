import re
from typing import List, Dict


def split_by_markdown_headers(text: str) -> List[Dict]:
    """
    Split markdown text by headers and keep section titles.
    """
    pattern = r"(##+ .*)"
    parts = re.split(pattern, text)

    chunks = []
    current_section = "Introduction"
    current_text = ""

    for part in parts:
        if part.startswith("##"):
            if current_text.strip():
                chunks.append({
                    "section": current_section,
                    "content": current_text.strip()
                })
            current_section = part.replace("#", "").strip()
            current_text = ""
        else:
            current_text += part

    if current_text.strip():
        chunks.append({
            "section": current_section,
            "content": current_text.strip()
        })

    return chunks


def chunk_documents(
    documents: List[Dict],
    max_tokens: int = 500
) -> List[Dict]:
    """
    Create section-aware chunks with metadata.
    """
    all_chunks = []

    for doc in documents:
        sections = split_by_markdown_headers(doc["text"])

        for sec in sections:
            words = sec["content"].split()
            for i in range(0, len(words), max_tokens):
                chunk_text = " ".join(words[i:i + max_tokens])

                chunk = {
                    "text": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "section": sec["section"]
                    }
                }
                all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    from loader import load_markdown_documents

    docs = load_markdown_documents("data")
    chunks = chunk_documents(docs)

    print(f"Generated {len(chunks)} chunks")
    print(chunks[0])
