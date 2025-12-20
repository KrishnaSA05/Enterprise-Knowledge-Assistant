from pathlib import Path
from typing import List, Dict

def load_markdown_documents(data_dir: str) -> List[Dict]:
    """
    Load markdown files and attach basic metadata.
    """
    documents = []
    data_path = Path(data_dir)

    for md_file in data_path.rglob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()

        document = {
            "text": text,
            "metadata": {
                "source_file": md_file.name,
                "department": md_file.parent.name,
                "path": str(md_file)
            }
        }

        documents.append(document)

    return documents


if __name__ == "__main__":
    docs = load_markdown_documents("data")
    print(f"Loaded {len(docs)} documents")
