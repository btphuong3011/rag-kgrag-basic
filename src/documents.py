import re
from pathlib import Path


def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                docs.append({"source": path.name, "text": text})
    return docs


def split_sentences(text: str) -> list[str]:
    # Tách câu tại khoảng trắng đứng sau dấu . ! ? hoặc tại ký tự xuống dòng
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    sentences = split_sentences(text)
    chunks, current = [], []
    for sent in sentences:
        if current and len(" ".join(current + [sent])) > chunk_size:
            chunks.append(" ".join(current))
            # Các câu cuối của đoạn vừa tạo (tổng độ dài không vượt quá overlap)
            # được giữ lại làm phần đầu của đoạn kế tiếp
            tail, length = [], 0
            for s in reversed(current):
                if length + len(s) > overlap:
                    break
                tail.insert(0, s)
                length += len(s)
            current = tail
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def make_chunks(docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    out = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            out.append({"id": f"{doc['source']}#{i}", "source": doc["source"], "text": piece})
    return out
