import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    chat_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    embed_model: str = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))
    # Thời gian nghỉ (giây) giữa hai lời gọi LLM khi xây đồ thị tri thức
    request_delay: float = float(os.getenv("REQUEST_DELAY", "4"))

    data_dir: Path = Path(os.getenv("DATA_DIR", ROOT / "data"))
    storage_dir: Path = Path(os.getenv("STORAGE_DIR", ROOT / "storage"))

    # Độ dài tối đa của một đoạn và của phần chồng lấn, tính theo ký tự
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    top_k: int = int(os.getenv("TOP_K", "3"))
    # Số bước duyệt tối đa tính từ thực thể gốc khi lấy đồ thị con
    kg_hops: int = int(os.getenv("KG_HOPS", "3"))
    kg_max_triples: int = int(os.getenv("KG_MAX_TRIPLES", "30"))
    # Ngưỡng độ tương đồng cosine để liên kết thực thể trong câu hỏi với đỉnh của đồ thị
    entity_match_threshold: float = float(os.getenv("ENTITY_MATCH_THRESHOLD", "0.75"))

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def kg_path(self) -> Path:
        return self.storage_dir / "kg.json"
