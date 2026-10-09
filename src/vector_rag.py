import chromadb

from .config import Config
from .prompts import ANSWER_SYSTEM, VECTOR_RAG_PROMPT

COLLECTION = "chunks"


class VectorRAG:
    def __init__(self, llm, cfg: Config, client=None):
        self.llm = llm
        self.cfg = cfg
        self.client = client or chromadb.PersistentClient(path=str(cfg.chroma_dir))

    def _collection(self):
        # Embedding do LLM tính sẵn nên tắt hàm embedding mặc định của Chroma;
        # chỉ mục HNSW sử dụng khoảng cách cosine
        return self.client.get_or_create_collection(
            name=COLLECTION, embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}})

    def build_index(self, chunks: list[dict]) -> int:
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = self._collection()
        vectors = self.llm.embed([c["text"] for c in chunks], task_type="RETRIEVAL_DOCUMENT")
        col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=vectors,
            metadatas=[{"source": c["source"]} for c in chunks],
        )
        return col.count()

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        k = k or self.cfg.top_k
        q_vec = self.llm.embed([question], task_type="RETRIEVAL_QUERY")[0]
        res = self._collection().query(query_embeddings=[q_vec], n_results=k)
        hits = []
        # Chroma trả về khoảng cách cosine; độ tương đồng cosine = 1 - khoảng cách
        for cid, text, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
            hits.append({"id": cid, "text": text, "score": round(1 - dist, 4)})
        return hits

    def answer(self, question: str) -> dict:
        hits = self.retrieve(question)
        context = "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)
        prompt = VECTOR_RAG_PROMPT.format(context=context, question=question)
        return {"answer": self.llm.generate(prompt, system=ANSWER_SYSTEM), "contexts": hits}
