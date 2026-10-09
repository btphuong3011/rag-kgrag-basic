import hashlib
import math

import chromadb
import pytest

from src.config import Config
from src.documents import chunk_text, load_documents, make_chunks
from src.kg_builder import KnowledgeGraph
from src.kg_rag import KGRAG
from src.vector_rag import VectorRAG

# Bộ ba cố định mà FakeLLM trả về khi đoạn văn chứa cụm từ khóa tương ứng
FAKE_FACTS = {
    "sinh ngày 14 tháng 3": [("Albert Einstein", "sinh_tại", "Ulm"),
                             ("Albert Einstein", "phát_triển", "Thuyết tương đối")],
    "thuộc bang": [("Ulm", "thuộc_quốc_gia", "Đức"), ("Ulm", "nằm_bên", "Sông Danube")],
    "sinh năm 1867": [("Marie Curie", "sinh_tại", "Warszawa")],
    "thủ đô": [("Warszawa", "là_thủ_đô_của", "Ba Lan"), ("Warszawa", "nằm_bên", "Sông Vistula")],
}


class FakeLLM:
    # Vector đặc trưng từ băm các n-gram ký tự (n = 3), chuẩn hóa về độ dài 1;
    # hai văn bản có nhiều n-gram chung cho độ tương đồng cosine cao
    def embed(self, texts, task_type=None):
        out = []
        for t in texts:
            v = [0.0] * 64
            s = t.casefold()
            for i in range(len(s) - 2):
                v[int(hashlib.md5(s[i:i + 3].encode()).hexdigest(), 16) % 64] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    def generate_json(self, prompt, system=None):
        if "CÂU HỎI:" in prompt:
            q = prompt.split("CÂU HỎI:")[-1]
            return {"entities": [n for n in ["Einstein", "Marie Curie"] if n in q]}
        rels = [r for key, triples in FAKE_FACTS.items() if key in prompt for r in triples]
        return {"entities": [], "relations": [{"head": h, "relation": r, "tail": t} for h, r, t in rels]}

    def generate(self, prompt, system=None, temperature=0.2):
        return "FAKE ANSWER"


@pytest.fixture
def cfg(tmp_path):
    return Config(storage_dir=tmp_path, request_delay=0, entity_match_threshold=0.5)


@pytest.fixture
def chunks(cfg):
    return make_chunks(load_documents(cfg.data_dir), cfg.chunk_size, cfg.chunk_overlap)


def test_chunk_size_and_overlap():
    text = " ".join(f"Câu số {i} có nội dung ngắn." for i in range(40))
    parts = chunk_text(text, chunk_size=200, overlap=60)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)
    last_sentence = parts[0].split(". ")[-1]
    assert last_sentence in parts[1]


def test_chunk_ids(chunks):
    assert chunks and all("#" in c["id"] for c in chunks)


def test_graph_dedup_and_roundtrip(tmp_path):
    kg = KnowledgeGraph()
    kg.add_triple("Albert Einstein", "sinh tại", "Ulm", "a#0")
    kg.add_triple("albert  einstein", "sinh_tại", "ulm", "b#0")
    assert kg.g.number_of_nodes() == 2 and kg.g.number_of_edges() == 1
    kg.save(tmp_path / "kg.json")
    kg2 = KnowledgeGraph.load(tmp_path / "kg.json")
    edge = list(kg2.g.edges(data=True))[0]
    assert edge[2]["sources"] == ["a#0", "b#0"]


def test_vector_rag_retrieve(cfg, chunks):
    rag = VectorRAG(FakeLLM(), cfg, client=chromadb.EphemeralClient())
    rag.build_index(chunks)
    hits = rag.retrieve("Warszawa là thủ đô của nước nào?", k=2)
    assert len(hits) == 2
    assert hits[0]["id"].startswith("warszawa")


def test_kg_rag_multi_hop(cfg, chunks):
    kgrag = KGRAG(FakeLLM(), cfg, client=chromadb.EphemeralClient())
    kgrag.build_index(chunks)
    result = kgrag.answer("Einstein sinh ra ở quốc gia nào?")
    # Einstein được liên kết với đỉnh Albert Einstein qua embedding; Đức cách đỉnh gốc 2 bước
    assert "Albert Einstein" in result["entities"]
    assert ("Albert Einstein", "sinh_tại", "Ulm") in result["triples"]
    assert ("Ulm", "thuộc_quốc_gia", "Đức") in result["triples"]
    assert result["answer"] == "FAKE ANSWER"
