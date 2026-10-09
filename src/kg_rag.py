import re
from collections import deque

import chromadb

from .config import Config
from .kg_builder import KnowledgeGraph, build_kg, normalize
from .prompts import ANSWER_SYSTEM, KG_RAG_PROMPT, QUESTION_ENTITY_PROMPT

ENTITY_COLLECTION = "entities"


class KGRAG:
    def __init__(self, llm, cfg: Config, client=None):
        self.llm = llm
        self.cfg = cfg
        self.client = client or chromadb.PersistentClient(path=str(cfg.chroma_dir))
        self.kg = KnowledgeGraph.load(cfg.kg_path) if cfg.kg_path.exists() else KnowledgeGraph()

    def _collection(self):
        return self.client.get_or_create_collection(
            name=ENTITY_COLLECTION, embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}})

    def build_index(self, chunks: list[dict], save: bool = True) -> str:
        self.kg = build_kg(chunks, self.llm, delay=self.cfg.request_delay)
        if save:
            self.kg.save(self.cfg.kg_path)
        # Embedding tên thực thể phục vụ bước liên kết thực thể trong câu hỏi với đỉnh của đồ thị
        try:
            self.client.delete_collection(ENTITY_COLLECTION)
        except Exception:
            pass
        keys = list(self.kg.g.nodes)
        if keys:
            names = [self.kg.name(k) for k in keys]
            vecs = self.llm.embed(names, task_type="SEMANTIC_SIMILARITY")
            self._collection().add(ids=keys, documents=names, embeddings=vecs)
        return self.kg.stats()

    def link_entities(self, question: str) -> list[str]:
        found = []
        q = question.casefold()
        # Bước 1: tìm tên đỉnh xuất hiện nguyên văn trong câu hỏi, có xét ranh giới từ
        for key in self.kg.g.nodes:
            if len(key) > 2 and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", q):
                found.append(key)
        # Bước 2: LLM trích thực thể từ câu hỏi; mỗi thực thể được khớp chính xác theo khóa
        # chuẩn hóa, nếu không có thì chọn đỉnh có độ tương đồng embedding cao nhất
        # khi độ tương đồng không nhỏ hơn entity_match_threshold
        try:
            mentions = self.llm.generate_json(
                QUESTION_ENTITY_PROMPT.format(question=question)).get("entities", [])
        except Exception:
            mentions = []
        unmatched = []
        for m in mentions:
            key = normalize(m)
            if key in self.kg.g:
                found.append(key)
            elif key:
                unmatched.append(m)
        if unmatched and self.kg.g.number_of_nodes():
            vecs = self.llm.embed(unmatched, task_type="SEMANTIC_SIMILARITY")
            res = self._collection().query(query_embeddings=vecs, n_results=1)
            for ids, dists in zip(res["ids"], res["distances"]):
                if ids and 1 - dists[0] >= self.cfg.entity_match_threshold:
                    found.append(ids[0])
        return list(dict.fromkeys(found))

    def subgraph_triples(self, seeds: list[str], hops: int | None = None,
                         max_triples: int | None = None) -> list[tuple[str, str, str]]:
        hops = self.cfg.kg_hops if hops is None else hops
        max_triples = max_triples or self.cfg.kg_max_triples
        g = self.kg.g
        # BFS từ các đỉnh gốc theo cả hai chiều cạnh; dist[v] là số bước từ đỉnh gốc gần nhất đến v
        dist = {s: 0 for s in seeds if s in g}
        queue = deque(dist)
        while queue:
            u = queue.popleft()
            if dist[u] >= hops:
                continue
            for v in list(g.successors(u)) + list(g.predecessors(u)):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        # Chỉ giữ cạnh có ít nhất một đầu mút cách đỉnh gốc dưới hops bước;
        # cạnh gần đỉnh gốc hơn được ưu tiên khi giới hạn theo max_triples
        edges = []
        for h, t, data in g.edges(data=True):
            if h in dist and t in dist and min(dist[h], dist[t]) < hops:
                edges.append((min(dist[h], dist[t]), h, data["relation"], t))
        edges.sort(key=lambda e: e[0])
        return [(self.kg.name(h), r, self.kg.name(t)) for _, h, r, t in edges[:max_triples]]

    def answer(self, question: str) -> dict:
        seeds = self.link_entities(question)
        triples = self.subgraph_triples(seeds)
        if not triples:
            return {"answer": "Không tìm thấy thực thể liên quan trong đồ thị tri thức.",
                    "entities": [], "triples": []}
        context = "\n".join(f"{h} --[{r}]--> {t}" for h, r, t in triples)
        prompt = KG_RAG_PROMPT.format(context=context, question=question)
        return {"answer": self.llm.generate(prompt, system=ANSWER_SYSTEM),
                "entities": [self.kg.name(s) for s in seeds], "triples": triples}
