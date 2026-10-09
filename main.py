import argparse

from src.config import Config
from src.documents import load_documents, make_chunks
from src.kg_builder import KnowledgeGraph


def get_llm(cfg):
    from src.llm import GeminiLLM
    return GeminiLLM(cfg)


def cmd_index(cfg, args):
    from src.kg_rag import KGRAG
    from src.vector_rag import VectorRAG

    docs = load_documents(cfg.data_dir)
    chunks = make_chunks(docs, cfg.chunk_size, cfg.chunk_overlap)
    print(f"Đọc {len(docs)} tài liệu -> {len(chunks)} đoạn")
    llm = get_llm(cfg)
    if args.only in (None, "rag"):
        n = VectorRAG(llm, cfg).build_index(chunks)
        print(f"[RAG] đã lưu {n} vector vào {cfg.chroma_dir}")
    if args.only in (None, "kg"):
        print("[KG] đang trích xuất thực thể và quan hệ...")
        stats = KGRAG(llm, cfg).build_index(chunks)
        print(f"[KG] đồ thị: {stats} -> {cfg.kg_path}")


def show_rag(result):
    print("\n=== RAG thường ===")
    for h in result["contexts"]:
        print(f"  - [{h['id']}] score={h['score']}: {h['text'][:90]}...")
    print("Trả lời:", result["answer"].strip())


def show_kg(result):
    print("\n=== KG-RAG ===")
    print("  Thực thể tìm được:", ", ".join(result["entities"]) or "(không có)")
    for h, r, t in result["triples"]:
        print(f"  - {h} --[{r}]--> {t}")
    print("Trả lời:", result["answer"].strip())


def ask(cfg, llm, question, mode):
    if mode in ("rag", "both"):
        from src.vector_rag import VectorRAG
        show_rag(VectorRAG(llm, cfg).answer(question))
    if mode in ("kg", "both"):
        from src.kg_rag import KGRAG
        show_kg(KGRAG(llm, cfg).answer(question))


def cmd_ask(cfg, args):
    ask(cfg, get_llm(cfg), args.question, args.mode)


def cmd_chat(cfg, args):
    llm = get_llm(cfg)
    print("Nhập câu hỏi (Enter trống để thoát).")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        ask(cfg, llm, q, args.mode)


def cmd_graph(cfg, args):
    from src.visualize import export_html
    if not cfg.kg_path.exists():
        print("Chưa có đồ thị, chạy `python main.py index` trước.")
        return
    kg = KnowledgeGraph.load(cfg.kg_path)
    print("Đồ thị:", kg.stats())
    for h, t, d in kg.g.edges(data=True):
        print(f"  {kg.name(h)} --[{d['relation']}]--> {kg.name(t)}")
    out = export_html(kg, cfg.storage_dir / "kg.html")
    print("Đã xuất:", out)


def main():
    parser = argparse.ArgumentParser(description="RAG cơ bản và KG-RAG cơ bản (Gemini)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="xây vector index và/hoặc knowledge graph")
    p.add_argument("--only", choices=["rag", "kg"], help="chỉ xây một loại")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="hỏi một câu")
    p.add_argument("question")
    p.add_argument("--mode", choices=["rag", "kg", "both"], default="both")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="hỏi đáp liên tục")
    p.add_argument("--mode", choices=["rag", "kg", "both"], default="both")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("graph", help="in và xuất knowledge graph ra HTML")
    p.set_defaults(func=cmd_graph)

    args = parser.parse_args()
    args.func(Config(), args)


if __name__ == "__main__":
    main()
