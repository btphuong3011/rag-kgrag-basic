# RAG cơ bản & KG-RAG cơ bản

Hai chương trình hỏi đáp trên tài liệu, cùng dùng Gemini làm LLM:

- **RAG thường (vector RAG):** chia tài liệu thành đoạn → embedding → lưu vào ChromaDB → tìm top-k đoạn gần câu hỏi nhất → LLM trả lời.
- **KG-RAG:** dùng LLM trích (thực thể, quan hệ, thực thể) từ từng đoạn → xây Knowledge Graph bằng NetworkX → tìm thực thể trong câu hỏi → lấy đồ thị con k-hop → LLM trả lời dựa trên các bộ ba.

Pipeline được cài đặt trực tiếp trên các thư viện nền tảng, không dùng framework LangChain/LlamaIndex.

## Kiến trúc

```
RAG thường
  Indexing:  tài liệu ─► chunking ─► embedding ─► ChromaDB (HNSW, cosine)
  Truy vấn:  câu hỏi ─► embedding ─► top-k đoạn ─► prompt ─► Gemini ─► trả lời

KG-RAG
  Indexing:  tài liệu ─► chunking ─► Gemini trích bộ ba (JSON) ─► gộp thực thể trùng ─► NetworkX graph
                                                                    └► embedding tên thực thể ─► ChromaDB
  Truy vấn:  câu hỏi ─► tìm thực thể (khớp tên / LLM + embedding) ─► BFS k-hop ─► bộ ba → văn bản ─► Gemini ─► trả lời
```

## Cấu trúc thư mục

```
├── main.py              # CLI: index / ask / chat / graph
├── src/
│   ├── config.py        # cấu hình, đọc từ .env
│   ├── llm.py           # wrapper Gemini: generate, generate_json, embed
│   ├── documents.py     # đọc file .txt/.md, chia đoạn có overlap
│   ├── prompts.py       # toàn bộ prompt
│   ├── vector_rag.py    # pipeline RAG thường
│   ├── kg_builder.py    # trích bộ ba bằng LLM, lớp KnowledgeGraph (NetworkX)
│   ├── kg_rag.py        # pipeline KG-RAG: liên kết thực thể, đồ thị con k-hop
│   └── visualize.py     # xuất đồ thị ra HTML (pyvis)
├── data/                # tài liệu mẫu (5 file .txt)
├── tests/test_offline.py  # test chạy không cần API key (FakeLLM)
└── storage/             # sinh ra khi chạy: ChromaDB + kg.json + kg.html
```

## Thư viện sử dụng

| Thư viện | Vai trò |
|---|---|
| `google-genai` | SDK chính thức của Gemini: sinh văn bản/JSON và tạo embedding |
| `chromadb` | Vector database, chỉ mục HNSW, lưu xuống đĩa |
| `networkx` | Lưu và duyệt Knowledge Graph (MultiDiGraph) |
| `pyvis` | Vẽ đồ thị tương tác ra file HTML |
| `python-dotenv` | Đọc API key và cấu hình từ file `.env` |
| `pytest` | Chạy test |

Model mặc định: `gemini-3.5-flash-lite` (sinh văn bản) và `gemini-embedding-001` (embedding, 768 chiều), cấu hình trong `.env`.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # rồi điền GEMINI_API_KEY
```

API key: https://aistudio.google.com/apikey

## Chạy

```bash
python main.py index                                   # xây cả vector index và knowledge graph
python main.py ask "Einstein sinh ra ở quốc gia nào?"  # so sánh RAG thường và KG-RAG
python main.py ask "..." --mode kg                     # chỉ chạy KG-RAG (rag | kg | both)
python main.py chat                                    # hỏi đáp liên tục
python main.py graph                                   # in các bộ ba, xuất storage/kg.html
```

Tài liệu đầu vào (.txt/.md) đặt trong `data/`; sau khi thay đổi tài liệu cần chạy lại `python main.py index`.

Câu hỏi mẫu (cần kết hợp thông tin từ nhiều tài liệu):

- Einstein sinh ra ở quốc gia nào? *(einstein.txt → ulm.txt)*
- Thành phố nơi Marie Curie sinh ra nằm bên con sông nào? *(marie_curie.txt → warszawa.txt)*
- Tổ chức nào trao giải Nobel Vật lý mà Einstein nhận năm 1921? *(einstein.txt → nobel.txt)*

## Test

```bash
python -m pytest -q
```

Test dùng `FakeLLM` thay cho Gemini, không cần API key. Phạm vi: chunking, gộp thực thể, lưu/đọc đồ thị, truy hồi vector, truy hồi 2 bước trên đồ thị.

## Hạn chế và hướng phát triển

- Bước xây KG gọi LLM một lần cho mỗi đoạn; giữa các lần gọi nghỉ `REQUEST_DELAY` giây và tự thử lại khi gặp lỗi 429.
- Gộp thực thể chỉ dựa trên tên chuẩn hóa (không phân biệt hoa thường), nên "Einstein" và "Albert Einstein" có thể thành hai đỉnh nếu LLM viết không nhất quán.
- KG-RAG chỉ đưa các bộ ba vào prompt, không kèm đoạn văn gốc, nên có thể mất chi tiết.
- Hướng mở rộng: lưu đồ thị vào Neo4j và truy vấn bằng Cypher; hybrid search (BM25 + vector) và reranking cho RAG thường; kết hợp bộ ba với đoạn văn gốc; phát hiện cộng đồng (Leiden) và tóm tắt như Microsoft GraphRAG.
