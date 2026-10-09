ANSWER_SYSTEM = (
    "Bạn là trợ lý hỏi đáp. Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp. "
    "Được phép kết hợp thông tin từ nhiều đoạn hoặc nhiều sự kiện trong ngữ cảnh để suy ra câu trả lời. "
    "Nếu ngữ cảnh không đủ thông tin, hãy nói rõ là không tìm thấy, không được bịa. "
    "Trả lời ngắn gọn bằng tiếng Việt."
)

VECTOR_RAG_PROMPT = """NGỮ CẢNH (các đoạn tài liệu liên quan):
{context}

CÂU HỎI: {question}

Trả lời câu hỏi, cuối câu trả lời ghi nguồn dạng [tên_file#số_đoạn]."""

KG_RAG_PROMPT = """NGỮ CẢNH (các sự kiện lấy từ đồ thị tri thức, dạng: thực thể --[quan hệ]--> thực thể):
{context}

CÂU HỎI: {question}

Hãy suy luận qua các sự kiện trên (có thể phải nối nhiều sự kiện) để trả lời.
Cuối câu trả lời liệt kê ngắn gọn chuỗi sự kiện đã dùng."""

EXTRACT_SYSTEM = (
    "Bạn là hệ thống trích xuất tri thức. Từ đoạn văn, hãy trích các thực thể và quan hệ "
    "giữa chúng để xây dựng đồ thị tri thức."
)

# Cặp {{ }} biểu diễn ký tự ngoặc nhọn trong chuỗi dùng với str.format
EXTRACT_PROMPT = """Quy tắc:
- Thực thể: người, tổ chức, địa điểm, sự kiện, giải thưởng, khái niệm, công trình... Dùng tên đầy đủ, nhất quán (ví dụ luôn viết "Albert Einstein", không viết "ông" hay "Einstein" lúc này lúc khác).
- Quan hệ: cụm động từ ngắn bằng tiếng Việt, viết thường, nối bằng dấu gạch dưới (ví dụ: sinh_tại, thuộc_quốc_gia, nhận_giải, làm_việc_tại).
- Chỉ trích thông tin có trong đoạn văn, không suy diễn thêm.

Trả về JSON đúng dạng:
{{"entities": [{{"name": "...", "type": "..."}}],
  "relations": [{{"head": "...", "relation": "...", "tail": "..."}}]}}

ĐOẠN VĂN:
{text}"""

QUESTION_ENTITY_PROMPT = """Liệt kê các thực thể (tên riêng, địa điểm, tổ chức, giải thưởng, khái niệm) được nhắc tới trong câu hỏi sau.
Trả về JSON dạng {{"entities": ["...", "..."]}}.

CÂU HỎI: {question}"""
