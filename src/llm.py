import json
import re
import time

from google import genai
from google.genai import errors, types

from .config import Config

# Tắt cơ chế tự động gọi hàm (automatic function calling) do chương trình không khai báo công cụ
NO_AFC = types.AutomaticFunctionCallingConfig(disable=True)


def _retry(fn, max_tries: int = 5):
    # Thử lại với thời gian chờ tăng theo hàm mũ khi gặp lỗi 429 (vượt hạn mức) hoặc 5xx (lỗi máy chủ)
    for attempt in range(max_tries):
        try:
            return fn()
        except errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < max_tries - 1:
                wait = 2 ** attempt * 5
                print(f"  [Gemini {e.code}] thử lại sau {wait}s...")
                time.sleep(wait)
            else:
                raise


def parse_json(text: str):
    text = text.strip()
    # Loại bỏ khối mã Markdown ```json ... ``` bao quanh nội dung JSON nếu có
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    return json.loads(text)


class GeminiLLM:
    def __init__(self, cfg: Config):
        if not cfg.api_key:
            raise RuntimeError("Chưa có GEMINI_API_KEY. Copy .env.example thành .env rồi điền key.")
        self.cfg = cfg
        self.client = genai.Client(api_key=cfg.api_key)

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, automatic_function_calling=NO_AFC)
        resp = _retry(lambda: self.client.models.generate_content(
            model=self.cfg.chat_model, contents=prompt, config=config))
        return resp.text or ""

    def generate_json(self, prompt: str, system: str | None = None):
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=0.0, response_mime_type="application/json",
            automatic_function_calling=NO_AFC)
        resp = _retry(lambda: self.client.models.generate_content(
            model=self.cfg.chat_model, contents=prompt, config=config))
        return parse_json(resp.text or "{}")

    # task_type: RETRIEVAL_DOCUMENT cho đoạn tài liệu, RETRIEVAL_QUERY cho câu hỏi,
    # SEMANTIC_SIMILARITY cho tên thực thể; văn bản được gửi theo lô tối đa 50 phần tử
    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]
            config = types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=self.cfg.embed_dim)
            resp = _retry(lambda: self.client.models.embed_content(
                model=self.cfg.embed_model, contents=batch, config=config))
            vectors.extend(e.values for e in resp.embeddings)
        return vectors
