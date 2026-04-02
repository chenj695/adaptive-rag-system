import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

_log = logging.getLogger(__name__)


class LLMReranker:
    def __init__(self):
        load_dotenv()

        token = os.getenv("GITHUB_TOKEN") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://models.github.ai/inference").rstrip("/")
        # 防止用户误写完整接口路径
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]

        self.model_name = os.getenv("GITHUB_MODEL", "openai/gpt-4.1")
        self.enabled = bool(token)

        if not self.enabled:
            _log.warning("LLM reranker disabled: no API token found.")
            self.client = None
            return

        self.client = OpenAI(
            api_key=token,
            base_url=base_url,
        )

    def _default_rank(self, docs: List[Dict]) -> List[float]:
        # 简单回退：按向量相似度映射成分数（distance 越大越相似则这样；如你相反可改）
        scores = []
        for d in docs:
            dist = float(d.get("distance", 0.0))
            scores.append(max(0.0, min(1.0, dist)))
        return scores

    def get_rank_for_multiple_blocks(self, query: str, texts: List[str]) -> List[float]:
        if not self.enabled or self.client is None:
            return self._default_rank([{"distance": 0.5}] * len(texts))

        if not texts:
            return []

        prompt = {
            "task": "rank_relevance",
            "instruction": (
                "Given a user query and candidate passages, output JSON only: "
                "{\"scores\": [float,...]} with one score per passage in [0,1], "
                "higher means more relevant."
            ),
            "query": query,
            "passages": [{"id": i, "text": t} for i, t in enumerate(texts)],
        }

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a strict JSON reranker."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                temperature=0,
                timeout=60,
            )

            content = (resp.choices[0].message.content or "").strip()
            # 去除代码块包裹
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:].strip()

            data = json.loads(content)
            scores = data.get("scores", [])

            # 长度纠正
            if not isinstance(scores, list):
                return self._default_rank([{"distance": 0.5}] * len(texts))

            fixed = []
            for i in range(len(texts)):
                if i < len(scores):
                    try:
                        v = float(scores[i])
                    except Exception:
                        v = 0.0
                else:
                    v = 0.0
                fixed.append(max(0.0, min(1.0, v)))
            return fixed

        except Exception as exc:
            _log.warning("reranker API failed, fallback to default rank: %s", exc)
            return self._default_rank([{"distance": 0.5}] * len(texts))

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict],
        documents_batch_size: int = 2,
        llm_weight: float = 0.7,
    ) -> List[Dict]:
        if not documents:
            return []

        llm_weight = max(0.0, min(1.0, llm_weight))
        vec_weight = 1.0 - llm_weight

        def process_batch(batch_docs: List[Dict]) -> List[Dict]:
            texts = [d.get("text", "") for d in batch_docs]
            scores = self.get_rank_for_multiple_blocks(query, texts)

            out = []
            for d, s in zip(batch_docs, scores):
                vec_score = float(d.get("distance", 0.0))
                final_score = vec_weight * vec_score + llm_weight * s
                out.append({**d, "llm_score": s, "final_score": final_score})
            return out

        # 分批
        batches = [
            documents[i:i + documents_batch_size]
            for i in range(0, len(documents), documents_batch_size)
        ]

        results = []
        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
            for batch_res in executor.map(process_batch, batches):
                results.extend(batch_res)

        results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return results
