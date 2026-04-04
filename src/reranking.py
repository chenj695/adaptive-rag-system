import json
import logging
import os
import time
from typing import List, Dict, Protocol

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

_log = logging.getLogger(__name__)


class RerankerProtocol(Protocol):
    def rerank_documents(
        self,
        query: str,
        documents: List[Dict],
        documents_batch_size: int = 6,
        llm_weight: float = 0.7,
    ) -> List[Dict]:
        ...


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars].rstrip() + "…"


def _looks_like_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "too many requests" in msg or "rate limit" in msg or "429" in msg:
        return True
    code = getattr(exc, "status_code", None)
    if code == 429:
        return True
    err = getattr(exc, "response", None)
    if err is not None and getattr(err, "status_code", None) == 429:
        return True
    return False


def _looks_like_payload_too_large(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "413" in msg or "tokens_limit" in msg or "too large" in msg or "request body too large" in msg:
        return True
    code = getattr(exc, "status_code", None)
    if code == 413:
        return True
    return False


class VectorOnlyReranker:
    """No cross-encoder / LLM: order by vector similarity only."""

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict],
        documents_batch_size: int = 6,
        llm_weight: float = 0.7,
    ) -> List[Dict]:
        if not documents:
            return []
        out = []
        for d in documents:
            v = float(d.get("distance", 0.0))
            out.append({**d, "llm_score": 0.0, "final_score": v})
        out.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return out


class BGEReranker:
    """Local cross-encoder rerank (e.g. BAAI/bge-reranker-base); no API quota."""

    def __init__(self):
        load_dotenv()
        self.model_name = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.model_path = os.getenv("BGE_RERANKER_PATH")
        self._max_query_chars = int(os.getenv("BGE_RERANKER_QUERY_CHARS", "512"))
        self._max_passage_chars = int(os.getenv("BGE_RERANKER_PASSAGE_CHARS", "2000"))
        self._predict_batch = int(os.getenv("BGE_RERANKER_PREDICT_BATCH", "16"))
        self._model = None

    def _device(self) -> str:
        requested = os.getenv("BGE_RERANKER_DEVICE", "").strip().lower()
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except ImportError:
            cuda_ok = False

        if not requested:
            return "cuda" if cuda_ok else "cpu"

        if requested in ("cuda", "gpu"):
            if not cuda_ok:
                _log.warning(
                    "BGE_RERANKER_DEVICE=%s but no CUDA GPU; using CPU for reranker",
                    os.getenv("BGE_RERANKER_DEVICE", "").strip() or "cuda",
                )
                return "cpu"
            return "cuda"

        return requested

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for BGE reranker. "
                "Install with: pip install sentence-transformers"
            ) from e

        path = (self.model_path or "").strip() or self.model_name
        device = self._device()
        _log.info("Loading BGE reranker from %s (device=%s)", path, device)
        self._model = CrossEncoder(path, device=device, trust_remote_code=True)

    def _scores_for_texts(self, query: str, texts: List[str]) -> List[float]:
        self._ensure_model()
        q = _clip_text(query, self._max_query_chars)
        pairs = [(q, _clip_text(t, self._max_passage_chars)) for t in texts]
        raw = self._model.predict(
            pairs,
            batch_size=max(1, self._predict_batch),
            show_progress_bar=False,
        )
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        # logits -> (0,1); avoids negative scores in fusion
        arr = np.clip(arr, -50.0, 50.0)
        norm = 1.0 / (1.0 + np.exp(-arr))
        return [float(x) for x in norm]

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict],
        documents_batch_size: int = 6,
        llm_weight: float = 0.7,
    ) -> List[Dict]:
        if not documents:
            return []
        llm_weight = max(0.0, min(1.0, llm_weight))
        vec_weight = 1.0 - llm_weight

        try:
            texts = [d.get("text", "") or "" for d in documents]
            scores = self._scores_for_texts(query, texts)
        except Exception as exc:
            _log.exception("BGE reranker failed, fallback to vector order: %s", exc)
            return VectorOnlyReranker().rerank_documents(
                query, documents, documents_batch_size, llm_weight
            )

        out = []
        for d, s in zip(documents, scores):
            vec_score = float(d.get("distance", 0.0))
            final_score = vec_weight * vec_score + llm_weight * s
            out.append({**d, "llm_score": s, "rerank_score": s, "final_score": final_score})
        out.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return out


class LLMReranker:
    def __init__(self):
        load_dotenv()

        token = os.getenv("GITHUB_TOKEN") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://models.github.ai/inference").rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]

        self.model_name = os.getenv("GITHUB_MODEL", "openai/gpt-4.1")
        self.enabled = bool(token)
        self._rerank_max_retries = int(os.getenv("RERANK_MAX_RETRIES", "3"))
        self._rerank_retry_base_sec = float(os.getenv("RERANK_RETRY_BASE_SEC", "2.0"))
        self._rerank_batch_delay_sec = float(os.getenv("RERANK_BATCH_DELAY_SEC", "1.5"))
        self._rerank_query_max_chars = int(os.getenv("RERANK_QUERY_MAX_CHARS", "512"))
        self._rerank_passage_max_chars = int(os.getenv("RERANK_PASSAGE_MAX_CHARS", "600"))
        self._rerank_batch_size = int(os.getenv("RERANK_BATCH_SIZE", "6"))

        if not self.enabled:
            _log.warning("LLM reranker disabled: no API token found.")
            self.client = None
            return

        self.client = OpenAI(
            api_key=token,
            base_url=base_url,
        )

    def _default_rank(self, docs: List[Dict]) -> List[float]:
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

        q_short = _clip_text(query, self._rerank_query_max_chars)
        texts_for_api = [_clip_text(t, self._rerank_passage_max_chars) for t in texts]

        prompt = {
            "task": "rank_relevance",
            "instruction": (
                "Given a user query and candidate passages, output JSON only: "
                "{\"scores\": [float,...]} with one score per passage in [0,1], "
                "higher means more relevant."
            ),
            "query": q_short,
            "passages": [{"id": i, "text": t} for i, t in enumerate(texts_for_api)],
        }

        user_content = json.dumps(prompt, ensure_ascii=False)
        max_retries = self._rerank_max_retries
        base_delay = self._rerank_retry_base_sec

        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a strict JSON reranker."},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    timeout=120,
                )

                content = (resp.choices[0].message.content or "").strip()
                if content.startswith("```"):
                    content = content.strip("`")
                    if content.startswith("json"):
                        content = content[4:].strip()

                data = json.loads(content)
                scores = data.get("scores", [])

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
                if attempt < max_retries and _looks_like_rate_limit(exc):
                    delay = base_delay * (2 ** attempt)
                    _log.warning(
                        "reranker rate limited, retry in %.1fs (attempt %s/%s): %s",
                        delay, attempt + 1, max_retries, exc,
                    )
                    time.sleep(delay)
                    continue
                if _looks_like_payload_too_large(exc):
                    _log.warning(
                        "reranker request too large (reduce RERANK_PASSAGE_MAX_CHARS / RERANK_BATCH_SIZE): %s",
                        exc,
                    )
                _log.warning("reranker API failed, fallback to default rank: %s", exc)
                return self._default_rank([{"distance": 0.5}] * len(texts))

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict],
        documents_batch_size: int = 6,
        llm_weight: float = 0.7,
    ) -> List[Dict]:
        if not documents:
            return []

        batch_size = min(documents_batch_size, self._rerank_batch_size)
        batch_size = max(1, batch_size)

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

        batches = [
            documents[i:i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]

        results: List[Dict] = []
        delay = self._rerank_batch_delay_sec
        for i, batch in enumerate(batches):
            if i > 0 and delay > 0:
                time.sleep(delay)
            results.extend(process_batch(batch))

        results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return results


def get_reranker() -> RerankerProtocol:
    """
    RERANK_BACKEND:
      - bge (default): local BAAI/bge-reranker-* CrossEncoder
      - llm: remote chat completions (GitHub Models / OpenAI-compatible)
      - none / vector: skip rerank, vector order only
    """
    load_dotenv()
    backend = os.getenv("RERANK_BACKEND", "bge").strip().lower()

    if backend in ("none", "off", "vector", "skip"):
        _log.info("Reranker: vector-only (RERANK_BACKEND=%s)", backend)
        return VectorOnlyReranker()

    if backend in ("llm", "github", "openai", "api"):
        r = LLMReranker()
        if r.enabled:
            _log.info("Reranker: LLM API (RERANK_BACKEND=%s)", backend)
            return r
        _log.warning("LLM reranker requested but no token; using vector-only")
        return VectorOnlyReranker()

    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
    except ImportError:
        _log.warning("sentence-transformers not installed; using vector-only rerank")
        return VectorOnlyReranker()

    br = BGEReranker()
    _log.info(
        "Reranker: BGE cross-encoder (%s); loads on first query",
        br.model_path or br.model_name,
    )
    return br
