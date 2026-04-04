"""GitHub Models Inference API Client - OpenAI-compatible"""
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Callable, TypeVar
import requests

_log = logging.getLogger(__name__)

T = TypeVar("T")


def _looks_like_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def _retry_on_rate_limit(
    fn: Callable[[], T],
    max_retries: int,
    base_delay_sec: float,
    label: str = "LLM",
) -> T:
    last: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt < max_retries and _looks_like_rate_limit(e):
                delay = base_delay_sec * (2 ** attempt)
                _log.warning(
                    "%s rate limited, sleeping %.1fs (attempt %s/%s): %s",
                    label,
                    delay,
                    attempt + 1,
                    max_retries,
                    e,
                )
                time.sleep(delay)
                continue
            raise
    assert last is not None
    raise last


def parse_json_object_from_llm_text(content: Optional[str]) -> Dict[str, Any]:
    """Extract and parse a JSON object from model output (handles ``` fences and preamble)."""
    if not content or not str(content).strip():
        raise ValueError("empty LLM content")
    s = str(content).strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip("\n").strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in LLM content")
    return json.loads(s[start : end + 1])


class GitHubModelsClient:
    """Client for GitHub Models Inference API"""
    
    BASE_URL = "https://models.github.ai/inference"
    API_VERSION = "2026-03-10"
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": self.API_VERSION,
            "Content-Type": "application/json"
        })
    
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Call GitHub Models chat completions endpoint"""
        
        url = f"{self.BASE_URL}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if response_format:
            payload["response_format"] = response_format
        
        response = self.session.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    
    def get_completion(self, **kwargs) -> str:
        """Get completion text"""
        result = self.chat_completion(**kwargs)
        return result["choices"][0]["message"]["content"]


class UnifiedLLMClient:
    """Unified LLM client supporting OpenAI and GitHub Models"""
    
    def __init__(self, api_provider: str = "openai", model: str = "o3-mini-2025-01-31"):
        self.api_provider = api_provider.lower()
        self.model = model
        
        self._openai_base_url = ""
        if self.api_provider in ["github", "github_models"]:
            self.client = GitHubModelsClient()
            # Use provided model or default to GPT-4.1
            if model == "o3-mini-2025-01-31":
                # Default fallback for GitHub Models
                self.model = os.getenv("GITHUB_MODEL", "openai/gpt-4.1")
        else:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            base = (os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")
            if base.endswith("/chat/completions"):
                base = base[: -len("/chat/completions")]
            self._openai_base_url = base
            kw = {"api_key": api_key}
            if base:
                kw["base_url"] = base
            self.client = OpenAI(**kw)
            self.model = model

    def _uses_github_inference_openai_sdk(self) -> bool:
        return (
            not isinstance(self.client, GitHubModelsClient)
            and "models.github.ai" in (self._openai_base_url or "").lower()
        )

    def _llm_max_retries(self) -> int:
        return int(os.getenv("LLM_COMPLETION_MAX_RETRIES", "6"))

    def _llm_rate_limit_base_sec(self) -> float:
        return float(os.getenv("LLM_RATE_LIMIT_BASE_SEC", "5.0"))
    
    def get_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """Get completion from LLM"""
        
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        if response_format:
            completion_kwargs["response_format"] = response_format
        
        if isinstance(self.client, GitHubModelsClient):
            return _retry_on_rate_limit(
                lambda: self.client.get_completion(**completion_kwargs),
                max_retries=self._llm_max_retries(),
                base_delay_sec=self._llm_rate_limit_base_sec(),
                label="GitHub Models",
            )
        else:
            return _retry_on_rate_limit(
                lambda: self.client.chat.completions.create(**completion_kwargs).choices[
                    0
                ].message.content,
                max_retries=self._llm_max_retries(),
                base_delay_sec=self._llm_rate_limit_base_sec(),
                label="OpenAI-compatible chat",
            )
    
    def parse_structured_output(
        self,
        messages: List[Dict[str, str]],
        response_schema: Any,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Get structured JSON output"""
        
        if isinstance(self.client, GitHubModelsClient):
            # GitHub Models - use JSON mode (output may still include markdown; parse robustly)
            content = self.get_completion(
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return parse_json_object_from_llm_text(content)

        mr = self._llm_max_retries()
        bd = self._llm_rate_limit_base_sec()
        skip_beta = self._uses_github_inference_openai_sdk() or (
            os.getenv("OPENAI_SKIP_BETA_PARSE", "").strip().lower()
            in ("1", "true", "yes")
        )

        if not skip_beta:
            try:
                completion = _retry_on_rate_limit(
                    lambda: self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=response_schema,
                        temperature=temperature,
                    ),
                    max_retries=mr,
                    base_delay_sec=bd,
                    label="beta.parse",
                )
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("message.parsed is None")
                return parsed.model_dump()
            except Exception as exc:
                _log.warning(
                    "beta.chat.completions.parse failed (%s); using JSON object mode",
                    exc,
                )

        try:
            response = _retry_on_rate_limit(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                ),
                max_retries=mr,
                base_delay_sec=bd,
                label="chat+json_object",
            )
            content = (response.choices[0].message.content or "").strip()
            return parse_json_object_from_llm_text(content)
        except Exception as exc2:
            _log.warning(
                "JSON object mode failed (%s); trying plain completion",
                exc2,
            )
        response = _retry_on_rate_limit(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            ),
            max_retries=mr,
            base_delay_sec=bd,
            label="chat+plain",
        )
        content = (response.choices[0].message.content or "").strip()
        return parse_json_object_from_llm_text(content)
