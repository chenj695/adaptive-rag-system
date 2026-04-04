"""GitHub Models Inference API Client - OpenAI-compatible"""
import json
import os
from typing import List, Dict, Any, Optional
import requests


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
            kw = {"api_key": api_key}
            if base:
                kw["base_url"] = base
            self.client = OpenAI(**kw)
            self.model = model
    
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
            return self.client.get_completion(**completion_kwargs)
        else:
            # OpenAI client
            response = self.client.chat.completions.create(**completion_kwargs)
            return response.choices[0].message.content
    
    def parse_structured_output(
        self,
        messages: List[Dict[str, str]],
        response_schema: Any,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Get structured JSON output"""
        
        if isinstance(self.client, GitHubModelsClient):
            # GitHub Models - use JSON mode (output may still include markdown; parse robustly)
            content = self.client.get_completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return parse_json_object_from_llm_text(content)
        else:
            # OpenAI - use beta.parse
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_schema,
                temperature=temperature
            )
            return completion.choices[0].message.parsed.model_dump()
