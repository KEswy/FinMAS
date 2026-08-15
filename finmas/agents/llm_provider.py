"""Unified LLM provider with structured JSON support and simple in-memory cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_OLLAMA_MODEL = "qwen2.5:32b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


@dataclass(slots=True)
class LLMResult:
    content: str
    model: str
    provider: str
    elapsed: float


class LLMProvider:
    """Provider-agnostic chat and JSON completion client."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: int = 180,
                 max_tokens: int = 2048, retries: int = 5,
                 cache_path: str = "data/cache/llm_cache.json") -> None:
        self.provider = (provider or self._detect_provider(api_key)).lower()
        self.model = model or (
            DEFAULT_DEEPSEEK_MODEL if self.provider == "deepseek" else DEFAULT_OLLAMA_MODEL
        )
        self.api_key = self._clean(api_key or os.environ.get("DEEPSEEK_API_KEY")
                                   or os.environ.get("LLM_API_KEY") or "")
        self.timeout = int(timeout)
        self.max_tokens = int(max_tokens)
        self.retries = int(retries)
        self._cache: Dict[str, LLMResult] = {}
        self._cost_usd = 0.0
        self.cache_path = Path(cache_path)
        self._load_disk_cache()

    @staticmethod
    def _clean(value: str) -> str:
        return "".join(ch for ch in (value or "") if 32 <= ord(ch) < 127)

    @staticmethod
    def _detect_provider(api_key: Optional[str]) -> str:
        explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if explicit:
            return explicit
        if api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY"):
            return "deepseek"
        return "ollama"

    @property
    def chat_url(self) -> str:
        if self.provider == "deepseek":
            base = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
            for suffix in ("/v1/chat/completions", "/chat/completions"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            return base + "/chat/completions"
        return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_CHAT_URL)

    def _load_disk_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            for key, value in raw.items():
                self._cache[key] = LLMResult(
                    content=value["content"],
                    model=value["model"],
                    provider=value["provider"],
                    elapsed=float(value.get("elapsed", 0.0)),
                )
        except Exception:
            return

    def _save_disk_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: asdict(value)
            for key, value in self._cache.items()
        }
        self.cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def chat(self, system: str, user: str, temperature: float = 0.2,
             json_mode: bool = False, use_cache: bool = True) -> LLMResult:
        key = hashlib.sha256(
            json.dumps(
                [self.provider, self.model, system, user, temperature, json_mode],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if use_cache and key in self._cache:
            return self._cache[key]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": float(temperature),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        if self.provider == "deepseek":
            if not self.api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is required for deepseek provider")
            payload["max_tokens"] = self.max_tokens
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        else:
            payload["keep_alive"] = "60m"
            payload["options"] = {"temperature": float(temperature),
                                  "num_predict": self.max_tokens}
            headers = {"Content-Type": "application/json"}

        data = json.dumps(payload).encode("utf-8")
        backoff = [1, 2, 5, 10, 20, 30, 30, 30, 30, 30]
        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            start = time.time()
            try:
                req = urllib.request.Request(self.chat_url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                content = self._extract_content(json.loads(body))
                result = LLMResult(
                    content=self._strip_thinking(content),
                    model=self.model,
                    provider=self.provider,
                    elapsed=time.time() - start,
                )
                self._cache[key] = result
                if use_cache:
                    self._save_disk_cache()
                self._cost_usd += self._estimate_cost(system, user, result.content)
                return result
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt + 1 >= self.retries:
                    break
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
        raise RuntimeError(f"LLM call failed: {last_err}")

    def chat_json(self, system: str, user: str, temperature: float = 0.1,
                  use_cache: bool = True) -> Dict[str, Any]:
        result = self.chat(system, user, temperature=temperature, json_mode=True,
                           use_cache=use_cache)
        parsed = self.parse_json(result.content)
        if parsed is None:
            raise ValueError(f"LLM did not return valid JSON: {result.content[:200]}")
        return parsed

    @staticmethod
    def _strip_thinking(content: str) -> str:
        return re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()

    @staticmethod
    def _extract_content(payload: Dict[str, Any]) -> str:
        if "choices" in payload:
            return payload["choices"][0]["message"]["content"]
        if "message" in payload:
            return payload["message"]["content"]
        raise KeyError("unknown LLM response shape")

    @staticmethod
    def parse_json(content: str) -> Optional[Dict[str, Any]]:
        text = (content or "").strip()
        for candidate in (text, text.strip("`")):
            try:
                value = json.loads(candidate)
                return value if isinstance(value, dict) else None
            except Exception:
                pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else None
            except Exception:
                pass
        return None

    @staticmethod
    def _estimate_cost(system: str, user: str, completion: str) -> float:
        """Approximate token cost for bookkeeping, not billing."""
        chars = len(system or "") + len(user or "") + len(completion or "")
        tokens = max(chars // 4, 1)
        return tokens * 0.0000001

    def usage(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "cache_size": len(self._cache),
            "estimated_cost_usd": round(self._cost_usd, 8),
        }
