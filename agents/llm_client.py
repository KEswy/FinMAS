"""
Unified LLM client for the financial-agent system.

Supports two providers with the same call interface:
  * ``ollama``  - local Ollama ``/api/chat`` (default, no API key required)
  * ``deepseek`` - DeepSeek's OpenAI-compatible ``/chat/completions`` API

The provider is auto-detected when ``DEEPSEEK_API_KEY`` (or ``LLM_API_KEY``)
is present. Explicit control is available through ``LLM_PROVIDER``.

Recommended DeepSeek V4 Flash setup:

    $env:DEEPSEEK_API_KEY="sk-..."
    $env:LLM_MODEL="deepseek-v4-flash"
    $env:MAX_CONCURRENCY="4"

All network calls use the standard library only, so no new dependency is
required and the client is safe to call from worker threads.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional


DEFAULT_OLLAMA_MODEL = "qwen2.5:32b"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


class LLMError(RuntimeError):
    """Raised when the unified LLM client cannot produce a response."""


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _retry_backoff(provider: str, retries: int) -> list[float]:
    raw = os.environ.get("LLM_RETRY_BACKOFF", "").strip()
    if raw:
        values = [max(float(x), 0.0) for x in raw.split(",") if x.strip()]
        if values:
            return values[: max(retries, 1)]

    if provider == "deepseek":
        backoff = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    else:
        # Ollama can be slow to restart after an OOM; keep the legacy generous
        # schedule while retaining the caller's retry-count limit.
        backoff = [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]
    return backoff[: max(retries, 1)]


def get_llm_config() -> dict:
    """Return provider/model/endpoint credentials, without making a request."""
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    ).strip()

    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit == "ollama":
        provider = "ollama"
    elif explicit in {"deepseek", "api", "openai"} or api_key:
        provider = "deepseek"
    else:
        provider = "ollama"

    if provider == "deepseek":
        base_url = os.environ.get(
            "DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL
        ).strip().rstrip("/")
        for suffix in ("/v1/chat/completions", "/chat/completions"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        chat_url = base_url + "/chat/completions"
        default_model = DEFAULT_DEEPSEEK_MODEL
    else:
        chat_url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_CHAT_URL).strip()
        base_url = chat_url.rsplit("/api/chat", 1)[0]
        default_model = DEFAULT_OLLAMA_MODEL

    model = (
        os.environ.get("LLM_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or default_model
    ).strip()

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "chat_url": chat_url,
        "api_key": api_key,
        "timeout": int(os.environ.get("LLM_TIMEOUT", "180")),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "2048")),
        "retries": int(os.environ.get("LLM_MAX_RETRIES", "5")),
    }


def _strip_thinking(content: str) -> str:
    """Strip ``<think>...</think>`` blocks used by reasoning models."""
    return re.sub(
        r"<think>.*?</think>", "", content or "", flags=re.DOTALL
    ).strip()


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
    return isinstance(
        exc,
        (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
         json.JSONDecodeError, KeyError, IndexError, TypeError),
    )


def _call_ollama(
    *,
    model: str,
    system: Optional[str],
    user: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout: int,
    chat_url: str,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "60m",
        "options": {"temperature": temperature},
    }
    if max_tokens is not None:
        payload["options"]["num_predict"] = int(max_tokens)

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        chat_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    result = json.loads(body)
    return _strip_thinking(result["message"]["content"])


def _call_deepseek(
    *,
    model: str,
    system: Optional[str],
    user: str,
    temperature: float,
    max_tokens: Optional[int],
    timeout: int,
    chat_url: str,
    api_key: str,
    json_mode: bool = False,
) -> str:
    if not api_key:
        raise LLMError(
            "DeepSeek API key is missing; set DEEPSEEK_API_KEY or LLM_API_KEY."
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        chat_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    result = json.loads(body)
    return _strip_thinking(result["choices"][0]["message"]["content"])


def chat_completion(
    system: Optional[str],
    user: str,
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    retries: Optional[int] = None,
    json_mode: bool = False,
) -> str:
    """One chat completion through the configured provider.

    ``system`` is optional, ``user`` is the main prompt. Returns the trimmed
    text content with reasoning blocks removed.
    """
    cfg = get_llm_config()
    model = (model or cfg["model"]).strip()
    temperature = cfg.get("default_temperature", 0.2) if temperature is None else temperature
    max_tokens = cfg["max_tokens"] if max_tokens is None else max_tokens
    timeout = cfg["timeout"] if timeout is None else timeout
    retries = cfg["retries"] if retries is None else retries
    retries = max(int(retries), 1)
    backoff = _retry_backoff(cfg["provider"], retries)

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if cfg["provider"] == "deepseek":
                return _call_deepseek(
                    model=model,
                    system=system,
                    user=user,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens) if max_tokens is not None else None,
                    timeout=int(timeout),
                    chat_url=cfg["chat_url"],
                    api_key=cfg["api_key"],
                    json_mode=json_mode,
                )
            return _call_ollama(
                model=model,
                system=system,
                user=user,
                temperature=float(temperature),
                max_tokens=int(max_tokens) if max_tokens is not None else None,
                timeout=int(timeout),
                chat_url=cfg["chat_url"],
            )
        except Exception as exc:  # noqa: BLE001 - normalized below
            last_err = exc
            if attempt >= retries or not _should_retry(exc):
                break
            wait = backoff[min(attempt - 1, len(backoff) - 1)]
            print(
                f"    [LLM {cfg['provider']} retry {attempt}/{retries}] "
                f"{type(exc).__name__}: {exc}; waiting {wait:.1f}s"
            )
            time.sleep(wait)

    raise LLMError(
        f"{cfg['provider']} call failed after {retries} attempt(s): {last_err}"
    ) from last_err


# Shorter alias used by older call sites / experiments.
call_llm = chat_completion
