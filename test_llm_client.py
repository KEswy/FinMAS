import json
import os

from agents.llm_client import (
    LLMError,
    _strip_thinking,
    chat_completion,
    get_llm_config,
)


def test_default_provider_is_ollama(monkeypatch):
    for key in ("DEEPSEEK_API_KEY", "LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL",
                "DEEPSEEK_MODEL"):
        monkeypatch.delenv(key, raising=False)
    cfg = get_llm_config()
    assert cfg["provider"] == "ollama"
    assert cfg["model"] == "qwen2.5:32b"


def test_deepseek_provider_autodetect(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    cfg = get_llm_config()
    assert cfg["provider"] == "deepseek"
    assert cfg["model"] == "deepseek-v4-flash"
    assert cfg["chat_url"] == "https://api.deepseek.com/chat/completions"


def test_strip_thinking():
    assert _strip_thinking("<think>hidden</think>final") == "final"
    assert _strip_thinking("plain") == "plain"


def test_deepseek_chat_completion(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-flash")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            payload = {
                "choices": [{
                    "message": {"content": "<think>reasoning</think>答案"}
                }]
            }
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("agents.llm_client.urllib.request.urlopen", fake_urlopen)
    result = chat_completion(system="system", user="user", retries=1)
    assert result == "答案"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "deepseek-v4-flash"


def test_missing_deepseek_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    try:
        chat_completion(system=None, user="x", retries=1)
    except LLMError as exc:
        assert "API key is missing" in str(exc)
    else:
        raise AssertionError("expected LLMError")
