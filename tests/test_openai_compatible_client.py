import json

from budget2success.clients.base import GenerationRequest
from budget2success.clients.factory import build_client
from budget2success.clients.openai_compatible import OpenAICompatibleClient


class _FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_client_parses_usage(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "model": "qwen-coder-32b",
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(base_url="http://localhost:8000/v1", api_key="EMPTY", timeout_seconds=9)

    response = client.generate(GenerationRequest(model="qwen-coder-32b", prompt="hi", max_tokens=12, temperature=0.2))

    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["timeout"] == 9
    assert captured["payload"]["max_tokens"] == 12
    assert response.text == "answer"
    assert response.prompt_tokens == 7
    assert response.completion_tokens == 3
    assert response.total_tokens == 10


def test_factory_builds_openai_compatible_client():
    client = build_client({"provider": "local_vllm", "base_url": "http://example.test/v1", "api_key": "EMPTY"})

    assert isinstance(client, OpenAICompatibleClient)
