import json

from budget2success.clients.base import GenerationRequest
from budget2success.clients.chat_gateway import ChatGatewayClient


def test_chat_gateway_builds_chat_completion_contract():
    client = ChatGatewayClient(api_key="test-key")
    request = GenerationRequest(
        model="DeepSeek-V3-0324",
        prompt="Say ok.",
        max_tokens=17,
        temperature=0,
        metadata={"top_p": 0.9},
    )

    http_request = client._build_http_request(request)
    payload = json.loads(http_request.data.decode("utf-8"))

    assert http_request.full_url == "http://localhost:8000/v1/chat/completions"
    assert http_request.get_method() == "POST"
    assert http_request.headers["Authorization"] == "Bearer test-key"
    assert http_request.headers["Content-type"] == "application/json"
    assert payload == {
        "model": "DeepSeek-V3-0324",
        "messages": [{"role": "user", "content": "Say ok."}],
        "temperature": 0,
        "stream": False,
        "max_tokens": 17,
        "top_p": 0.9,
    }
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "functions" not in payload


def test_chat_gateway_omits_top_p_when_not_requested():
    client = ChatGatewayClient(api_key="test-key", base_url="http://localhost:8000/v1")
    request = GenerationRequest(model="gemini-2.0-flash-lite-001", prompt="Say ok.", max_tokens=8)

    payload = json.loads(client._build_http_request(request).data.decode("utf-8"))

    assert "top_p" not in payload
    assert payload["stream"] is False
