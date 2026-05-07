from budget2success.clients.base import GenerationRequest
from budget2success.clients.chat_gateway import ChatGatewayClient


def test_provider_parser_preserves_finish_reason_variants():
    client = ChatGatewayClient(api_key="key", base_url="http://example.test")
    request = GenerationRequest(model="m", prompt="p", max_tokens=8)

    response = client._parse_response(
        {
            "model": "m-provider",
            "choices": [{"message": {"content": "done"}, "finishReason": "MAX_TOKENS"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 8, "total_tokens": 10},
        },
        request,
    )

    assert response.text == "done"
    assert response.finish_reason == "MAX_TOKENS"
    assert response.completion_tokens == 8
