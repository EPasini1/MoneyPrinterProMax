from unittest.mock import Mock

import pytest
from ollama import ResponseError

import gpt


@pytest.mark.parametrize("fallback", [False, True])
def test_thinking_disabled_for_chat_and_generate(monkeypatch: pytest.MonkeyPatch, fallback: bool) -> None:
    monkeypatch.setenv("OLLAMA_THINK", "true")  # Even a conflicting setting cannot enable thinking.
    client = Mock()
    client.chat.return_value = {"message": {"content": "Narration"}}
    client.generate.return_value = {"response": "Narration"}
    if fallback:
        client.chat.side_effect = ResponseError("endpoint unavailable", status_code=404)
    monkeypatch.setattr(gpt, "_ollama_client", lambda: client)
    assert gpt.generate_response("Prompt", "test") == "Narration"
    assert client.chat.call_args.kwargs["think"] is False
    assert client.chat.call_args.kwargs["stream"] is False
    if fallback:
        assert client.generate.call_args.kwargs["think"] is False
    else:
        client.generate.assert_not_called()


@pytest.mark.parametrize("legacy_error", [
    TypeError("chat() got an unexpected keyword argument 'think'"),
    ResponseError('unknown field "think"', status_code=400),
    ResponseError("model does not support thinking", status_code=400),
])
def test_legacy_think_compatibility(legacy_error: Exception) -> None:
    request = Mock(side_effect=[legacy_error, "Narration"])
    assert gpt._ollama_request(request, model="test", stream=False) == "Narration"
    assert request.call_args_list[0].kwargs["think"] is False
    assert "think" not in request.call_args_list[1].kwargs


@pytest.mark.parametrize("error", [TypeError("unrelated implementation bug"), ResponseError("invalid model", status_code=400)])
def test_unrelated_errors_do_not_trigger_legacy_retry(error: Exception) -> None:
    request = Mock(side_effect=error)
    with pytest.raises(type(error)):
        gpt._ollama_request(request, model="test")
    request.assert_called_once()
