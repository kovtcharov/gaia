# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Claude provider - no embeddings support."""

from typing import Iterator, Optional, Union

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

from ..base_client import LLMClient


class ClaudeProvider(LLMClient):
    """Claude (Anthropic) provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-6",
        system_prompt: Optional[str] = None,
        **_kwargs,
    ):
        if anthropic is None:
            raise ImportError(
                "anthropic package is required for ClaudeProvider. "
                "Install it with: pip install anthropic"
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._system_prompt = system_prompt

    @property
    def provider_name(self) -> str:
        return "Claude"

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Iterator[str]]:
        # If caller passed explicit 'messages', use those instead of prompt
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        # Extract system messages from the messages list (Claude uses separate system param)
        system_msgs = [m for m in messages if m.get("role") == "system"]
        chat_msgs = [m for m in messages if m.get("role") != "system"]
        if system_msgs:
            # Combine system messages and set as system prompt for this call
            kwargs["system"] = "\n".join(m["content"] for m in system_msgs)
        return self.chat(
            chat_msgs,
            model=model,
            stream=stream,
            **kwargs,
        )

    # Parameters supported by Claude Messages API
    _ALLOWED_PARAMS = {
        "model", "messages", "max_tokens", "stream", "system",
        "temperature", "top_p", "top_k", "stop_sequences", "metadata",
        "tools", "tool_choice",
    }

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Iterator[str]]:
        # Filter out unsupported parameters (e.g. OpenAI-style 'stop')
        # and remap where possible
        if "stop" in kwargs:
            # Remap OpenAI 'stop' to Claude 'stop_sequences'
            kwargs.setdefault("stop_sequences", kwargs.pop("stop"))
        filtered = {k: v for k, v in kwargs.items() if k in self._ALLOWED_PARAMS}

        # Ensure max_tokens is set (required by Claude API)
        if "max_tokens" not in filtered:
            filtered["max_tokens"] = 16384

        # Build parameters for Anthropic messages.create
        params = {
            "model": model or self._model,
            "messages": messages,
            "stream": stream,
            **filtered,
        }
        # Claude API requires system prompt as separate parameter, not in messages
        if self._system_prompt:
            params["system"] = self._system_prompt

        response = self._client.messages.create(**params)
        if stream:
            return self._handle_stream(response)
        return response.content[0].text

    # embed() inherited from ABC - raises NotSupportedError

    def vision(self, images: list[bytes], prompt: str, **kwargs) -> str:
        import base64

        # Claude supports vision via messages
        image_b64 = base64.b64encode(images[0]).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.chat(messages, **kwargs)

    # get_performance_stats() inherited from ABC - raises NotSupportedError
    # load_model() inherited from ABC - raises NotSupportedError
    # unload_model() inherited from ABC - raises NotSupportedError

    def _handle_stream(self, response) -> Iterator[str]:
        for chunk in response:
            if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                yield chunk.delta.text
