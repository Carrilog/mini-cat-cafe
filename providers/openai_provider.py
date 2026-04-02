from __future__ import annotations
"""OpenAI-compatible provider (works with OpenAI, DeepSeek, Qwen, etc.)."""
import os
from typing import Any

from openai import AsyncOpenAI

from core.llm_interface import BaseLLMProvider
from core.types import LLMResponse, Message, Role, ToolCall


class OpenAICompatibleProvider(BaseLLMProvider):
    """Works with any OpenAI-compatible API: OpenAI, DeepSeek, Qwen, Moonshot, etc."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        env_key: str = "OPENAI_API_KEY",
    ):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ[env_key],
            base_url=base_url,
        )

    def get_model_name(self) -> str:
        return self.model

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        api_messages = [
            {"role": m.role.value, "content": m.content} for m in messages
        ]

        params: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", 16000),
        }
        if tools:
            params["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]

        response = await self.client.chat.completions.create(**params)
        choice = response.choices[0].message

        tool_calls = []
        if choice.tool_calls:
            import json
            for tc in choice.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                        call_id=tc.id,
                    )
                )

        return LLMResponse(
            content=choice.content or "",
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            model=self.model,
        )
