import os
import json
from types import SimpleNamespace
from typing import List, Optional, Dict, Any
from openai import OpenAI
from anthropic import Anthropic
from lib.messages import (
    AnyMessage,
    AIMessage,
    BaseMessage,
    UserMessage,
)
from lib.tooling import Tool


class LLM:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        tools: Optional[List[Tool]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "openai",
        timeout: float = 30.0,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.timeout = timeout
        self.tools: Dict[str, Tool] = {
            tool.name: tool for tool in (tools or [])
        }

        if self.provider not in {"openai", "anthropic", "claude"}:
            raise ValueError(
                "provider must be 'openai', 'anthropic', or 'claude'"
            )

        if api_key is None:
            if self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY")

        if self.provider == "openai":
            self.model = model or "gpt-4o-mini"
            if base_url is None:
                base_url = os.getenv("OPENAI_BASE_URL")
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout) if api_key else OpenAI(base_url=base_url, timeout=timeout)
        else:
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
            self.client = Anthropic(api_key=api_key, timeout=timeout) if api_key else Anthropic(timeout=timeout)

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def _build_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        if self.provider == "openai":
            return self._build_openai_payload(messages)
        return self._build_anthropic_payload(messages)

    def _build_openai_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [m.dict() for m in messages],
            "temperature": self.temperature,
        }

        if self.tools:
            payload["tools"] = [tool.dict() for tool in self.tools.values()]
            payload["tool_choice"] = "auto"

        return payload

    def _build_anthropic_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        system_messages = []
        api_messages = []

        for message in messages:
            role = getattr(message, "role", None)
            content = message.content or ""

            if role == "system":
                system_messages.append(content)
            elif role == "tool":
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": content,
                    }],
                })
            elif role == "assistant" and getattr(message, "tool_calls", None):
                api_messages.append({
                    "role": "assistant",
                    "content": self._anthropic_assistant_content(message),
                })
            elif role in {"user", "assistant"}:
                api_messages.append({"role": role, "content": content})
            else:
                raise ValueError(f"Unsupported message role for Anthropic: {role}")

        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": 2048,
            "temperature": self.temperature,
        }

        if system_messages:
            payload["system"] = "\n\n".join(system_messages)

        if self.tools:
            payload["tools"] = [self._anthropic_tool_schema(tool) for tool in self.tools.values()]
            payload["tool_choice"] = {"type": "auto"}

        return payload

    def _anthropic_tool_schema(self, tool: Tool) -> Dict[str, Any]:
        function = tool.dict()["function"]
        return {
            "name": function["name"],
            "description": function["description"],
            "input_schema": function["parameters"],
        }

    def _anthropic_assistant_content(self, message: AIMessage) -> List[Dict[str, Any]]:
        content = []
        if message.content:
            content.append({"type": "text", "text": message.content})

        for call in message.tool_calls or []:
            function = self._tool_call_function(call)
            content.append({
                "type": "tool_use",
                "id": self._tool_call_id(call),
                "name": function.name,
                "input": json.loads(function.arguments),
            })

        return content

    def _tool_call_id(self, call: Any) -> str:
        if isinstance(call, dict):
            return call["id"]
        return call.id

    def _tool_call_function(self, call: Any) -> Any:
        if isinstance(call, dict):
            return SimpleNamespace(**call["function"])
        return call.function

    def _convert_input(self, input: Any) -> List[BaseMessage]:
        if isinstance(input, str):
            return [UserMessage(content=input)]
        elif isinstance(input, BaseMessage):
            return [input]
        elif isinstance(input, list):
            if all(isinstance(m, BaseMessage) for m in input):
                return input
            if all(callable(getattr(m, "dict", None)) for m in input):
                return input
        raise ValueError(f"Invalid input type {type(input)}.")

    def invoke(self, input: str | BaseMessage | List[BaseMessage]) -> AIMessage:
        messages = self._convert_input(input)
        payload = self._build_payload(messages)

        if self.provider == "openai":
            response = self.client.chat.completions.create(**payload)
            choice = response.choices[0]
            message = choice.message
            content = message.content
            tool_calls = getattr(message, "tool_calls", None)
        else:
            response = self.client.messages.create(**payload)
            content = self._anthropic_text_content(response)
            tool_calls = self._anthropic_tool_calls(response)

        return AIMessage(
            content=content,
            tool_calls=tool_calls,
        )

    def _anthropic_text_content(self, response: Any) -> Optional[str]:
        text_parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        return "\n".join(text_parts) if text_parts else None

    def _anthropic_tool_calls(self, response: Any) -> Optional[List[Any]]:
        tool_calls = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(SimpleNamespace(
                    id=block.id,
                    type="function",
                    function=SimpleNamespace(
                        name=block.name,
                        arguments=json.dumps(block.input or {}),
                    ),
                ))
        return tool_calls or None
