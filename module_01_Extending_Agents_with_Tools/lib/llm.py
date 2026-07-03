import os
import re
import json
import uuid
import subprocess
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

_CLAUDE_EXE = os.path.join(
    os.path.expanduser("~"),
    r".vscode\extensions\anthropic.claude-code-2.1.200-win32-x64\resources\native-binary\claude.exe",
)


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

        if self.provider not in {"openai", "anthropic", "claude", "claude_code"}:
            raise ValueError(
                "provider must be 'openai', 'anthropic', 'claude', or 'claude_code'"
            )

        if self.provider == "claude_code":
            self.model = model or os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6")
            self.client = None
            return

        auth_token = None
        if api_key is None:
            if self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key is None:
                    auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")

        if self.provider == "openai":
            self.model = model or "gpt-4o-mini"
            if base_url is None:
                base_url = os.getenv("OPENAI_BASE_URL")
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout) if api_key else OpenAI(base_url=base_url, timeout=timeout)
        else:
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            if auth_token:
                self.client = Anthropic(auth_token=auth_token, timeout=timeout)
            elif api_key:
                self.client = Anthropic(api_key=api_key, timeout=timeout)
            else:
                self.client = Anthropic(timeout=timeout)

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    # ── Claude Code CLI provider ──────────────────────────────────────────

    def _cli_invoke(self, messages: List[BaseMessage]) -> AIMessage:
        parts = []

        if self.tools:
            tool_lines = []
            for t in self.tools.values():
                fd = t.dict()["function"]
                params = ", ".join(fd["parameters"]["properties"].keys())
                desc = fd["description"].split("\n")[0]
                tool_lines.append(f"  - {fd['name']}({params}): {desc}")
            parts.append(
                "You have access to these tools. When you need to call one, "
                "output ONLY this on its own line (nothing else before or after):\n"
                '<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>\n\n'
                "Available tools:\n" + "\n".join(tool_lines)
            )

        for msg in messages:
            role = getattr(msg, "role", "user")
            content = msg.content or ""
            if role == "system":
                parts.insert(0, f"[System instructions: {content}]\n")
            elif role == "user":
                parts.append(f"Human: {content}")
            elif role == "assistant":
                if getattr(msg, "tool_calls", None):
                    tc = msg.tool_calls[0]
                    args = json.loads(tc.function.arguments)
                    parts.append(
                        f'Assistant: <tool_call>{{"name": "{tc.function.name}", "arguments": {json.dumps(args)}}}</tool_call>'
                    )
                else:
                    parts.append(f"Assistant: {content}")
            elif role == "tool":
                parts.append(f"Tool result ({msg.name}): {content}\n\nNow give your final answer to the user.")

        prompt = "\n".join(parts)

        result = subprocess.run(
            [_CLAUDE_EXE, "--print", "--output-format", "json", "--tools", "none", "--model", self.model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=self.timeout + 60,
        )

        try:
            output = json.loads(result.stdout)
            text = output.get("result", "")
        except json.JSONDecodeError:
            text = result.stdout.strip()

        match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if match and self.tools:
            try:
                tc_data = json.loads(match.group(1))
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_call = SimpleNamespace(
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(
                        name=tc_data["name"],
                        arguments=json.dumps(tc_data["arguments"]),
                    ),
                )
                return AIMessage(content=None, tool_calls=[tool_call])
            except (json.JSONDecodeError, KeyError):
                pass

        return AIMessage(content=text)

    # ── Standard SDK providers ────────────────────────────────────────────

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

        if self.provider == "claude_code":
            return self._cli_invoke(messages)

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
