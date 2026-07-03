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

    def _run_cli(self, prompt: str, settings: Optional[str] = None, json_schema: Optional[str] = None) -> str:
        cmd = [_CLAUDE_EXE, "--print", "--output-format", "json", "--tools", "none", "--model", self.model]
        if settings:
            cmd += ["--settings", settings]
        if json_schema:
            cmd += ["--json-schema", json_schema]
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout + 60)
        try:
            return json.loads(result.stdout).get("result", "").strip()
        except json.JSONDecodeError:
            return result.stdout.strip()

    def _cli_invoke(self, messages: List[BaseMessage]) -> AIMessage:
        system_parts = []
        conversation = []
        last_role = None

        for msg in messages:
            role = getattr(msg, "role", "user")
            content = msg.content or ""
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                conversation.append(("user", content))
                last_role = "user"
            elif role == "assistant":
                if getattr(msg, "tool_calls", None):
                    tc = msg.tool_calls[0]
                    args = json.loads(tc.function.arguments)
                    conversation.append(("assistant_tool", (tc.function.name, args)))
                else:
                    conversation.append(("assistant", content))
                last_role = "assistant"
            elif role == "tool":
                conversation.append(("tool_result", (msg.name, content)))
                last_role = "tool"

        # ── Case 1: last message is a tool result → get final answer ──────
        if last_role == "tool":
            parts = []
            if system_parts:
                parts.append("\n".join(system_parts))
            for kind, val in conversation:
                if kind == "user":
                    parts.append(f"User: {val}")
                elif kind == "assistant":
                    parts.append(f"Assistant: {val}")
                elif kind == "assistant_tool":
                    name, args = val
                    parts.append(f"Assistant called tool {name!r} with {args}")
                elif kind == "tool_result":
                    name, result = val
                    parts.append(f"Tool {name!r} returned: {result}")
            parts.append("Now give a concise final answer to the user using the tool result.")
            return AIMessage(content=self._run_cli("\n".join(parts)))

        # ── Case 2: tools registered → decide which tool to call ──────────
        if self.tools:
            tool_list = []
            for t in self.tools.values():
                fd = t.dict()["function"]
                params = list(fd["parameters"]["properties"].keys())
                tool_list.append(f'{fd["name"]}({", ".join(params)}): {fd["description"].split(chr(10))[0]}')

            # Step 1: ask the model to identify the tool call
            user_msg = conversation[-1][1] if conversation and conversation[-1][0] == "user" else ""
            system_ctx = ("\n".join(system_parts) + "\n") if system_parts else ""
            decision_prompt = (
                f"{system_ctx}"
                f"Available tools:\n" + "\n".join(f"  - {t}" for t in tool_list) + "\n\n"
                f"User request: {user_msg}\n\n"
                "Reply with ONLY a JSON object like this (no other text):\n"
                '{"name":"<tool_name>","arguments":{"<param>":"<value>"}}\n'
                "If no tool is needed, reply with: null"
            )

            schema = json.dumps({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            })

            raw = self._run_cli(decision_prompt, json_schema=schema)

            tc_data = None
            if raw and raw.lower() != "null":
                try:
                    tc_data = json.loads(raw)
                except json.JSONDecodeError:
                    m = re.search(r'\{"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*(\{[^}]*\})', raw, re.DOTALL)
                    if m:
                        try:
                            tc_data = {"name": m.group(1), "arguments": json.loads(m.group(2))}
                        except json.JSONDecodeError:
                            pass

            if tc_data and "name" in tc_data and tc_data["name"] in self.tools:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                return AIMessage(content=None, tool_calls=[SimpleNamespace(
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(
                        name=tc_data["name"],
                        arguments=json.dumps(tc_data.get("arguments", {})),
                    ),
                )])

        # ── Case 3: plain conversation ─────────────────────────────────────
        parts = []
        if system_parts:
            parts.append("\n".join(system_parts))
        for kind, val in conversation:
            if kind == "user":
                parts.append(f"Human: {val}")
            elif kind == "assistant":
                parts.append(f"Assistant: {val}")
        return AIMessage(content=self._run_cli("\n".join(parts)))

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
