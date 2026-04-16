"""DeepSeek-reasoner driven agent implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from agent_base import AgentInput, AgentOutput, BaseAgent
from utils.action_parser import parse_action_output


def _load_env() -> None:
    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parent / ".env",
        current_file.parent.parent / ".env",
        current_file.parent.parent.parent / ".env",
        Path.cwd() / ".env",
    ]

    for dotenv_path in candidates:
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)
            return


_load_env()


class Agent(BaseAgent):
    """Competition agent powered by DeepSeek reasoner model."""

    def act(self, input_data: AgentInput) -> AgentOutput:
        if not self.api_key:
            raise RuntimeError("VLM_API_KEY is empty. Please configure it in .env or env vars.")

        messages = self.generate_messages(input_data)
        response = self._call_api(messages)

        raw_content = self._extract_content(response)
        parsed = parse_action_output(raw_content)
        usage = self.extract_usage_info(response)

        return AgentOutput(
            action=parsed.action,
            parameters=parsed.parameters,
            raw_output=raw_content,
            usage=usage,
        )

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        history_json = json.dumps(input_data.history_actions[-5:], ensure_ascii=False)

        system_prompt = (
            "You are a mobile GUI agent. "
            "Return one next-step action in strict JSON only. "
            "Do not output markdown or any extra text."
        )
        user_prompt = (
            "Task instruction:\n"
            f"{input_data.instruction}\n\n"
            "Current screenshot metadata:\n"
            f"width={input_data.current_image.width}, height={input_data.current_image.height}\n"
            f"step_count={input_data.step_count}\n\n"
            f"Recent history actions (JSON): {history_json}\n\n"
            "Action schema:\n"
            "1) CLICK: {\"action\":\"CLICK\",\"parameters\":{\"point\":[x,y]}}\n"
            "2) TYPE: {\"action\":\"TYPE\",\"parameters\":{\"text\":\"...\"}}\n"
            "3) SCROLL: {\"action\":\"SCROLL\",\"parameters\":{\"start_point\":[x1,y1],\"end_point\":[x2,y2]}}\n"
            "4) OPEN: {\"action\":\"OPEN\",\"parameters\":{\"app_name\":\"...\"}}\n"
            "5) COMPLETE: {\"action\":\"COMPLETE\",\"parameters\":{}}\n\n"
            "Coordinate rule: all x/y must be integer in [0,1000]."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_content(self, response: object) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("model response has no choices")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError("model response choice has no message")

        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("model response content is empty")

        return content.strip()
