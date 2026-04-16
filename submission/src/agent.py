"""DeepSeek-reasoner driven agent implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from agent_base import AgentInput, AgentOutput, BaseAgent
from utils.action_parser import parse_action_output

DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=False)


class Agent(BaseAgent):
    """Competition agent powered by DeepSeek reasoner model."""

    def _initialize(self) -> None:
        self._client = OpenAI(base_url=self.api_url, api_key=self.api_key)

    def act(self, input_data: AgentInput) -> AgentOutput:
        if not self.api_key:
            raise RuntimeError("VLM_API_KEY is empty. Please configure it in .env or env vars.")

        messages = self._build_messages(input_data)
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0,
        )

        content = self._extract_content(response)
        parsed = parse_action_output(content)
        usage = self.extract_usage_info(response)

        return AgentOutput(
            action=parsed.action,
            parameters=parsed.parameters,
            raw_output=content,
            usage=usage,
        )

    def _build_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        image = input_data.current_image
        history_text = json.dumps(input_data.history_actions[-5:], ensure_ascii=False)

        system_prompt = (
            "You are a mobile GUI agent. "
            "Return one next-step action in strict JSON only. "
            "Do not output markdown or extra commentary."
        )

        user_prompt = (
            "Task instruction:\n"
            f"{input_data.instruction}\n\n"
            "Current screenshot metadata:\n"
            f"width={image.width}, height={image.height}\n\n"
            f"Current step index: {input_data.step_count}\n"
            f"Recent action history (JSON): {history_text}\n\n"
            "Allowed actions and parameters:\n"
            "1) CLICK: {\"action\": \"CLICK\", \"parameters\": {\"point\": [x, y]}}\n"
            "2) TYPE: {\"action\": \"TYPE\", \"parameters\": {\"text\": \"...\"}}\n"
            "3) SCROLL: {\"action\": \"SCROLL\", \"parameters\": {\"start_point\": [x1, y1], \"end_point\": [x2, y2]}}\n"
            "4) OPEN: {\"action\": \"OPEN\", \"parameters\": {\"app_name\": \"...\"}}\n"
            "5) COMPLETE: {\"action\": \"COMPLETE\", \"parameters\": {}}\n\n"
            "Coordinate constraints:\n"
            "- All coordinates must be relative integers in [0, 1000].\n"
            "- Output must be a single JSON object with keys action and parameters."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_content(self, response: Any) -> str:
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
