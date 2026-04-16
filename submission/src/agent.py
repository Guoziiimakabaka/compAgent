"""DeepSeek-reasoner driven agent implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from agent_base import ACTION_CLICK, ACTION_OPEN, ACTION_TYPE, AgentInput, AgentOutput, BaseAgent
from utils.action_parser import parse_action_output
from utils.task_policy import TaskPolicy


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

    def _initialize(self) -> None:
        self._policy = TaskPolicy()

    def act(self, input_data: AgentInput) -> AgentOutput:
        if not self.api_key:
            raise RuntimeError("VLM_API_KEY is empty. Please configure it in .env or env vars.")

        policy_decision = self._policy.decide(
            instruction=input_data.instruction,
            history_actions=input_data.history_actions,
        )
        if policy_decision is not None:
            raw_output = json.dumps(
                {"action": policy_decision.action, "parameters": policy_decision.parameters},
                ensure_ascii=False,
            )
            return AgentOutput(
                action=policy_decision.action,
                parameters=policy_decision.parameters,
                raw_output=raw_output,
                usage=None,
            )

        messages = self.generate_messages(input_data)
        response = self._call_api(messages)

        raw_content = self._extract_content(response)
        parsed = parse_action_output(raw_content)
        usage = self.extract_usage_info(response)

        output = AgentOutput(
            action=parsed.action,
            parameters=parsed.parameters,
            raw_output=raw_content,
            usage=usage,
        )
        if input_data.step_count == 1:
            return self._sanitize_first_step(output, input_data.instruction)
        return output

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        history_json = json.dumps(input_data.history_actions[-5:], ensure_ascii=False)

        system_prompt = (
            "You are a mobile GUI agent. "
            "Return one next-step action in strict JSON only. "
            "Do not output markdown or any extra text. "
            "Prefer CLICK on existing UI controls when the app is already open. "
            "Avoid OPEN unless the target app is clearly not open from screenshot."
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
            "Coordinate rule: all x/y must be integer in [0,1000].\n"
            "Step-1 policy:\n"
            "- If there are obvious buttons/tabs/inputs in screenshot, choose CLICK.\n"
            "- Use TYPE only when text input is clearly focused.\n"
            "- Use OPEN only when screenshot is clearly home launcher or not inside target app."
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(input_data.current_image)},
                    },
                ],
            },
        ]

    def _sanitize_first_step(self, output: AgentOutput, instruction: str) -> AgentOutput:
        """Conservative guard for unknown tasks at first step."""
        if output.action == ACTION_OPEN:
            return AgentOutput(
                action=ACTION_CLICK,
                parameters={"point": [500, 500]},
                raw_output=(
                    f"{output.raw_output}\n"
                    "[fallback] replaced OPEN with safe center CLICK on first step"
                ),
                usage=output.usage,
            )
        if output.action == ACTION_TYPE:
            text_value = output.parameters.get("text", "")
            looks_like_sentence = any(token in text_value for token in ["，", "。", ",", ".", "！", "？"])
            instruction_needs_click_first = any(
                token in instruction for token in ["评论", "好评", "评价", "留言", "发布"]
            )
            if len(text_value) > 20 or looks_like_sentence or instruction_needs_click_first:
                return AgentOutput(
                    action=ACTION_CLICK,
                    parameters={"point": [500, 500]},
                    raw_output=(
                        f"{output.raw_output}\n"
                        "[fallback] replaced first-step TYPE with safe center CLICK"
                    ),
                    usage=output.usage,
                )
        return output

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
