"""DeepSeek-reasoner driven agent implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from agent_base import (
    ACTION_CLICK,
    ACTION_OPEN,
    ACTION_TYPE,
    AgentInput,
    AgentOutput,
    BaseAgent,
    UsageInfo,
)
from utils.action_parser import ActionParseError, parse_action_output
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
        usage = self.extract_usage_info(response)
        parsed = self._parse_with_retry(raw_content, input_data, usage)

        output = AgentOutput(
            action=parsed.action,
            parameters=parsed.parameters,
            raw_output=raw_content,
            usage=usage,
        )
        if input_data.step_count == 1:
            return self._sanitize_first_step(output, input_data)
        return output

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        history_json = json.dumps(input_data.history_actions[-5:], ensure_ascii=False)

        system_prompt = (
            "You are a mobile GUI agent. "
            "Return one next-step action in strict JSON only. "
            "Do not output markdown or any extra text. "
            "Prefer CLICK on existing UI controls when the app is already open. "
            "Avoid OPEN unless the target app is clearly not open from screenshot. "
            "Never output long TYPE content on step 1."
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
            "- Use OPEN only when screenshot is clearly home launcher or not inside target app.\n"
            "- If uncertain at step 1, output CLICK instead of OPEN/TYPE."
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

    def _sanitize_first_step(self, output: AgentOutput, input_data: AgentInput) -> AgentOutput:
        """Conservative guard and recovery for unknown tasks at first step."""
        if output.action == ACTION_CLICK:
            return output

        if output.action == ACTION_OPEN and self._instruction_explicit_open(input_data.instruction):
            return output

        recovered = self._retry_first_step_click(input_data, output)
        if recovered is not None:
            return recovered

        if output.action == ACTION_TYPE:
            text_value = output.parameters.get("text", "")
            looks_like_sentence = any(token in text_value for token in ["，", "。", ",", ".", "！", "？"])
            instruction_needs_click_first = any(
                token in input_data.instruction for token in ["评论", "好评", "评价", "留言", "发布"]
            )
            if len(text_value) <= 20 and not looks_like_sentence and not instruction_needs_click_first:
                return output

        return AgentOutput(
            action=ACTION_CLICK,
            parameters={"point": [500, 500]},
            raw_output=(
                f"{output.raw_output}\n"
                "[fallback] replaced first-step non-CLICK with safe center CLICK"
            ),
            usage=output.usage,
        )

    def _retry_first_step_click(
        self,
        input_data: AgentInput,
        current_output: AgentOutput,
    ) -> AgentOutput | None:
        history_json = json.dumps(input_data.history_actions[-5:], ensure_ascii=False)
        prompt = (
            "Task instruction:\n"
            f"{input_data.instruction}\n\n"
            f"step_count={input_data.step_count}\n"
            f"Recent history actions: {history_json}\n\n"
            "Previous model output was not acceptable for step 1:\n"
            f"{current_output.raw_output}\n\n"
            "Now output ONLY one CLICK action in strict JSON:\n"
            "{\"action\":\"CLICK\",\"parameters\":{\"point\":[x,y]}}\n"
            "Rules:\n"
            "- x/y must be integer in [0,1000]\n"
            "- choose a visible tappable control in screenshot\n"
            "- do not output OPEN/TYPE/SCROLL/COMPLETE"
        )
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "You are a mobile GUI agent. "
                    "For this retry you must output CLICK JSON only."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(input_data.current_image)},
                    },
                ],
            },
        ]

        retry_response = self._call_api(retry_messages)
        retry_raw = self._extract_content(retry_response)
        retry_usage = self.extract_usage_info(retry_response)
        merged_usage = self._merge_usage(current_output.usage, retry_usage)
        retry_parsed = parse_action_output(retry_raw)

        if retry_parsed.action == ACTION_CLICK:
            return AgentOutput(
                action=retry_parsed.action,
                parameters=retry_parsed.parameters,
                raw_output=(
                    f"{current_output.raw_output}\n"
                    "[retry-first-step]\n"
                    f"{retry_raw}"
                ),
                usage=merged_usage,
            )

        if retry_parsed.action == ACTION_OPEN and self._instruction_explicit_open(input_data.instruction):
            return AgentOutput(
                action=retry_parsed.action,
                parameters=retry_parsed.parameters,
                raw_output=(
                    f"{current_output.raw_output}\n"
                    "[retry-first-step]\n"
                    f"{retry_raw}"
                ),
                usage=merged_usage,
            )

        return None

    def _parse_with_retry(
        self,
        raw_content: str,
        input_data: AgentInput,
        usage: UsageInfo | None,
    ):
        try:
            return parse_action_output(raw_content)
        except ActionParseError:
            repair_messages = self._build_repair_messages(input_data, raw_content)
            repair_response = self._call_api(repair_messages)
            repair_raw = self._extract_content(repair_response)
            repair_usage = self.extract_usage_info(repair_response)
            merged_usage = self._merge_usage(usage, repair_usage)
            if merged_usage is not None:
                usage.input_tokens = merged_usage.input_tokens
                usage.output_tokens = merged_usage.output_tokens
                usage.total_tokens = merged_usage.total_tokens
                usage.cached_tokens = merged_usage.cached_tokens
                usage.reasoning_tokens = merged_usage.reasoning_tokens
            return parse_action_output(repair_raw)

    def _build_repair_messages(self, input_data: AgentInput, raw_content: str) -> List[Dict[str, Any]]:
        prompt = (
            "Task instruction:\n"
            f"{input_data.instruction}\n\n"
            "Previous output failed parser:\n"
            f"{raw_content}\n\n"
            "Rewrite into strict JSON with this schema only:\n"
            "{\"action\":\"CLICK|TYPE|SCROLL|OPEN|COMPLETE\",\"parameters\":{...}}\n"
            "Do not include any extra text."
        )
        return [
            {
                "role": "system",
                "content": "You are a formatter that outputs strict action JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(input_data.current_image)},
                    },
                ],
            },
        ]

    def _instruction_explicit_open(self, instruction: str) -> bool:
        open_keywords = ["打开", "启动", "进入", "open"]
        return any(keyword in instruction.lower() for keyword in open_keywords)

    def _merge_usage(self, left: UsageInfo | None, right: UsageInfo | None) -> UsageInfo | None:
        if left is None:
            return right
        if right is None:
            return left
        return UsageInfo(
            input_tokens=left.input_tokens + right.input_tokens,
            output_tokens=left.output_tokens + right.output_tokens,
            total_tokens=left.total_tokens + right.total_tokens,
            cached_tokens=left.cached_tokens + right.cached_tokens,
            reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
        )

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
