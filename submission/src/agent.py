"""DeepSeek-reasoner driven agent implementation."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from agent_base import AgentInput, AgentOutput, BaseAgent
from utils.action_parser import parse_action_output

DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=False)


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
