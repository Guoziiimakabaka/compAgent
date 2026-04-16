"""Minimal runnable GUI agent implementation."""

from __future__ import annotations

import json

from agent_base import AgentInput, AgentOutput, BaseAgent
from utils.decision_policy import RuleBasedPlanner


class Agent(BaseAgent):
    """Competition agent with a deterministic rule-based policy."""

    def _initialize(self) -> None:
        self._planner = RuleBasedPlanner()

    def act(self, input_data: AgentInput) -> AgentOutput:
        decision = self._planner.plan(
            instruction=input_data.instruction,
            image=input_data.current_image,
            step_count=input_data.step_count,
            history_actions=input_data.history_actions,
        )
        raw_output = json.dumps(
            {
                "action": decision.action,
                "parameters": decision.parameters,
                "reason": decision.reason,
            },
            ensure_ascii=False,
        )
        return AgentOutput(
            action=decision.action,
            parameters=decision.parameters,
            raw_output=raw_output,
        )


