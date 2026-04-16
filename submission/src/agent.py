"""Competition submission agent implementation placeholder."""

from agent_base import BaseAgent, AgentInput, AgentOutput, ACTION_COMPLETE


class Agent(BaseAgent):
    def act(self, input_data: AgentInput) -> AgentOutput:
        # Placeholder to keep submission package runnable before strategy implementation.
        return AgentOutput(action=ACTION_COMPLETE, parameters={}, raw_output="TODO")
