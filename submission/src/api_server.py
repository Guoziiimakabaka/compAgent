"""FastAPI service for local demo and API smoke tests."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field
from PIL import Image

from agent import Agent
from agent_base import AgentInput


class PredictRequest(BaseModel):
    """Input contract for one-step action prediction."""

    instruction: str = Field(min_length=1)
    image_base64: str = Field(min_length=1)
    step_count: int = Field(default=0, ge=0)
    history_actions: List[Dict[str, Any]] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """Normalized action response."""

    action: str
    parameters: Dict[str, Any]
    raw_output: str


app = FastAPI(title="Competition Agent API", version="0.1.0")
agent = Agent()


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode base64 image payload to RGB PIL image."""

    payload = image_base64
    if image_base64.startswith("data:"):
        _, payload = image_base64.split(",", maxsplit=1)

    binary = base64.b64decode(payload, validate=True)
    image = Image.open(io.BytesIO(binary))
    return image.convert("RGB")


@app.get("/health")
def health() -> Dict[str, str]:
    """Service liveness endpoint."""

    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run one-step GUI action prediction."""

    image = decode_base64_image(request.image_base64)
    agent_input = AgentInput(
        instruction=request.instruction,
        current_image=image,
        step_count=request.step_count,
        history_actions=request.history_actions,
    )
    output = agent.act(agent_input)

    return PredictResponse(
        action=output.action,
        parameters=output.parameters,
        raw_output=output.raw_output,
    )
