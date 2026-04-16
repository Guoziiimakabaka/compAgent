"""Call local FastAPI app directly for smoke testing."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient

from api_server import app


def load_sample_image_base64() -> str:
    """Load one local test image and convert it to base64."""

    repo_root = Path(__file__).resolve().parents[2]
    offline_dir = repo_root / "code-for-student" / "test_data" / "offline"
    sample_image = next(offline_dir.rglob("*.png"))
    image_bytes = sample_image.read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


def main() -> None:
    """Send one predict request and print the response."""

    client = TestClient(app)
    payload = {
        "instruction": "打开抖音并搜索周杰伦",
        "image_base64": load_sample_image_base64(),
        "step_count": 0,
        "history_actions": [],
    }

    response = client.post("/predict", json=payload)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
