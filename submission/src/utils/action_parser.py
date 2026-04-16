"""Utilities for parsing model output into valid competition actions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from agent_base import (
    ACTION_CLICK,
    ACTION_COMPLETE,
    ACTION_OPEN,
    ACTION_SCROLL,
    ACTION_TYPE,
    VALID_ACTIONS,
)


ACTION_ALIAS = {
    "click": ACTION_CLICK,
    "type": ACTION_TYPE,
    "scroll": ACTION_SCROLL,
    "open": ACTION_OPEN,
    "complete": ACTION_COMPLETE,
}


@dataclass(frozen=True)
class ParsedAction:
    """Normalized action parsed from model output."""

    action: str
    parameters: Dict[str, Any]


class ActionParseError(ValueError):
    """Raised when model output cannot be parsed into valid action."""



def parse_action_output(raw_text: str) -> ParsedAction:
    """Parse model output text into validated action and parameters."""

    payload = _load_payload(raw_text)
    action = _normalize_action(payload.get("action"))
    parameters = _normalize_parameters(action=action, parameters=payload.get("parameters"))
    return ParsedAction(action=action, parameters=parameters)



def _load_payload(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ActionParseError("model output is empty")

    candidates = [text]

    code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_block:
        candidates.insert(0, code_block.group(1).strip())

    brace_block = re.search(r"(\{[\s\S]*\})", text)
    if brace_block:
        candidates.append(brace_block.group(1).strip())

    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded

    raise ActionParseError("model output is not valid JSON object")



def _normalize_action(raw_action: Any) -> str:
    if not isinstance(raw_action, str):
        raise ActionParseError("field 'action' must be string")

    candidate = raw_action.strip()
    if not candidate:
        raise ActionParseError("field 'action' is empty")

    upper_candidate = candidate.upper()
    if upper_candidate in VALID_ACTIONS:
        return upper_candidate

    alias_value = ACTION_ALIAS.get(candidate.lower())
    if alias_value:
        return alias_value

    raise ActionParseError(f"unsupported action: {raw_action}")



def _normalize_parameters(action: str, parameters: Any) -> Dict[str, Any]:
    if parameters is None:
        parameters = {}

    if not isinstance(parameters, dict):
        raise ActionParseError("field 'parameters' must be object")

    if action == ACTION_CLICK:
        point = _normalize_point(parameters.get("point"))
        return {"point": point}

    if action == ACTION_TYPE:
        text_value = parameters.get("text")
        if text_value is None:
            text_value = parameters.get("content")
        if not isinstance(text_value, str):
            raise ActionParseError("TYPE requires string field 'text'")
        return {"text": text_value}

    if action == ACTION_SCROLL:
        start_point = _normalize_point(parameters.get("start_point"))
        end_point = _normalize_point(parameters.get("end_point"))
        return {
            "start_point": start_point,
            "end_point": end_point,
        }

    if action == ACTION_OPEN:
        app_name = parameters.get("app_name")
        if app_name is None:
            app_name = parameters.get("app")
        if not isinstance(app_name, str):
            raise ActionParseError("OPEN requires string field 'app_name'")
        return {"app_name": app_name}

    if action == ACTION_COMPLETE:
        return {}

    raise ActionParseError(f"unknown action during parameter normalization: {action}")



def _normalize_point(value: Any) -> List[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ActionParseError("point must be [x, y]")

    x_value = _normalize_coord(value[0])
    y_value = _normalize_coord(value[1])
    return [x_value, y_value]



def _normalize_coord(value: Any) -> int:
    if not isinstance(value, (int, float)):
        raise ActionParseError("coordinate must be number")
    rounded = int(round(value))
    return max(0, min(1000, rounded))
