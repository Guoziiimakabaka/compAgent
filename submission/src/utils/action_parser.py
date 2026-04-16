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

    shorthand_payload = _try_parse_shorthand(text)
    if shorthand_payload is not None:
        return shorthand_payload

    function_payload = _try_parse_function_call(text)
    if function_payload is not None:
        return function_payload

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


def _try_parse_shorthand(text: str) -> Dict[str, Any] | None:
    """
    Parse shorthand style output, for example:
    - CLICK:[[100,200]]
    - TYPE:['hello']
    - OPEN:['淘宝']
    - COMPLETE:[]
    """
    match = re.search(
        r"\b(CLICK|TYPE|SCROLL|OPEN|COMPLETE)\s*[:：]\s*([^\n\r]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    action_raw = match.group(1).upper()
    body = match.group(2).strip()

    if action_raw == ACTION_CLICK:
        numbers = _extract_numbers(body)
        if len(numbers) < 2:
            return None
        return {
            "action": ACTION_CLICK,
            "parameters": {"point": [numbers[0], numbers[1]]},
        }

    if action_raw == ACTION_SCROLL:
        numbers = _extract_numbers(body)
        if len(numbers) < 4:
            return None
        return {
            "action": ACTION_SCROLL,
            "parameters": {
                "start_point": [numbers[0], numbers[1]],
                "end_point": [numbers[2], numbers[3]],
            },
        }

    if action_raw == ACTION_OPEN:
        text_value = _extract_quoted_text(body)
        if text_value is None:
            return None
        return {"action": ACTION_OPEN, "parameters": {"app_name": text_value}}

    if action_raw == ACTION_TYPE:
        text_value = _extract_quoted_text(body)
        if text_value is None:
            return None
        return {"action": ACTION_TYPE, "parameters": {"text": text_value}}

    if action_raw == ACTION_COMPLETE:
        return {"action": ACTION_COMPLETE, "parameters": {}}

    return None


def _try_parse_function_call(text: str) -> Dict[str, Any] | None:
    """
    Parse function-call style output, for example:
    - click(point='<point>100 200</point>')
    - type(content='搜索词')
    - open(app_name='淘宝')
    """
    match = re.search(
        r"\b(click|type|scroll|open|complete)\s*\(([\s\S]*?)\)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    action_name = match.group(1).lower()
    arguments = match.group(2).strip()
    action = ACTION_ALIAS.get(action_name)
    if action is None:
        return None

    if action == ACTION_CLICK:
        numbers = _extract_numbers(arguments)
        if len(numbers) < 2:
            return None
        return {"action": ACTION_CLICK, "parameters": {"point": [numbers[0], numbers[1]]}}

    if action == ACTION_SCROLL:
        numbers = _extract_numbers(arguments)
        if len(numbers) < 4:
            return None
        return {
            "action": ACTION_SCROLL,
            "parameters": {
                "start_point": [numbers[0], numbers[1]],
                "end_point": [numbers[2], numbers[3]],
            },
        }

    if action == ACTION_OPEN:
        app_name = _extract_key_value_string(arguments, ["app_name", "app"])
        if app_name is None:
            app_name = _extract_quoted_text(arguments)
        if app_name is None:
            return None
        return {"action": ACTION_OPEN, "parameters": {"app_name": app_name}}

    if action == ACTION_TYPE:
        text_value = _extract_key_value_string(arguments, ["text", "content"])
        if text_value is None:
            text_value = _extract_quoted_text(arguments)
        if text_value is None:
            return None
        return {"action": ACTION_TYPE, "parameters": {"text": text_value}}

    if action == ACTION_COMPLETE:
        return {"action": ACTION_COMPLETE, "parameters": {}}

    return None


def _extract_key_value_string(arguments: str, keys: List[str]) -> str | None:
    for key in keys:
        pattern = rf"{key}\s*=\s*['\"]([^'\"]+)['\"]"
        match = re.search(pattern, arguments, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_quoted_text(text: str) -> str | None:
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1).strip()
    return None


def _extract_numbers(text: str) -> List[int]:
    values = re.findall(r"-?\d+(?:\.\d+)?", text)
    return [int(round(float(value))) for value in values]



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
