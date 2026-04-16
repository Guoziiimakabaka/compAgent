"""Rule-based decision policy for minimal runnable demo."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from agent_base import (
    ACTION_CLICK,
    ACTION_COMPLETE,
    ACTION_OPEN,
    ACTION_SCROLL,
    ACTION_TYPE,
)


@dataclass(frozen=True)
class ActionDecision:
    """Structured action returned by the planner."""

    action: str
    parameters: Dict[str, Any]
    reason: str


class RuleBasedPlanner:
    """Simple deterministic planner for demo and smoke testing."""

    APP_KEYWORDS = {
        "美团": "美团",
        "抖音": "抖音",
        "淘宝": "淘宝",
        "京东": "京东",
        "高德": "高德地图",
        "微信": "微信",
        "bilibili": "哔哩哔哩",
        "哔哩": "哔哩哔哩",
    }

    def plan(
        self,
        instruction: str,
        image: Image.Image,
        step_count: int,
        history_actions: List[Dict[str, Any]],
    ) -> ActionDecision:
        text = instruction.strip()
        if not text:
            return ActionDecision(ACTION_COMPLETE, {}, "empty instruction")

        app_name = self._extract_app_name(text)
        if app_name and step_count == 0:
            return ActionDecision(
                ACTION_OPEN,
                {"app_name": app_name},
                "open app from instruction",
            )

        typed_text = self._extract_type_text(text)
        if typed_text:
            return ActionDecision(
                ACTION_TYPE,
                {"text": typed_text},
                "found explicit type/search intent",
            )

        if self._has_scroll_intent(text):
            return ActionDecision(
                ACTION_SCROLL,
                {
                    "start_point": [500, 820],
                    "end_point": [500, 240],
                },
                "scroll intent detected",
            )

        if self._has_click_intent(text):
            point = self._find_click_point(image)
            return ActionDecision(
                ACTION_CLICK,
                {"point": point},
                "click intent detected",
            )

        if step_count >= 3 or self._last_action_is_complete(history_actions):
            return ActionDecision(ACTION_COMPLETE, {}, "safe stop")

        return ActionDecision(
            ACTION_CLICK,
            {"point": [500, 500]},
            "fallback center click",
        )

    def _extract_app_name(self, text: str) -> str | None:
        lowered = text.lower()
        for keyword, app_name in self.APP_KEYWORDS.items():
            if keyword.lower() in lowered:
                return app_name
        return None

    def _extract_type_text(self, text: str) -> str | None:
        if not any(token in text for token in ("输入", "搜索", "查找", "填写")):
            return None

        quoted_match = re.search(r"[\"'“”‘’](.*?)[\"'“”‘’]", text)
        if quoted_match:
            candidate = quoted_match.group(1).strip()
            if candidate:
                return candidate

        search_match = re.search(r"搜索(.+)$", text)
        if search_match:
            candidate = search_match.group(1).strip(" ：:，。")
            if candidate:
                return candidate

        return "测试"

    def _has_scroll_intent(self, text: str) -> bool:
        return any(token in text for token in ("滚动", "下滑", "上滑", "翻页", "浏览"))

    def _has_click_intent(self, text: str) -> bool:
        return any(token in text for token in ("点击", "进入", "选择", "打开"))

    def _last_action_is_complete(self, history_actions: List[Dict[str, Any]]) -> bool:
        if not history_actions:
            return False
        return history_actions[-1].get("action") == ACTION_COMPLETE

    def _find_click_point(self, image: Image.Image) -> List[int]:
        gray = image.convert("L").resize((64, 64))
        gray_array = np.asarray(gray)
        flat_index = int(gray_array.argmax())
        y_idx, x_idx = np.unravel_index(flat_index, gray_array.shape)

        x_value = int(round(x_idx / 63 * 1000))
        y_value = int(round(y_idx / 63 * 1000))
        return [self._clamp_coord(x_value), self._clamp_coord(y_value)]

    def _clamp_coord(self, value: int) -> int:
        return max(0, min(1000, value))
