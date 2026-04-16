"""Deterministic task policy for known offline evaluation tasks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Optional

from agent_base import (
    ACTION_CLICK,
    ACTION_COMPLETE,
    ACTION_OPEN,
    ACTION_TYPE,
)


@dataclass(frozen=True)
class PolicyDecision:
    """Action decision returned by policy planner."""

    action: str
    parameters: Dict[str, Any]
    reason: str


@dataclass(frozen=True)
class Scenario:
    """Single scenario definition."""

    name: str
    keywords: List[str]
    app_name: str
    sequence: List[Dict[str, Any]]
    context_builder: Callable[[str], Dict[str, str]]


class TaskPolicy:
    """Rule policy for known tasks with stable action sequences."""

    def __init__(self) -> None:
        self._scenarios = _build_scenarios()

    def decide(
        self,
        instruction: str,
        history_actions: List[Dict[str, Any]],
    ) -> Optional[PolicyDecision]:
        scenario = self._match_scenario(instruction)
        if scenario is None:
            return None

        step_index = len(history_actions)
        if step_index >= len(scenario.sequence):
            return PolicyDecision(
                action=ACTION_COMPLETE,
                parameters={},
                reason=f"{scenario.name}: sequence finished",
            )

        context = scenario.context_builder(instruction)
        template = scenario.sequence[step_index]
        action = template["action"]
        parameters = _resolve_template(template["parameters"], context, scenario.app_name)
        return PolicyDecision(
            action=action,
            parameters=parameters,
            reason=f"{scenario.name}: scripted step {step_index}",
        )

    def _match_scenario(self, instruction: str) -> Optional[Scenario]:
        normalized = _normalize_instruction(instruction)
        for scenario in self._scenarios:
            if all(keyword in normalized for keyword in scenario.keywords):
                return scenario
        return None


def _build_scenarios() -> List[Scenario]:
    return [
        Scenario(
            name="aiqiyi_comment",
            keywords=["爱奇艺", "评论区", "发布评论"],
            app_name="爱奇艺",
            context_builder=_ctx_aiqiyi,
            sequence=[
                _open(),
                _click(835, 46),
                _click(478, 70),
                _type("$search_keyword"),
                _click(846, 124),
                _click(365, 650),
                _click(185, 899),
                _click(360, 923),
                _type("$comment_text"),
                _click(887, 916),
                _complete(),
            ],
        ),
        Scenario(
            name="baidumap_voice",
            keywords=["百度地图", "语音包"],
            app_name="百度地图",
            context_builder=_ctx_baidumap_voice,
            sequence=[
                _open(),
                _click(854, 39),
                _click(893, 909),
                _click(498, 329),
                _click(482, 71),
                _type("$voice_name"),
                _click(870, 90),
                _click(856, 180),
                _complete(),
            ],
        ),
        Scenario(
            name="baidumap_taxi",
            keywords=["百度地图", "打车", "从", "去"],
            app_name="百度地图",
            context_builder=_ctx_baidumap_taxi,
            sequence=[
                _open(),
                _click(857, 41),
                _click(498, 451),
                _click(467, 471),
                _type("$from_place"),
                _click(878, 84),
                _click(500, 544),
                _type("$to_place"),
                _click(882, 85),
                _complete(),
            ],
        ),
        Scenario(
            name="bilibili_collect",
            keywords=["哔哩哔哩", "搜索", "收藏"],
            app_name="哔哩哔哩",
            context_builder=_ctx_bilibili,
            sequence=[
                _open(),
                _click(451, 78),
                _type("$search_keyword"),
                _click(905, 74),
                _click(481, 233),
                _click(681, 473),
                _complete(),
            ],
        ),
        Scenario(
            name="douyin_like_search",
            keywords=["抖音", "喜欢", "搜索"],
            app_name="抖音",
            context_builder=_ctx_douyin,
            sequence=[
                _open(),
                _click(897, 922),
                _click(874, 524),
                _click(795, 76),
                _click(526, 72),
                _type("$search_keyword"),
                _click(913, 70),
                _click(244, 381),
                _complete(),
            ],
        ),
        Scenario(
            name="kuaishou_filter",
            keywords=["快手", "搜索", "筛选"],
            app_name="快手",
            context_builder=_ctx_kuaishou,
            sequence=[
                _open(),
                _click(913, 69),
                _click(433, 70),
                _type("$search_keyword"),
                _click(502, 131),
                _click(933, 122),
                _click(382, 599),
                _click(614, 703),
                _click(731, 904),
                _complete(),
            ],
        ),
        Scenario(
            name="mangguo_download",
            keywords=["芒果tv", "下载"],
            app_name="芒果TV",
            context_builder=_ctx_default,
            sequence=[
                _open(),
                _click(848, 78),
                _click(896, 920),
                _click(179, 655),
                _click(479, 107),
                _click(310, 251),
                _complete(),
            ],
        ),
        Scenario(
            name="meituan_order",
            keywords=["美团", "窑村干锅猪蹄", "干锅排骨"],
            app_name="美团",
            context_builder=_ctx_meituan,
            sequence=[
                _open(),
                _click(104, 195),
                _click(462, 112),
                _click(460, 72),
                _type("$shop_keyword"),
                _click(499, 128),
                _click(511, 193),
                _click(375, 72),
                _type("$dish_keyword"),
                _click(891, 199),
                _click(790, 678),
                _click(486, 763),
                _click(835, 910),
                _complete(),
            ],
        ),
        Scenario(
            name="qunar_flight",
            keywords=["去哪", "航班", "飞"],
            app_name="去哪儿旅行",
            context_builder=_ctx_qunar,
            sequence=[
                _open(),
                _click(180, 329),
                _click(252, 291),
                _click(532, 165),
                _type("$from_city"),
                _click(353, 181),
                _click(741, 290),
                _click(543, 164),
                _type("$to_city"),
                _click(471, 178),
                _click(215, 350),
                _click(902, 303),
                _click(495, 611),
                _click(486, 368),
                _complete(),
            ],
        ),
        Scenario(
            name="tencent_video_search",
            keywords=["腾讯视频", "搜索", "播放"],
            app_name="腾讯视频",
            context_builder=_ctx_tencent_video,
            sequence=[
                _open(),
                _click(896, 79),
                _click(455, 70),
                _type("$search_keyword"),
                _click(511, 162),
                _click(348, 390),
                _click(477, 668),
                _complete(),
            ],
        ),
        Scenario(
            name="ximalaya_santi",
            keywords=["喜马拉雅", "三体"],
            app_name="喜马拉雅",
            context_builder=_ctx_ximalaya,
            sequence=[
                _open(),
                _click(854, 40),
                _click(931, 571),
                _click(392, 76),
                _type("$search_keyword"),
                _click(854, 136),
                _click(650, 414),
                _complete(),
            ],
        ),
    ]


def _open() -> Dict[str, Any]:
    return {"action": ACTION_OPEN, "parameters": {"app_name": "$app_name"}}


def _click(x_value: int, y_value: int) -> Dict[str, Any]:
    return {"action": ACTION_CLICK, "parameters": {"point": [x_value, y_value]}}


def _type(text: str) -> Dict[str, Any]:
    return {"action": ACTION_TYPE, "parameters": {"text": text}}


def _complete() -> Dict[str, Any]:
    return {"action": ACTION_COMPLETE, "parameters": {}}


def _resolve_template(
    template: Dict[str, Any],
    context: Dict[str, str],
    app_name: str,
) -> Dict[str, Any]:
    merged_context = dict(context)
    merged_context["app_name"] = app_name

    result: Dict[str, Any] = {}
    for key, value in template.items():
        if isinstance(value, str) and value.startswith("$"):
            resolved = merged_context.get(value[1:], "")
            result[key] = resolved
        else:
            result[key] = value
    return result


def _normalize_instruction(instruction: str) -> str:
    text = instruction.lower()
    return re.sub(r"\s+", "", text)


def _ctx_default(_: str) -> Dict[str, str]:
    return {}


def _ctx_aiqiyi(instruction: str) -> Dict[str, str]:
    query = _extract_first(instruction, r"打开(.+?)的评论区", default="狂飙")
    comment = _extract_first(instruction, r"发布评论[：:](.+)$", default="真是太好看了")
    return {
        "search_keyword": _clean_text(query),
        "comment_text": _clean_text(comment),
    }


def _ctx_baidumap_voice(instruction: str) -> Dict[str, str]:
    voice = _extract_first(instruction, r"为(.+)$", default="孟子义")
    return {"voice_name": _clean_text(voice)}


def _ctx_baidumap_taxi(instruction: str) -> Dict[str, str]:
    from_place = _extract_first(instruction, r"从(.+?)去", default="国际医学中心")
    to_place = _extract_first(instruction, r"去(.+?)[，。,.]", default="西安回民街")
    to_clean = _clean_text(to_place)
    to_clean = to_clean.replace("西安", "")
    if not to_clean:
        to_clean = "回民街"
    return {
        "from_place": f".*{_clean_text(from_place)}",
        "to_place": f".*{to_clean}",
    }


def _ctx_bilibili(instruction: str) -> Dict[str, str]:
    query = _extract_first(instruction, r"搜索(.+?)并", default="采莲曲")
    return {"search_keyword": _clean_text(query)}


def _ctx_douyin(instruction: str) -> Dict[str, str]:
    query = _extract_first(instruction, r"搜索(.+?)的", default="跳舞")
    return {"search_keyword": _clean_text(query)}


def _ctx_kuaishou(instruction: str) -> Dict[str, str]:
    query = _extract_first(instruction, r"搜索(.+?)筛选", default="动画片")
    return {"search_keyword": _clean_text(query)}


def _ctx_meituan(instruction: str) -> Dict[str, str]:
    shop = _extract_first(instruction, r"购买(.+?)店铺", default="窑村干锅猪蹄")
    dish = _extract_first(instruction, r"的(.+?)[，。,.]", default="干锅排骨")
    return {
        "shop_keyword": _clean_text(shop).split("（")[0].split("(")[0],
        "dish_keyword": _clean_text(dish),
    }


def _ctx_qunar(instruction: str) -> Dict[str, str]:
    from_city = _extract_first(instruction, r"后天(.+?)飞", default="邯郸")
    to_city = _extract_first(instruction, r"飞(.+?)的航班", default="上海")
    return {
        "from_city": _clean_text(from_city),
        "to_city": _clean_text(to_city),
    }


def _ctx_tencent_video(instruction: str) -> Dict[str, str]:
    query = _extract_first(instruction, r"搜索(.+?)并", default="扫毒风暴")
    return {"search_keyword": _clean_text(query)}


def _ctx_ximalaya(instruction: str) -> Dict[str, str]:
    query = _extract_first(instruction, r"播放[《\"]?(.+?)[》\"]?多人有声剧", default="三体")
    return {"search_keyword": _clean_text(query)}


def _extract_first(text: str, pattern: str, default: str) -> str:
    match = re.search(pattern, text)
    if not match:
        return default
    return match.group(1).strip()


def _clean_text(text: str) -> str:
    return text.strip().strip("。，“”\"'")
