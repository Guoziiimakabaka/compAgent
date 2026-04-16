# 比赛完整执行步骤（实操版）

## 1. 环境准备
1. 使用 `Python 3.10.12` 创建虚拟环境。
2. 安装依赖：`pip install -r submission/src/requirements.txt`。
3. 配置模型环境变量（如果需要在线模型）：`VLM_API_KEY`、`DEBUG_MODEL_ID`、`DEBUG_API_URL`。

## 2. 接口契约确认
1. 读取并锁定标准动作集合：`CLICK`、`TYPE`、`SCROLL`、`OPEN`、`COMPLETE`。
2. 所有坐标使用相对坐标，取值范围 `[0, 1000]`。
3. 输出结构固定为：`action` + `parameters`。

## 3. 先做最小可跑链路
1. 先实现一个规则版 Agent，保证 `act()` 可稳定产出合法动作。
2. 用 `FastAPI` 暴露 `/predict` 接口，入参为指令、截图、步数和历史动作。
3. 编写本地自调用脚本，对 API 发起请求并检查响应字段是否完整。

## 4. 离线评测联调
1. 使用主办方评测脚本：`code-for-student/test_runner.py`。
2. 先跑通全流程，再分析失败 case。
3. 每轮只改一类问题（解析、策略、动作参数），避免混合改动导致定位困难。

## 5. 逐步提升精度
1. 在规则版可跑后，替换为多模态模型推理。
2. 做好输出解析与动作归一化，确保非法格式立即失败。
3. 保留规则兜底，降低模型不稳定输出导致的崩溃概率。

## 6. 提交前检查
1. 再次执行离线评测，确认无接口错误。
2. 检查 `src/` 下文件齐全：`agent_base.py`、`agent.py`、`utils/`、`requirements.txt`。
3. 在 `doc/` 补齐设计说明。
4. 打包后确认解压总大小不超过 `20MB`。

## 7. 最终提交结构
1. `submission/doc/`：算法设计说明与实验结论。
2. `submission/src/`：可运行 Agent 代码与依赖列表。
3. 压缩为 `submission.zip` 并上传。
