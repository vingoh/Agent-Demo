import re
from typing import List
from my_llm import MyLLM
from tool_executor import ToolExecutor
from dotenv import load_dotenv
from search_tool import search

load_dotenv()

# ======================== Prompt 模板 ========================

PLANNER_PROMPT_TEMPLATE = """
你是一个任务规划专家。你的职责是将用户的复杂问题拆解为一系列清晰、可执行的子步骤。

请严格按照以下格式输出你的计划，每个步骤占一行:
Step 1: <第一步要做的事情>
Step 2: <第二步要做的事情>
...
Step N: 根据以上所有步骤的结果，汇总并给出最终答案。

要求:
- 每个步骤应当是独立、具体、可执行的。
- 最后一个步骤必须是"汇总并给出最终答案"。
- 步骤数量应当合理，不要过多也不要过少（通常2-5步）。

用户的问题是:
{question}
"""

EXECUTOR_PROMPT_TEMPLATE = """
你是一个任务执行专家。你需要完成当前子任务，同时理解它在整体计划中的位置。

用户的原始问题:
{question}

整体执行计划:
{plan}

当前子任务:
{step}

前序步骤的执行历史:
{history}

可用工具如下:
{tools}

请严格按照以下格式回应:
Thought: 你的思考过程，分析当前子任务需要怎么做。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`: 调用一个可用工具来获取信息。
- `Finish[结果]`: 当你已经能够给出当前子任务的结果时。

注意:
- 如果当前子任务可以通过执行历史中的信息直接回答，请直接使用 Finish[结果]。
- 如果需要查询外部信息，请先调用工具，再根据工具返回的结果给出 Finish[结果]。
"""


# ======================== Planner 类 ========================

class Planner:
    """
    计划器：接收用户问题，调用 LLM 生成结构化的多步骤执行计划。
    """

    def __init__(self, llm_client: MyLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> List[str]:
        """
        将用户问题分解为有序的子步骤列表。
        """
        print("[PLANNER][START] 正在为问题生成计划...")
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.think(messages=messages)

        if not response:
            print("[PLANNER][ERROR] LLM 未能返回有效的计划。")
            return []

        steps = self._parse_steps(response)
        print(f"[PLANNER][DONE] 共生成 {len(steps)} 个步骤:")
        for i, step in enumerate(steps, 1):
            print(f"  Step {i}: {step}")
        return steps

    def _parse_steps(self, text: str) -> List[str]:
        """从 LLM 输出中提取 Step N: ... 格式的步骤。"""
        matches = re.findall(r"Step\s+\d+[:：]\s*(.+)", text)
        return [m.strip() for m in matches if m.strip()]


# ======================== Executor 类 ========================

class Executor:
    """
    执行器：逐个执行子任务，按需调用工具，返回每步的执行结果。
    """

    def __init__(self, llm_client: MyLLM, tool_executor: ToolExecutor, max_retries: int = 3):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_retries = max_retries

    def execute(self, step: str, history: str, question: str, plan: str) -> str:
        """
        执行单个子任务。可能经历"调用工具 -> 获取观察 -> 总结结果"的过程。
        """
        print(f"[EXECUTOR][START] 正在执行子任务: {step}")
        tools_desc = self.tool_executor.getAvailableTools()

        prompt = EXECUTOR_PROMPT_TEMPLATE.format(
            question=question,
            plan=plan,
            step=step,
            history=history if history else "（暂无历史记录，这是第一个步骤）",
            tools=tools_desc,
        )
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(1, self.max_retries + 1):
            tag = f"[EXECUTOR][ATTEMPT {attempt}]"
            response = self.llm_client.think(messages=messages)

            if not response:
                print(f"{tag} LLM 未能返回有效响应。")
                continue

            thought, action = self._parse_output(response)

            if thought:
                print(f"{tag}[THOUGHT] {thought}")

            if not action:
                print(f"{tag}[WARN] 未能解析出有效的 Action。")
                continue

            print(f"{tag}[ACTION] {action}")

            if action.startswith("Finish"):
                finish_match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                if finish_match:
                    result = finish_match.group(1).strip()
                    print(f"[EXECUTOR][DONE] {result}")
                    return result
                print(f"{tag}[WARN] Finish 格式无效: {action}")
                continue

            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                print(f"{tag}[WARN] Action 格式无效，重试。")
                continue

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误: 未找到名为 '{tool_name}' 的工具。"
            else:
                print(f"{tag}[TOOL_CALL] {tool_name}[{tool_input}]")
                observation = tool_function(tool_input)

            print(f"{tag}[OBSERVATION] {observation}")

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}\n\n请根据以上观察结果，给出当前子任务的最终结果，使用 Finish[结果] 格式。"})

        print("[EXECUTOR][FAIL] 达到最大重试次数，未能完成子任务。")
        return "（该步骤未能成功执行）"

    def _parse_output(self, text: str):
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None


# ======================== PlanAndSolveAgent 类 ========================

class PlanAndSolveAgent:
    """
    Plan-and-Solve 智能体（v2）：先通过 Planner 生成计划，再通过 Executor 逐步执行，
    最后一个步骤的输出即为最终答案（无需额外的 summarize 阶段）。
    """

    def __init__(self, planner: Planner, executor: Executor):
        self.planner = planner
        self.executor = executor

    def run(self, question: str) -> str:
        print(f"[AGENT][START] Question: {question}")

        # 阶段一：生成计划
        steps = self.planner.plan(question)
        if not steps:
            print("[AGENT][ERROR] 计划生成失败，流程终止。")
            return None

        # 阶段二：逐步执行，最后一步的结果即为最终答案
        plan_str = "\n".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))
        step_results = []
        for i, step in enumerate(steps, 1):
            print(f"\n{'='*60}")
            print(f"[AGENT][EXECUTE] Step {i}/{len(steps)}: {step}")
            print(f"{'='*60}")

            history = self._build_history(step_results)
            result = self.executor.execute(step, history, question=question, plan=plan_str)
            step_results.append({"step": step, "result": result})
            print(f"[AGENT][STEP {i} RESULT] {result}")

        final_answer = step_results[-1]["result"]
        print(f"\n[AGENT][FINAL ANSWER] {final_answer}")
        return final_answer

    def _build_history(self, step_results: list) -> str:
        """将已完成步骤及其结果格式化为历史记录字符串。"""
        if not step_results:
            return ""
        lines = []
        for i, sr in enumerate(step_results, 1):
            lines.append(f"Step {i}: {sr['step']}")
            lines.append(f"Result: {sr['result']}")
            lines.append("")
        return "\n".join(lines)


# ======================== 主入口 ========================

if __name__ == '__main__':
    # 1. 初始化 LLM 客户端与工具执行器
    llm_client = MyLLM()
    tool_executor = ToolExecutor()

    # 2. 注册工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_description, search)

    # 3. 创建各组件
    planner = Planner(llm_client=llm_client)
    executor = Executor(llm_client=llm_client, tool_executor=tool_executor)
    agent = PlanAndSolveAgent(planner=planner, executor=executor)

    # 4. 运行 Plan-and-Solve Agent
    question = "英伟达最新的GPU型号是什么？它相比上一代有哪些提升？"
    print(f"[AGENT][MAIN] Question: {question}")
    agent.run(question)
