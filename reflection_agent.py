import json
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from memory_store import MemoryStore
from my_llm import MyLLM
from search_tool import search
from tool_executor import ToolExecutor

load_dotenv()


EXECUTION_PROMPT_TEMPLATE = """
你是执行模块（Execution Module），负责先产出一个初稿答案。

用户问题:
{question}

近期记忆:
{memory}

可用工具:
{tools}

请严格按如下格式输出:
Thought: 你对当前步骤的思考
Action: 只能是以下两种之一
- {{tool_name}}[{{tool_input}}]
- Finish[初稿答案]

要求:
- 能直接回答就使用 Finish[...]
- 需要外部信息时先调用工具，基于 Observation 再给出 Finish[...]
"""


REFLECTION_PROMPT_TEMPLATE = """
你是反思模块（Reflection Module），你的职责是审查“初稿答案”的质量，不要直接给最终答案。

用户问题:
{question}

初稿答案:
{draft_answer}

近期记忆:
{memory}

请只输出 JSON（不要额外文本），格式如下:
{{
  "verdict": "pass 或 revise",
  "strengths": ["..."],
  "issues": ["..."],
  "missing_info": ["..."],
  "improvement_plan": ["..."]
}}

要求:
- verdict=pass 表示质量足够高，可直接沿用
- verdict=revise 表示需要优化
- issues 和 improvement_plan 要可执行，避免空泛表述
"""


OPTIMIZATION_PROMPT_TEMPLATE = """
你是优化模块（Optimization Module），需要根据反思报告对初稿答案进行全局优化。

用户问题:
{question}

初稿答案:
{draft_answer}

反思报告(JSON):
{critique}

近期记忆:
{memory}

请严格按如下格式输出:
Decision: KEEP 或 REWRITE
OptimizedAnswer: <最终答案文本>

规则:
- 当反思报告 verdict=pass 且无实质问题时，可使用 KEEP
- 否则使用 REWRITE，并给出更准确、完整、可读的最终答案
"""


class ExecutionModule:
    """
    执行模块：产出初稿答案，按需调用工具。
    """

    def __init__(self, llm_client: MyLLM, tool_executor: ToolExecutor, max_steps: int = 4):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps

    def execute(self, question: str, memory_text: str) -> Dict[str, Any]:
        print(f"[EXECUTION][START] question={question}")
        tools_desc = self.tool_executor.getAvailableTools() or "（暂无可用工具）"
        prompt = EXECUTION_PROMPT_TEMPLATE.format(
            question=question,
            memory=memory_text,
            tools=tools_desc,
        )
        messages = [{"role": "user", "content": prompt}]
        trace: List[Dict[str, str]] = []

        for step in range(1, self.max_steps + 1):
            tag = f"[EXECUTION][STEP {step}]"
            response = self.llm_client.think(messages=messages)
            if not response:
                print(f"{tag}[WARN] LLM empty response.")
                continue

            thought, action = self._parse_output(response)
            if thought:
                print(f"{tag}[THOUGHT] {thought}")

            if not action:
                print(f"{tag}[WARN] no Action parsed.")
                continue

            print(f"{tag}[ACTION] {action}")

            if action.startswith("Finish"):
                final = self._parse_finish(action)
                if final:
                    print(f"{tag}[DONE] draft answer ready.")
                    trace.append({"step": str(step), "action": action, "observation": "N/A"})
                    return {"draft_answer": final, "trace": trace}
                print(f"{tag}[WARN] invalid Finish format.")
                continue

            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                print(f"{tag}[WARN] invalid tool action format.")
                continue

            tool_func = self.tool_executor.getTool(tool_name)
            if not tool_func:
                observation = f"错误: 未找到名为 '{tool_name}' 的工具。"
            else:
                print(f"{tag}[TOOL_CALL] {tool_name}[{tool_input}]")
                observation = tool_func(tool_input)

            print(f"{tag}[OBSERVATION] {observation}")
            trace.append({"step": str(step), "action": action, "observation": observation})

            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请基于观察继续，若信息足够请使用 Finish[初稿答案]。",
                }
            )

        print("[EXECUTION][END] max steps reached.")
        fallback = "（执行模块未能在限制步数内产出可靠初稿）"
        return {"draft_answer": fallback, "trace": trace}

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if not match:
            return None, None
        return match.group(1), match.group(2)

    def _parse_finish(self, action_text: str) -> Optional[str]:
        match = re.match(r"Finish\[(.*)\]", action_text, re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()


class ReflectionModule:
    """
    反思模块：评估初稿并给出结构化改进建议。
    """

    def __init__(self, llm_client: MyLLM):
        self.llm_client = llm_client

    def reflect(self, question: str, draft_answer: str, memory_text: str) -> Dict[str, Any]:
        print("[REFLECTION][START]")
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            question=question,
            draft_answer=draft_answer,
            memory=memory_text,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.think(messages=messages)
        if not response:
            print("[REFLECTION][WARN] empty response, use fallback critique.")
            return self._fallback_critique("LLM empty response.")

        critique = self._parse_critique_json(response)
        if critique is None:
            print("[REFLECTION][WARN] invalid JSON critique, use fallback.")
            return self._fallback_critique(response)

        critique.setdefault("verdict", "revise")
        critique.setdefault("strengths", [])
        critique.setdefault("issues", [])
        critique.setdefault("missing_info", [])
        critique.setdefault("improvement_plan", [])
        print(f"[REFLECTION][DONE] verdict={critique.get('verdict')}")
        return critique

    def _parse_critique_json(self, text: str) -> Optional[Dict[str, Any]]:
        raw = text.strip()
        block_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if block_match:
            raw = block_match.group(1).strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _fallback_critique(self, raw_text: str) -> Dict[str, Any]:
        return {
            "verdict": "revise",
            "strengths": ["初稿已给出可用的基础回答。"],
            "issues": ["反思输出解析失败，无法确认质量。"],
            "missing_info": ["建议补充关键事实与依据。"],
            "improvement_plan": ["结合用户问题重写答案，保证完整性和可验证性。"],
            "raw_reflection": raw_text,
        }


class OptimizationModule:
    """
    优化模块：根据反思结果生成最终答案。
    """

    def __init__(self, llm_client: MyLLM):
        self.llm_client = llm_client

    def optimize(
        self,
        question: str,
        draft_answer: str,
        critique: Dict[str, Any],
        memory_text: str,
    ) -> Dict[str, str]:
        print("[OPTIMIZATION][START]")
        prompt = OPTIMIZATION_PROMPT_TEMPLATE.format(
            question=question,
            draft_answer=draft_answer,
            critique=json.dumps(critique, ensure_ascii=False, indent=2),
            memory=memory_text,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.think(messages=messages)
        if not response:
            print("[OPTIMIZATION][WARN] empty response, fallback to draft.")
            return {"decision": "KEEP", "optimized_answer": draft_answer}

        decision, answer = self._parse_optimization_output(response)
        if not answer:
            print("[OPTIMIZATION][WARN] parse failed, fallback to draft.")
            return {"decision": "KEEP", "optimized_answer": draft_answer}

        print(f"[OPTIMIZATION][DONE] decision={decision}")
        return {"decision": decision, "optimized_answer": answer}

    def _parse_optimization_output(self, text: str) -> Tuple[str, Optional[str]]:
        decision_match = re.search(r"Decision:\s*(KEEP|REWRITE)", text)
        answer_match = re.search(r"OptimizedAnswer:\s*(.*)$", text, re.DOTALL)
        decision = decision_match.group(1).strip() if decision_match else "REWRITE"
        answer = answer_match.group(1).strip() if answer_match else None
        return decision, answer


class ReflectionAgent:
    """
    Reflection 范式主编排:
    Execution -> Reflection -> Optimization
    """

    def __init__(
        self,
        execution_module: ExecutionModule,
        reflection_module: ReflectionModule,
        optimization_module: OptimizationModule,
        memory_store: MemoryStore,
        max_reflections: int = 3,
    ):
        self.execution_module = execution_module
        self.reflection_module = reflection_module
        self.optimization_module = optimization_module
        self.memory_store = memory_store
        self.max_reflections = max(1, max_reflections)

    def run(self, question: str) -> str:
        print(f"[AGENT][START] Question: {question}")
        self.memory_store.clear_session()

        # 阶段一：Execution
        exec_memory = self.memory_store.format_recent_for_prompt(
            limit=5, include_session=True, include_persistent=True
        )
        exec_result = self.execution_module.execute(question=question, memory_text=exec_memory)
        draft_answer = exec_result["draft_answer"]

        self.memory_store.add_record(
            question=question,
            stage="execution",
            content=draft_answer,
            meta={"trace": exec_result.get("trace", [])},
        )

        # 阶段二/三：Reflection-Optimization 迭代
        current_answer = draft_answer
        early_stopped = False
        for round_idx in range(1, self.max_reflections + 1):
            print(f"[AGENT][ROUND {round_idx}] start reflection.")
            reflection_memory = self.memory_store.format_recent_for_prompt(
                limit=8, include_session=True, include_persistent=True
            )
            critique = self.reflection_module.reflect(
                question=question,
                draft_answer=current_answer,
                memory_text=reflection_memory,
            )

            verdict = str(critique.get("verdict", "revise")).strip().lower() or "revise"
            self.memory_store.add_record(
                question=question,
                stage="reflection",
                content=json.dumps(critique, ensure_ascii=False),
                meta={"round": round_idx, "verdict": verdict},
            )
            print(f"[REFLECTION][ROUND {round_idx}] verdict={verdict}")

            if verdict == "pass":
                early_stopped = True
                print(
                    f"[AGENT][EARLY_STOP] round={round_idx} verdict=pass, skip optimization."
                )
                break

            print(f"[AGENT][ROUND {round_idx}] start optimization.")
            optimization_memory = self.memory_store.format_recent_for_prompt(
                limit=10, include_session=True, include_persistent=True
            )
            optimized = self.optimization_module.optimize(
                question=question,
                draft_answer=current_answer,
                critique=critique,
                memory_text=optimization_memory,
            )
            current_answer = optimized["optimized_answer"]

            decision = str(optimized.get("decision", "REWRITE")).strip().upper() or "REWRITE"
            self.memory_store.add_record(
                question=question,
                stage="optimization",
                content=current_answer,
                meta={"round": round_idx, "decision": decision},
            )
            print(f"[OPTIMIZATION][ROUND {round_idx}] decision={decision}")

        if not early_stopped:
            print(f"[AGENT][INFO] reached max_reflections={self.max_reflections}.")

        print(f"[AGENT][FINAL] {current_answer}")
        return current_answer


if __name__ == "__main__":
    llm_client = MyLLM()
    tool_executor = ToolExecutor()
    memory_store = MemoryStore(memory_file="./memory/reflection_memory.json")

    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_description, search)

    execution_module = ExecutionModule(llm_client=llm_client, tool_executor=tool_executor, max_steps=4)
    reflection_module = ReflectionModule(llm_client=llm_client)
    optimization_module = OptimizationModule(llm_client=llm_client)

    agent = ReflectionAgent(
        execution_module=execution_module,
        reflection_module=reflection_module,
        optimization_module=optimization_module,
        memory_store=memory_store,
        max_reflections=3,
    )

    question = "英伟达最新的GPU型号是什么？它相比上一代有哪些提升？"
    print(f"[AGENT][MAIN] Question: {question}")
    agent.run(question)
