# `plan_and_solve_agent.py` 类图与运行时序

本文档描述 [plan_and_solve_agent.py](../plan_and_solve_agent.py) 中的核心类型及其协作方式。可在支持 Mermaid 的编辑器中预览（如 Cursor / VS Code Markdown 预览）。

## 类图（静态结构）

说明：`PlanAndSolveAgent` 通过组合持有 `Planner` 与 `Executor`；`Planner` 与 `Executor` 依赖 `MyLLM`；`Executor` 额外依赖 `ToolExecutor` 以注册与调用工具。无继承关系。

```mermaid
classDiagram
  direction TB
  class PlanAndSolveAgent {
    +Planner planner
    +Executor executor
    +run(str question) str
    -_build_history(list step_results) str
  }
  class Planner {
    +MyLLM llm_client
    +plan(str question) List~str~
    -_parse_steps(str text) List~str~
  }
  class Executor {
    +MyLLM llm_client
    +ToolExecutor tool_executor
    +int max_retries
    +execute(str step, str history, str question, str plan) str
    -_parse_output(str text) tuple
    -_parse_action(str action_text) tuple
  }
  class MyLLM
  class ToolExecutor
  PlanAndSolveAgent *-- Planner : composition
  PlanAndSolveAgent *-- Executor : composition
  Planner --> MyLLM : uses
  Executor --> MyLLM : uses
  Executor --> ToolExecutor : uses
```

## 序列图（`run` 主流程）

说明：`run` 先调用 `planner.plan` 得到步骤列表，再对每一步调用 `executor.execute`；前序步骤结果经 `_build_history` 拼成历史字符串传入下一步。

```mermaid
sequenceDiagram
  participant User
  participant Agent as PlanAndSolveAgent
  participant Planner
  participant Executor
  participant LLM as MyLLM
  participant Tools as ToolExecutor

  User->>Agent: run(question)
  Agent->>Planner: plan(question)
  Planner->>LLM: think(messages)
  LLM-->>Planner: response
  Planner-->>Agent: steps[]

  loop each step in steps
    Agent->>Agent: _build_history(step_results)
    Agent->>Executor: execute(step, history, question, plan)
    Executor->>Tools: getAvailableTools() / getTool(name)
    loop until Finish or max_retries
      Executor->>LLM: think(messages)
      LLM-->>Executor: Thought / Action
      alt Action is tool call
        Executor->>Tools: tool_function(input)
        Tools-->>Executor: observation
      else Action is Finish
        Executor-->>Agent: step result
      end
    end
    Agent->>Agent: append step_results
  end

  Agent-->>User: final_answer (last step result)
```
