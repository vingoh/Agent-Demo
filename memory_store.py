import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class MemoryStore:
    """
    独立记忆模块:
    - session_memory: 当前进程内短时记忆
    - persistent_memory: 本地 JSON 文件持久化记忆
    """

    def __init__(self, memory_file: str = "./memory/reflection_memory.json"):
        self.memory_file = memory_file
        self.session_memory: List[Dict[str, Any]] = []
        self.persistent_memory: List[Dict[str, Any]] = []
        self.log_prefix = "[MEMORY]"
        self.load()

    def _log(self, action: str, message: str):
        print(f"{self.log_prefix}[{action}] {message}")

    def add_record(
        self,
        question: str,
        stage: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
        auto_save: bool = True,
    ) -> Dict[str, Any]:
        """
        添加一条记忆记录，同时写入短时层和持久层。
        """
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": (question or "").strip(),
            "stage": (stage or "").strip(),
            "content": (content or "").strip(),
            "meta": meta or {},
        }
        self.session_memory.append(record)
        self.persistent_memory.append(record)
        self._log(
            "ADD",
            (
                f"stage={record['stage']} content_len={len(record['content'])} "
                f"session_count={len(self.session_memory)} "
                f"persistent_count={len(self.persistent_memory)} auto_save={auto_save}"
            ),
        )

        if auto_save:
            self.save()

        return record

    def search(
        self,
        keyword: str,
        stage: Optional[str] = None,
        include_session: bool = True,
        include_persistent: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        按关键字（可选阶段）检索记忆。
        """
        keyword = (keyword or "").strip().lower()
        stage = (stage or "").strip()
        if not keyword and not stage:
            self._log("SEARCH", "keyword/stage both empty, returned=0")
            return []

        sources: List[Dict[str, Any]] = []
        if include_session:
            sources.extend(self.session_memory)
        if include_persistent:
            sources.extend(self.persistent_memory)

        results: List[Dict[str, Any]] = []
        for record in sources:
            content = str(record.get("content", "")).lower()
            question = str(record.get("question", "")).lower()
            record_stage = str(record.get("stage", ""))

            keyword_hit = (not keyword) or (keyword in content or keyword in question)
            stage_hit = (not stage) or (record_stage == stage)
            if keyword_hit and stage_hit:
                results.append(record)
        self._log(
            "SEARCH",
            (
                f"keyword={'<empty>' if not keyword else keyword} "
                f"stage={'<any>' if not stage else stage} "
                f"sources_count={len(sources)} matched={len(results)}"
            ),
        )
        return results

    def clear_session(self):
        """
        清空本次运行短时记忆，不影响持久层。
        """
        before = len(self.session_memory)
        self.session_memory = []
        self._log("CLEAR_SESSION", f"before={before} after=0")

    def load(self):
        """
        从持久化文件加载记忆。
        文件不存在时静默初始化为空列表。
        """
        if not os.path.exists(self.memory_file):
            self.persistent_memory = []
            self._log("LOAD", f"file_not_found path={self.memory_file} initialized_empty")
            return

        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.persistent_memory = data
                self._log("LOAD", f"path={self.memory_file} loaded={len(self.persistent_memory)}")
            else:
                self._log("WARN", "memory file content is not a list. reset to empty.")
                self.persistent_memory = []
        except Exception as e:
            self._log("WARN", f"failed to load memory file path={self.memory_file}: {e}")
            self.persistent_memory = []

    def save(self):
        """
        将持久层记忆写入本地文件。
        """
        try:
            parent_dir = os.path.dirname(self.memory_file)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.persistent_memory, f, ensure_ascii=False, indent=2)
            self._log("SAVE", f"path={self.memory_file} saved={len(self.persistent_memory)}")
        except Exception as e:
            self._log("ERROR", f"failed to save memory file path={self.memory_file}: {e}")

    def format_recent_for_prompt(
        self,
        limit: int = 5,
        include_session: bool = True,
        include_persistent: bool = True,
    ) -> str:
        """
        将近期记忆格式化为提示词可用文本。
        """
        records: List[Dict[str, Any]] = []

        if include_persistent:
            records.extend(self.persistent_memory[-max(limit, 0) :])
        if include_session:
            records.extend(self.session_memory[-max(limit, 0) :])

        self._log(
            "FORMAT",
            (
                f"limit={max(limit, 0)} include_session={include_session} "
                f"include_persistent={include_persistent} records={len(records)}"
            ),
        )

        if not records:
            return "（暂无可用记忆）"

        lines: List[str] = []
        for idx, record in enumerate(records, 1):
            lines.append(
                f"[{idx}] {record.get('timestamp', '')} | {record.get('stage', '')} | Q: {record.get('question', '')}"
            )
            lines.append(f"    {record.get('content', '')}")
        return "\n".join(lines)


if __name__ == "__main__":
    store = MemoryStore()
    store.add_record(
        question="示例问题",
        stage="execution",
        content="这是一次执行阶段的示例记录。",
        meta={"source": "demo"},
    )
    print("--- Recent Session Memory ---")
    print(store.format_recent_for_prompt(limit=3, include_session=True, include_persistent=False))
