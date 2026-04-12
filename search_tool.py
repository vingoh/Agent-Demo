import os
import sys

from dotenv import load_dotenv
from serpapi import SerpApiClient

# 加载 .env 文件中的环境变量
load_dotenv()

def search(query: str) -> str:
    """
    一个基于SerpApi的网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"[TOOL][Search][REQUEST] query={query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            print("[TOOL][Search][ERROR] SERPAPI_API_KEY 未配置。")
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            print("[TOOL][Search][RESULT] source=answer_box_list")
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            print("[TOOL][Search][RESULT] source=answer_box")
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            print("[TOOL][Search][RESULT] source=knowledge_graph")
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            print("[TOOL][Search][RESULT] source=organic_results top=3")
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        print("[TOOL][Search][RESULT] no_data")
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        print(f"[TOOL][Search][ERROR] exception={e}")
        return f"搜索时发生错误: {e}"


if __name__ == "__main__":
    # 用法:
    # python search_tool.py "今天北京天气"
    # 或直接运行后手动输入问题
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("请输入要搜索的问题: ").strip()

    if not query:
        print("错误: 查询内容不能为空。")
        sys.exit(1)

    print("--- 搜索结果 ---")
    print(search(query))
