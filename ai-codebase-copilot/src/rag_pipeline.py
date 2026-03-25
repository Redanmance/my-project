import re
import logging
import time
import traceback
from typing import List, Dict, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAGPipeline")

class RAGPipeline:
    """
    RAG Pipeline 模块：负责编排检索和生成流程，提供带引用的代码问答。
    """

    def __init__(self, retriever: Any, llm_client: Any, system_prompt: Optional[str] = None):
        """
        初始化 Pipeline。
        
        :param retriever: 具有 retrieve(query, k, score_threshold) 方法的实例
        :param llm_client: 具有 generate(prompt, temperature, max_tokens) 方法的实例
        :param system_prompt: 自定义系统提示词，若为 None 则使用默认值
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.system_prompt = system_prompt or (
            "You are a professional AI software engineering assistant specialized in codebase analysis.\n"
            "Your task is to answer the user's question using ONLY the provided code snippets.\n"
            "Strict rules:\n"
            "1. If the answer is not in the context, say 'I don't have enough information in the codebase to answer this'.\n"
            "2. For every claim you make, cite the source file and line number using the format [filename.py:line].\n"
            "3. Keep your answer concise and technically accurate."
        )

    async def answer(
        self, 
        query: str, 
        k: int = 5, 
        temperature: float = 0.0, 
        max_tokens: int = 1024,
        score_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        端到端问答主方法。
        """
        try:
            # 1. 检索阶段
            start_retrieve = time.time()
            snippets = self.retriever.retrieve(query, k=k, score_threshold=score_threshold)
            retrieve_duration = time.time() - start_retrieve

            if not snippets:
                return {
                    "answer": "I'm sorry, I couldn't find any relevant code snippets in the repository to answer your question.",
                    "references": [],
                    "success": True,
                    "error": None
                }

            # 2. 上下文构建
            context_str = self._build_context(snippets)

            # 3. 提示词构造
            full_prompt = self._build_prompt(context_str, query)

            # 4. LLM 调用
            start_llm = time.time()
            raw_answer = await self.llm_client.generate(
                prompt=full_prompt, 
                temperature=temperature, 
                max_tokens=max_tokens
            )
            llm_duration = time.time() - start_llm

            # 5. 引用提取与元数据对齐
            structured_refs = self._extract_and_match_references(raw_answer, snippets)

            logger.info(f"RAG 流程完成. 检索耗时: {retrieve_duration:.2f}s, LLM 耗时: {llm_duration:.2f}s")

            return {
                "answer": raw_answer,
                "references": structured_refs,
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"RAG Pipeline 运行异常: {str(e)}")
            traceback_logger = logging.getLogger("traceback")
            traceback_logger.error(traceback.format_exc())
            return {
                "answer": "",
                "references": [],
                "success": False,
                "error": str(e)
            }

    def _build_context(self, snippets: List[Dict[str, Any]]) -> str:
        """
        将检索到的片段格式化为 LLM 可读的上下文字符串。
        """
        formatted_snippets = []
        for i, snip in enumerate(snippets):
            meta = snip['metadata']
            header = f"--- Snippet {i+1} | File: {meta['file_path']} | Lines: {meta['start_line']}-{meta['end_line']} ---"
            content = snip['text']
            formatted_snippets.append(f"{header}\n{content}\n")
        
        return "\n".join(formatted_snippets)

    def _build_prompt(self, context: str, query: str) -> str:
        """
        组装最终发送给 LLM 的提示词。
        """
        return (
            f"{self.system_prompt}\n\n"
            f"Context Information:\n{context}\n\n"
            f"User Question: {query}\n\n"
            f"Assistant Answer:"
        )

    def _extract_and_match_references(self, answer: str, original_snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        解析 LLM 回答中的引用标记，并与原始元数据匹配。
        匹配格式示例: [file_name.py:12] 或 [file_name.py:12-20]
        """
        # 正则匹配格式: [文件名:行号]
        pattern = r"\[([\w\/\.\-]+):(\d+)(?:-(\d+))?\]"
        matches = re.findall(pattern, answer)
        
        unique_refs = set()
        structured_references = []

        for file_name, start_line, end_line in matches:
            ref_key = f"{file_name}:{start_line}"
            if ref_key in unique_refs:
                continue
                
            line_num = int(start_line)
            
            # 尝试在检索到的原始片段中寻找匹配项
            for snip in original_snippets:
                meta = snip['metadata']
                # 匹配逻辑：文件名相同且行号在范围内
                if file_name in meta['file_path'] and meta['start_line'] <= line_num <= meta['end_line']:
                    structured_references.append({
                        "file_path": meta['file_path'],
                        "start_line": meta['start_line'],
                        "end_line": meta['end_line'],
                        "snippet_preview": snip['text'][:200] + "...",
                        "score": snip.get('score', 0.0)
                    })
                    unique_refs.add(ref_key)
                    break
                    
        return structured_references

# --- 接口协议定义 (Mock 示例) ---

class MockLLMClient:
    """模拟 LLM 客户端"""
    async def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        # 模拟包含引用的回答
        return (
            "The `CodeRetriever` class is initialized by loading a FAISS index from a path [retriever.py:30]. "
            "It also performs a consistency check between the index and the metadata [retriever.py:45]."
        )

class MockRetriever:
    """模拟检索器"""
    def retrieve(self, query: str, k: int, score_threshold: float):
        return [
            {
                "text": "class CodeRetriever:\n    def __init__(self, ...):\n        self.index = faiss.read_index(path)",
                "metadata": {"file_path": "src/retriever.py", "start_line": 25, "end_line": 50},
                "score": 0.95
            }
        ]

# --- 测试运行 ---
if __name__ == "__main__":
    import asyncio
    import traceback

    async def main():
        # 初始化组件
        retriever = MockRetriever()
        llm = MockLLMClient()
        pipeline = RAGPipeline(retriever, llm)

        # 执行问答
        result = await pipeline.answer("How is the index loaded?")
        
        print("\n=== LLM Answer ===")
        print(result['answer'])
        print("\n=== Structured References ===")
        for ref in result['references']:
            print(f"- {ref['file_path']} (Lines {ref['start_line']}-{ref['end_line']})")

    asyncio.run(main())