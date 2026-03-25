import uuid
import time
import logging
import os 
from typing import List, Dict, Any, Optional, Union
import asyncio

# 导入核心组件
from src.repo_loader import clone_repo
from src.code_parser import get_code_files
from src.code_splitter import CodeSplitter
from src.embedder import CodeEmbedder
from src.vector_store import FAISSStore
from src.retriever import CodeRetriever
from src.rag_pipeline import RAGPipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QASystem")

class QASession:
    """会话对象，存储单个用户的对话上下文"""
    def __init__(self, session_id: str, repo_id: Optional[str] = None):
        self.session_id = session_id
        self.repo_id = repo_id
        self.history: List[Dict[str, Any]] = []  # 存储格式: {"q": str, "a": str, "refs": list}
        self.created_at = time.time()
        self.last_active = time.time()

    def update_activity(self):
        self.last_active = time.time()

    def get_history_context(self, max_turns: int = 3) -> str:
        """格式化最近几轮对话历史，用于增强 Prompt"""
        recent_history = self.history[-max_turns:]
        if not recent_history:
            return ""
        
        context = "\n--- Recent Conversation History ---\n"
        for turn in recent_history:
            context += f"User: {turn['q']}\nAI: {turn['a']}\n"
        return context

class QASystem:
    """
    问答中枢模块：管理多会话、多仓库，并协调检索与生成过程。
    """
    def __init__(self, registry: Dict[str, Any], llm_client: Any = None):
        """
        初始化 QA 系统。
        
        :param registry: 仓库注册表，格式为 {repo_id: RAGPipelineInstance}
        :param llm_client: 用于生成回答的 LLM 客户端实例
        """
        self.sessions: Dict[str, QASession] = {}
        self.repo_registry = registry  # 维护 repo_id 到 RAGPipeline 的映射
        self.llm_client = llm_client
        self.session_expiry_seconds = 3600  # 1小时过期
        
        # 初始化组件（单例模式，共用模型以节省资源）
        self.splitter = CodeSplitter()
        self.embedder = CodeEmbedder()

    def load_repository(self, repo_url: str, force_reindex: bool = False) -> str:
        """
        端到端加载仓库：克隆 -> 解析 -> 切分 -> 向量化 -> 索引 -> 注册 Pipeline。
        
        :param repo_url: 仓库 Git URL
        :param force_reindex: 是否强制重新索引（即使本地已有索引）
        :return: 仓库 ID (repo_id)
        """
        # 1. 克隆仓库
        local_path = clone_repo(repo_url)
        repo_name = os.path.basename(local_path)
        repo_id = repo_name # 简单起见，使用仓库名作为 ID
        
        if repo_id in self.repo_registry and not force_reindex:
            logger.info(f"Repository {repo_id} already loaded and indexed.")
            return repo_id

        logger.info(f"--- Starting end-to-end indexing for: {repo_id} ---")

        # 2. 解析文件
        code_files = get_code_files(local_path)
        
        # 3. 代码切分
        # 注意：CodeSplitter.split_repository 是批量处理多个文件的方法
        all_chunks = []
        for file_path in code_files:
            chunks = self.splitter.split_file(file_path)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")

        # 4. 向量化
        embeddings, metadatas = self.embedder.embed_chunks(all_chunks)

        # 5. 构建向量库
        vector_store = FAISSStore(dimension=self.embedder.embedding_dim)
        vector_store.add(embeddings, metadatas)
        
        # 保存到磁盘（可选）
        vector_store.save_to_disk(repo_id)

        # 6. 初始化检索器与 Pipeline
        # 从 vector_store 中提取文本和元数据供检索器使用
        chunk_texts = [c['text'] for c in all_chunks]
        chunk_metadata = [c['metadata'] for c in all_chunks]
        
        # 为了简单起见，我们直接使用 FAISSStore 内存中的数据初始化检索器
        # 实际上可以通过 save_to_disk 后的文件路径来初始化
        index_file = f"data/vector_store/{repo_id}.faiss"
        retriever = CodeRetriever(
            embedder=self.embedder,
            index_path=index_file,
            chunk_texts=chunk_texts,
            chunk_metadata=chunk_metadata
        )
        
        pipeline = RAGPipeline(retriever=retriever, llm_client=self.llm_client)
        
        # 7. 注册
        self.repo_registry[repo_id] = pipeline
        logger.info(f"✅ Repository {repo_id} successfully loaded and indexed.")
        
        return repo_id

    def _cleanup_expired_sessions(self):
        """清理过期的会_话"""
        now = time.time()
        expired_ids = [
            sid for sid, sess in self.sessions.items() 
            if now - sess.last_active > self.session_expiry_seconds
        ]
        for sid in expired_ids:
            del self.sessions[sid]
            logger.info(f"Session {sid} has expired and been cleaned up.")

    def get_or_create_session(self, session_id: Optional[str]) -> QASession:
        """获取现有会话或创建新会话"""
        self._cleanup_expired_sessions()
        
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.update_activity()
            return session
        
        new_id = session_id or str(uuid.uuid4())
        new_session = QASession(new_id)
        self.sessions[new_id] = new_session
        logger.info(f"Created new session: {new_id}")
        return new_session

    async def ask(
        self, 
        question: str, 
        session_id: Optional[str] = None, 
        repo_id: Optional[str] = None,
        k: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 512,
        score_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        执行问答的核心接口。
        """
        # 1. 初始化会话
        session = self.get_or_create_session(session_id)
        
        # 2. 仓库绑定与校验
        current_repo_id = repo_id or session.repo_id
        if not current_repo_id:
            return self._make_error_response("No repository specified. Please load a repository first.", session.session_id)
        
        if current_repo_id not in self.repo_registry:
            return self._make_error_response(f"Repository '{current_repo_id}' is not loaded or indexed.", session.session_id)
        
        # 更新会话绑定的仓库
        session.repo_id = current_repo_id
        pipeline = self.repo_registry[current_repo_id]

        try:
            # 3. 构造增强提问（包含历史上下文）
            history_str = session.get_history_context(max_turns=3)
            # 在某些实现中，history 会传给 pipeline 内部处理，这里我们简单拼接
            enhanced_query = question
            if history_str:
                enhanced_query = f"{history_str}\nNew Question: {question}"

            # 4. 调用 RAG Pipeline 获取答案
            # 注意：这里的 pipeline 是在构建索引阶段初始化好的 RAGPipeline 实例
            start_time = time.time()
            result = await pipeline.answer(
                query=enhanced_query,
                k=k,
                temperature=temperature,
                max_tokens=max_tokens,
                score_threshold=score_threshold
            )
            duration = time.time() - start_time

            # 5. 更新会话历史
            if result.get("success"):
                session.history.append({
                    "q": question,
                    "a": result["answer"],
                    "refs": result["references"]
                })
                logger.info(f"QA Success for session {session.session_id}. Took {duration:.2f}s")

            # 6. 组装返回结果
            return {
                "answer": result["answer"],
                "references": result["references"],
                "session_id": session.session_id,
                "repo_id": current_repo_id,
                "success": result["success"],
                "error": result.get("error")
            }

        except Exception as e:
            logger.error(f"QA System Error: {str(e)}")
            return self._make_error_response(f"Internal system error: {str(e)}", session.session_id)

    def _make_error_response(self, msg: str, session_id: str) -> Dict[str, Any]:
        return {
            "answer": "",
            "references": [],
            "session_id": session_id,
            "success": False,
            "error": msg
        }

# --- 使用示例 (Mock 演示) ---

async def main_demo():
    # 1. 模拟一个已经初始化好的 Pipeline
    class MockPipeline:
        async def answer(self, **kwargs):
            return {
                "answer": "The Graph class handles spatial connections.",
                "references": [{"file_path": "graph.py", "start_line": 10, "end_line": 20}],
                "success": True
            }

    # 2. 注册仓库
    registry = {"my_cool_project": MockPipeline()}
    qa_sys = QASystem(registry)

    # 3. 第一轮提问
    print("--- Round 1 ---")
    resp1 = await qa_sys.ask("What does the Graph class do?", repo_id="my_cool_project")
    print(f"Session: {resp1['session_id']}\nAnswer: {resp1['answer']}")

    # 4. 第二轮提问 (携带 session_id，支持历史)
    print("\n--- Round 2 ---")
    resp2 = await qa_sys.ask("And how do I initialize it?", session_id=resp1['session_id'])
    print(f"Answer: {resp2['answer']}")

if __name__ == "__main__":
    asyncio.run(main_demo())