import logging
import time
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Union

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CodeRetriever")

class CodeRetriever:
    """
    代码检索模块：负责将查询转换为向量，并在 FAISS 索引中检索相关的代码片段。
    """
    def __init__(
        self, 
        embedder: Any, 
        index_path: str, 
        chunk_texts: List[str], 
        chunk_metadata: List[Dict[str, Any]]
    ):
        """
        初始化检索器。
        
        :param embedder: 嵌入器实例，必须包含 encode_queries 方法。
        :param index_path: FAISS 索引文件路径。
        :param chunk_texts: 原始代码片段文本列表（与索引顺序一致）。
        :param chunk_metadata: 代码片段元数据列表（与索引顺序一致）。
        """
        self.embedder = embedder
        try:
            self.index = faiss.read_index(index_path)
            logger.info(f"成功加载 FAISS 索引，包含 {self.index.ntotal} 个向量。")
        except Exception as e:
            logger.error(f"加载 FAISS 索引失败: {str(e)}")
            raise e

        self.chunk_texts = chunk_texts
        self.chunk_metadata = chunk_metadata

        # 验证数据一致性
        if self.index.ntotal != len(self.chunk_texts) or self.index.ntotal != len(self.chunk_metadata):
            logger.error("数据不一致：索引中的向量数与文本/元数据列表长度不匹配！")
            # 这种情况下建议抛出异常，防止返回错误的检索结果
            raise ValueError("Consistency check failed: Index size doesn't match data size.")

    def retrieve(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: float = -1.0
    ) -> List[Dict[str, Any]]:
        """
        执行单条查询检索。
        
        :param query: 用户查询字符串。
        :param k: 返回的最大结果数。
        :param score_threshold: 相似度阈值过滤。
        :return: 检索结果列表。
        """
        # 1. 查询预处理
        if not query or not query.strip():
            logger.warning("收到空查询，跳过检索。")
            return []

        start_time = time.time()
        
        try:
            # 2. 查询编码
            # 确保输入是列表，并获取向量 (1, dim)
            query_vector = self.embedder.encode_queries([query])
            query_vector = np.array(query_vector).astype('float32')

            # 3. 向量检索
            # scores: 相似度分数, indices: 索引 ID
            scores, indices = self.index.search(query_vector, k)
            
            # 4. 结果组装与后处理
            results = self._process_results(indices[0], scores[0], k, score_threshold)
            
            duration = time.time() - start_time
            logger.info(f"查询检索完成。耗时: {duration:.4f}s, 返回结果数: {len(results)}")
            return results

        except Exception as e:
            logger.error(f"检索过程中发生异常: {str(e)}")
            return []

    def retrieve_batch(
        self, 
        queries: List[str], 
        k: int = 5, 
        score_threshold: float = -1.0
    ) -> List[List[Dict[str, Any]]]:
        """
        批量查询检索。
        """
        valid_queries = [q for q in queries if q and q.strip()]
        if not valid_queries:
            return [[] for _ in queries]

        try:
            query_vectors = self.embedder.encode_queries(valid_queries)
            query_vectors = np.array(query_vectors).astype('float32')
            
            all_scores, all_indices = self.index.search(query_vectors, k)
            
            batch_results = []
            for i in range(len(valid_queries)):
                res = self._process_results(all_indices[i], all_scores[i], k, score_threshold)
                batch_results.append(res)
            return batch_results
            
        except Exception as e:
            logger.error(f"批量检索异常: {str(e)}")
            return [[] for _ in queries]

    def _process_results(
        self, 
        indices: np.ndarray, 
        scores: np.ndarray, 
        k_request: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        内部方法：映射 ID 并处理边界/阈值逻辑。
        """
        processed = []
        
        for idx, score in zip(indices, scores):
            # 处理 FAISS 返回的无效索引 (-1)
            if idx == -1:
                continue
                
            # 处理索引越界（严重错误）
            if idx >= len(self.chunk_texts):
                logger.error(f"严重错误：检索到的索引 {idx} 超出存储范围。")
                continue

            # 分数阈值过滤
            # 注意：若 FAISS 使用的是 IndexFlatIP 且向量已归一化，score 即为余弦相似度
            if score < threshold:
                continue

            # 构造结果字典
            item = {
                "score": float(score),
                "text": self.chunk_texts[idx],
                "metadata": self.chunk_metadata[idx]
            }
            processed.append(item)

        # 记录 K 值请求多于实际存在数的情况
        if len(processed) < k_request and self.index.ntotal >= k_request:
            logger.debug(f"检索结果不足 {k_request} 个（含阈值过滤），实际返回 {len(processed)} 个。")

        return processed

# --- 模拟使用示例 ---
if __name__ == "__main__":
    # 模拟 Embedder 类
    class MockEmbedder:
        def encode_queries(self, queries):
            # 返回随机向量用于测试
            return np.random.rand(len(queries), 128).astype('float32')

    # 模拟数据
    mock_texts = ["def hello(): print('world')", "class Runner: pass", "import os"]
    mock_meta = [
        {"file_path": "a.py", "start_line": 1, "end_line": 2, "language": "python"},
        {"file_path": "b.py", "start_line": 5, "end_line": 6, "language": "python"},
        {"file_path": "c.py", "start_line": 1, "end_line": 1, "language": "python"}
    ]
    
    # 创建模拟索引
    dim = 128
    index = faiss.IndexFlatIP(dim)
    index.add(np.random.rand(3, dim).astype('float32'))
    faiss.write_index(index, "test.index")

    # 初始化 Retriever
    retriever = CodeRetriever(
        embedder=MockEmbedder(),
        index_path="test.index",
        chunk_texts=mock_texts,
        chunk_metadata=mock_meta
    )

    # 测试检索
    results = retriever.retrieve("How to say hello?", k=2, score_threshold=0.1)
    for r in results:
        print(f"Score: {r['score']:.4f} | File: {r['metadata']['file_path']} | Text: {r['text']}")