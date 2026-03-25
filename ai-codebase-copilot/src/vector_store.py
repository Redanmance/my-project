import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

class FAISSStore:
    def __init__(self, dimension: int):
        """
        初始化 FAISS 向量存储 (默认使用 IndexFlatIP，即内积计算。当向量 L2 归一化时，等价于余弦相似度)
        
        :param dimension: 向量的维度 (例如 BAAI/bge-small-en-v1.5 是 384)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadatas: List[Dict] = []
        
    def add(self, embeddings: np.ndarray, metadatas: List[Dict]) -> None:
        """
        将向量和对应的元数据批量添加到 FAISS 索引中
        
        :param embeddings: 形状为 (num_chunks, dimension) 的 numpy 数组 (应已 L2 归一化)
        :param metadatas: 包含 chunk 信息的字典列表，长度必须与 embeddings 相同
        """
        if len(embeddings) == 0:
            print("⚠️ 警告: 传入的向量为空，跳过添加。")
            return
            
        if len(embeddings) != len(metadatas):
            raise ValueError(f"❌ 数量不匹配: 向量数 ({len(embeddings)}) 与 元数据数 ({len(metadatas)}) 不一致！")
            
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"❌ 维度不匹配: 期望 {self.dimension} 维，实际传入 {embeddings.shape[1]} 维！")

        # FAISS 要求输入必须是 float32 类型的 C-contiguous 数组
        embeddings_np = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        # 添加到 FAISS 索引
        self.index.add(embeddings_np)
        
        # 保存元数据 (保持顺序与索引 ID 一一对应)
        self.metadatas.extend(metadatas)
        print(f"✅ 成功添加 {len(embeddings)} 条向量。当前索引总数: {self.index.ntotal}")

    def search(self, query_vectors: np.ndarray, top_k: int = 5) -> List[List[Dict]]:
        """
        批量检索最相似的 Top-K 向量
        
        :param query_vectors: 查询向量，形状为 (num_queries, dimension) 或 (dimension,)
        :param top_k: 返回的最相似结果数量
        :return: 列表的列表。外层对应每个查询，内层为 Top-K 结果字典 (包含 score 和 metadata)
        """
        if self.index.ntotal == 0:
            print("⚠️ 警告: 索引为空，无法检索。")
            return [[] for _ in range(len(query_vectors))] if query_vectors.ndim == 2 else [[]]

        # 确保查询向量是二维且为 float32
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        if query_vectors.shape[1] != self.dimension:
            raise ValueError(f"❌ 查询向量维度 ({query_vectors.shape[1]}) 与索引维度 ({self.dimension}) 不匹配！")

        query_np = np.ascontiguousarray(query_vectors, dtype=np.float32)
        
        # 限制 top_k 不能超过索引中实际的向量总数
        actual_k = min(top_k, self.index.ntotal)
        
        # D 为相似度分数矩阵 (Inner Product), I 为索引 ID 矩阵
        distances, indices = self.index.search(query_np, actual_k)
        
        results = []
        for i in range(len(query_np)): # 遍历每个查询
            query_result = []
            for j in range(actual_k):  # 遍历该查询的 Top-K
                idx = indices[i][j]
                score = distances[i][j]
                
                # FAISS 若找不到足够的邻居，会返回 -1 的索引
                if idx != -1:
                    query_result.append({
                        "score": float(score),
                        "metadata": self.metadatas[idx]
                    })
            results.append(query_result)
            
        return results

    def save_to_disk(self, repo_name: str, save_dir: str = "data/vector_store") -> None:
        """持久化保存 FAISS 索引和元数据"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        index_file = save_path / f"{repo_name}.faiss"
        meta_file = save_path / f"{repo_name}_meta.pkl"
        
        # 保存 FAISS 索引
        faiss.write_index(self.index, str(index_file))
        
        # 保存元数据
        with open(meta_file, "wb") as f:
            pickle.dump(self.metadatas, f)
            
        print(f"💾 向量库已持久化保存至:\n  - {index_file}\n  - {meta_file}")

    @classmethod
    def load_from_disk(cls, repo_name: str, load_dir: str = "data/vector_store") -> Optional["FAISSStore"]:
        """从磁盘加载 FAISS 索引和元数据 (类方法)"""
        index_file = Path(load_dir) / f"{repo_name}.faiss"
        meta_file = Path(load_dir) / f"{repo_name}_meta.pkl"
        
        if not (index_file.exists() and meta_file.exists()):
            print(f"⚠️ 未找到 {repo_name} 的本地缓存向量库。")
            return None
            
        # 读取 FAISS 索引以获取维度
        index = faiss.read_index(str(index_file))
        dimension = index.d
        
        # 实例化自身
        store = cls(dimension=dimension)
        store.index = index
        
        # 读取元数据
        with open(meta_file, "rb") as f:
            store.metadatas = pickle.load(f)
            
        print(f"🔄 成功加载本地向量库: {repo_name} (包含 {store.index.ntotal} 条向量, 维度: {dimension})")
        return store


# ================= 测试与验证模块 =================
if __name__ == "__main__":
    print("\n--- FAISS 向量存储模块测试 ---")
    DIMENSION = 4
    
    # 生成模拟数据 (归一化后的随机向量)
    np.random.seed(42)
    raw_vectors = np.random.rand(3, DIMENSION).astype(np.float32)
    # L2 归一化 (模拟 Embedder 的输出)
    norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    normalized_vectors = raw_vectors / norms

    dummy_metadatas = [
        {"file_path": "main.py", "text": "def main(): pass"},
        {"file_path": "utils.py", "text": "def add(a, b): return a + b"},
        {"file_path": "api.py", "text": "@app.route('/home')"}
    ]

    # 1. 初始化并添加向量
    print("\n1. 测试初始化与添加")
    store = FAISSStore(dimension=DIMENSION)
    store.add(normalized_vectors, dummy_metadatas)
    assert store.index.ntotal == 3, "索引总数不正确！"

    # 2. 检索一致性测试 (用原向量去搜索，Top-1 应该是它自己，且 Score 接近 1.0)
    print("\n2. 测试检索一致性")
    query_vector = normalized_vectors[1] # 取第二个向量
    results = store.search(query_vector, top_k=2)
    
    top1_result = results[0][0]
    print(f"搜索结果 Top-1: Score = {top1_result['score']:.6f}, 文件 = {top1_result['metadata']['file_path']}")
    assert top1_result['metadata']['file_path'] == "utils.py", "元数据对齐错误！"
    assert np.isclose(top1_result['score'], 1.0, atol=1e-5), "自身检索相似度应接近 1.0！"

    # 3. 保存与加载正确性
    print("\n3. 测试持久化保存与加载")
    repo_name = "test_faiss_repo"
    store.save_to_disk(repo_name=repo_name)
    
    loaded_store = FAISSStore.load_from_disk(repo_name=repo_name)
    assert loaded_store is not None, "加载失败！"
    assert loaded_store.index.ntotal == 3, "加载后的索引数量不匹配！"
    assert len(loaded_store.metadatas) == 3, "加载后的元数据数量不匹配！"

    # 4. 边界测试：查询数量大于索引库总数
    print("\n4. 边界测试 (top_k > ntotal)")
    boundary_results = loaded_store.search(normalized_vectors[0], top_k=10)
    assert len(boundary_results[0]) == 3, f"预期返回 3 条，实际返回 {len(boundary_results[0])} 条"
    print("✅ 所有测试通过！")