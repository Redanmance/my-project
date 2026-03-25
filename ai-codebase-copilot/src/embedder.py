import os
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

class CodeEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        初始化嵌入模型
        :param model_name: HuggingFace 上的模型名称，默认为 BAAI/bge-small-en-v1.5
        """
        self.model_name = model_name
        self.device = self._get_device()
        print(f"⏳ 正在加载 Embedding 模型: {self.model_name} (设备: {self.device})...")
        
        # 加载模型
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ 模型加载完成! 向量维度: {self.embedding_dim}")

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        将查询文本列表转换为向量。
        
        :param queries: 查询字符串列表
        :return: 向量矩阵 (num_queries, embedding_dim)
        """
        # 复用内部带重试机制的编码方法
        return self._encode_with_retry(queries, batch_size=32)

    def _get_device(self) -> str:
        """自动检测可用的计算设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # 兼容 Apple Silicon (M1/M2/M3)
        else:
            return "cpu"

    def embed_chunks(
        self, 
        chunks: List[Dict], 
        batch_size: int = 64
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        将文本 chunks 转换为归一化的向量
        
        :param chunks: 包含 'text' 和 'metadata' 的字典列表
        :param batch_size: 批处理大小
        :return: (向量矩阵 numpy 数组, 元数据列表)
        """
        if not chunks:
            print("⚠️ 警告: 输入的 chunks 列表为空！")
            return np.array([]), []

        print(f"🚀 开始向量化 {len(chunks)} 个代码片段...")
        
        # 1. 提取文本和元数据
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # 2. 批量编码并处理 OOM (内存溢出)
        try:
            embeddings = self._encode_with_retry(texts, batch_size)
        except Exception as e:
            print(f"❌ 向量化失败: {e}")
            raise

        print(f"✅ 向量化完成! 矩阵形状: {embeddings.shape}")
        return embeddings, metadatas

    def _encode_with_retry(self, texts: List[str], batch_size: int) -> np.ndarray:
        """内部方法：执行编码，若发生 OOM 则减半 batch_size 重试"""
        try:
            # normalize_embeddings=True 确保输出为 L2 归一化向量，便于余弦相似度计算
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        except torch.cuda.OutOfMemoryError as e:
            if batch_size > 4:
                new_batch_size = batch_size // 2
                print(f"⚠️ 发生 GPU 显存溢出 (OOM)！正在清理缓存，将 batch_size 降至 {new_batch_size} 后重试...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._encode_with_retry(texts, new_batch_size)
            else:
                raise RuntimeError("Batch size 已降至最低，仍然发生 OOM，请考虑使用更小的模型或截断文本。") from e

    def save_to_disk(
        self, 
        embeddings: np.ndarray, 
        metadatas: List[Dict], 
        repo_name: str, 
        save_dir: str = "data/embeddings"
    ) -> None:
        """
        将向量和元数据持久化保存到本地，防止后续步骤失败导致重新计算
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        embed_file = save_path / f"embeddings_{repo_name}.npy"
        meta_file = save_path / f"metadata_{repo_name}.pkl"

        np.save(embed_file, embeddings)
        with open(meta_file, "wb") as f:
            pickle.dump(metadatas, f)
            
        print(f"💾 数据已持久化保存至:\n  - {embed_file}\n  - {meta_file}")

    def load_from_disk(self, repo_name: str, load_dir: str = "data/embeddings") -> Optional[Tuple[np.ndarray, List[Dict]]]:
        """
        尝试从本地加载已保存的向量和元数据（缓存机制）
        """
        embed_file = Path(load_dir) / f"embeddings_{repo_name}.npy"
        meta_file = Path(load_dir) / f"metadata_{repo_name}.pkl"

        if embed_file.exists() and meta_file.exists():
            print(f"🔄 检测到本地缓存，正在加载 {repo_name} 的向量数据...")
            embeddings = np.load(embed_file)
            with open(meta_file, "rb") as f:
                metadatas = pickle.load(f)
            return embeddings, metadatas
        return None


# ================= 测试与验证模块 =================
if __name__ == "__main__":
    # 模拟从上一模块 (code_splitter) 输出的 chunks
    dummy_chunks = [
        {
            "text": "def add(a, b):\n    return a + b",
            "metadata": {"file_path": "math.py", "start_line": 1, "end_line": 2, "language": "python"}
        },
        {
            "text": "class User:\n    def __init__(self, name):\n        self.name = name",
            "metadata": {"file_path": "models.py", "start_line": 10, "end_line": 12, "language": "python"}
        },
        {
            "text": "import os\nprint(os.getcwd())",
            "metadata": {"file_path": "main.py", "start_line": 5, "end_line": 6, "language": "python"}
        }
    ]

    # 1. 初始化模型 (这里使用了 bge-small-en-v1.5，效果很好且轻量)
    embedder = CodeEmbedder(model_name="BAAI/bge-small-en-v1.5")

    # 2. 向量化处理
    embeddings, metadatas = embedder.embed_chunks(dummy_chunks, batch_size=2)

    # 3. 验证测试点
    print("\n--- 验证测试 ---")
    
    # 验证点 1：维度检查
    print(f"1. 维度检查: 期望 (3, 384), 实际 {embeddings.shape}")
    assert embeddings.shape == (3, 384), "维度不符合预期！"
    
    # 验证点 2：归一化验证 (L2 范数应接近 1)
    sample_norm = np.linalg.norm(embeddings[0])
    print(f"2. 归一化验证: Chunk 0 的 L2 范数 = {sample_norm:.6f} (应极度接近 1.0)")
    assert np.isclose(sample_norm, 1.0, atol=1e-5), "向量未正确归一化！"
    
    # 验证点 3：数据持久化
    print("3. 数据持久化测试:")
    embedder.save_to_disk(embeddings, metadatas, repo_name="test_repo")
    
    # 验证点 4：读取缓存测试
    cached_data = embedder.load_from_disk(repo_name="test_repo")
    if cached_data:
        print("4. 缓存读取测试: 成功读取！")