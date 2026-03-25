import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tree_sitter_languages import get_parser

class CodeSplitter:
    def __init__(self, max_lines: int = 500):
        self.max_lines = max_lines
        # 映射文件后缀到 tree-sitter 语言名称
        self.lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rs': 'rust',
        }
        # 定义不同语言中代表“逻辑单元”的节点类型
        self.node_types = {
            'python': ['class_definition', 'function_definition'],
            'javascript': ['class_declaration', 'function_declaration', 'method_definition'],
            'typescript': ['class_declaration', 'function_declaration', 'method_definition'],
            'java': ['class_declaration', 'method_declaration'],
            'go': ['function_declaration', 'method_declaration'],
            'cpp': ['class_specifier', 'function_definition'],
        }

    def _get_language(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        return self.lang_map.get(ext)

    def split_file(self, file_path: str) -> List[Dict[str, Any]]:
        """对单个文件进行切分"""
        lang_name = self._get_language(file_path)
        if not lang_name:
            # 如果不支持该语言，退回到简单的按行切分（此处可扩展）
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return []

        # 获取解析器
        parser = get_parser(lang_name)
        tree = parser.parse(bytes(content, "utf8"))
        root_node = tree.root_node

        chunks = []
        visited_bytes = 0
        
        # 提取所有定义的逻辑单元节点
        interesting_nodes = []
        self._collect_nodes(root_node, lang_name, interesting_nodes)
        
        # 按照在文件中的先后顺序排序
        interesting_nodes.sort(key=lambda x: x.start_byte)

        # 遍历节点并生成 chunk
        content_bytes = content.encode('utf-8')
        for node in interesting_nodes:
            # 1. 处理节点之前的“间隙”代码（如 import, 全局变量）
            if node.start_byte > visited_bytes:
                gap_text = content_bytes[visited_bytes:node.start_byte].decode('utf-8', errors='ignore').strip()
                if len(gap_text) > 20: # 过滤掉过小的无意义间隙
                    chunks.append(self._create_chunk(gap_text, file_path, visited_bytes, node.start_byte, content, lang_name, is_global=True))

            # 2. 处理节点本身（类或函数）
            node_text = content_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            chunks.append(self._create_chunk(node_text, file_path, node.start_byte, node.end_byte, content, lang_name))
            
            visited_bytes = node.end_byte

        # 3. 处理文件末尾剩余的代码
        if visited_bytes < len(content_bytes):
            remaining_text = content_bytes[visited_bytes:].decode('utf-8', errors='ignore').strip()
            if len(remaining_text) > 20:
                chunks.append(self._create_chunk(remaining_text, file_path, visited_bytes, len(content_bytes), content, lang_name, is_global=True))

        return chunks

    def _collect_nodes(self, node, lang_name, nodes_list):
        """递归提取特定类型的节点"""
        target_types = self.node_types.get(lang_name, [])
        if node.type in target_types:
            nodes_list.append(node)
            # 如果是类，我们通常已经拿到了整个类。
            # 如果需要把类里的方法也独立切分，可以继续递归，但这里先保持逻辑单元完整。
            return 
        
        for child in node.children:
            self._collect_nodes(child, lang_name, nodes_list)

    def _create_chunk(self, text, file_path, start_byte, end_byte, full_content, lang, is_global=False) -> Dict[str, Any]:
        """封装 chunk 格式"""
        # 计算行号 (tree-sitter 的行号从0开始，我们习惯从1开始)
        start_line = full_content.count('\n', 0, start_byte) + 1
        end_line = full_content.count('\n', 0, end_byte) + 1
        
        return {
            "text": text,
            "metadata": {
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "language": lang,
                "type": "global_scope" if is_global else "logical_unit"
            }
        }

    def split_repository(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """处理整个仓库的文件列表"""
        all_chunks = []
        for path in file_paths:
            print(f"--- 正在切分文件: {path}")
            file_chunks = self.split_file(path)
            all_chunks.extend(file_chunks)
        return all_chunks

if __name__ == "__main__":
    # 测试代码
    splitter = CodeSplitter()
    # 假设你有一个本地的 Python 文件
    test_file = "repo_loader.py" 
    if os.path.exists(test_file):
        chunks = splitter.split_file(test_file)
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i} ({chunk['metadata']['type']}) ---")
            print(f"Lines: {chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}")
            print(chunk['text'][:100] + "...")