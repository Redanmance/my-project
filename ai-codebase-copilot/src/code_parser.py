import os
from pathlib import Path
from typing import List, Tuple

def get_code_files(
    repo_path: str, 
    extensions: Tuple[str, ...] = (
        '.py', '.js', '.java', '.ts', '.go', '.rs', 
        '.c', '.cpp', '.h', '.hpp', '.cs', '.php', '.rb'
    )
) -> List[str]:
    """
    递归遍历目录，收集指定后缀的代码文件路径。
    
    参数:
        repo_path: 仓库本地根目录。
        extensions: 需要收集的文件后缀元组（需带点，如 .py）。
        
    返回:
        List[str]: 文件的绝对路径列表。
    """
    code_files = []
    
    # 定义需要忽略的目录名（不仅是 .git，还有依赖包和构建目录）
    ignore_dirs = {'.git', 'node_modules', '__pycache__', 'venv', 'dist', 'build', '.idea', '.vscode'}
    
    # 将后缀统一转为小写，以便后续进行不区分大小写的匹配
    target_extensions = tuple(ext.lower() for ext in extensions)

    # os.walk 会递归遍历
    for root, dirs, files in os.walk(repo_path):
        # --- 关键优化：目录剪枝 ---
        # 通过修改 dirs 列表，os.walk 就不会进入这些被剔除的目录
        # 必须使用 dirs[:] = ... 这种切片赋值方式来原地修改列表
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            # 检查文件后缀（转为小写匹配）
            if file.lower().endswith(target_extensions):
                full_path = os.path.join(root, file)
                # 使用 abspath 确保返回的是标准绝对路径
                code_files.append(os.path.abspath(full_path))

    return code_files

if __name__ == "__main__":
    # 测试代码
    # 假设你当前目录下有一个 data 文件夹
    test_path = "./data/repos" 
    if os.path.exists(test_path):
        print(f"--- 正在扫描目录: {test_path}")
        files = get_code_files(test_path)
        print(f"--- 找到代码文件数量: {len(files)}")
        for f in files[:5]:  # 只打印前5个示例
            print(f"示例文件: {f}")
    else:
        print(f"路径不存在: {test_path}")