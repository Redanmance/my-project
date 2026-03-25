import git
from pathlib import Path

def clone_repo(repo_url: str, target_dir: str = "data/repos") -> str:
    """
    将指定的 GitHub/Git 仓库克隆到本地。
    
    :param repo_url: 仓库的 URL (例如: https://github.com/user/repo.git)
    :param target_dir: 存放仓库的本地父级目录，默认为 "data/repos"
    :return: 克隆成功后的本地仓库绝对或相对路径
    """
    # 1. 从 URL 中提取仓库名，并处理可能带有 .git 后缀的情况
    repo_name = repo_url.rstrip('/').split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]  # 移除 .git 后缀使目录名更干净

    # 2. 拼接目标路径
    save_path = Path(target_dir) / repo_name

    # 3. 检查路径是否存在，存在则直接返回
    if save_path.exists() and save_path.is_dir():
        print(f"✅ 仓库已存在，跳过克隆: {save_path}")
        return str(save_path)

    # 确保父级目录 "data/repos" 存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 4 & 5. 使用浅克隆 (depth=1) 并捕获网络/URL等异常
    try:
        print(f"⏳ 正在克隆仓库 (浅克隆): {repo_name} ...")
        git.Repo.clone_from(repo_url, save_path, depth=1)
        print(f"✅ 克隆完成: {save_path}")
        return str(save_path)
        
    except git.exc.GitCommandError as e:
        # 捕获 Git 命令相关的错误（如 URL 无效、网络断开、权限被拒等）
        print(f"❌ 克隆失败，Git错误: {e}")
        # 清理可能产生的空文件夹，避免下次误判为“已存在”
        if save_path.exists():
            import shutil
            shutil.rmtree(save_path, ignore_errors=True)
        # 重新抛出异常，交由上层业务逻辑处理（例如提示用户检查URL）
        raise
        
    except Exception as e:
        # 捕获其他未知异常
        print(f"❌ 克隆时发生未知错误: {e}")
        raise

# ================= 测试代码 =================
if __name__ == "__main__":
    # 测试有效URL
    test_url = "https://github.com/tiangolo/fastapi.git"
    try:
        local_path = clone_repo(test_url)
        print(f"仓库就绪，路径为: {local_path}")
    except Exception:
        print("处理失败。")