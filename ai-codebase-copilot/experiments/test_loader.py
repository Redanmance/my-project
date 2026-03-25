import os
import time
import logging
import sys
import asyncio

# 设置 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 获取项目根目录并添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qa_system import QASystem
# 导入实际的 OpenAI 客户端
try:
    from src.openai_client import OpenAIClient 
except ImportError:
    print("⚠️ 无法导入 src.openai_client，请检查文件是否存在于 src/ 目录下。")
    OpenAIClient = None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 端到端测试函数
# ----------------------------------------------------------------------
async def run_e2e_test():
    print("="*50)
    print("🚀 开始 AI Codebase Copilot 端到端总体测试")
    print("="*50)

    # 1. 初始化系统
    print("\n[阶段 1] 初始化系统...")
    start_time = time.time()

    # 仓库注册表
    repo_registry = {}

    # 从环境变量读取 LLM 配置
    api_key = os.getenv("OPENAI_API_KEY", "sk-ltqjyrbgbqownyyrhepzhnosbgckzswzxnlhdvqazsrzjbvw")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")  # 例如 http://localhost:8000/v1
    model_name = os.getenv("OPENAI_MODEL_NAME", "Qwen/Qwen3-8B")

    # 创建 LLM 客户端（如果未提供 base_url，则使用 OpenAI 官方）
    llm_client = OpenAIClient(
        api_key=api_key,
        model=model_name,
        base_url=base_url
    )

    # 初始化 QASystem
    qa = QASystem(registry=repo_registry, llm_client=llm_client)
    print(f"✅ 系统初始化完成，耗时: {time.time() - start_time:.2f}s")

    # 2. 加载测试仓库
    test_repo_path = "/home/annaa/agent_programme/data/repos/flask"
    print(f"\n[阶段 2] 加载仓库: {test_repo_path}...")
    start_time = time.time()

    repo_id = qa.load_repository(test_repo_path)

    assert repo_id is not None, "❌ 仓库加载失败，repo_id 为空"
    print(f"✅ 仓库加载及向量化完成，repo_id: {repo_id}，耗时: {time.time() - start_time:.2f}s")

    # 3. 单轮问答测试
    question_1 = "Where is the Session class defined, and what is its main purpose?"
    print(f"\n[阶段 3] 单轮问答测试 - Q: {question_1}")
    start_time = time.time()

    response_1 = await qa.ask(question=question_1, repo_id=repo_id, session_id="test_session_1")

    print(f"✅ 问答完成，耗时: {time.time() - start_time:.2f}s")
    print("-" * 30)
    print("🤖 答案:")
    print(response_1.get('answer', 'No answer generated.'))
    print("-" * 30)
    print("📚 引用来源:")
    for ref in response_1.get('references', []):
        print(f"  - {ref['file_path']} (Lines {ref.get('start_line')} - {ref.get('end_line')})")

    # 验证输出结构
    assert 'answer' in response_1, "❌ 返回结果缺失 'answer' 字段"
    assert len(response_1.get('references', [])) > 0, "❌ 检索模块似乎未返回任何引用"

    # 4. 多轮对话测试（可选，保持原有逻辑）
    question_2 = "Can you show me the code for its __init__ method?"
    print(f"\n[阶段 4] 多轮对话(上下文)测试 - Q: {question_2}")
    start_time = time.time()

    response_2 = await qa.ask(question=question_2, repo_id=repo_id, session_id="test_session_1")

    print(f"✅ 多轮问答完成，耗时: {time.time() - start_time:.2f}s")
    print("-" * 30)
    print("🤖 答案:")
    print(response_2.get('answer', 'No answer generated.'))
    print("-" * 30)

    # 5. 清理与收尾
    print("\n[阶段 5] 清理与收尾...")
    print("\n🎉 端到端总体测试圆满通过！")

if __name__ == "__main__":
    asyncio.run(run_e2e_test())