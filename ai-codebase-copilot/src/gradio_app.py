import gradio as gr
import os
import asyncio
import sys
from typing import List, Tuple

# 尝试加载 .env 文件，如果未安装 python-dotenv 则跳过
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 将项目根目录加入 Python 搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入项目中的真实组件
from src.repo_loader import clone_repo
from src.code_parser import get_code_files
from src.code_splitter import CodeSplitter
from src.embedder import CodeEmbedder
from src.vector_store import FAISSStore
from src.retriever import CodeRetriever
from src.rag_pipeline import RAGPipeline
from src.qa_system import QASystem
from src.openai_client import OpenAIClient  # 你提供的真实 LLM 客户端

# ==========================================
# 全局配置（从环境变量读取，或直接填写）
# ==========================================
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "your-api-key-here")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")

# ==========================================
# 创建全局 QASystem 实例（单例）
# ==========================================
# 初始化 LLM 客户端
llm_client = OpenAIClient(
    api_key=SILICONFLOW_API_KEY,
    model=LLM_MODEL,
    base_url=OPENAI_BASE_URL
)

# 初始化仓库注册表（字典，存储已加载的仓库及其 Pipeline）
repo_registry = {}

# 创建 QASystem 实例，传入注册表和 LLM 客户端
qa_system = QASystem(registry=repo_registry, llm_client=llm_client)

# ==========================================
# Gradio UI 逻辑
# ==========================================

class CodebaseCopilotUI:
    def __init__(self):
        self.current_repo_id = None

    def process_repository(self, repo_url: str):
        """加载仓库（同步，因为 QASystem.load_repository 是同步的）"""
        if not repo_url or "github.com" not in repo_url:
            yield "❌ 错误: 请输入有效的 GitHub 仓库 URL。", None
            return

        repo_name = repo_url.split("/")[-1]
        yield f"正在从 {repo_url} 克隆仓库...", None

        try:
            # 调用 QASystem 的真实加载流程（克隆、解析、切分、向量化、索引）
            repo_id = qa_system.load_repository(repo_url, force_reindex=False)
            self.current_repo_id = repo_id
            # 关键修复：返回系统生成的 repo_id 而不是原始 URL 中的 repo_name
            yield f"✅ 仓库 '{repo_id}' 加载成功！您可以开始提问了。", repo_id
        except Exception as e:
            yield f"❌ 仓库加载失败: {str(e)}", None

    async def chat(self, message: str, history: List[dict], repo_id: str):
        """处理用户消息，调用 QASystem.ask 获取答案"""
        if not repo_id:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "⚠️ 请先在上方加载一个 GitHub 仓库。"})
            return history, ""

        # 调用 QASystem 的异步问答接口
        result = await qa_system.ask(
            question=message,
            session_id=None,           # 可以让 Gradio 自动生成，也可以自己管理
            repo_id=repo_id,
            k=5,                       # 可以从高级设置中获取
            temperature=0.0,
            max_tokens=1024,
            score_threshold=0.2
        )

        if result["success"]:
            full_response = result["answer"]
            # 添加引用信息（如果存在）
            if result["references"]:
                refs_md = "\n\n**引用来源:**\n"
                for ref in result["references"]:
                    # 生成真实的 GitHub 链接
                    link = f"https://github.com/{repo_id}/blob/main/{ref['file_path']}#L{ref['start_line']}"
                    refs_md += f"- [`{ref['file_path']}:{ref['start_line']}`]({link})\n"
                full_response += refs_md
        else:
            full_response = f"❌ 系统错误: {result['error']}"

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": full_response})
        return history, ""

# ==========================================
# 构建 Gradio 界面
# ==========================================

def create_ui():
    ui_logic = CodebaseCopilotUI()

    with gr.Blocks(title="AI Codebase Copilot") as demo:
        repo_state = gr.State(None)

        gr.Markdown("# 🤖 AI Codebase Copilot")
        gr.Markdown("输入 GitHub 仓库地址，让 AI 帮你阅读代码。")

        with gr.Row():
            with gr.Column(scale=4):
                repo_url = gr.Textbox(
                    label="GitHub 仓库 URL",
                    placeholder="https://github.com/username/repo",
                    container=False
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("🚀 加载仓库", variant="primary")

        status_display = gr.Markdown("*等待加载仓库...*")

        chatbot = gr.Chatbot(label="对话历史", height=500)

        with gr.Row():
            msg = gr.Textbox(
                label="输入您的问题",
                placeholder="例如：这个仓库的架构是怎么设计的？",
                scale=9
            )
            submit_btn = gr.Button("发送", scale=1, variant="primary")

        with gr.Accordion("高级设置", open=False):
            k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="检索片段数量 (k)")
            temp_slider = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="LLM 温度 (Temperature)")

        # 加载仓库（同步生成器）
        load_btn.click(
            ui_logic.process_repository,
            inputs=[repo_url],
            outputs=[status_display, repo_state]
        )

        # 发送消息（异步）
        submit_btn.click(
            ui_logic.chat,
            inputs=[msg, chatbot, repo_state],
            outputs=[chatbot, msg]
        )
        msg.submit(
            ui_logic.chat,
            inputs=[msg, chatbot, repo_state],
            outputs=[chatbot, msg]
        )

        gr.ClearButton([msg, chatbot], value="🗑️ 清空对话")

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft()
    )