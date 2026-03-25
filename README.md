# 智能代码问答系统

## 📖 项目简介

CodeAsk 是一个**基于 RAG（检索增强生成）的代码库智能问答系统**。用户只需提供一个 GitHub 仓库 URL，即可向系统提问关于该仓库的任何问题，例如“这个项目的整体架构是什么？”、“Graph 类的作用是什么？”、“如何运行这个项目？”。系统会从代码中检索相关片段，并由大语言模型生成带有来源引用的答案。

本项目完全在本地或自托管环境中运行，支持多种编程语言的代码理解，是学习 AI 工程（RAG、Embedding、向量检索）的理想实战项目。

---

## ✨ 主要功能

- 🔗 **GitHub 仓库自动加载**：输入仓库 URL，系统自动克隆并解析代码。
- 🔍 **代码智能切分**：基于抽象语法树（AST）按函数、类等语义单元切分代码，保证检索质量。
- 🧠 **向量化存储**：使用 `sentence-transformers` 生成代码片段的向量表示，并存入 FAISS 索引。
- 💬 **自然语言问答**：用户提问后，系统检索最相关的代码片段，调用 LLM 生成答案，并标注来源文件及行号。
- 🖥️ **交互式 Web UI**：基于 Gradio 构建，提供简洁友好的对话界面。
- 📚 **多轮对话支持**：维护会话历史，支持上下文连续提问。
- 🔗 **可点击引用**：答案中的引用自动转换为指向 GitHub 源代码的链接。

---

## 🏗️ 系统架构

```
GitHub Repo
    │
    ▼
Repo Downloader ────► Code Parser ────► Code Splitter (AST)
    │
    ▼
Embedding Model (sentence-transformers)
    │
    ▼
FAISS Vector DB
    │
    ▼
Retriever (语义搜索)
    │
    ▼
RAG Pipeline (LLM 生成)
    │
    ▼
Web UI (Gradio)
```

---

## 🛠️ 技术栈

| 组件 | 技术选择 |
|------|----------|
| 编程语言 | Python 3.9+ |
| 代码解析 | tree-sitter (支持多语言 AST) |
| 嵌入模型 | BAAI/bge-small-en (384维) |
| 向量数据库 | FAISS (CPU/GPU) |
| RAG 框架 | 自研 (LangChain 可选) |
| 大语言模型 | OpenAI API / 本地模型 (如 Ollama) |
| Web UI | Gradio |
| 版本控制 | GitPython |
| 依赖管理 | pip + conda (可选) |

---

## 🚀 快速开始

### 1. 环境准备

确保系统已安装 Python 3.9+ 和 Git。

```bash
# 克隆项目
git clone https://github.com/yourusername/ai-codebase-copilot.git
cd ai-codebase-copilot

# 创建 conda 环境（推荐）
conda create -n codebase-copilot python=3.10 -y
conda activate codebase-copilot

# 安装依赖
pip install -r requirements.txt
```

> 注意：若使用 OpenAI API，请设置环境变量 `OPENAI_API_KEY`。若使用本地模型，请配置 `LOCAL_LLM_ENDPOINT`。

### 2. 运行系统

#### 启动 Web UI
```bash
python app/gradio_app.py
```
然后在浏览器中打开 `http://127.0.0.1:7860`。

#### 使用命令行测试
```python
from src.qa_system import QASystem

qa = QASystem()
repo_path = qa.load_repo("https://github.com/tiangolo/fastapi")
answer = qa.ask("What is FastAPI?")
print(answer["answer"])
print(answer["references"])
```

---

## 📁 项目结构

```
ai-codebase-copilot/
│
├── data/                     # 数据存储目录
│   └── repos/                # 克隆的仓库
│
├── src/                      # 核心源码
│   ├── repo_loader.py        # 仓库下载模块
│   ├── code_parser.py        # 代码文件遍历
│   ├── code_splitter.py      # 代码切分（基于 AST）
│   ├── embedder.py           # 向量化模块
│   ├── vector_store.py       # FAISS 索引管理
│   ├── retriever.py          # 检索模块
│   ├── rag_pipeline.py       # RAG 生成流程
│   └── qa_system.py          # 高层问答接口 + 会话管理
│
├── app/                      # 应用入口
│   └── gradio_app.py         # Gradio Web UI
│
├── experiments/              # 实验与测试脚本
│   ├── test_loader.py
│   ├── test_splitter.py
│   └── e2e_test.py
│
├── requirements.txt          # Python 依赖
├── README.md                 # 项目说明
└── LICENSE
```

---

## 💡 使用示例

1. 在 Web UI 中输入 GitHub 仓库 URL，例如 `https://github.com/tiangolo/fastapi`。
2. 等待系统加载（首次需克隆并构建索引，耗时取决于仓库大小）。
3. 输入问题，例如：
   - “What is the purpose of the `APIRouter` class?”
   - “How to run the test suite?”
   - “Where is the dependency injection implemented?”
4. 系统返回答案，并附上引用（如 `fastapi/routing.py:120`），点击即可跳转到对应代码行。

---


**Happy Coding! 🚀**
