import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    支持 OpenAI API 及其兼容接口的客户端（如 vLLM, LocalAI, Ollama 等）
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: int = 60
    ):
        """
        :param api_key: API 密钥，优先使用传入参数，其次使用环境变量 SILICONFLOW_API_KEY 或 OPENAI_API_KEY
        :param model: 模型名称（默认使用 Qwen2.5-7B）
        :param base_url: API 基础地址（默认使用 SiliconFlow 镜像地址）
        :param timeout: 请求超时时间（秒）
        """
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        # 延迟导入 openai，避免在没有安装时出错
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError("请安装 openai 库: pip install openai")

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> str:
        """
        调用 OpenAI 兼容接口生成回答
        """
        client = self.openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API 调用失败: {e}")
            raise