import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()
class ChatOpenRouter(ChatOpenAI):
    """
    Convenience wrapper around ChatOpenAI that preconfigures OpenRouter settings.
    """

    def __init__(
        self,
        model: str = "google/gemma-3n-e2b-it:free",
        *,
        openai_api_key: Optional[str] = None,
        **kwargs,
    ):
        api_key = openai_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for ChatOpenRouter.")

        headers = kwargs.pop("default_headers", {})
        headers.setdefault("HTTP-Referer", os.getenv("OPENROUTER_SITE_URL", ""))
        headers.setdefault("X-Title", os.getenv("OPENROUTER_APP_NAME", "RequirementExtraction"))

        super().__init__(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(api_key),
            default_headers=headers,
            **kwargs,
        )

if __name__=="__main__":
    api_key = os.getenv("Testing")
    openrouter_model = ChatOpenRouter(
        model="qwen/qwen3-14b:free",
        openai_api_key=api_key
    )
    print(openrouter_model.invoke('What is the fastest car in the world ?'))