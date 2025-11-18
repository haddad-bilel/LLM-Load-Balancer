from __future__ import annotations

import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from utils import get_provider, get_model_name, assert_api_keys_present


def get_llm(temperature: float = 0.2, max_tokens: Optional[int] = None):
    """
    Returns a LangChain ChatModel configured for OpenRouter or Groq.
    - For OpenRouter, uses OpenAI-compatible client with base_url.
    - For Groq, uses official ChatGroq integration.
    """
    assert_api_keys_present()
    provider = get_provider()
    model = get_model_name()
    if provider == "openrouter":
        # ChatOpenAI with OpenRouter-compatible base_url
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            # Optional routing preferences:
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "RequirementExtraction"),
            },
        )
    elif provider == "groq":
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    else:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )


