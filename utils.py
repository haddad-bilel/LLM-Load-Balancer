import os
from typing import Literal, Optional

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

ProviderName = Literal["openrouter", "groq", "gemini"]


def get_provider() -> ProviderName:
    provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    if provider not in ("openrouter", "groq", "gemini"):
        raise ValueError("LLM_PROVIDER must be 'openrouter', 'groq', or 'gemini'")
    return provider  # type: ignore[return-value]


def get_model_name(
    default_openrouter: str = "openrouter/auto",
    default_groq: str = "llama-3.1-70b-versatile",
    default_gemini: str = "gemini-2.5-flash",
) -> str:
    model = os.getenv("LLM_MODEL")
    if model:
        return model
    provider = get_provider()
    if provider == "openrouter":
        return default_openrouter
    if provider == "groq":
        return default_groq
    return default_gemini


def assert_api_keys_present() -> None:
    provider = get_provider()
    if provider == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter provider.")
    elif provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required for Groq provider.")
    else:
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini provider.")


