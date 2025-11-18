from __future__ import annotations

import asyncio
import os

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from LB import LoadBalancer


async def main() -> None:
    groq_client = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.2,
    )

    gemini_client = ChatOpenAI(
        model="google/gemma-3n-e2b-it:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["Testing"],
        temperature=0.2,
    )
    gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv('GOOGLE_API_KEY','')
)
    openrouter_model = ChatOpenRouter(model="mistralai/mistral-small-3.2-24b-instruct:free",
    openai_api_key=os.getenv('Testing'))
    print(openrouter_model.invoke('what is the capital of Tunisia ?'))
    lb = LoadBalancer.from_models(
        [groq_client, gemini_client,gemini_model],
        strategy="round_robin",
    )

    prompts = [
        "Give one sentence defining model governance.",
        "List an evaluation metric for LLM security.",
    ]
    semaphore = asyncio.Semaphore(1)
    async def run_prompt(p: str) -> None:
        async with semaphore:
            response = await lb.agenerate(p)
            print(f"{p}\n=> {response.content if hasattr(response, 'content') else response}\n")

    await asyncio.gather(*(run_prompt(p) for p in prompts))


if __name__ == "__main__":
    asyncio.run(main())


