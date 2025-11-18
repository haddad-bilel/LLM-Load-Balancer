from __future__ import annotations

import asyncio
import itertools
import os
import random
from typing import Iterable, List, Literal, Any
from dotenv import load_dotenv
import time as t
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
load_dotenv()
Strategy = Literal["round_robin", "random"]


class GeminiLoadBalancer:
    """
    Minimal load balancer across multiple Gemini chat models.

    Uses LangChain ChatGoogleGenerativeAI clients internally and supports
    either round-robin or random routing.
    """

    def __init__(
        self,
        model_names: Iterable[str],
        *,
        strategy: Strategy = "round_robin",
        temperature: float = 0.2,
    ) -> None:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        names = list(model_names)
        if len(names) < 2:
            raise ValueError("Provide at least two model identifiers.")

        self.strategy = strategy
        clients: List[Any] = []
        for name in names:
            if isinstance(name, str) and name.startswith("groq:"):
                if not groq_key:
                    raise RuntimeError("GROQ_API_KEY must be set to use Groq models.")
                groq_model = name.split("groq:", 1)[1]
                clients.append(
                    ChatGroq(
                        model=groq_model,
                        temperature=temperature,
                        api_key=groq_key,
                    )
                )
            else:
                if not openrouter_key:
                    raise RuntimeError("GOOGLE_API_KEY must be set to use Gemini models.")
                clients.append(
                    ChatOpenAI(
                        model=name,
                        temperature=temperature,
                        base_url="https://openrouter.ai/api/v1",
                        api_key=openrouter_key,
                    )
                )

        self.clients = clients
        self._cycle = itertools.cycle(range(len(self.clients)))

    def _pick_client(self) -> Any:
        if self.strategy == "random":
            return random.choice(self.clients)
        # default round robin
        idx = next(self._cycle)
        return self.clients[idx]

    async def agenerate(self, prompt: str) -> str:
        client = self._pick_client()
        response = await client.ainvoke([HumanMessage(content=prompt)])
        #t.sleep(20)
        return response.content if hasattr(response, "content") else str(response)


async def main() -> None:
    prompts = [
        "Give me three product requirements for a secure chat platform.",
        "Summarize the key risks when deploying a medical AI assistant.",
        "List evaluation criteria for testing an onboarding chatbot.",
        "Suggest KPIs for monitoring a customer support LLM.",
    ]

    # Mix Groq + Gemini: prefix Groq models with 'groq:'
    models = ["groq:llama-3.3-70b-versatile", "mistralai/mistral-small-3.1-24b-instruct:free"]

    lb = GeminiLoadBalancer(models, strategy=os.getenv("LB_STRATEGY", "round_robin"))
    semaphore = asyncio.Semaphore(2)
    async def handle(prompt: str) -> None:
        async with semaphore:
            t.sleep(15)
            result = await lb.agenerate(prompt)
            print(f"\nPrompt: {prompt}\nResponse: {result}\n{'-' * 60}")

    await asyncio.gather(*(handle(p) for p in prompts))


if __name__ == "__main__":
    asyncio.run(main())


