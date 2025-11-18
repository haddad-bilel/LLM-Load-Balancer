from __future__ import annotations
import asyncio
import itertools
import os
import random
from typing import Iterable, List, Literal, Any, Sequence
from dotenv import load_dotenv
import time as t
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import pydantic
load_dotenv()
Strategy = Literal["round_robin", "random"]


class LoadBalancer:
    """
    Minimal load balancer across multiple chat models.

    Uses LangChain ChatGoogleGenerativeAI clients internally and supports
    either round-robin or random routing.
    """

    def __init__(
        self,
        model_names: Iterable[str] | None = None,
        *,
        strategy: Strategy = "round_robin",
        temperature: float = 0.2,
    ) -> None:
        names = list(model_names or [])
        if len(names) < 2:
            raise ValueError("Provide at least two model identifiers.")

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        self.strategy = strategy
        self.temperature = temperature
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
                    raise RuntimeError("OPENROUTER_API_KEY must be set to use OpenRouter models.")
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

    @classmethod
    def from_models(
        cls,
        models: Sequence[Any],
        *,
        strategy: Strategy = "round_robin",
    ) -> "LoadBalancer":
        client_list = list(models)
        if len(client_list) < 2:
            raise ValueError("Provide at least two model clients.")
        instance = cls.__new__(cls)
        instance.clients = client_list
        instance.strategy = strategy
        instance.temperature = None
        instance._cycle = itertools.cycle(range(len(client_list)))
        return instance
    def _pick_client(self) -> Any:
        if self.strategy == "random":
            return random.choice(self.clients)
        # default round robin
        idx = next(self._cycle)
        return self.clients[idx]

    async def agenerate(self, prompt: str) -> str:
        client = self._pick_client()
        response = await client.ainvoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else response

    def with_structured_output(self, model_schema: type[pydantic.BaseModel]) -> "LoadBalancer":
        """
        Wrap each client with LangChain's structured output helper.

        Usage:
            lb = GeminiLoadBalancer(...).with_structured_output(MySchema)
        """
        wrapped = []
        for client in self.clients:
            if hasattr(client,'with_structured_output'):
                wrapped.append(client.with_structured_output(model_schema))
            else:
                wrapped.append(client)
        self.clients = wrapped
        return self