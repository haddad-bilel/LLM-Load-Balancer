from __future__ import annotations

from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from config import get_llm

def build_relationship_analyzer() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You analyze relationships between documents: overlaps, dependencies, contradictions, and traceability links.",
            ),
            (
                "human",
                "Given these document-level summaries with their sources, identify relationships.\n"
                "- Overlaps: common concepts across docs\n"
                "- Dependencies: doc B depends on A\n"
                "- Contradictions: conflicting statements\n"
                "- Traceability: map requirements to sources\n\n"
                "Provide a concise structured analysis in JSON with keys: overlaps, dependencies, contradictions, traceability.\n\n"
                "{doc_summaries}",
            ),
        ]
    )
    llm = get_llm(temperature=0.1)
    parser = StrOutputParser()
    return prompt | llm | parser


async def analyze_relationships(doc_summaries: List[str]) -> str:
    runnable = build_relationship_analyzer()
    return await runnable.ainvoke({"doc_summaries": "\n\n".join(doc_summaries)})


