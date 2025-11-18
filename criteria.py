from __future__ import annotations

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from config import get_llm

def build_criteria_generator() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You generate objective, testable evaluation criteria for LLMs grounded in requirements.",
            ),
            (
                "human",
                "You are given: (1) a synthesized corpus summary, and (2) an analysis of relationships between documents.\n"
                "Produce a JSON array of evaluation criteria. Each item should include:\n"
                "- id: stable identifier\n"
                "- name: short label\n"
                "- description: what is evaluated and why\n"
                "- priority: High|Medium|Low\n"
                "- metrics: concrete measurement approach (automatic if possible)\n"
                "- inputs: example prompts or input schema\n"
                "- expected_behavior: what the ideal model should do\n"
                "- references: sources that justify this criterion\n\n"
                "Corpus summary:\n{corpus_summary}\n\n"
                "Relationships:\n{relationships}\n",
            ),
        ]
    )
    llm = get_llm(temperature=0.1)
    parser = StrOutputParser()
    return prompt | llm | parser


async def generate_criteria(corpus_summary: str, relationships: str) -> str:
    runnable = build_criteria_generator()
    return await runnable.ainvoke(
        {
            "corpus_summary": corpus_summary,
            "relationships": relationships,
        }
    )


