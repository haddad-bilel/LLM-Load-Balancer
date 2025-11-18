from __future__ import annotations

import asyncio
from typing import Iterable, List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
import time
from config import get_llm


def build_chunk_summarizer() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that writes concise, faithful summaries. "
                "Summaries should retain key entities, metrics, constraints, and open questions. "
                "Use bullet points where appropriate. Avoid repeating boilerplate.",
            ),
            (
                "human",
                "Summarize the following chunk in 6-10 bullet points:\n\n"
                "{chunk}\n\n"
                "Metadata: {metadata}",
            ),
        ]
    )
    llm = get_llm(temperature=0.2)
    parser = StrOutputParser()
    return prompt | llm | parser


async def summarize_chunk(runnable: Runnable, chunk: Document) -> Tuple[Document, str]:
    text = chunk.page_content
    metadata = chunk.metadata or {}
    
    summary = await runnable.ainvoke({"chunk": text, "metadata": metadata})
    #time.sleep(5)
    return (chunk, summary)


async def summarize_chunks_parallel(chunks: List[Document], concurrency: int = 8) -> List[Tuple[Document, str]]:
    runnable = build_chunk_summarizer()

    semaphore = asyncio.Semaphore(concurrency)
    results: List[Tuple[Document, str]] = []

    async def worker(doc: Document):
        async with semaphore:
            return await summarize_chunk(runnable, doc)

    tasks = [asyncio.create_task(worker(c)) for c in chunks]
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    return results


def build_corpus_summarizer() -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert technical writer. Write an executive summary for product and ML teams.",
            ),
            (
                "human",
                "You are given multiple chunk-level summaries from a corpus of documents. "
                "Synthesize them into a single cohesive summary organized by themes. "
                "Include: key objectives, constraints, assumptions, data sources, evaluation targets, and risks.\n\n"
                "Chunk summaries:\n{chunk_summaries}",
            ),
        ]
    )
    llm = get_llm(temperature=0.2)
    parser = StrOutputParser()
    return prompt | llm | parser


async def synthesize_corpus_summary(chunk_summaries: List[str]) -> str:
    summarizer = build_corpus_summarizer()
    return await summarizer.ainvoke({"chunk_summaries": "\n\n---\n\n".join(chunk_summaries)})


