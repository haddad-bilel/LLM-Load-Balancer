from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from ingestion import load_documents
from chunking import chunk_documents
from summarization import summarize_chunks_parallel, synthesize_corpus_summary
from relationships import analyze_relationships
from criteria import generate_criteria


class PipelineResult:
    def __init__(
        self,
        chunk_summaries: List[Tuple[Document, str]],
        corpus_summary: str,
        relationships: str,
        criteria: str,
    ) -> None:
        self.chunk_summaries = chunk_summaries
        self.corpus_summary = corpus_summary
        self.relationships = relationships
        self.criteria = criteria

    def to_json(self) -> str:
        serializable = {
            "chunk_summaries": [
                {
                    "source": cs[0].metadata.get("source"),
                    "start_index": cs[0].metadata.get("start_index"),
                    "summary": cs[1],
                }
                for cs in self.chunk_summaries
            ],
            "corpus_summary": self.corpus_summary,
            "relationships": self.relationships,
            "criteria": self.criteria,
        }
        return json.dumps(serializable, indent=2, ensure_ascii=False)


async def run_pipeline(input_path: str | Path, chunk_size: int = 1500, chunk_overlap: int = 200) -> PipelineResult:
    # 1) Ingest
    documents = load_documents(input_path)
    if not documents:
        raise RuntimeError("No documents found to process.")

    # 2) Chunk
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Chunks: {len(chunks)}")
    # 3) Summarize chunks in parallel
    chunk_summaries = await summarize_chunks_parallel(chunks,concurrency=2)
    print(f"Chunk summaries: {len(chunk_summaries)}")
    # 4) Synthesize overall summary
    summaries_only = [s for (_, s) in chunk_summaries]
    corpus_summary = await synthesize_corpus_summary(summaries_only)
    print(f"Corpus summary: {corpus_summary}")
    # 5) Build per-document summaries (aggregate chunk summaries by source)
    by_source: Dict[str, List[str]] = {}
    for doc, summ in chunk_summaries:
        src = str(doc.metadata.get("source"))
        by_source.setdefault(src, []).append(summ)
    doc_level_summaries: List[str] = [
        f"Source: {src}\n\n" + "\n\n".join(parts) for src, parts in by_source.items()
    ]
     
    # 6) Analyze relationships
    relationships = await analyze_relationships(doc_level_summaries)

    # 7) Generate evaluation criteria
    criteria = await generate_criteria(corpus_summary, relationships)

    return PipelineResult(
        chunk_summaries=chunk_summaries,
        corpus_summary=corpus_summary,
        relationships=relationships,
        criteria=criteria,
    )


