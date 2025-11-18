from typing import List,Tuple
from ingestion import load_documents    
from chunking import chunk_documents
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
import asyncio
import time as t
from summarization import summarize_chunk
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def build_chunk_summarizer()-> Runnable:
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
    parser = StrOutputParser()
    return prompt|model|parser



async def summarize_chunks_parallel(chunks: List[Document], concurrency: int = 2) -> List[Tuple[Document, str]]:
    runnable = build_chunk_summarizer()

    semaphore = asyncio.Semaphore(concurrency)
    results: List[Tuple[Document, str]] = []

    async def worker(doc: Document):
        async with semaphore:
            t.sleep(10)
            return await summarize_chunk(runnable, doc)

    tasks = [asyncio.create_task(worker(c)) for c in chunks]
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    return results


async def main():
    docs = load_documents("Data")
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    print(len(chunks))
    runnable = build_chunk_summarizer()
    start = t.time()
    summary = await summarize_chunks_parallel(chunks,concurrency=3)
    print(f'Lasted : {t.time()-start}')
    print("/"*6+"\n",len(summary))
    return chunks

if __name__ == "__main__":
    asyncio.run(main())
