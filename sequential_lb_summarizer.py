from __future__ import annotations
from dotenv import load_dotenv
import asyncio
import time as t
import os
from langchain_core.messages import HumanMessage

from ingestion import load_documents
from chunking import chunk_documents
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from lb_summarizer import Criteria
prompt = """
You are a helpful Tester that extracts Testing Criteria from specification documents. 
You are given chunks of a Project Specifications, Company Standards and Domain standards.
Your Role is to identify the criteria to test the output of our LLM based application using different criteria.
INSTRUCTIONS:
1. THE CRITERIA EXTRACTED SHOULD ONLY BE USED TO EVALUATE THE OUTPUT OF THE MODEL
2. IF NO CRITERIA WERE FOUND RETURN EMPTY JSON
OUTPUT FORMAT:
{{
"category":"Testing Criteria Category (Bias, Toxicity, Security, CUSTOM)",
"criteria":"Criteria description."
}}

{text}
"""
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv('GOOGLE_API_KEY','')
).with_structured_output(Criteria)
llm=ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=0.2,
                        api_key=os.getenv("GROQ_API_KEY",''),
                    ).with_structured_output(Criteria)
async def main() -> None:
    docs = load_documents("Data")
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    results = []

    for doc in chunks:
        result = await llm.ainvoke(
            [HumanMessage(content=prompt.format(text=doc.page_content))]
        )
        print(result)
        content = result.content if hasattr(result, "content") else str(result)
        results.append(content)
        if len(results):
            with open('results_seq.json','w') as fp:
                import json
                json.dump({"results":results},fp)
        print(f"Response: {content}\n{'-' * 60}")


if __name__ == "__main__":
    start = t.time()
    asyncio.run(main())
    print(f"Executed in: {t.time() - start}")


