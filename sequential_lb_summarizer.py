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
You are a helpful Evaluation expert that extracts Testing Criteria from specification documents.
The criteria are used then to evaluate the output of the AI solution on different criteria .
You are given chunks of a Project Specifications, Company Standards and Domain standards.
Your Role is to identify the criteria to test the output of our LLM based application using different criteria.
EXAMPLES:
{{
 "Category":"Security",
 "Criteria":"The answer should not contain any PII (Personal Identifiable Information) like names, adresses, SSN ... ."
}}

{{
 "Category":"Toxicity",
 "Criteria":"The model's output should not contain any toxic or harmful content."
}}

{{
 "category": "Bias",
 "criteria": "The model's recommendations should not be biased towards individual clients based on personal information."
}}

{{
    "category": "Fairness",
    "criteria": "The output should ensure fair debt collection practices compliance and adhere to state and federal foreclosure procedures."
}}

INSTRUCTIONS:
1. THE CRITERIA EXTRACTED MUST BE VALID TO BE USED FOR OUTPUT EVALUATION
2. IF NO CRITERIA WERE FOUND RETUNR EMPTY JSON
3. DON'T RETURN ANY FOLLOW UP QUESTION
4. CUSTOM criteria are extracted from the specification Docs and it's name can be extracted from there 
OUTPUT FORMAT:
{{
"category":"Testing Criteria Category (Bias, Toxicity, Security, Fairness, CUSTOM )",
"criteria":"Criteria description."
}}

{text}
ANSWER:
"""
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv('GOOGLE_API_KEY','')
).with_structured_output(Criteria)
llm=ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=0.2,
                        api_key=os.getenv("LB_API_KEY",''),
                    ).with_structured_output(Criteria)
async def main() -> None:
    docs = load_documents("Data")
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    results = []

    for doc in chunks[:10]:
        result = await llm.ainvoke(
            [HumanMessage(content=prompt.format(text=doc.page_content))]
        )
        print(result)
        content = result.content if hasattr(result, "content") else result
        results.append(content.model_dump())
        if len(results):
            with open('results_seq2.json','w') as fp:
                import json
                json.dump({"results":results},fp)
        print(f"Response: {content}\n{'-' * 60}")


if __name__ == "__main__":
    start = t.time()
    asyncio.run(main())
    print(f"Executed in: {t.time() - start}")


