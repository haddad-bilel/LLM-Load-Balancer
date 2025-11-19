import os
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import BaseModel
from LB import LoadBalancer
from ingestion import load_documents
from chunking import chunk_documents
import asyncio
import time as t

from langchain_openrouter import ChatOpenRouter

prompt = """
You are a helpful Tester that extracts LLM's Testing Criteria from specification documents.
The criteria are used then to evaluate the output of the LLM or the AI solution .
You are given chunks of a Project Specifications, Company Standards and Domain standards.
Your Role is to identify the criteria to test the output of our LLM based application using different criteria.
EXAMPLES:
{{
 "Category":"Security",
 "Criteria":"The answer should not contain any PII like names, adresses, SSN ... ."
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
1. THE CRITERIA EXTRACTED SHOULD ONLY BE USED TO EVALUATE THE OUTPUT OF THE MODEL
2. IF NO CRITERIA WERE FOUND RETUNR EMPTY JSON
3. DON'T RETURN ANY FOLLOW UP QUESTION
4. CUSTOM criteria are extracted from the specification Docs and it name can be extracted from there
OUTPUT FORMAT:
{{
"category":"Testing Criteria Category (Bias, Toxicity, Security, Fairness, CUSTOM )",
"criteria":"Criteria description."
}}

{text}
ANSWER:
"""
class Criteria(BaseModel):
    category:str
    criteria:str



async def main():
    docs = load_documents("Data")
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    openrouter_model = ChatOpenRouter(model="meta-llama/llama-3.3-70b-instruct:free",openai_api_key=os.environ["Testing"],
        temperature=0.2,)
    gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv('GOOGLE_API_KEY','')
    )
    
    lb = LoadBalancer(model_names=["groq:llama-3.3-70b-versatile","groq:llama-3.3-70b-versatile"], strategy="round_robin")._add_models([openrouter_model,gemini_model]).with_structured_output(Criteria)
    semaphore = asyncio.Semaphore(4)
    results = []
    async def handle(text:str):
        async with semaphore:
        #t.sleep(5)
            result = await lb.agenerate(prompt.format(text=text))
            print(type(result))
            results.append(result.model_dump())
            print(f"Response: {result}\n{'-' * 60}")
    await asyncio.gather(*(handle(doc.page_content) for doc in chunks))
    if len(results):
        with open('results2.json','w') as fp:
            import json
            json.dump({"results":results},fp)
if __name__=='__main__':
    start = t.time()
    asyncio.run(main())
    print(f"Executed in: {t.time()-start}")