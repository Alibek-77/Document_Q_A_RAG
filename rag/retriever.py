from rank_bm25 import BM25Okapi
from models.schemas import Question,MultiQueryResponse
from sentence_transformers import CrossEncoder
import numpy as np
import os,json
from rag.indexer import collection
from rag.generator import generate_answer
from config import gemini_client
from google import genai
from google.genai import types
from dotenv import load_dotenv
reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def reranking_document(question:Question,documents:list[str]):
    pairs=[(question.question,doc) for doc in documents]
    scores=reranker.predict(pairs)
    ranked=sorted(
        zip(documents,scores),
        key=lambda x:x[1],
        reverse=True
    )
    return [doc for doc,score in ranked[:question.top_k]]
def naive_rag(q:Question):
    result=collection.query(
        query_texts=[q.question],
        n_results=q.top_k
    )
    context="\n\n".join(result["documents"][0])
    print(context)
    #generator
    return generate_answer(context=context,question=q.question)
def rewrite_query_rag(question:Question):
    rewrite_query_response=gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
Ты помощник который улучшает поисковые запросы.

Перепиши запрос пользователя чтобы он был более детальным и точным для поиска в документации.

Верни только переписанный запрос, без объяснений.

Запрос пользователя:{question}
"""
    )
    new_query=rewrite_query_response.text
    result=collection.query(
        query_texts=[new_query],
        n_results=3
    )
    context="\n\n".join(result["documents"][0])
    #generator answer
    return generate_answer(question.question,context)
def multi_query_rag(question:Question):
    multiple_query_response=gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Сгенерируй 3 разных варианта этого вопроса для поиска в документации.

Каждый вариант должен искать ту же информацию но с разных углов.

Вопрос от пользователя: {question.question}""",
config=types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=MultiQueryResponse
)
    )
    data=json.loads(multiple_query_response.text)
    queries=data["queries"]
    print(queries)
    queries.append(question.question)
    all_docs=set()
    for query in queries:
        result=collection.query(query_texts=[query],n_results=2)
        for res in result["documents"][0]:
            all_docs.add(res)
    context="\n\n".join(all_docs)
    print(context)
    return generate_answer(question.question,context=context)
    #generation answer
def hybrid_search(question:Question,alpha:float=0.5,):
    pass