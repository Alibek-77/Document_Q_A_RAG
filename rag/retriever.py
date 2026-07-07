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
def get_all_documents():
    result = collection.get()
    return result["documents"] if result["documents"] else []
def hybrid_search(question:Question,alpha:float=0.5):
    all_docs=get_all_documents()
    bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
    bm25_scores = np.array(bm25.get_scores(question.question.lower().split()))
    if bm25_scores.max()>0:
        bm25_scores=bm25_scores/bm25_scores.max()
    n= min(20, len(all_docs))
    result=collection.query(
        query_texts=[question.question],
        n_results=n
    )
    found_docs=result["documents"][0]
    found_distances=result["distances"][0]
    vector_scores=np.zeros(len(all_docs))
    for doc,dist in zip(found_docs,found_distances):
        if doc in all_docs:
            idx=all_docs.index(doc)
            vector_scores[idx]=1/(1+dist)
    if vector_scores.max()>0:
        vector_scores=vector_scores/vector_scores.max()
    combined=alpha*vector_scores+(1-alpha)*bm25_scores
    top_indices=np.argsort(combined)[::-1][:question.top_k*3]
    top_docs=[all_docs[i] for i in top_indices]
    final_docs=reranking_document(question,top_docs)
    context="\n\n".join(final_docs)
    return generate_answer(question.question,context)