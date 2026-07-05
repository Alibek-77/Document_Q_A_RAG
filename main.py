from fastapi import HTTPException,FastAPI
from models.schemas import DocumentUpload,Answer,Question
from rag.indexer import indexing_document
from rag.retriever import naive_rag,multi_query_rag,rewrite_query_rag,hybrid_search
app=FastAPI(title="Document Q&A API")
@app.post("/documents/upload",status_code=201)
def upload_document(doc:DocumentUpload):
    chunks_count=indexing_document(doc)
    return {"message":f"Indexed {chunks_count} chunks","source":doc.source}
@app.post("/ask",response_model=Answer)
def ask_question(q:Question):
    if q.method.lower()=="hybrid":
        answer=hybrid_search(q)
    elif q.method.lower()=="naive":
        answer=naive_rag(q)
    elif q.method.lower()=="multi_query":
        answer=multi_query_rag(q)
    else:
        answer=rewrite_query_rag(q)
    return Answer(
        answer=answer,
        method_used=q.method,
    )