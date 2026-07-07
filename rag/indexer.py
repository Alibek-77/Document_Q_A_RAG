from models.schemas import DocumentUpload
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import chromadb
load_dotenv()
genai_ef=embedding_functions.GoogleGeminiEmbeddingFunction()
client=chromadb.PersistentClient("./chroma_db")
collection=client.get_or_create_collection(name="documents",embedding_function=genai_ef)
def chunking_document(text:str,chunk_size:int=400,overlap:int=50):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunks.append(text[start:end])
        start=end-overlap
    return chunks 
def indexing_document(document:DocumentUpload):
    chunks=chunking_document(document.text)
    collection.add(
        ids=[f"{document.source}_chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
    )
    return len(chunks)
