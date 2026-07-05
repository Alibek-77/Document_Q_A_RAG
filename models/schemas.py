from pydantic import BaseModel,Field
from typing import Optional
class DocumentUpload(BaseModel):
    text:str=Field(max_length=10000)
    source:str
class Question(BaseModel):
    question:str
    method:str="hybrid"
    top_k:int=3
class Answer(BaseModel):
    answer:str
    method_used:str
class MultiQueryResponse(BaseModel):
    queries:list[str]