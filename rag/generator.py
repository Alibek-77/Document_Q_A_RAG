from config import gemini_client
from models.schemas import Answer
def generate_answer(question:str,context:str):
    response=gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Отвечай только на основе предоставленного контекста.
Если информации нет в контексте — скажи что не знаешь.
Контекст:
{context}
Вопрос пользователя: {question}
"""
    )
    return response.text