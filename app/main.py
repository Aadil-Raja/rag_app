from fastapi import FastAPI
from models import Query
from rag import generate_answer

app = FastAPI()

@app.post("/ask")
async def ask_question(query: Query):
    answer = generate_answer(query.question)
    return {"answer": answer}
