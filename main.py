from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import routing_agent
import json

app = FastAPI(
    title="Gemini Aptitude Solver API",
    description="Backend powered by LangChain + Gemini for Aptitude Solving",
    version="1.0"
)

# Allow Chrome Extension access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # or ["chrome-extension://<ID>"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str
    GoogleApiKey: str
    ProblemType: str
@app.get("/")
def MainGet():
    return {"msg": "Works!!!..."}

@app.post("/ask")
async def ask_gemini(data: Question):
    """
    Receives a question from the Chrome extension or frontend,
    calls your LangChain + Gemini agent,
    and returns the final answer.
    """
    try:
        print("Received question:", data.question)
        answer = routing_agent(data.question, data.GoogleApiKey, data.ProblemType)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

    
