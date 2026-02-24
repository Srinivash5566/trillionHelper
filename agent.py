from langchain_core.messages import SystemMessage, HumanMessage
from google.api_core.exceptions import ResourceExhausted
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from promptsTemplates import SYSTEM_PROMPT, aptitude_prompt
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import StructuredTool

import API_KEY as key

Gemini_model = "gemini-2.5-flash"
groq_model_name = "llama-3.1-8b-instant"

def get_llm_model(api_key: str):
    llm = ChatGroq(
        api_key=api_key,
        model=groq_model_name,
        temperature=0,
        max_tokens=256
    )
    return llm

load_dotenv()
# - coding (writing code, algorithms, programming tasks)
def solve_technical(question: str, api_key: str) -> str:
    llm = get_llm_model(api_key)

    prompt = f"""
You are a technical expert.

Rules:
- Give a clear, concise, correct answer
- No unnecessary verbosity
- Output ONLY the final answer

Question:
"{question}"
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
# ---_-_--_-_-_----_-_-_---__---_-_-_-_-____---__---___--_-__-_-__-_______-__-____________________-__-_____-___-
def solve_reasoning(question: str, api_key: str) -> str:
    try:
        llm = get_llm_model(api_key)

        val = ""
        try:
            search = SerpAPIWrapper(serpapi_api_key=key.API_KEY_WEB)
            val = search.run(question)
        except Exception:
            # Ignore web search failure for reasoning questions
            val = ""
        # print(val if val else "No web search results")
        prompt = f"""
You are a logical reasoning and technical 1 mark expert.

Rules:
- Think internally step by step
- Output ONLY the final answer
- No explanation

Problem:
"{question}"

Websearch results:
"{val}"
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    except ResourceExhausted:
        return "Api key quota exceeded. Please try again later."
    except Exception as e:
        return f"Error: {str(e)}"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

db = Chroma(
    persist_directory="chroma_stores/aptitude_formulas",
    embedding_function=embeddings,
    collection_name="aptitude_formulas"
)


def stage1_retrieve(question: str, k: int = 6):
    return db.similarity_search(
        query=question,
        k=k
    )


def extract_candidate_topics(docs):
    seen = {}
    for d in docs:
        topic = d.metadata.get("topic")
        if topic and topic not in seen:
            seen[topic] = d
    return list(seen.keys())

def build_formula_context(docs):
    return "\n\n".join(d.page_content for d in docs)
# -____--_----_--_-_----_-_--_-_-_-__-_-___--_--_-_-_--_----_-_-__-_-_----_-_-_-_--_-_-____-_-_-_-_-_-_-____-____-_-
def solve_question(question: str, ApiKey: str) -> str:
    """
    Build final prompt, send to Gemini through LangChain,
    and return the single final answer as a string.
    """

    try:
        docs = stage1_retrieve(question)
        formula_context = build_formula_context(docs)
        llm = get_llm_model(ApiKey)
        # Format the user problem into the template
        final_prompt = aptitude_prompt.format(
            context=formula_context,
            question=question
        )
        # print(final_prompt)
        # Create full message stack
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=final_prompt)
        ]
        # Call Gemini model
        response = llm.invoke(messages)
        return response.content.strip()
        # Return the model’s content (string)
    except ResourceExhausted:
        return "Api key quota exceeded. Please try again later."
    except Exception as e:
        return f"Error: {str(e)}"



def routing_agent(question: str, api_key: str, problem_type: str) -> str:
    if problem_type == "aptitude_formula":
        return solve_question(question, api_key)
    elif problem_type == "logical_reasoning":   
        return solve_reasoning(question, api_key)
    elif problem_type == "technical":
        return solve_technical(question, api_key)
PROBLEM_TYPES = [
    "aptitude_formula",
    "logical_reasoning",
    "technical"
]

if __name__ == "__main__":
    test_question = "If the sum of two numbers is 20 and their difference is 4, what are the numbers?"
    # print(routing_agent(test_question, "APIKEY",problem_type="aptitude_formula"))
    print("----",routing_agent(test_question, key.Groq_api_key,problem_type="aptitude_formula"))