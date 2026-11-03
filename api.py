"""
api.py
------
Optional FastAPI backend for the Document Intelligence Chatbot.
Run separately in a second terminal:

    python api.py

The Streamlit app (main.py) runs independently and does NOT require this.
"""

import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

working_dir = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(working_dir, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    config_path = os.path.join(working_dir, "config.json")
    if os.path.isfile(config_path):
        try:
            config_data = json.load(open(config_path))
            GROQ_API_KEY = config_data.get("GROQ_API_KEY", "")
        except Exception:
            GROQ_API_KEY = ""

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Import helpers from main (functions only – no Streamlit side-effects at import time)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

app = FastAPI(title="Document Intelligence Chatbot API")


class MessageRequest(BaseModel):
    message: str


def _setup_vectorstore():
    persist_directory = os.path.join(working_dir, "vector_db_dir")
    embeddings = HuggingFaceEmbeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def _chat_chain(vectorstore):
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=False,
        return_source_documents=True,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chatbot(request: MessageRequest):
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured."}
    message = request.message
    vectorstore = _setup_vectorstore()
    chain = _chat_chain(vectorstore)
    response = chain({"question": message})
    return {"response": response["answer"]}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
