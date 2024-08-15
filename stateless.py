from json import load
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
load_dotenv('api.env')

def llamaEmbed():
    llama_embed = OllamaEmbeddings(model="llama3.1")
    return llama_embed


def contentLoad(embeddings):
    vector_db = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"),
        embedding=embeddings
    )
    return vector_db



embeddings = llamaEmbed()
vectorestore = contentLoad(embeddings=embeddings)
chat = ChatOllama(verbose=True, temperature=0, model="llama3.1")
QA = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=vectorestore.as_retriever()
)   

res = QA.invoke("Give a specific definition for artificial intelligence.")
print(res) 