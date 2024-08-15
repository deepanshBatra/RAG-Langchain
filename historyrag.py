import os
import warnings
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore

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



warnings.filterwarnings("ignore")
chat_list = []

if __name__ == "__main__":
    embeddings = llamaEmbed()
    vectorestore = contentLoad(embeddings=embeddings)
    chat = ChatOllama(verbose=True, temperature=0, model="llama3.1")
    
    # Initialize the Conversational Retrieval Chain
    QA = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=vectorestore.as_retriever()
    )

    # First question
    res = QA({"question": "What are the learning processes of machine learning and number them.", "chat_history": chat_list})
    print(res["answer"])  # Print the result from the response

    history = (res["question"], res["answer"])
    chat_list.append(history)

    # Follow-up question
    res = QA({"question": "Can you please elaborate more on learning process number 2?", "chat_history": chat_list})
    print(res["answer"])  # Print the result from the response
