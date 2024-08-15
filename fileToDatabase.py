from json import load
import os
from re import split
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv('api.env')

def pdf_loader(file_path):
    loader = PyPDFLoader(file_path=file_path)
    content= loader.load()
    return content


def text_splitter(content, chunk_size, chunk_overlap):
    split_text = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = split_text.split_documents(content)
    return chunks


def ollama_embeddings(chunks):
    ollama_embed = OllamaEmbeddings(model="llama3.1")
    return ollama_embed

def vectore_database(embeddings, chunks):
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=os.environ.get("INDEX_NAME")
    )


content = pdf_loader("A_Brief_Introduction_To_AI.pdf")
split_text = text_splitter(content=content, chunk_size=1000, chunk_overlap=200)
embeddings = ollama_embeddings(split_text)
db = vectore_database(embeddings=embeddings, chunks=split_text)

