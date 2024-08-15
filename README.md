
# AI-Powered Document Processing and Q&A System

This project leverages AI models and Pinecone's vector database to process PDF documents, generate embeddings, and enable stateless and conversational retrieval-based Q&A systems. The project is divided into three main components:

## Components

### 1. `fileToDatabase.py`
This script loads a PDF file, splits its content into chunks, generates embeddings using the Ollama model, and stores these embeddings in a Pinecone vector database.

- **Functions:**
  - `pdf_loader(file_path)`: Loads content from a PDF file.
  - `text_splitter(content, chunk_size, chunk_overlap)`: Splits the loaded text into chunks.
  - `ollama_embeddings(chunks)`: Generates embeddings for the text chunks using the Ollama model.
  - `vectore_database(embeddings, chunks)`: Stores the generated embeddings in the Pinecone vector database.

### 2. `stateless.py`
This script sets up a stateless Q&A system using the stored embeddings. It enables you to ask questions related to the stored documents and get specific answers.

- **Functions:**
  - `llamaEmbed()`: Initializes the Ollama embeddings.
  - `contentLoad(embeddings)`: Loads the embeddings into the Pinecone vector store.
  - **Process:** Uses the `RetrievalQA` chain to answer questions based on the stored document embeddings.

### 3. `historyrag.py`
This script builds a conversational Q&A system, allowing for multi-turn interactions with context persistence across questions.

- **Functions:**
  - `llamaEmbed()`: Initializes the Ollama embeddings.
  - `contentLoad(embeddings)`: Loads the embeddings into the Pinecone vector store.
  - **Process:** Uses the `ConversationalRetrievalChain` to maintain a conversation history and provide contextual answers based on the document embeddings.



## Usage

- **fileToDatabase.py**: Run this script to process and store the document embeddings in Pinecone.
- **stateless.py**: Use this script to ask stateless questions based on the stored documents.
- **historyrag.py**: Run this script for a conversational Q&A system that remembers context across multiple questions.
