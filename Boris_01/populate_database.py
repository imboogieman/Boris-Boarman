# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 

import shutil
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Ensure the OpenAI API key is set
openai_api_key = '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'
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please ensure it is in your .env file or passed directly.")
print(f"OpenAI API Key Loaded: {openai_api_key[:5]}...")


# Path to the directory containing markdown files
DATA_PATH = "D:\\Whatdahack71\\Boris Boarman MVP\\Bot App\\Boris Data"
CHROMA_PATH = "chroma"

# Loading documents to local memory for further processing
def load_documents():
    try:
        loader = DirectoryLoader(DATA_PATH, glob="*.md")
        documents = loader.load()
        if not documents:
            print("No documents were loaded. Please check the DATA_PATH and ensure the directory contains .md files.")
        else:
            print(f"Loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
        return []

# Splitting the text into chunks for more efficient analysis
def split_text(documents: list[Document]):
    if not documents:
        print("No documents to split.")
        return []

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        if len(chunks) > 10:
            document = chunks[10]
            print("Chunk 10 content:", document.page_content)
            print("Chunk 10 metadata:", document.metadata)
        
        return chunks
    except Exception as e:
        print(f"An error occurred while splitting documents: {e}")
        return []

# Create vector store for storing embeddings
def save_to_chroma(chunks: list[Document]):
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        # Create a new DB from the documents using OpenAIEmbeddings with the API key
        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(openai_api_key=openai_api_key), persist_directory=CHROMA_PATH
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    except Exception as e:
        print(f"An error occurred while saving to Chroma: {e}")

# Main execution
if __name__ == "__main__":
    documents = load_documents()
    chunks = split_text(documents)

    if chunks:
        print(f"Processing completed. Total chunks created: {len(chunks)}")
        save_to_chroma(chunks)
    else:
        print("No chunks were created.")
