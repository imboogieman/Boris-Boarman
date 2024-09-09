from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.evaluation import load_evaluator

def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

from langchain_openai import OpenAIEmbeddings

openai_api_key = "your_openai_api_key_here"

db = Chroma.from_documents(
    chunks, OpenAIEmbeddings(openai_api_key=openai_api_key), persist_directory=CHROMA_PATH
)
