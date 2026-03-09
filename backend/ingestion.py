import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load API keys from the .env file
load_dotenv()

def ingest_pdf(file_path: str):
    """Loads a PDF, chunks it, and upserts vectors to Pinecone."""
    
    # 1. Load the raw document
    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split the document into chunks
    # We use an overlap of 200 to ensure sentences aren't cut in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")

    # 3. Initialize the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    # 4. Embed and push to Pinecone
    print(f"Pushing vectors to Pinecone index: '{index_name}'...")
    PineconeVectorStore.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        index_name=index_name
    )
    print("Ingestion complete! Data is ready for retrieval.")

if __name__ == "__main__":
    # Test the function by pointing it to your local data folder
    test_file_path = "./data/sample_handbook.pdf"
    
    if os.path.exists(test_file_path):
        ingest_pdf(test_file_path)
    else:
        print(f"Please place a PDF at {test_file_path} to test.")