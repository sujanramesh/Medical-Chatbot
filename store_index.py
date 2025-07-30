from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Load and process data
extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
print(f"Loaded {len(text_chunks)} text chunks.")

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{index_name}' created and ready.")

# Load documents into Pinecone vector store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
print("Documents successfully embedded and stored.")
