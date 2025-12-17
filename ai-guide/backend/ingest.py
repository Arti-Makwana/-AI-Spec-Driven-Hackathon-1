import os
import qdrant_client 
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from sentence_transformers import SentenceTransformer 
from qdrant_client.models import PointStruct, VectorParams, Distance

# You might need to add this if you haven't already:
from dotenv import load_dotenv
load_dotenv() 

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, BOOK_DOCS_PATH

# Initialize the local embedding model
local_model = SentenceTransformer('all-MiniLM-L6-v2') 
EMBEDDING_DIM = 384 

def generate_embedding(text: str) -> list[float]:
    """Generates an embedding vector for the given text using the local model."""
    embedding = local_model.encode(text)
    return embedding.tolist()

def load_and_split_documents(docs_path: str) -> list[dict]:
    """Loads and splits markdown files into chunks."""
    print(f"Starting data ingestion from: {docs_path}")
    all_chunks = []
    md = MarkdownIt()
    
    if not os.path.isdir(docs_path):
        print(f"Error: Path not found or is not a directory: {docs_path}")
        return all_chunks

    for root, _, files in os.walk(docs_path):
        for filename in files:
            if filename.endswith(('.md', '.mdx')):
                file_path = os.path.join(root, filename)
                print(f"  - Reading file: {filename}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    html = md.render(content)
                    soup = BeautifulSoup(html, 'html.parser')
                    chunk_text = soup.get_text()
                    
                    if chunk_text.strip():
                        doc_id = os.path.relpath(file_path, docs_path)
                        all_chunks.append({
                            "text": chunk_text,
                            "source": doc_id,
                            "id": len(all_chunks)
                        })
                except Exception as e:
                    print(f"  - Error processing {filename}: {e}")

    if not all_chunks:
        print("No content found to ingest. Check BOOK_DOCS_PATH in config.py.")

    return all_chunks

def ingest_data():
    """Handles the entire ingestion process."""
    
    # 1. Load and Split Documents
    chunks = load_and_split_documents(BOOK_DOCS_PATH)
    if not chunks:
        return

    print(f"Total chunks generated: {len(chunks)}")
    
    # 2. Setup Qdrant Client 
    client = qdrant_client.QdrantClient(
        url="https://fb3daab4-91bf-4bb3-9515-594865aae1c7.europe-west3-0.gcp.cloud.qdrant.io",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.f_2rDhtqiIZ900qC4PlnmZMElzM_CWVoftAZnEu__kA"
)
    
    print("Setting up Qdrant collection...")
    try:
       client.recreate_collection(
    collection_name="my_book_rag", # Change this from QDRANT_COLLECTION_NAME
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
)
    except Exception as e:
        print(f"Error setting up Qdrant collection: {e}. Check if your Docker container is running.")
        return

    # 3. Generate Embeddings and Prepare Points
    points = []
    print("Generating embeddings and preparing points...")
    for i, chunk in enumerate(chunks):
        print(f"  - Generating embedding for chunk {i+1}/{len(chunks)}", end='\r')
        vector = generate_embedding(chunk["text"])
        
        points.append(
            PointStruct(
                id=chunk["id"],
                vector=vector,
                payload={"text": chunk["text"], "source": chunk["source"]},
            )
        )
    print("\n")

    # 4. Upload to Qdrant
    print(f"Uploading {len(points)} vectors to Qdrant...")
    client.upsert(
        collection_name="my_book_rag",
        wait=True,
        points=points,
    )

    print("\n✅ Data ingestion complete!")

if __name__ == "__main__":
    # Ensure this function call is NOT indented!
    ingest_data()