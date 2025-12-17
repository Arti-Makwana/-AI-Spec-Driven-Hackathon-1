from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import qdrant_client
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()

# 1. ALLOW THE WEBSITE TO CONNECT (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. INITIALIZE AI MODEL AND DATABASE (This fixes the "not defined" error)
local_model = SentenceTransformer('all-MiniLM-L6-v2')
client = qdrant_client.QdrantClient(host="localhost", port=6333)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running"}

@app.post("/query")
async def process_query(request: QueryRequest):
    # Convert user question to numbers
    query_vector = local_model.encode(request.query).tolist()
    
    # Search the 6 points in your 'my_book_rag' collection
    search_result = client.search(
        collection_name="my_book_rag",
        query_vector=query_vector,
        limit=3
    )
    
    if search_result:
        # Get the best matching text from your points
        best_match = search_result[0].payload['text']
        sources = [res.payload['source'] for res in search_result]
        
        return {
            "final_answer": best_match,
            "sources": list(set(sources))
        }
    
    return {"final_answer": "No relevant info found.", "sources": []}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)