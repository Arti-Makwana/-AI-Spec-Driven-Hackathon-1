import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import time
from openai import AsyncOpenAI
import numpy as np
from numpy.linalg import norm 

# --- 1. Load environment variables FIRST ---
from dotenv import os
from dotenv import load_dotenv
load_dotenv()

# This allows it to work BOTH locally (.env) and on Render (Environment Variables)
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure this matches your Render secret name
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "my_book_rag")

# --- 2. Import configuration SECOND ---
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, OPENAI_API_KEY, OPENAI_MODEL 

# --- Initialize FastAPI App ---
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
# ... (other imports)

# 27: Add CORS Middleware
app.add_middleware(
CORSMiddleware,
    # Add your GitHub Pages URL here!
    # If running locally, you only need 'http://localhost:3000' for testing
  # Update line 29 in main.py
allow_origins=[
    "http://localhost:3000",
    "https://Arti-Makwana.github.io" # Add your GitHub Pages domain here
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- IMPORTANT: Configure CORS ---
origins = [
   "http://localhost:3000",  # Docusaurus development server
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Global Resources ---
# 1. Qdrant Client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# 2. Embedding Model (Must match the one used in ingest.py)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. OpenRouter Client
openai_client = AsyncOpenAI(
    api_key="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",
    base_url=("https://openrouter.ai/api/v1")
)

# Pydantic schema for incoming query requests
class QueryRequest(BaseModel):
    query: str

# Function to generate embedding for the user's query
def embed_query(query: str):
    return embedding_model.encode(query).tolist()

# --- Core RAG Logic ---

# --- In your main.py file ---

# --- In your main.py file, replace the entire @app.post("/query") function ---
# --- In your main.py file, replace the entire @app.post("/query") function ---
# --- In main.py ---
# --- In your main.py file, replace the entire @app.post("/query") function ---
@app.post("/query")
async def process_query(request: QueryRequest):
    query_vector = None
    
    try:
        start_time = time.time()
        user_query = request.query
        
        query_vector = embed_query(user_query)

        # 2. Retrieve relevant context from Qdrant
        search_result = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vector,  # <--- Correct argument name
            limit=3
        )
        
        # 3. Extract and concatenate context and sources
        context = ""
        valid_sources = []

        # FINAL FIX: Accessing the search results via the standard '.points' attribute
        # This handles the 'QueryResponse' object returned by query_points()
        try:
            points_list = search_result.points # <--- Use .points attribute
        except AttributeError:
            # Fallback for very old/unusual clients, which might just use the result directly
            points_list = search_result
        
        # NOTE: The loop MUST use 'for hit in points_list:'
        for hit in points_list:
            
            # We use an arbitrary threshold of 0.5 to filter irrelevant results
            if hit.score >= 0.5:
                
                # Access data using the .payload attribute
                text_chunk = hit.payload.get('text', 'No content found') 
                source = hit.payload.get('source', 'Unknown')
                
                context += text_chunk + "\n---\n\n"
                
                if source and source not in valid_sources:
                    valid_sources.append(source)

        # 4. Check if context was found
        if not context:
            return {"final_answer": "I couldn't find any relevant information in the documentation to answer your question.", "sources": []}

        # --- STEP 5: Define the prompt (MUST be defined before params) ---
        prompt = f"""You are an expert technical documentation assistant. Use the following context, delimited by triple backticks, to answer the user's question.

If the answer is not present in the provided context, you must politely state that you cannot find the relevant information.

Context: ```{context}```
Question: {user_query}
"""
        
        # --- STEP 6: Prepare and CALL THE LLM ---
        # Create parameters dictionary for OpenRouter/old client compatibility
        params = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful and concise documentation assistant. Cite your sources."},
                {"role": "user", "content": prompt} # <-- Now 'prompt' is defined!
            ],
            "temperature": 0.0
        }

        # Use **params to pass all keyword arguments correctly
        # FIX: Use the asynchronous method '.acreate'
# FIX: The modern async method is just '.create()'
        response = await openai_client.chat.completions.create(**params)        
        # 6. Extract the final answer content from the response object
        final_answer = response.choices[0].message.content # <--- Must be this exact syntax
        
        # 7. Return the final structured response
        return {
            "final_answer": final_answer,
            "sources": valid_sources,
        }

    except Exception as e:
        # Catch any errors (API key issues, timeouts, etc.)
        print(f"An error occurred during query processing: {e}")
        # Return a generic 500 error message
        raise HTTPException(status_code=500, detail="Internal server error: Could not process the query.")