import os 
# The load_dotenv() call must NOT be here!

# Qdrant Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "my_book_rag"

# OpenAI Configuration (Pulls from the environment set by main.py)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "mistralai/mistral-7b-instruct:free"

# Application/Book Configuration
# File path where your book markdown files are located
BOOK_DOCS_PATH = r"C:\Users\DELL\my-ai-project\my-book\docs\tutorial-basics"

# Verification
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in the environment (check your .env file).")

print("Configuration loaded successfully.")