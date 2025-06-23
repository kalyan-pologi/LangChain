from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain."
]
# Embed the documents
result = embeddings_model.embed_documents(documents)
print(f"Embedding: {result}")





