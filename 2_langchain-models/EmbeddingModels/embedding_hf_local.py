from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")

# Initialize the Hugging Face embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"use_auth_token": hf_token}
)
text = "What is the capital of France?"
vector = embeddings_model.embed_query(text)
print(f"Embedding: {vector}")

documents = [
    "Paris is the capital of France.",  
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain."
]
# Embed the documents
result = embeddings_model.embed_documents(documents)
print(f"Embeddings: {result}")

