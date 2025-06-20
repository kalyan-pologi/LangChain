from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

result = embeddings_model.embed_query("What is the capital of France?")
print(f"Embedding: {result}")





