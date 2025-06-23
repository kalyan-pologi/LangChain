# ------------------ IMPORTS ------------------
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEmbeddings

# ------------------ LOAD .env ------------------
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env")

# ------------------ 1. OPENAI CHAT MODEL ------------------
print("\n🧠 OpenAI Chat Model (GPT-4):")
openai_chat = ChatOpenAI(model_name="gpt-4", temperature=0.7)
response = openai_chat.invoke("What is the capital of France?")
print("Response:", response.content)

# ------------------ 2. OPENAI CLASSIC MODEL ------------------
print("\n🧠 OpenAI LLM (GPT-3.5 Turbo):")
classic_llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
response = classic_llm.invoke("What is the capital of France?")
print("Response:", response.content)

# ------------------ 3. ANTHROPIC (CLAUDE) ------------------
print("\n🧠 Anthropic Chat Model (Claude 2):")
claude = ChatAnthropic(model_name="claude-2", temperature=0.7)
response = claude.invoke("What is the capital of France?")
print("Response:", response.content)

# ------------------ 4. GOOGLE GENERATIVE AI ------------------
print("\n🧠 Google Gemini (1.5 Flash):")
gemini = ChatGoogleGenerativeAI(model_name="gemini-1.5-flash", temperature=0.7)
response = gemini.invoke("What is the capital of France?")
print("Response:", response.content)

# ------------------ 5. HUGGINGFACE CHAT MODEL ------------------
print("\n🧠 Hugging Face (TinyLlama):")
hf_llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_length": 512, "temperature": 0.7}
)
chat_model = ChatHuggingFace(llm=hf_llm)
response = chat_model.invoke("What is the capital of France?")
print("Response:", response.content)

# ------------------ 6. OPENAI EMBEDDINGS ------------------
print("\n🔎 OpenAI Embeddings + Similarity Search:")
openai_embed = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

docs = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]
query = "tell me about bumrah"
doc_embeddings = openai_embed.embed_documents(docs)
query_embedding = openai_embed.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
top_index = int(np.argmax(similarities))
print("Query:", query)
print("Most Relevant Document:", docs[top_index])
print("Similarity Score:", similarities[top_index])

# ------------------ 7. HUGGINGFACE EMBEDDINGS ------------------
print("\n🔎 Hugging Face Embeddings (MiniLM):")
hf_embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"use_auth_token": hf_token}
)

query = "What is the capital of France?"
query_vec = hf_embed.embed_query(query)
print("Query Embedding:", query_vec)

docs = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain."
]
doc_vecs = hf_embed.embed_documents(docs)
print("Document Embeddings:", doc_vecs)

# ------------------ 8. OPENAI QUERY EMBEDDING ONLY ------------------
print("\n🔎 OpenAI Single Query Embedding:")
query_result = openai_embed.embed_query("What is the capital of France?")
print("Embedding Vector:", query_result)
