from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


# Load environment variables from .env (should contain HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

# Check if the API token is loaded
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")

# # Initialize the Hugging Face Endpoint model
# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     huggingfacehub_api_token=hf_token,
# )
# chat_model = ChatHuggingFace(llm=llm)
# # Invoke the model
# result = chat_model.invoke("What is the capital of France?")
# print(f"Response: {result.content}")


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_length": 512,
        "temperature": 0.7
    }
)
chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is the capital of France?")
print(response.content)