from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the Google Generative AI chat model
chat_model = ChatGoogleGenerativeAI(model_name="gemini-1.5-flash", temperature=0.7) 

# Invoke the chat model with a prompt and print the response
result = chat_model.invoke("What is the capital of France?")

print(f"Response: {result.content}")

