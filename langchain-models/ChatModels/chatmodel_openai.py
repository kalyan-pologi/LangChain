from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI chat model
chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.7)

# Invoke the chat model with a prompt and print the response
result = chat_model.invoke("What is the capital of France?")
print(result.content)


 