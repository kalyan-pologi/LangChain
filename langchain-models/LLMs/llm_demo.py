from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI model
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Invoke the model with a prompt and print the response
result = llm.invoke("What is the capital of France?")
print(f"Response: {result.content}")
