from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the Anthropic chat model
chat_model = ChatAnthropic(model_name="claude-2", temperature=0.7)

# Invoke the chat model with a prompt and print the response
result = chat_model.invoke("What is the capital of France?")

print(f"Response: {result.content}")

