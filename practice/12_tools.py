import os
import json
import requests
from typing import Annotated, Type
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from langchain_community.tools import DuckDuckGoSearchRun, ShellTool
from langchain.agents import initialize_agent, AgentType

load_dotenv()

# ------------------------- BUILT-IN TOOLS -------------------------

print("\n🔍 DuckDuckGo Search Tool:")
search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke("top news in india today")
print(result)

print("\n🖥️ Shell Tool:")
shell_tool = ShellTool()
print(shell_tool.invoke("echo Hello World"))

# ------------------------- CUSTOM TOOL: @tool -------------------------

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

print("\n🔧 Multiply Tool (@tool):")
print(multiply.invoke({'a': 3, 'b': 5}))

# ------------------------- STRUCTURED TOOL -------------------------

class MultiplyInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")

def multiply_func(a: int, b: int) -> int:
    return a * b

structured_multiply = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)
print("\n🔧 StructuredTool Result:")
print(structured_multiply.invoke({"a": 3, "b": 3}))

# ------------------------- BASE TOOL -------------------------

class MultiplyTool(BaseTool):
    name = "multiply"
    description = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

base_tool = MultiplyTool()
print("\n🔧 BaseTool Result:")
print(base_tool.invoke({'a': 3, 'b': 3}))

# ------------------------- TOOLKIT EXAMPLE -------------------------

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

class MathToolkit:
    def get_tools(self):
        return [add, multiply]

toolkit = MathToolkit()
print("\n🧰 Toolkit:")
for tool in toolkit.get_tools():
    print(f"{tool.name} - {tool.description}")

# ------------------------- TOOL BINDING WITH LLM -------------------------

llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([multiply])

messages = [HumanMessage(content="can you multiply 3 with 1000")]
response = llm_with_tools.invoke(messages)
messages.append(response)
tool_response = multiply.invoke(response.tool_calls[0])
messages.append(tool_response)

final_response = llm_with_tools.invoke(messages)
print("\n🧠 LLM Response After Tool Execution:")
print(final_response.content)

# ------------------------- MULTI-STEP TOOL USE: Currency Conversion -------------------------

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This tool fetches the currency conversion factor between base and target currencies
    """
    url = f'https://v6.exchangerate-api.com/v6/YOUR_API_KEY/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Converts the amount using the provided rate
    """
    return base_currency_value * conversion_rate

# Tool binding
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage("What is the conversion factor between INR and USD, and based on that convert 10 INR to USD.")]
response = llm_with_tools.invoke(messages)
messages.append(response)

for call in response.tool_calls:
    if call["name"] == "get_conversion_factor":
        tool_msg1 = get_conversion_factor.invoke(call)
        rate = json.loads(tool_msg1.content).get("conversion_rate", 1.0)
        messages.append(tool_msg1)
    elif call["name"] == "convert":
        call["args"]["conversion_rate"] = rate
        tool_msg2 = convert.invoke(call)
        messages.append(tool_msg2)

final_response = llm_with_tools.invoke(messages)
print("\n💱 Currency Conversion Final Response:")
print(final_response.content)

# ------------------------- AGENT SETUP -------------------------

agent_executor = initialize_agent(
    tools=[get_conversion_factor, convert],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent_input = "Can you convert 10 INR to USD using live rate?"
print("\n🤖 Agent-Based Conversion:")
agent_result = agent_executor.invoke({"input": agent_input})
print(agent_result['output'])
