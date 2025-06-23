# ------------------- Imports -------------------
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# ------------------- Load environment -------------------
load_dotenv()

# ------------------- HuggingFace Model (Gemma 2B) -------------------
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
gemma = ChatHuggingFace(llm=llm)

# ------------------- 1. JSON Output Parser -------------------
print("\n🧠 JSON Output Parser (5 Facts)")
json_parser = JsonOutputParser()
json_prompt = PromptTemplate(
    template="Give me 5 facts about {topic} \n{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": json_parser.get_format_instructions()}
)
json_chain = json_prompt | gemma | json_parser
print(json_chain.invoke({"topic": "black hole"}))

# ------------------- 2. Pydantic Parser (Structured Person Info) -------------------
print("\n👤 PydanticOutputParser (Fictional Person)")
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City of the person")

pydantic_parser = PydanticOutputParser(pydantic_object=Person)
pydantic_prompt = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person \n{format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()}
)
pydantic_chain = pydantic_prompt | gemma | pydantic_parser
print(pydantic_chain.invoke({"place": "Sri Lankan"}))

# ------------------- 3. Chained Prompts: Detailed → Summary -------------------
print("\n📝 Prompt Chaining: Report → Summary")
report_prompt = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)
summary_prompt = PromptTemplate(
    template="Write a 5 line summary on the following text.\n{text}",
    input_variables=["text"]
)

# Step-by-step prompt execution
detailed = report_prompt.invoke({"topic": "black hole"})
detailed_result = gemma.invoke(detailed)
summary = summary_prompt.invoke({"text": detailed_result.content})
summary_result = gemma.invoke(summary)
print(summary_result.content)

# ------------------- 4. OpenAI Equivalent Chaining -------------------
print("\n🔁 OpenAI Equivalent Chaining (with StrOutputParser)")
openai_model = ChatOpenAI()
str_parser = StrOutputParser()
openai_chain = report_prompt | openai_model | str_parser | summary_prompt | openai_model | str_parser
print(openai_chain.invoke({"topic": "black hole"}))

# ------------------- 5. StructuredOutputParser (Fielded Facts) -------------------
print("\n📦 StructuredOutputParser (3 Named Facts)")
response_schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]
structured_parser = StructuredOutputParser.from_response_schemas(response_schema)

structured_prompt = PromptTemplate(
    template="Give 3 fact about {topic} \n{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": structured_parser.get_format_instructions()}
)

structured_chain = structured_prompt | gemma | structured_parser
print(structured_chain.invoke({"topic": "black hole"}))
