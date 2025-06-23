from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal, Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# ------------------------
# 1. Pydantic Example
# ------------------------
print("\n--- Pydantic Model Example ---")

class Student(BaseModel):
    name: str = 'nitish'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='CGPA of student')

new_student = {'age': '32', 'email': 'abc@gmail.com'}
student = Student(**new_student)

student_dict = student.model_dump()
print("Age:", student_dict['age'])
print("Student JSON:", student.model_dump_json())

# ------------------------
# 2. TypedDict Example
# ------------------------
print("\n--- TypedDict Example ---")

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {'name': 'nitish', 'age': 35}
print(new_person)

# ------------------------
# 3. Structured Output with ChatOpenAI
# ------------------------
print("\n--- LangChain Structured Output with ChatOpenAI ---")

model = ChatOpenAI()

class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes in the review")
    summary: str = Field(description="Brief summary")
    sentiment: Literal["pos", "neg"] = Field(description="Sentiment")
    pros: Optional[list[str]] = Field(default=None, description="List of pros")
    cons: Optional[list[str]] = Field(default=None, description="List of cons")
    name: Optional[str] = Field(default=None, description="Reviewer name")

structured_model = model.with_structured_output(Review)

review_text = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. 
The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches. 
The 200MP camera with night mode is stunning, but 100x zoom isn't very clear.

However, it’s too big for one-handed use and comes with Samsung bloatware. 
Also, it's quite expensive.

Pros:
- Powerful processor
- Excellent camera
- Great battery
- S-Pen support

Review by Nitish Singh
"""

result = structured_model.invoke(review_text)
print(result)

# ------------------------
# 4. Structured Output with TinyLlama
# ------------------------
print("\n--- LangChain Structured Output with HuggingFace TinyLlama ---")

hf_llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
hf_model = ChatHuggingFace(llm=hf_llm)
hf_structured = hf_model.with_structured_output(Review)

hf_result = hf_structured.invoke(review_text)
print(hf_result)

# ------------------------
# 5. Structured Output with TypedDict + ChatOpenAI
# ------------------------
print("\n--- TypedDict + LangChain Structured Output ---")

class ReviewDict(TypedDict):
    key_themes: Annotated[list[str], "Key themes in the review"]
    summary: Annotated[str, "Brief summary"]
    sentiment: Annotated[Literal["pos", "neg"], "Sentiment"]
    pros: Annotated[Optional[list[str]], "List of pros"]
    cons: Annotated[Optional[list[str]], "List of cons"]
    name: Annotated[Optional[str], "Reviewer name"]

typed_model = model.with_structured_output(ReviewDict)
typed_result = typed_model.invoke(review_text)
print("Name from TypedDict output:", typed_result['name'])
