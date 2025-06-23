# -------------------- Imports --------------------
import os
from dotenv import load_dotenv
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# -------------------- Load .env --------------------
load_dotenv()

# -------------------- 1. CharacterTextSplitter on PDF --------------------
print("\n📄 CharacterTextSplitter + PDF Example")

pdf_loader = PyPDFLoader("dl-curriculum.pdf")
pdf_docs = pdf_loader.load()

char_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''  # No separator for max granularity
)

char_chunks = char_splitter.split_documents(pdf_docs)
print(f"Chunks created from PDF: {len(char_chunks)}")
print("Sample chunk:", char_chunks[1].page_content[:250])

# -------------------- 2. Markdown-Aware Recursive Split --------------------
print("\n📝 RecursiveCharacterTextSplitter on Markdown Text")

markdown_text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data.

## Features
- Add new students
- View student details

## Tech Stack
- Python 3.10+
- No external dependencies

## Getting Started
Clone the repo and run.
"""

md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0,
)

md_chunks = md_splitter.split_text(markdown_text)
print(f"Chunks from Markdown: {len(md_chunks)}")
print("First chunk:", md_chunks[0])

# -------------------- 3. Python Code Split --------------------
print("\n🐍 RecursiveCharacterTextSplitter on Python Code")

code_text = '''
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_details(self):
        return self.name

    def is_passing(self):
        return self.grade >= 6.0

student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())
'''

py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

code_chunks = py_splitter.split_text(code_text)
print(f"Python Code Chunks: {len(code_chunks)}")
print("Sample chunk:\n", code_chunks[1])

# -------------------- 4. Semantic Chunker Example --------------------
print("\n🧠 SemanticChunker Example (OpenAI Embeddings)")

sample_text = """
Farmers were working hard in the fields. The Indian Premier League (IPL) is the biggest cricket league in the world.

Terrorism is a big danger to peace and safety. To fight terrorism, we need strong laws and alert security forces.
"""

semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3,
)

semantic_chunks = semantic_splitter.create_documents([sample_text])
print(f"Semantic Chunks Created: {len(semantic_chunks)}")
for doc in semantic_chunks:
    print("Chunk:\n", doc.page_content.strip(), "\n---")

# -------------------- 5. Natural Language Text --------------------
print("\n🚀 Natural Language Text with RecursiveCharacterTextSplitter")

nl_text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push boundaries.

These missions have not only expanded our knowledge of the universe but also advanced technologies like GPS and medical imaging.
"""

nl_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
)

nl_chunks = nl_splitter.split_text(nl_text)
print(f"Natural Language Chunks: {len(nl_chunks)}")
print(nl_chunks[0])
