# ------------------ IMPORTS ------------------
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    WebBaseLoader
)

# ------------------ Load environment variables ------------------
load_dotenv()

# ------------------ Initialize model and parser ------------------
model = ChatOpenAI()
parser = StrOutputParser()

# ------------------ PromptTemplate Example ------------------
summary_prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

qa_prompt = PromptTemplate(
    template='Answer the following question:\n{question}\nFrom the following text:\n{text}',
    input_variables=['question', 'text']
)

# ------------------ 1. CSV Loader ------------------
print("\n📄 CSV File Loader Example:")
csv_loader = CSVLoader(file_path='Social_Network_Ads.csv')
csv_docs = csv_loader.load()

print(f"Loaded {len(csv_docs)} rows.")
print(csv_docs[1])  # Show second record

# ------------------ 2. PDF Loader (Single File) ------------------
print("\n📕 PDF File Loader Example:")
pdf_loader = PyPDFLoader('dl-curriculum.pdf')
pdf_docs = pdf_loader.load()

print(f"Total pages: {len(pdf_docs)}")
print("Page 1 content preview:", pdf_docs[0].page_content[:200])
print("Page 2 metadata:", pdf_docs[1].metadata)

# ------------------ 3. Directory PDF Loader ------------------
print("\n📚 Directory PDF Loader Example:")
dir_loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

dir_docs = dir_loader.lazy_load()

print("Sample from one PDF file:")
for doc in dir_docs:
    print("Text Preview:", doc.page_content[:150])
    print("Metadata:", doc.metadata)
    break  # just to limit output

# ------------------ 4. Text File Loader ------------------
print("\n📜 Text File Loader Example:")
text_loader = TextLoader('cricket.txt', encoding='utf-8')
text_docs = text_loader.load()

print(f"Loaded {len(text_docs)} text documents.")
print("Text Preview:", text_docs[0].page_content[:200])
print("Metadata:", text_docs[0].metadata)

# Run summary chain on poem
summary_chain = summary_prompt | model | parser
print("\n📝 Poem Summary:")
print(summary_chain.invoke({'poem': text_docs[0].page_content}))

# ------------------ 5. Web Page Loader ------------------
print("\n🌐 Web Page Loader Example:")
web_loader = WebBaseLoader(
    'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
)
web_docs = web_loader.load()

print("Web Page Content Preview:", web_docs[0].page_content[:300])

# Ask a question about the loaded web content
qa_chain = qa_prompt | model | parser
question = "What is the product that we are talking about?"
print("\n🔍 Web-Based Q&A:")
print(qa_chain.invoke({'question': question, 'text': web_docs[0].page_content}))
