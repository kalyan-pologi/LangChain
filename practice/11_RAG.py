import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

# -------------------- STEP 1a: Get Transcript --------------------
video_id = "Gfr50f6ZBvo"

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
except TranscriptsDisabled:
    raise Exception("No captions available for this video.")

# -------------------- STEP 1b: Split Text --------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(f"✅ Split into {len(chunks)} chunks")

# -------------------- STEP 1c: Embed and Store --------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

# -------------------- STEP 2: Setup Retriever --------------------
retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 4}
)

# -------------------- STEP 3: Prompt Template --------------------
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

# -------------------- STEP 4: LLM & Manual Retrieval + Generation --------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

question = "Is the topic of nuclear fusion discussed in this video? If yes, then what was discussed?"
retrieved_docs = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})
answer = llm.invoke(final_prompt)

print("\n🤖 Answer:")
print(answer.content)

# -------------------- STEP 5: Build Full Chain --------------------
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# Example queries
print("\n📌 Chain Result: Who is Demis?")
print(main_chain.invoke("Who is Demis"))

print("\n📌 Chain Result: Summarize the video")
print(main_chain.invoke("Can you summarize the video?"))
