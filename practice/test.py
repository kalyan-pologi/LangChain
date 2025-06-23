
# ─────────────────────────────────────────────────────────────────────
# 0. 🔍 OBSERVABILITY & TRACING
# --------------------------------------------------------------------
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
from langchain.callbacks.tracers import LangChainTracer
tracer = LangChainTracer(project_name="demo-kitchen-sink")

# ─────────────────────────────────────────────────────────────────────
# 1. 🤖 LLM & CHAT MODELS
# --------------------------------------------------------------------
from langchain_community.llms import OpenAI, HuggingFaceHub
from langchain_community.chat_models import ChatOpenAI, ChatHuggingFace

llm = OpenAI(model_name="gpt-4o-mini", temperature=0.2)
chat_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

hf_llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.3, "max_length": 512}
)

chat_hf = ChatHuggingFace(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
)

print("≈ tokens:", llm.get_num_tokens("Hello LangChain!"))

# ─────────────────────────────────────────────────────────────────────
# 1a. 🛡️ MODERATION / GUARDRAILS
# --------------------------------------------------------------------
from langchain.chains import OpenAIModerationChain
moderation = OpenAIModerationChain()
assert moderation.run("A harmless request")["safe"]

# ─────────────────────────────────────────────────────────────────────
# 2. 🔑 EMBEDDINGS
# --------------------------------------------------------------------
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("vector me!")

# ─────────────────────────────────────────────────────────────────────
# 3. 🧠 VECTOR STORES
# --------------------------------------------------------------------
from langchain_community.vectorstores import FAISS
docs = ["alpha", "bravo", "charlie"]
faiss = FAISS.from_texts(docs, embeddings)
retriever = faiss.as_retriever(search_type="mmr")

from langchain.retrievers.self_query import SelfQueryRetriever
from langchain.retrievers.parent_document import ParentDocumentRetriever

self_query = SelfQueryRetriever.from_llm(llm=llm, vectorstore=faiss)

parent_ret = ParentDocumentRetriever(
    vectorstore=faiss,
    doc_min_chunk_chars=400,
    doc_max_chunk_chars=1500,
)

# ─────────────────────────────────────────────────────────────────────
# 4. 📝 PROMPTS & CHAINS
# --------------------------------------------------------------------
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA, ConversationChain

prompt = PromptTemplate.from_template("Q: {question}\nA:")
qa_chain = LLMChain(llm=llm, prompt=prompt, callbacks=[tracer])
print(qa_chain.predict(question="What is LangChain?"))

rag_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, callbacks=[tracer]
)
print(rag_chain.invoke({"query": "alpha"}))

# ─────────────────────────────────────────────────────────────────────
# 5. 🧠 MEMORY
# --------------------------------------------------------------------
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "hi"}, {"output": "hey"})
chat = ConversationChain(llm=chat_llm, memory=memory)
print(chat.predict(input="How are you?"))

# ─────────────────────────────────────────────────────────────────────
# 6. 🔗 LCEL RUNNABLES
# --------------------------------------------------------------------
from langchain_core.runnables import RunnableSequence

pipeline = prompt | llm

print("— stream —")
for token in pipeline.stream({"question": "Explain LCEL"}):
    print(token, end="", flush=True)

batch_out = pipeline.batch([{"question": q} for q in ("one", "two", "three")])
print("\n— batch —", batch_out)

# ─────────────────────────────────────────────────────────────────────
# 7. ⚡ CACHING
# --------------------------------------------------------------------
from langchain.cache import InMemoryCache
import langchain
langchain.cache = InMemoryCache()

# ─────────────────────────────────────────────────────────────────────
# 8. 🛠️ TOOLS & AGENTS
# --------------------------------------------------------------------
from langchain.tools import tool, StructuredTool
from langchain.agents import initialize_agent, AgentType

@tool
def shout(text: str) -> str:
    "Repeat text loudly."
    return text.upper()

calc_tool = StructuredTool.from_function(
    lambda x, y: str(x + y), name="adder",
    description="Add two numbers",
)

agent = initialize_agent(
    tools=[shout, calc_tool],
    llm=chat_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[tracer],
)
print(agent.invoke({"input": "adder(3,4)"}))

# ─────────────────────────────────────────────────────────────────────
# 9. ♻️ SERIALIZE / RELOAD
# --------------------------------------------------------------------
from pathlib import Path
from langchain_core.runnables import Runnable

yaml_path = Path("qa_chain.yaml")
yaml_path.write_text(qa_chain.to_yaml())

loaded_chain = Runnable.deserialize(yaml_path.read_text())
print(loaded_chain.invoke({"question": "Persistence test"}))

# ─────────────────────────────────────────────────────────────────────
# 10. 🌐 FASTAPI WRAPPER
# --------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

@app.get("/qa")
async def qa(q: str):
    async def streamer():
        async for tok in rag_chain.astream({"query": q}):
            yield tok
    return StreamingResponse(streamer(), media_type="text/plain")

# ─────────────────────────────────────────────────────────────────────
# 11. 🧰 UTILITY HELPERS
# --------------------------------------------------------------------
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
print(splitter.split_text("A long text…" * 50)[:2])

# ─────────────────────────────────────────────────────────────────────
# 12. 🔄 CONCURRENCY DEMO
# --------------------------------------------------------------------
import asyncio
async def async_demo():
    results = await asyncio.gather(*[
        rag_chain.ainvoke({"query": v})
        for v in ("alpha", "bravo", "charlie")
    ])
    print("async results:", results)

# ─────────────────────────────────────────────────────────────────────
# 13. 🧪 HUGGING FACE TESTS
# --------------------------------------------------------------------
from langchain_core.messages import HumanMessage

print("HF LLM Output:", hf_llm("Translate English to French: 'The sky is blue.'"))
hf_response = chat_hf([HumanMessage(content="Tell me a joke about AI.")])
print("HF Chat Output:", hf_response.content)

# ─────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# --------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(async_demo())
    # uvicorn.run("langchain_kitchen_sink:app", host="127.0.0.1", port=8000, reload=True)
