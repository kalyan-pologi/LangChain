import os
from dotenv import load_dotenv

# LangChain components
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()  # Ensure your .env has OPENAI_API_KEY

# -------------------- Wikipedia Retriever --------------------
print("\n📚 Wikipedia Retriever")
wiki_retriever = WikipediaRetriever(top_k_results=2, lang="en")
wiki_query = "the geopolitical history of india and pakistan from the perspective of a chinese"
wiki_docs = wiki_retriever.invoke(wiki_query)
for i, doc in enumerate(wiki_docs):
    print(f"\n--- Wikipedia Result {i+1} ---")
    print(doc.page_content[:500], "...")

# -------------------- Vector Store with Chroma --------------------
print("\n🔍 Vector Store (Chroma + OpenAI Embeddings)")
embedding_model = OpenAIEmbeddings()

chroma_docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

chroma_store = Chroma.from_documents(
    documents=chroma_docs,
    embedding=embedding_model,
    collection_name="my_collection"
)

chroma_retriever = chroma_store.as_retriever(search_kwargs={"k": 2})
query = "What is Chroma used for?"
results = chroma_retriever.invoke(query)
for i, doc in enumerate(results):
    print(f"\n--- Chroma Result {i+1} ---")
    print(doc.page_content)

# -------------------- MMR (Maximal Marginal Relevance) --------------------
print("\n🤖 MMR Retriever with FAISS")

mmr_docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

mmr_store = FAISS.from_documents(mmr_docs, embedding_model)
mmr_retriever = mmr_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)
mmr_query = "What is langchain?"
mmr_results = mmr_retriever.invoke(mmr_query)
for i, doc in enumerate(mmr_results):
    print(f"\n--- MMR Result {i+1} ---")
    print(doc.page_content)

# -------------------- MultiQuery Retriever --------------------
print("\n🧠 MultiQuery Retriever")

multi_docs = [
    Document(page_content="Regular walking boosts heart health.", metadata={"source": "H1"}),
    Document(page_content="Leafy greens help detox the body.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep repairs the body.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness lowers cortisol.", metadata={"source": "H4"}),
    Document(page_content="Hydration supports metabolism.", metadata={"source": "H5"}),
    Document(page_content="Solar energy balances demand.", metadata={"source": "I1"}),
    Document(page_content="Python is a readable programming language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis converts sunlight to energy.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was in Qatar.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime.", metadata={"source": "I5"}),
]

multi_vectorstore = FAISS.from_documents(multi_docs, embedding_model)
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=multi_vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)

query = "How to improve energy levels and maintain balance?"
similarity_results = multi_vectorstore.similarity_search(query, k=5)
multiquery_results = multi_retriever.invoke(query)

print("\n📈 Similarity Retriever:")
for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

print("\n🔍 MultiQuery Retriever:")
for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

# -------------------- Contextual Compression Retriever --------------------
print("\n📦 Contextual Compression Retriever")

comp_docs = [
    Document(page_content=(
        "The Grand Canyon is one of the most visited natural wonders in the world. "
        "Photosynthesis is the process by which green plants convert sunlight into energy. "
        "Millions of tourists travel to see it every year."
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        "In medieval Europe, castles were built primarily for defense. "
        "The chlorophyll in plant cells captures sunlight during photosynthesis."
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        "Basketball was invented by Dr. James Naismith. Originally played with a soccer ball and peach baskets."
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        "The history of cinema began in the 1800s. Silent films were the earliest form. "
        "Photosynthesis does not occur in animal cells."
    ), metadata={"source": "Doc4"})
]

compression_store = FAISS.from_documents(comp_docs, embedding_model)
base_retriever = compression_store.as_retriever(search_kwargs={"k": 5})

compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-3.5-turbo"))
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

compression_query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(compression_query)

for i, doc in enumerate(compressed_results):
    print(f"\n--- Compressed Result {i+1} ---")
    print(doc.page_content)
