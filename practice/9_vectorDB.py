import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# --------------------- Step 1: Define IPL Player Documents ---------------------
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
]

# --------------------- Step 2: Initialize Chroma Vector Store ---------------------
vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='my_chroma_db',
    collection_name='sample'
)

# --------------------- Step 3: Add Documents ---------------------
vector_store.add_documents(docs)

# --------------------- Step 4: View Stored Documents ---------------------
print("\n📄 All Documents with Metadata and Embeddings:")
docs_data = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
print(docs_data)

# --------------------- Step 5: Perform Similarity Search ---------------------
print("\n🔍 Top 2 Results for: 'Who among these are a bowler?'")
results = vector_store.similarity_search(query='Who among these are a bowler?', k=2)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

# --------------------- Step 6: Search with Similarity Score ---------------------
print("\n📊 Similarity Scores:")
scored_results = vector_store.similarity_search_with_score(query='Who among these are a bowler?', k=2)
for i, (doc, score) in enumerate(scored_results):
    print(f"\n--- Result {i+1} ---")
    print(f"Score: {score}")
    print(f"Content: {doc.page_content}")

# --------------------- Step 7: Filter Search by Metadata ---------------------
print("\n🎯 Search for players in Chennai Super Kings:")
filtered_results = vector_store.similarity_search_with_score(
    query="",  # Blank query returns top-k documents filtered by metadata
    filter={"team": "Chennai Super Kings"}
)
for i, (doc, score) in enumerate(filtered_results):
    print(f"\n--- Chennai Player {i+1} ---")
    print(doc.page_content)

# --------------------- Step 8: Update a Document (Example: Kohli's Info) ---------------------
updated_doc = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

# Note: Replace the ID below with the actual document ID from vector_store.get()
document_id_to_update = '09a39dc6-3ba6-4ea7-927e-fdda591da5e4'
vector_store.update_document(document_id=document_id_to_update, document=updated_doc)

# --------------------- Step 9: Delete a Document ---------------------
vector_store.delete(ids=[document_id_to_update])

# --------------------- Step 10: Final Store Status ---------------------
print("\n📂 Final Vector Store State:")
print(vector_store.get(include=['documents', 'metadatas']))
