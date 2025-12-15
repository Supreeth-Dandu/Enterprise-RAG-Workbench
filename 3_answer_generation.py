from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


docs_block = chr(10).join(
    [
        f"- SOURCE: {doc.metadata.get('source')}{chr(10)}  TEXT: {doc.page_content}"
        for doc in relevant_docs
    ]
)

# building one big prompt: question + retrieved context
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents: {docs_block}

Please provide a clear, helpful answer using only the information from these documents.

Requirements:
- In your answer, cite sources like [1], [2] for each claim.
- After the answer, include a "Evidence" section with 1-2 short quotes per cited source.


If you can't find the answer in the documents, say: "I don't have enough information to answer that question based on the provided documents."
"""
            # chr(10) is a newline
            #[f"- {doc.page_content}" for doc in relevant_docs]
            # adds source + text

# Create a ChatOpenAI model that generates the final answer
model = ChatOpenAI(model="gpt-4o")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."), # sets behavior
    HumanMessage(content=combined_input),  # contains the entire question + documents blob
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
 