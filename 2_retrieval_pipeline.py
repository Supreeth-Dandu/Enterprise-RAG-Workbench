from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

persistent_directory = "db/chroma_db"  # Points to the saved Chroma DB(has stored vectors + metadata)

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small") 
   #we should use the same embedding model we used when we created the DB

# Loading the existing vector store   
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model, # knows how to embed user query
    collection_metadata={"hnsw:space": "cosine"}  #algo: cosin similarity
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

#Creating a retriever (top-k search)
retriever = db.as_retriever(search_kwargs={"k": 5}) 
# its going to retrieve the top 5 chunks with highes similarity scores to the user's query embedding 

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#            # prevents garbage context and helps reduce hallucinations later
#     }
# )

# Actually retrieves the docs
relevant_docs = retriever.invoke(query)
 # Each doc has: doc.page_content: (chunk) && doc.metadata: (source file path, and any other metadata we stored)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
