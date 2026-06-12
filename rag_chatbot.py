import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# 1. Simulate an unstructured document (Your resume mentions PDF text extraction)
print("Step 1: Loading unstructured document data...")
with open("company_policy.txt", "w") as f:
    f.write("Praveen Reddy is a Computer Science Engineer graduated in 2026. "
            "He qualified the GATE 2026 exam and specializes in AIML.")

loader = TextLoader("company_policy.txt")
documents = loader.load()

# 2. Chunking: Break the text into smaller pieces so it fits in the LLM's memory
print("Step 2: Chunking document into smaller segments...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. Vector Embeddings & Semantic Search (Using FAISS as listed on your resume)
print("Step 3: Generating vector embeddings and indexing into FAISS...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(texts, embeddings)

# 4. Connect to an Open-Source LLM (Using a free Hugging Face Hub token)
# Note: You can get a free token from huggingface.co
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_free_token_here"
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})

# 5. Create the RAG Chain: Connect the Vector Store retrieval to the LLM generation
print("Step 4: Setting up the Retrieval-Augmented Generation chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 1})
)

# 6. Ask a question about your specific document
question = "What exam did Praveen Reddy qualify in 2026?"
print(f"\nUser Question: {question}")
response = qa_chain.run(question)
print(f"Chatbot Response: {response}")
