import os
# Changed TextLoader to PyPDFLoader to handle real PDF documents
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# 1. Load a real PDF document from your laptop folder
print("Step 1: Extracting text from real PDF document...")

# Place any PDF file in the same folder as this script and type its name here
pdf_filename = "data_structures_notes.pdf" 

if not os.path.exists(pdf_filename):
    print(f"Error: Please drop a PDF file named '{pdf_filename}' into this folder first!")
    exit()
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Chunking: Break the text into smaller pieces
print("Step 2: Chunking document into smaller segments...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. Vector Embeddings & Semantic Search
print("Step 3: Generating vector embeddings and indexing into FAISS...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(texts, embeddings)

# 4. Connect to an Open-Source LLM (Requires your actual free token)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})

# 5. Create the RAG Chain
print("Step 4: Setting up the Retrieval-Augmented Generation chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}) # Retrieves top 2 matching context blocks
)

# 6. Ask a question about YOUR specific PDF
question = "What is the main topic discussed in this document?"
print(f"\nUser Question: {question}")
response = qa_chain.run(question)
print(f"Chatbot Response: {response}")
