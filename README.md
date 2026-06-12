# Intelligent Document Chatbot (RAG-based)

An end-to-end Retrieval-Augmented Generation (RAG) system built to enable precise question-answering over unstructured documents using LangChain and FAISS.

## Key Features
* **Semantic Document Chunking:** Implements LangChain text splitters to optimize context windows.
* **Vector Indexing:** Utilizes FAISS to index and perform high-speed similarity matching on document text embeddings.
* **Contextual Generation:** Channels queries through an open-source LLM by injecting retrieved source documents directly into the context window.

## Technical Stack
* **Framework:** LangChain
* **Vector Database:** FAISS
* **Embeddings:** Hugging Face Transformers
* **Language:** Python
