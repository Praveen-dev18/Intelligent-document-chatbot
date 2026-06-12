# Intelligent Document Chatbot (RAG-based)
An end-to-end Retrieval-Augmented Generation (RAG) system built to enable precise question-answering over unstructured documents while eliminating model hallucinations.

## Key Features
* **Semantic Document Chunking:** Implements LangChain's `RecursiveCharacterTextSplitter` to optimize context windows.
* **Vector Indexing:** Utilizes `FAISS` (Facebook AI Similarity Search) to index and perform high-speed mathematical similarity matching on document text embeddings.
* **Contextual Anchor Generation:** Channels queries through an open-source LLM by injecting retrieved source documents directly into the context window.

## Technical Stack
* **Framework:** LangChain
* **Vector Database:** FAISS
* **Embeddings:** Hugging Face Transformers (`all-MiniLM-L6-v2`)
* **Language:** Python 

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/Praveen-dev18/intelligent-document-chatbot.git](https://github.com/Praveen-dev18/intelligent-document-chatbot.git)
