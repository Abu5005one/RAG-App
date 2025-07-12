# RAG-App
A simple rag based chatbot on Kerala Tourism

How it works
- Uses a custom text file about Kerala tourism
- Splits it into chunks and stores as vector embeddings in Qdrant
- Uses `google/flan-t5-base` to generate responses
- UI built with Streamlit


Technologies Used
- LangChain
- Hugging Face Transformers
- Qdrant (via Docker)
- Streamlit

