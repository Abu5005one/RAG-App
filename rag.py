import os
import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.chains import RetrievalQA


file_path = "files_test.txt"
if not os.path.exists(file_path):
    st.error("File not found.")
    st.stop()

loader = TextLoader(file_path)
document = loader.load()

for doc in document:
    doc.metadata["source"] = "files_test.txt"


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(document)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


qdrant = Qdrant.from_documents(
    chunks,
    embeddings,
    collection_name="ie_sample_data",
    url="http://localhost:6333",
    prefer_grpc=True
)


model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)  # ✅ Fixed typo
llm = HuggingFacePipeline(pipeline=generator)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=qdrant.as_retriever(),
    return_source_documents=True
)

st.set_page_config(page_title="RAG Test")
st.title("KERALA TOURISM Q&A BOT")

user_query = st.text_input("Ask a question on Kerala Tourism (e.g. Top destinations in Kerala?)")

if user_query:
    with st.spinner(" Thinking..."):
        result = qa_chain({"query": user_query})

        st.subheader(" Answer")
        st.write(result["result"])




