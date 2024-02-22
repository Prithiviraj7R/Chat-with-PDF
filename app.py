import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def get_pdf_content(documents):
    raw_text = ""

    for document in documents:
        pdf_reader = PdfReader(document)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

    return raw_text


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_storage


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:", layout="wide")
    st.image("templates/baasha.jpg")

    st.header("Hi, I am Baasha, a PDF ChatBot")
    st.text_input("How can I help you today?")

    with st.sidebar:
        st.subheader("PDF documents")
        documents = st.file_uploader(
            "Upload your PDF files", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Run"):
            with st.spinner("Processing..."):
                # extract text from pdf documents
                extracted_text = get_pdf_content(documents)
                # convert text to chunks of data
                text_chunks = get_chunks(extracted_text)
                # create vector embeddings
                vector_embeddings = get_embeddings(text_chunks)


if __name__ == "__main__":
    main()
