import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv


def get_pdf_content(documents):
    raw_text = ""

    for document in documents:
        pdf_reader = PdfReader(document)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

    return raw_text


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
                ## extract text from pdf documents
                extracted_text = get_pdf_content(documents)
                st.write(extracted_text)


if __name__ == "__main__":
    main()
