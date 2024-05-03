import streamlit as st 
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
import time
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "<ACCESSTOKEN>"

if 'uploaded_pdfs' not in st.session_state:
    st.session_state.uploaded_pdfs = []

def load_doc_and_qa(pdf_doc):
    try:
        if pdf_doc is not None:
            pdf_reader = PdfReader(pdf_doc)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
                print(text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        text = text_splitter.create_documents(text)
        embedding = HuggingFaceEmbeddings()
        db = Chroma.from_documents(text, embedding)
      
        llm = Ollama(model="llama3")
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

        st.session_state.uploaded_pdfs.append((pdf_doc, chain))

        return 'Document has been successfully loaded'
    except Exception as e:
        return f"Error loading the document: {str(e)}"

# Streamlit UI
st.title("ChatPDF")
st.write("Upload PDF Files, then click on Load PDF File.")
st.write("Once the documents have been loaded, you can begin chatting with each PDF :)")

pdf_docs = st.file_uploader("Load PDF files", type=["pdf"], accept_multiple_files=True)
status = st.empty()

if pdf_docs is not None:
    load_pdfs = st.button('Load PDF files')
    if load_pdfs:
        for pdf_doc in pdf_docs:
            with st.spinner(f"Loading PDF: {pdf_doc.name}..."):
                status.text(load_doc_and_qa(pdf_doc))
                st.success(f"PDF '{pdf_doc.name}' loaded successfully")

for pdf_doc, qa_instance in st.session_state.uploaded_pdfs:
    st.write(f"Processing PDF: {pdf_doc.name}")
    form = st.form(key=f'question_form_{pdf_doc.name}')
    input_question = form.text_input("Type in your question", key=f'input_question_{pdf_doc.name}')
    submit_query = form.form_submit_button("Submit")

    if submit_query:
        start_time = time.time()  
        output = qa_instance.run(input_question)
        end_time = time.time()  
        processing_time = end_time - start_time 
        st.text("Output:")
        st.write(output)
        st.text(f"Time taken to process the query: {processing_time :.2f} seconds")
