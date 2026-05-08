import streamlit as st
import os
import tempfile
import easyocr
import cv2
import fitz
import numpy as np
from PIL import Image

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq

# تحميل قارئ النصوص (EasyOCR)
@st.cache_resource # لإبقاء القارئ في الذاكرة وتسريع التطبيق
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    return processed

def extract_text_from_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    processed = preprocess_image(image)
    results = reader.readtext(
        processed, detail=1, paragraph=True,
        text_threshold=0.7, low_text=0.4, contrast_ths=0.1
    )
    text = "\n".join([r[1] for r in results])
    return text

def extract_text_from_pdf(pdf_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name
    doc = fitz.open(tmp_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    os.unlink(tmp_path) # حذف الملف المؤقت بعد الانتهاء
    return text

def create_vector_store(text):
    docs = [Document(page_content=text)]
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

def build_qa_chain(vectorstore):
    # جلب المفتاح السري من إعدادات Streamlit Cloud الآمنة
    api_key = st.secrets["GROQ_API_KEY"]
    
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
    )
    return qa_chain

def main():
    st.set_page_config(page_title="Med-Eye", layout="centered")
    st.title("Med-Eye")
    st.write("Upload an image or PDF, then ask questions about it.")

    uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file is not None:
        if "extracted_text" not in st.session_state:
            with st.spinner("Extracting text..."):
                if "pdf" in uploaded_file.type:
                    st.session_state.extracted_text = extract_text_from_pdf(uploaded_file)
                else:
                    st.session_state.extracted_text = extract_text_from_image(uploaded_file)

        st.subheader("Extracted Text")
        st.text_area("OCR Result", st.session_state.extracted_text, height=200)

        if "qa_chain" not in st.session_state:
            with st.spinner("Creating AI knowledge base..."):
                vectorstore = create_vector_store(st.session_state.extracted_text)
                st.session_state.qa_chain = build_qa_chain(vectorstore)

        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Generating answer..."):
                answer = st.session_state.qa_chain.run(query)
                st.subheader("Answer")
                st.write(answer)

if __name__ == "__main__":
    main()
