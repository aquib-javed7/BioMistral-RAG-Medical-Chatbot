import streamlit as st
from huggingface_hub import hf_hub_download
import os
import tempfile

from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ------------------ Streamlit UI Setup ------------------
st.set_page_config(page_title='🧠 BioMistral RAG Medical Assistant')
st.title('🩺 BioMistral RAG Medical Chatbot')
st.markdown('Upload a medical PDF and ask medical questions based on its content.')

# ------------------ Model Setup ------------------
MODEL_PATH = "BioMistral-7B.Q4_K_M.gguf"

# Safe auto-download (1st time only)
if not os.path.exists(MODEL_PATH):
    st.warning("📦 Downloading BioMistral model (~4GB). Please wait 3–5 mins...")
    try:
        MODEL_PATH = hf_hub_download(
            repo_id="QuantFactory/BioMistral-7B-GGUF",
            filename="BioMistral-7B.Q4_K_M.gguf",
            local_dir=".",
            local_dir_use_symlinks=False,
        )
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()
else:
    st.success("✅ Model found locally. Skipping download.")

# Load model with llama-cpp
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.2,
    max_tokens=2048,
    top_p=1,
    n_ctx=4096,
    verbose=False
)

# Prompt template
template = """<context>
You are a helpful and concise Medical Assistant that answers based on context and query.
</s>
<user>
{query}
</s>
<assistant>
"""
prompt = ChatPromptTemplate.from_template(template)

# ------------------ PDF Upload + RAG ------------------
uploaded_file = st.file_uploader("📄 Upload Medical PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.info("📚 Loading PDF and generating context...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()

    st.info("🔍 Creating vector embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Input query
    question = st.text_input("❓ Ask your medical question:")
    if question:
        st.markdown("### 🤖 Answer")
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(question)
            st.write(response)
else:
    st.info("👆 Please upload a PDF to get started.")
