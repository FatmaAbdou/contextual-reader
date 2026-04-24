import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="Contextual Reader", page_icon="📚", layout="wide")

st.title("📚 Contextual Reader")
st.markdown("Upload any PDF and ask questions about it.")

class SimpleVectorStore:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.chunks = []
        self.embeddings = np.array([])

    def add_documents(self, chunks):
        self.chunks = chunks
        texts = [chunk.page_content for chunk in chunks]
        self.embeddings = np.array(self.embeddings_model.embed_documents(texts))

    def similarity_search(self, query, k=5):
        query_emb = np.array(self.embeddings_model.embed_query(query))
        similarities = np.dot(self.embeddings, query_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb))
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.chunks[i] for i in top_indices]

with st.sidebar:
    st.header("Status")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ GOOGLE_API_KEY not found in .env file")
    else:
        st.success("✅ Google API key loaded")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing PDF... (this may take a few minutes)"):
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        st.info(f"Split into {len(chunks)} text chunks")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key
        )

        vectorstore = SimpleVectorStore(embeddings)
        vectorstore.add_documents(chunks)
        st.session_state.vectorstore = vectorstore

        os.remove("temp.pdf")
        st.success("✅ PDF processed! You can now ask questions.")

if st.session_state.vectorstore is not None:
    st.header("Ask about your document")

    # ✅ Use one of the models from your list (without the 'models/' prefix)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # or gemini-2.5-flash, gemini-2.5-pro
        temperature=0.3,
        google_api_key=api_key
    )

    user_question = st.text_input("Your question:", placeholder="e.g., What is the main theme?")

    if user_question:
        with st.spinner("Thinking..."):
            docs = st.session_state.vectorstore.similarity_search(user_question, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""Answer the question based only on the following context from the book. If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {user_question}

Answer:"""

            response = llm.invoke(prompt)
            st.markdown("### Answer")
            st.write(response.content)

            with st.expander("📖 See relevant passages from the book"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
else:
    st.info("👈 Upload a PDF to begin.")