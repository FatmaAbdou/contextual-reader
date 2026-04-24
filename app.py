import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError, AttributeError):
        return os.getenv("GOOGLE_API_KEY")

api_key = get_api_key()
if not api_key:
    st.error("❌ GOOGLE_API_KEY missing. Add to .env (local) or Streamlit secrets.")
    st.stop()

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
        try:
            # HuggingFace embeddings (no API key, runs locally)
            self.embeddings = np.array(self.embeddings_model.embed_documents(texts))
        except Exception as e:
            st.error(f"❌ Embedding error: {type(e).__name__}: {str(e)}")
            raise

    def similarity_search(self, query, k=5):
        try:
            query_emb = np.array(self.embeddings_model.embed_query(query))
        except Exception as e:
            st.error(f"❌ Query embedding error: {e}")
            return []
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.chunks[i] for i in top_indices]

with st.sidebar:
    st.header("Status")
    st.success("✅ Google API key loaded for chat")
    st.info("🔍 Using HuggingFace embeddings (free, local, no API key)")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing PDF... (may take 2-3 minutes for a book)"):
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        st.info(f"Split into {len(chunks)} text chunks")

        # ---------- HuggingFace Embeddings ----------
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = SimpleVectorStore(embeddings)
        vectorstore.add_documents(chunks)
        st.session_state.vectorstore = vectorstore

        os.remove("temp.pdf")
        st.success("✅ PDF processed! You can now ask questions.")

if st.session_state.vectorstore is not None:
    st.header("Ask about your document")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    user_question = st.text_input("Your question:")
    if user_question:
        with st.spinner("Thinking..."):
            try:
                docs = st.session_state.vectorstore.similarity_search(user_question, k=5)
                if not docs:
                    st.warning("No relevant passages found. Try rephrasing your question.")
                else:
                    context = "\n\n".join([doc.page_content for doc in docs])
                    prompt = f"""Answer based only on the context below. If the answer is not there, say "I don't have enough information."

Context:
{context}

Question: {user_question}

Answer:"""
                    response = llm.invoke(prompt)
                    st.write(response.content)
                    with st.expander("📖 Source passages"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Source {i+1}:**")
                            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
            except Exception as e:
                st.error(f"Chat error: {type(e).__name__}: {str(e)}")
else:
    st.info("👈 Upload a PDF to begin.")