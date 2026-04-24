import streamlit as st
import os
import numpy as np
import pandas as pd
import nltk
import re
import requests
import random
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import plotly.express as px

# ---------- Page config (wide layout) ----------
st.set_page_config(page_title="Contextual Reader", page_icon="📚", layout="wide")

# ---------- Custom CSS for better UI ----------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9edf2 100%);
    }
    /* Card style for metric boxes */
    div[data-testid="stMetricValue"] {
        background: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #ffffffcc;
        border-radius: 30px;
        padding: 5px;
        backdrop-filter: blur(4px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 30px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a6fa5;
        color: white;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.9);
        border-right: 1px solid #ddd;
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helper functions ----------
def get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError, AttributeError):
        return os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def load_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return stopwords.words('english')

stop_words = load_nltk_data()

# ---------- Vector store class ----------
class SimpleVectorStore:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.chunks = []
        self.embeddings = np.array([])

    def add_documents(self, chunks):
        self.chunks = chunks
        texts = [chunk.page_content for chunk in chunks]
        try:
            self.embeddings = np.array(self.embeddings_model.embed_documents(texts))
        except Exception as e:
            st.error(f"❌ Embedding failed: {e}")
            raise

    def similarity_search(self, query, k=12):
        query_emb = np.array(self.embeddings_model.embed_query(query))
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.chunks[i] for i in top_indices]

# ---------- Session state ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "saved_qa" not in st.session_state:
    st.session_state.saved_qa = []
if "book_stats" not in st.session_state:
    st.session_state.book_stats = None
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "cover_image" not in st.session_state:
    st.session_state.cover_image = None

# ---------- Sidebar with cover image and info ----------
with st.sidebar:
    st.markdown("## 📖 Contextual Reader")
    st.markdown("---")
    
    # Book cover upload or placeholder
    st.subheader("📸 Book Cover")
    uploaded_cover = st.file_uploader("Upload cover image (optional)", type=["jpg", "jpeg", "png"])
    if uploaded_cover is not None:
        st.session_state.cover_image = uploaded_cover
        st.image(uploaded_cover, use_container_width=True)
    else:
        if st.session_state.cover_image is None:
            st.image("https://via.placeholder.com/300x400?text=No+Cover", use_container_width=True)
        else:
            st.image(st.session_state.cover_image, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Engine")
    st.info("🔍 Embeddings: `all-MiniLM-L6-v2` (lightweight)\n\n✅ LLM: Gemini 2.5 Flash")
    st.caption(f"📅 Session started: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🗑️ Clear all saved data", use_container_width=True):
        st.session_state.saved_qa = []
        st.session_state.vectorstore = None
        st.session_state.book_stats = None
        st.session_state.full_text = ""
        st.rerun()

# ---------- Main title and subtitle ----------
st.title("📚 Contextual Reader")
st.markdown("#### *Your personal AI reading companion – ask, analyse, learn*")

# ---------- File upload (central area) ----------
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("📄 Upload a PDF document", type="pdf", label_visibility="collapsed")
with col2:
    st.markdown("")

# ---------- Process PDF only once ----------
if uploaded_file is not None:
    if st.session_state.vectorstore is None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("📖 Processing PDF (this will take a few minutes for long books)..."):
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            full_text = " ".join([doc.page_content for doc in documents])
            st.session_state.full_text = full_text

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            st.info(f"📄 **{len(chunks)} text chunks** created")

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = SimpleVectorStore(embeddings)
            vectorstore.add_documents(chunks)
            st.session_state.vectorstore = vectorstore

            # Statistics
            words = re.findall(r'\b\w+\b', full_text.lower())
            unique_words = set(words)
            sentences = nltk.sent_tokenize(full_text)
            avg_sentence_len = np.mean([len(w.split()) for w in sentences]) if sentences else 0
            reading_time_min = len(words) / 250

            st.session_state.book_stats = {
                "characters": len(full_text),
                "words": len(words),
                "unique_words": len(unique_words),
                "sentences": len(sentences),
                "avg_sentence_len": round(avg_sentence_len, 1),
                "reading_time": round(reading_time_min, 1),
            }

            os.remove("temp.pdf")
            st.success("✅ **PDF processed successfully!** You can now use all features.")
            st.balloons()
    else:
        st.info("📚 **Book already loaded.** Ask questions below or explore other tabs.")

# ---------- Main tabs (only if book loaded) ----------
if st.session_state.vectorstore is not None:
    api_key = get_api_key()
    if not api_key:
        st.error("❌ GOOGLE_API_KEY missing. Add to .env (local) or Streamlit secrets.")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    # Create tabs with icons
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬  Ask", "📊  Dashboard", "📌  Saved Q&A", "📖  Vocabulary", "🧠  Quiz"])

    # ---------- TAB 1: Ask ----------
    with tab1:
        st.header("Ask about your book")
        user_question = st.text_input("Your question:", placeholder="e.g., Describe Mr. Darcy's personality", label_visibility="collapsed")
        if user_question:
            with st.spinner("🤔 Thinking..."):
                try:
                    docs = st.session_state.vectorstore.similarity_search(user_question, k=12)
                    if not docs:
                        st.warning("No relevant passages found. Try rephrasing or using more specific terms.")
                    else:
                        context = "\n\n".join([doc.page_content for doc in docs])
                        prompt = f"""Answer based only on the context below. If the answer is not there, say "I don't have enough information."

Context:
{context}

Question: {user_question}

Answer:"""
                        response = llm.invoke(prompt)
                        st.markdown("### Answer")
                        st.success(response.content)

                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button("💾 Save Q&A"):
                                st.session_state.saved_qa.append({"question": user_question, "answer": response.content})
                                st.toast("Saved!", icon="✅")
                        with col2:
                            with st.expander("📖 Show source passages"):
                                for i, doc in enumerate(docs):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.caption(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))
                except Exception as e:
                    st.error(f"Error: {e}")
                    if "429" in str(e):
                        st.warning("⚠️ Gemini quota exhausted. Try again tomorrow or use a new Google Cloud project.")

    # ---------- TAB 2: Dashboard ----------
    with tab2:
        st.header("Book Dashboard")
        stats = st.session_state.book_stats
        if stats:
            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📄 Characters", f"{stats['characters']:,}")
            m2.metric("📝 Words", f"{stats['words']:,}")
            m3.metric("🔤 Unique Words", f"{stats['unique_words']:,}")
            m4.metric("⏱️ Reading time (min)", stats['reading_time'])
            m5, m6, m7 = st.columns(3)
            m5.metric("📖 Sentences", f"{stats['sentences']:,}")
            m6.metric("📏 Avg sentence length", f"{stats['avg_sentence_len']} words")
            m7.metric("📊 Readability (Flesch)", "N/A (future)")

            st.markdown("---")
            st.subheader("👥 Character mentions")
            default_chars = ["Elizabeth", "Darcy", "Jane", "Bingley", "Wickham", "Collins", "Lydia", "Mrs Bennet"]
            char_input = st.text_input("Character names (comma separated)", value=",".join(default_chars))
            if char_input:
                names = [n.strip() for n in char_input.split(",")]
                counts = {}
                text_lower = st.session_state.full_text.lower()
                for name in names:
                    pattern = r'\b' + re.escape(name.lower()) + r'\b'
                    counts[name] = len(re.findall(pattern, text_lower))
                if counts:
                    fig = px.bar(x=list(counts.keys()), y=list(counts.values()), title="", color_discrete_sequence=["#4a6fa5"])
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)

            # Random quote
            st.markdown("---")
            if st.button("📜 Show random passage"):
                random_chunk = random.choice(st.session_state.vectorstore.chunks).page_content
                st.info(f"*{random_chunk[:800]}...*")
        else:
            st.info("No stats yet – upload a book first.")

    # ---------- TAB 3: Saved Q&A ----------
    with tab3:
        st.header("Saved Q&A")
        if st.session_state.saved_qa:
            df = pd.DataFrame(st.session_state.saved_qa)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download as CSV", csv, "saved_qa.csv", "text/csv")
        else:
            st.info("No saved Q&A yet. Use the 💾 button in the Ask tab.")

    # ---------- TAB 4: Vocabulary ----------
    with tab4:
        st.header("Vocabulary Builder")
        if st.session_state.full_text:
            words = re.findall(r'\b[a-z]{3,}\b', st.session_state.full_text.lower())
            word_freq = Counter(words)
            rare_words = {w: c for w, c in word_freq.items() if w not in stop_words and 1 <= c <= 3}
            rare_words_sorted = sorted(rare_words.items(), key=lambda x: x[1])[:50]

            st.write(f"🔍 Found **{len(rare_words_sorted)}** rare words (frequency ≤ 3). Select a word to see its definition.")
            vocab_df = pd.DataFrame(rare_words_sorted, columns=["Word", "Frequency"])
            st.dataframe(vocab_df, use_container_width=True)

            selected_word = st.selectbox("Choose a word to define", [w for w, _ in rare_words_sorted])
            if selected_word:
                try:
                    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{selected_word}"
                    resp = requests.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        definition = data[0]['meanings'][0]['definitions'][0]['definition']
                        st.success(f"**{selected_word}**: {definition}")
                    else:
                        st.warning("Definition not found.")
                except:
                    st.warning("Could not fetch definition.")

            if st.button("📤 Export vocabulary (CSV for Anki)"):
                anki_df = pd.DataFrame(rare_words_sorted, columns=["Word", "Frequency"])
                anki_csv = anki_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download", anki_csv, "vocabulary.csv", "text/csv")
        else:
            st.info("No text loaded – upload a book first.")

    # ---------- TAB 5: Quiz ----------
    with tab5:
        st.header("AI‑Generated Quiz")
        if st.button("🎲 Generate 5 multiple‑choice questions"):
            with st.spinner("Generating quiz..."):
                try:
                    chunks = st.session_state.vectorstore.chunks
                    sample_chunks = chunks[:20]
                    context = "\n\n".join([c.page_content for c in sample_chunks])
                    quiz_llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0.5,
                        google_api_key=api_key
                    )
                    prompt = f"""Generate 5 multiple‑choice questions based on the following text. Each question must have 4 options (A, B, C, D) and indicate the correct answer. Return in plain text with each question on a new line, formatted like:

1. Question text?
   A) option1
   B) option2
   C) option3
   D) option4
   Answer: A

Text:
{context[:8000]}"""
                    response = quiz_llm.invoke(prompt)
                    st.markdown("### 📝 Quiz")
                    st.success(response.content)
                except Exception as e:
                    st.error(f"Quiz generation failed: {e}")
                    if "429" in str(e):
                        st.warning("Quota exhausted – try again later.")
else:
    st.info("👈 Upload a PDF to start your journey.")