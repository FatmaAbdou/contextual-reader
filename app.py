import streamlit as st
import os
import numpy as np
import pandas as pd
import nltk
import re
import json
import requests
from collections import Counter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import matplotlib.pyplot as plt
import plotly.express as px

# ---------- Load API key ----------
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

# ---------- NLTK setup (cached) ----------
@st.cache_resource
def load_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    from nltk.corpus import stopwords
    return stopwords.words('english')

stop_words = load_nltk_data()

# ---------- Simple vector store with numpy ----------
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

    def similarity_search(self, query, k=12):   # Increased k to 12
        query_emb = np.array(self.embeddings_model.embed_query(query))
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.chunks[i] for i in top_indices]

# ---------- Sidebar ----------
st.set_page_config(page_title="Contextual Reader", page_icon="📚", layout="wide")
st.title("📚 Contextual Reader")
st.markdown("Upload any PDF – ask questions, analyse style, build vocabulary, and more.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.info("🔍 Using HuggingFace `all-mpnet-base-v2` embeddings (more accurate, slightly slower).")
    st.success("✅ Google Gemini 2.5 Flash for answers")
    st.caption("💡 Quota limit: 20 requests/day on free tier. Use a new API key if exhausted.")
    if st.button("Clear saved Q&A"):
        st.session_state.saved_qa = []
        st.rerun()

# ---------- Session state ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "saved_qa" not in st.session_state:
    st.session_state.saved_qa = []
if "book_stats" not in st.session_state:
    st.session_state.book_stats = None
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

# ---------- File upload & processing ----------
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing PDF... (may take a few minutes for a long book)"):
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        full_text = " ".join([doc.page_content for doc in documents])
        st.session_state.full_text = full_text

        # Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        st.info(f"📄 Split into {len(chunks)} text chunks")

        # Better embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        vectorstore = SimpleVectorStore(embeddings)
        vectorstore.add_documents(chunks)
        st.session_state.vectorstore = vectorstore

        # Compute book statistics
        words = re.findall(r'\b\w+\b', full_text.lower())
        unique_words = set(words)
        sentences = nltk.sent_tokenize(full_text)
        avg_sentence_len = np.mean([len(w.split()) for w in sentences]) if sentences else 0
        reading_time_min = len(words) / 250  # 250 wpm

        st.session_state.book_stats = {
            "characters": len(full_text),
            "words": len(words),
            "unique_words": len(unique_words),
            "sentences": len(sentences),
            "avg_sentence_len": round(avg_sentence_len, 1),
            "reading_time": round(reading_time_min, 1),
        }

        os.remove("temp.pdf")
        st.success("✅ PDF processed! You can now explore the tabs below.")

# ---------- Tabs ----------
if st.session_state.vectorstore is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Ask", "📊 Dashboard", "📌 Saved Q&A", "📖 Vocabulary", "🧠 Quiz"])

    # ---------- TAB 1: Ask questions ----------
    with tab1:
        st.header("Ask about your document")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        user_question = st.text_input("Your question:", placeholder="e.g., Who is Jane Bennet?")
        if user_question:
            with st.spinner("Thinking..."):
                try:
                    docs = st.session_state.vectorstore.similarity_search(user_question, k=12)
                    if not docs:
                        st.warning("No relevant passages found. Try rephrasing.")
                    else:
                        context = "\n\n".join([doc.page_content for doc in docs])
                        prompt = f"""Answer based only on the context below. If the answer is not there, say "I don't have enough information."

Context:
{context}

Question: {user_question}

Answer:"""
                        response = llm.invoke(prompt)
                        st.markdown("### Answer")
                        st.write(response.content)

                        # Save Q&A if user wants
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button("💾 Save this Q&A"):
                                st.session_state.saved_qa.append({"question": user_question, "answer": response.content})
                                st.success("Saved!")
                        with col2:
                            with st.expander("📖 Source passages"):
                                for i, doc in enumerate(docs):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                except Exception as e:
                    st.error(f"Chat error: {e}")
                    if "429" in str(e):
                        st.info("⚠️ You exceeded your daily Gemini quota. Wait until tomorrow or use a new API key.")

    # ---------- TAB 2: Dashboard ----------
    with tab2:
        st.header("Book Dashboard")
        stats = st.session_state.book_stats
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📄 Characters", f"{stats['characters']:,}")
            col2.metric("📝 Words", f"{stats['words']:,}")
            col3.metric("🔤 Unique Words", f"{stats['unique_words']:,}")
            col4.metric("⏱️ Reading time (min)", stats['reading_time'])
            col5, col6, col7 = st.columns(3)
            col5.metric("📖 Sentences", f"{stats['sentences']:,}")
            col6.metric("📏 Avg sentence length", f"{stats['avg_sentence_len']} words")
            col7.metric("📊 Readability (Flesch)", "N/A (to implement)")

            # Character mention bar chart
            st.subheader("Character mentions (enter names below)")
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
                    fig = px.bar(x=list(counts.keys()), y=list(counts.values()), title="Character frequency")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No stats yet – upload a PDF first.")

    # ---------- TAB 3: Saved Q&A ----------
    with tab3:
        st.header("Saved Questions & Answers")
        if st.session_state.saved_qa:
            df = pd.DataFrame(st.session_state.saved_qa)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download as CSV", csv, "saved_qa.csv", "text/csv")
        else:
            st.info("No saved Q&A yet. Use the 'Save' button in the Ask tab.")

    # ---------- TAB 4: Vocabulary Builder ----------
    with tab4:
        st.header("Vocabulary Builder")
        if st.session_state.full_text:
            words = re.findall(r'\b[a-z]{3,}\b', st.session_state.full_text.lower())
            word_freq = Counter(words)
            # Remove stopwords and keep words with frequency 1-3 (rare)
            rare_words = {w: c for w, c in word_freq.items() if w not in stop_words and 1 <= c <= 3}
            rare_words_sorted = sorted(rare_words.items(), key=lambda x: x[1])[:50]

            st.write(f"Found **{len(rare_words_sorted)}** rare words (frequency ≤ 3). Click a word to see definition.")

            vocab_df = pd.DataFrame(rare_words_sorted, columns=["Word", "Frequency"])
            st.dataframe(vocab_df, use_container_width=True)

            # Lookup definition using FreeDictionary API
            selected_word = st.selectbox("Select a word to define", [w for w, _ in rare_words_sorted])
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

            # Export for Anki
            if st.button("📤 Export vocabulary to Anki (CSV)"):
                anki_df = pd.DataFrame(rare_words_sorted, columns=["Word", "Frequency"])
                anki_csv = anki_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", anki_csv, "vocabulary.csv", "text/csv")
        else:
            st.info("No text loaded – upload a PDF first.")

    # ---------- TAB 5: Quiz Generator ----------
    with tab5:
        st.header("AI‑Generated Quiz")
        if st.button("Generate 5 multiple‑choice questions"):
            with st.spinner("Creating quiz..."):
                # Retrieve a random sample of chunks (e.g., first 20 chunks)
                chunks = st.session_state.vectorstore.chunks
                sample_chunks = chunks[:20]  # simple – could randomize
                context = "\n\n".join([c.page_content for c in sample_chunks])
                llm = ChatGoogleGenerativeAI(
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
{context[:8000]}  # limit to avoid token overflow
"""
                try:
                    response = llm.invoke(prompt)
                    st.markdown("### Quiz")
                    st.write(response.content)
                except Exception as e:
                    st.error(f"Quiz generation failed: {e}")
else:
    st.info("👈 Upload a PDF to begin.")