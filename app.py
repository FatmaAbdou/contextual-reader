import streamlit as st
import os
import numpy as np
import pandas as pd
import nltk
import re
import requests
import random
from collections import Counter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
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

# ---------- NLTK setup ----------
@st.cache_resource
def load_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return stopwords.words('english')

stop_words = load_nltk_data()

# ---------- Vector store ----------
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

# ---------- Session state for multiple books ----------
if "books" not in st.session_state:
    st.session_state.books = {}          # name -> {vectorstore, full_text, stats, chunks}
if "current_book" not in st.session_state:
    st.session_state.current_book = None
if "saved_qa" not in st.session_state:
    st.session_state.saved_qa = []

# ---------- Sidebar: book management ----------
with st.sidebar:
    st.title("📚 My Library")
    
    # Upload new book
    uploaded_file = st.file_uploader("Add a new book (PDF)", type="pdf")
    if uploaded_file:
        book_name = st.text_input("Book name (e.g., Pride and Prejudice)", 
                                  value=uploaded_file.name.replace(".pdf", ""))
        if st.button("Process and add to library"):
            if book_name and book_name not in st.session_state.books:
                with st.spinner(f"Processing {book_name}..."):
                    # Save temp PDF
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    loader = PyPDFLoader("temp.pdf")
                    documents = loader.load()
                    full_text = " ".join([doc.page_content for doc in documents])
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=100,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vectorstore = SimpleVectorStore(embeddings)
                    vectorstore.add_documents(chunks)
                    
                    # Stats
                    words = re.findall(r'\b\w+\b', full_text.lower())
                    unique_words = set(words)
                    sentences = nltk.sent_tokenize(full_text)
                    avg_sentence_len = np.mean([len(w.split()) for w in sentences]) if sentences else 0
                    reading_time_min = len(words) / 250
                    stats = {
                        "characters": len(full_text),
                        "words": len(words),
                        "unique_words": len(unique_words),
                        "sentences": len(sentences),
                        "avg_sentence_len": round(avg_sentence_len, 1),
                        "reading_time": round(reading_time_min, 1),
                    }
                    
                    st.session_state.books[book_name] = {
                        "vectorstore": vectorstore,
                        "full_text": full_text,
                        "stats": stats,
                        "chunks": chunks
                    }
                    os.remove("temp.pdf")
                    st.success(f"✅ {book_name} added to library!")
                    st.session_state.current_book = book_name
                    st.rerun()
            else:
                st.error("Book name required or already exists.")
    
    st.markdown("---")
    st.subheader("Your Books")
    if st.session_state.books:
        for bname in st.session_state.books.keys():
            col1, col2 = st.columns([3,1])
            with col1:
                if st.button(f"📖 {bname}", key=f"select_{bname}"):
                    st.session_state.current_book = bname
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{bname}"):
                    del st.session_state.books[bname]
                    if st.session_state.current_book == bname:
                        st.session_state.current_book = None
                    st.rerun()
    else:
        st.info("No books yet. Upload a PDF to start.")
    
    st.markdown("---")
    if st.button("Clear all saved Q&A"):
        st.session_state.saved_qa = []
        st.success("Cleared!")

# ---------- Main area ----------
st.title("Contextual Reader")
st.markdown("Ask questions, explore vocabulary, and analyse your books.")

if st.session_state.current_book and st.session_state.current_book in st.session_state.books:
    book = st.session_state.books[st.session_state.current_book]
    vectorstore = book["vectorstore"]
    full_text = book["full_text"]
    stats = book["stats"]
    chunks = book["chunks"]
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Ask", "📊 Dashboard", "📌 Saved Q&A", "📖 Vocabulary", "🧠 Quiz"])
    
    with tab1:
        st.header(f"Ask about {st.session_state.current_book}")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)
        user_question = st.text_input("Your question:", placeholder="e.g., Who is Jane Bennet?")
        if user_question:
            with st.spinner("Thinking..."):
                try:
                    docs = vectorstore.similarity_search(user_question, k=12)
                    if not docs:
                        st.warning("No relevant passages found.")
                    else:
                        context = "\n\n".join([doc.page_content for doc in docs])
                        prompt = f"Answer based only on the context. If not there, say 'I don't have enough information.'\n\nContext:\n{context}\n\nQuestion: {user_question}\nAnswer:"
                        response = llm.invoke(prompt)
                        st.success(response.content)
                        if st.button("💾 Save this Q&A"):
                            st.session_state.saved_qa.append({"book": st.session_state.current_book, "question": user_question, "answer": response.content})
                            st.toast("Saved!", icon="✅")
                        with st.expander("📖 Source passages"):
                            for i, doc in enumerate(docs):
                                st.markdown(f"**Source {i+1}:**")
                                st.caption(doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""))
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        st.header("Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Characters", f"{stats['characters']:,}")
        c2.metric("Words", f"{stats['words']:,}")
        c3.metric("Unique Words", f"{stats['unique_words']:,}")
        c4.metric("Reading Time (min)", stats['reading_time'])
        c5, c6 = st.columns(2)
        c5.metric("Sentences", f"{stats['sentences']:,}")
        c6.metric("Avg Sentence Length", f"{stats['avg_sentence_len']} words")
        
        st.subheader("Character Mentions")
        default_chars = ["Elizabeth", "Darcy", "Jane", "Bingley", "Wickham", "Collins"]
        char_input = st.text_input("Character names (comma separated)", value=",".join(default_chars))
        if char_input:
            names = [n.strip() for n in char_input.split(",")]
            counts = {}
            text_lower = full_text.lower()
            for name in names:
                pattern = r'\b' + re.escape(name.lower()) + r'\b'
                counts[name] = len(re.findall(pattern, text_lower))
            if counts:
                fig = px.bar(x=list(counts.keys()), y=list(counts.values()), color_discrete_sequence=["#4a6fa5"])
                st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📜 Show random passage"):
            random_chunk = random.choice(chunks).page_content
            st.info(random_chunk[:800] + "...")
    
    with tab3:
        st.header("Saved Q&A")
        if st.session_state.saved_qa:
            df = pd.DataFrame(st.session_state.saved_qa)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "saved_qa.csv", "text/csv")
        else:
            st.info("No saved Q&A yet.")
    
    with tab4:
        st.header("Vocabulary Builder")
        words = re.findall(r'\b[a-z]{3,}\b', full_text.lower())
        word_freq = Counter(words)
        rare_words = {w: c for w, c in word_freq.items() if w not in stop_words and 1 <= c <= 3}
        rare_sorted = sorted(rare_words.items(), key=lambda x: x[1])[:50]
        st.write(f"Found {len(rare_sorted)} rare words.")
        vocab_df = pd.DataFrame(rare_sorted, columns=["Word", "Frequency"])
        st.dataframe(vocab_df, use_container_width=True)
        selected = st.selectbox("Get definition", [w for w, _ in rare_sorted])
        if selected:
            try:
                url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{selected}"
                resp = requests.get(url)
                if resp.status_code == 200:
                    definition = resp.json()[0]['meanings'][0]['definitions'][0]['definition']
                    st.success(f"**{selected}**: {definition}")
                else:
                    st.warning("Definition not found.")
            except:
                st.warning("Could not fetch definition.")
        if st.button("Export vocabulary (CSV)"):
            anki_csv = vocab_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download", anki_csv, "vocabulary.csv", "text/csv")
    
    with tab5:
        st.header("Quiz Generator")
        if st.button("Generate 5 questions"):
            with st.spinner("Generating..."):
                try:
                    sample_chunks = chunks[:20]
                    context = "\n\n".join([c.page_content for c in sample_chunks])
                    quiz_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=api_key)
                    prompt = f"Generate 5 multiple-choice questions from this text. Format: 1. Question? A) ... B) ... C) ... D) ... Answer: X\n\nText:\n{context[:8000]}"
                    response = quiz_llm.invoke(prompt)
                    st.markdown("### Quiz")
                    st.success(response.content)
                except Exception as e:
                    st.error(f"Failed: {e}")
else:
    st.info("👈 Select a book from the sidebar or upload a new one.")