import streamlit as st
import os
import numpy as np
import pandas as pd
import nltk
import re
import requests
import json
import base64
from io import BytesIO
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import plotly.express as px
from PIL import Image

# ---------- Page config ----------
st.set_page_config(page_title="Contextual Reader", page_icon="📚", layout="wide")

# ---------- Helper functions ----------
def get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except:
        return os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def load_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    return stopwords.words('english')

stop_words = load_nltk_data()

# ---------- Library persistence ----------
LIBRARY_FILE = "book_library.json"

def load_library():
    if os.path.exists(LIBRARY_FILE):
        with open(LIBRARY_FILE, "r") as f:
            return json.load(f)
    return []

def save_library(library):
    with open(LIBRARY_FILE, "w") as f:
        json.dump(library, f, indent=2)

def add_book_to_library(filename, word_count, stats, cover_b64=None, status="unread", progress=0):
    library = load_library()
    for book in library:
        if book["filename"] == filename:
            return
    library.append({
        "filename": filename,
        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "word_count": word_count,
        "stats": stats,
        "cover_b64": cover_b64,
        "status": status,
        "progress": progress
    })
    save_library(library)

def update_book_status(filename, status=None, progress=None):
    library = load_library()
    for book in library:
        if book["filename"] == filename:
            if status is not None:
                book["status"] = status
            if progress is not None:
                book["progress"] = progress
    save_library(library)

def delete_book_from_library(filename):
    library = load_library()
    library = [b for b in library if b["filename"] != filename]
    save_library(library)

# ---------- Vector store ----------
class SimpleVectorStore:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.chunks = []
        self.embeddings = np.array([])

    def add_documents(self, chunks):
        self.chunks = chunks
        texts = [c.page_content for c in chunks]
        self.embeddings = np.array(self.embeddings_model.embed_documents(texts))

    def similarity_search(self, query, k=12):
        q_emb = np.array(self.embeddings_model.embed_query(query))
        sim = np.dot(self.embeddings, q_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb))
        idx = np.argsort(sim)[-k:][::-1]
        return [self.chunks[i] for i in idx]

# ---------- Session state ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "saved_qa" not in st.session_state:
    st.session_state.saved_qa = []
if "book_stats" not in st.session_state:
    st.session_state.book_stats = None
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "chunks_metadata" not in st.session_state:
    st.session_state.chunks_metadata = []  # list of (chunk_text, page_number)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📖 Contextual Reader")
    if st.button("🗑️ Clear current book"):
        st.session_state.vectorstore = None
        st.session_state.book_stats = None
        st.session_state.full_text = ""
        st.session_state.current_filename = None
        st.session_state.chunks_metadata = []
        st.rerun()
    if st.button("📤 Export library (JSON)"):
        lib = load_library()
        if lib:
            json_str = json.dumps(lib, indent=2)
            st.download_button("Download library", json_str, "book_library.json", "application/json")
        else:
            st.info("No library to export.")
    uploaded_lib = st.file_uploader("📥 Import library (JSON)", type="json")
    if uploaded_lib:
        try:
            new_lib = json.load(uploaded_lib)
            save_library(new_lib)
            st.success("Library imported! Refresh to see changes.")
            st.rerun()
        except:
            st.error("Invalid JSON file.")

# ---------- Main area ----------
st.title("📚 Contextual Reader")
st.markdown("*Your personal AI reading companion*")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    if st.session_state.vectorstore is None or st.session_state.current_filename != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            full_text = " ".join([d.page_content for d in documents])
            st.session_state.full_text = full_text

            # Collect chunks with page numbers (optional)
            chunks_meta = []
            for doc in documents:
                # split each page into chunks (rough)
                page_text = doc.page_content
                splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(page_text)
                for s in splits:
                    chunks_meta.append((s, doc.metadata.get("page", 0)+1))
            st.session_state.chunks_metadata = chunks_meta

            # Use only text for vector store
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            st.info(f"📄 Split into {len(chunks)} chunks")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = SimpleVectorStore(embeddings)
            vectorstore.add_documents(chunks)
            st.session_state.vectorstore = vectorstore
            st.session_state.current_filename = uploaded_file.name

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
            st.session_state.book_stats = stats

            # Cover image (optional)
            cover_b64 = None
            cover_img = st.sidebar.file_uploader("Upload cover for this book", type=["jpg","png"], key="cover_upload")
            if cover_img:
                img = Image.open(cover_img)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                cover_b64 = base64.b64encode(buffered.getvalue()).decode()

            add_book_to_library(uploaded_file.name, len(words), stats, cover_b64)
            os.remove("temp.pdf")
            st.success(f"✅ Processed: {uploaded_file.name}")
            st.balloons()

# ---------- Tabs ----------
if st.session_state.vectorstore is not None:
    api_key = get_api_key()
    if not api_key:
        st.error("❌ GOOGLE_API_KEY missing.")
        st.stop()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)

    tabs = st.tabs(["💬 Ask", "📊 Dashboard", "📌 Saved Q&A", "📖 Vocabulary", "🧠 Quiz", "📚 Reading Tracker", "✍️ Summarize"])

    # ---------- TAB 1: Ask ----------
    with tabs[0]:
        st.header("Ask about this book")
        question = st.text_input("Your question:", placeholder="e.g., Describe Mr. Darcy")
        if question:
            with st.spinner("Thinking..."):
                try:
                    docs = st.session_state.vectorstore.similarity_search(question, k=12)
                    if not docs:
                        st.warning("No relevant passages found.")
                    else:
                        context = "\n\n".join([d.page_content for d in docs])
                        prompt = f"Answer based on context:\n{context}\n\nQuestion: {question}\nAnswer:"
                        response = llm.invoke(prompt)
                        st.success(response.content)
                        if st.button("💾 Save Q&A"):
                            st.session_state.saved_qa.append({"question": question, "answer": response.content})
                            st.toast("Saved!", icon="✅")
                        with st.expander("📖 Source passages"):
                            for i, d in enumerate(docs):
                                st.caption(f"Source {i+1}: {d.page_content[:500]}...")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ---------- TAB 2: Dashboard ----------
    with tabs[1]:
        st.header("Current Book Dashboard")
        if st.session_state.book_stats:
            s = st.session_state.book_stats
            cols = st.columns(4)
            cols[0].metric("Characters", f"{s['characters']:,}")
            cols[1].metric("Words", f"{s['words']:,}")
            cols[2].metric("Unique words", f"{s['unique_words']:,}")
            cols[3].metric("Reading time (min)", s['reading_time'])
            st.subheader("Character mentions")
            default = ["Elizabeth", "Darcy", "Jane", "Bingley", "Wickham", "Collins", "Lydia"]
            chars = st.text_input("Character names (comma separated)", value=",".join(default))
            if chars:
                names = [n.strip() for n in chars.split(",")]
                counts = {}
                text_lower = st.session_state.full_text.lower()
                for n in names:
                    pattern = r'\b' + re.escape(n.lower()) + r'\b'
                    counts[n] = len(re.findall(pattern, text_lower))
                fig = px.bar(x=list(counts.keys()), y=list(counts.values()), color_discrete_sequence=["#4a6fa5"])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No book loaded.")

    # ---------- TAB 3: Saved Q&A ----------
    with tabs[2]:
        st.header("Saved Q&A")
        if st.session_state.saved_qa:
            df = pd.DataFrame(st.session_state.saved_qa)
            st.dataframe(df)
            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, "saved_qa.csv")
        else:
            st.info("No saved Q&A yet.")

    # ---------- TAB 4: Vocabulary ----------
    with tabs[3]:
        st.header("Vocabulary Builder")
        if st.session_state.full_text:
            words = re.findall(r'\b[a-z]{3,}\b', st.session_state.full_text.lower())
            freq = Counter(words)
            rare = {w: c for w, c in freq.items() if w not in stop_words and 1 <= c <= 3}
            rare_sorted = sorted(rare.items(), key=lambda x: x[1])[:50]
            st.dataframe(pd.DataFrame(rare_sorted, columns=["Word", "Frequency"]), use_container_width=True)
            word = st.selectbox("Get definition", [w for w, _ in rare_sorted])
            if word:
                try:
                    r = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
                    if r.status_code == 200:
                        defn = r.json()[0]['meanings'][0]['definitions'][0]['definition']
                        st.success(f"**{word}**: {defn}")
                except:
                    st.warning("Definition not found.")
        else:
            st.info("No text loaded.")

    # ---------- TAB 5: Interactive Quiz ----------
    with tabs[4]:
        st.header("AI‑Generated Quiz")
        if "quiz_questions" not in st.session_state:
            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = {}
        if st.button("Generate new quiz (5 questions)"):
            with st.spinner("Generating quiz..."):
                try:
                    sample = [c.page_content for c in st.session_state.vectorstore.chunks[:20]]
                    context = "\n\n".join(sample)
                    prompt = f"""Generate 5 multiple-choice questions based on the following text. Return a JSON list with each question containing: "question", "options" (list of 4 strings), "correct" (0-based index of correct option). Example: [{{"question": "What is X?", "options": ["A", "B", "C", "D"], "correct": 1}}]

Text:
{context[:8000]}"""
                    response = llm.invoke(prompt)
                    # Parse JSON from response
                    text = response.content
                    # Find JSON part
                    import json
                    start = text.find('[')
                    end = text.rfind(']')+1
                    if start != -1 and end != -1:
                        json_str = text[start:end]
                        st.session_state.quiz_questions = json.loads(json_str)
                        st.session_state.quiz_answers = {}
                        st.success("Quiz generated!")
                    else:
                        st.error("Could not parse quiz. Try again.")
                except Exception as e:
                    st.error(f"Error: {e}")
        if st.session_state.quiz_questions:
            for i, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                options = q['options']
                selected = st.radio(f"Select answer for Q{i+1}", options, key=f"quiz_{i}", index=None)
                st.session_state.quiz_answers[i] = selected
            if st.button("Submit answers"):
                correct_count = 0
                for i, q in enumerate(st.session_state.quiz_questions):
                    user_ans = st.session_state.quiz_answers.get(i)
                    correct_opt = q['options'][q['correct']]
                    if user_ans == correct_opt:
                        correct_count += 1
                        st.success(f"Q{i+1}: Correct!")
                    else:
                        st.error(f"Q{i+1}: Incorrect. Correct answer: {correct_opt}")
                st.info(f"Score: {correct_count}/{len(st.session_state.quiz_questions)}")
        else:
            st.info("Click 'Generate new quiz' to start.")

    # ---------- TAB 6: Reading Tracker ----------
    with tabs[5]:
        st.header("Reading Tracker")
        library = load_library()
        if not library:
            st.info("No books in library. Upload a PDF to add it.")
        else:
            # Display each book with status and progress
            for book in library:
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                    with col1:
                        if book.get("cover_b64"):
                            st.image(base64.b64decode(book["cover_b64"]), width=80)
                        else:
                            st.image("https://via.placeholder.com/80x120?text=No+Cover", width=80)
                    with col2:
                        st.write(f"**{book['filename']}**")
                        st.caption(f"Uploaded: {book['upload_date']}")
                        st.caption(f"Words: {book['word_count']:,}")
                    with col3:
                        status = st.selectbox("Status", ["unread", "reading", "finished"], key=f"status_{book['filename']}", index=["unread","reading","finished"].index(book.get("status","unread")))
                        progress = st.number_input("Progress (pages)", min_value=0, max_value=1000, value=book.get("progress",0), key=f"progress_{book['filename']}")
                        if st.button("Update", key=f"update_{book['filename']}"):
                            update_book_status(book['filename'], status, progress)
                            st.rerun()
                    with col4:
                        if st.button("Delete", key=f"del_{book['filename']}"):
                            delete_book_from_library(book['filename'])
                            st.rerun()
                st.divider()

    # ---------- TAB 7: Summarize ----------
    with tabs[6]:
        st.header("Summarize Chapters or Paragraphs")
        st.write("You can summarize a range of pages or a specific chapter (if page numbers are known).")
        # Use the stored chunks with page numbers (approximate)
        if st.session_state.chunks_metadata:
            # Build page ranges from chunks
            pages = sorted(set([page for _, page in st.session_state.chunks_metadata]))
            min_page = min(pages) if pages else 1
            max_page = max(pages) if pages else 1
            page_range = st.slider("Select page range", min_value=min_page, max_value=max_page, value=(min_page, min(min_page+10, max_page)))
            if st.button("Summarize selected pages"):
                selected_text = []
                for text, page in st.session_state.chunks_metadata:
                    if page_range[0] <= page <= page_range[1]:
                        selected_text.append(text)
                if selected_text:
                    context = "\n\n".join(selected_text)[:8000]
                    with st.spinner("Generating summary..."):
                        try:
                            prompt = f"Summarize the following text concisely:\n\n{context}"
                            response = llm.invoke(prompt)
                            st.success(response.content)
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("No text in that page range.")
        else:
            st.info("Page information not available. Use manual text input below.")
            manual_text = st.text_area("Or paste text to summarize")
            if st.button("Summarize custom text") and manual_text:
                with st.spinner("Summarizing..."):
                    try:
                        prompt = f"Summarize the following text concisely:\n\n{manual_text[:8000]}"
                        response = llm.invoke(prompt)
                        st.success(response.content)
                    except Exception as e:
                        st.error(f"Error: {e}")

else:
    st.info("👈 Upload a PDF to start.")