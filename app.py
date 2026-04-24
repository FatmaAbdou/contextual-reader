import streamlit as st
import os
import numpy as np
import pandas as pd
import nltk
import re
import requests
import json
import base64
import hashlib
import random
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

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

def add_book_to_library(filename, word_count, stats, cover_b64=None, cover_url=None, status="unread", progress=0):
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
        "cover_url": cover_url,
        "status": status,
        "progress": progress
    })
    save_library(library)

def update_book_status(filename, status=None, progress=None):
    library = load_library()
    found = False
    for book in library:
        if book["filename"] == filename:
            if status is not None:
                book["status"] = status
            if progress is not None:
                book["progress"] = progress
            found = True
            break
    if found:
        save_library(library)
    return found

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
if "saved_summaries" not in st.session_state:
    st.session_state.saved_summaries = []
if "saved_quizzes" not in st.session_state:
    st.session_state.saved_quizzes = []
if "book_stats" not in st.session_state:
    st.session_state.book_stats = None
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "chunks_metadata" not in st.session_state:
    st.session_state.chunks_metadata = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_context_docs" not in st.session_state:
    st.session_state.last_context_docs = []
if "rare_words" not in st.session_state:
    st.session_state.rare_words = []
if "word_definitions" not in st.session_state:
    st.session_state.word_definitions = {}
if "custom_word_lookup" not in st.session_state:
    st.session_state.custom_word_lookup = ""
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = {}
if "cover_url_temp" not in st.session_state:
    st.session_state.cover_url_temp = ""
if "imported" not in st.session_state:
    st.session_state.imported = False

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📖 Contextual Reader")
    st.markdown("---")
    
    st.subheader("📸 Book Cover (for current upload)")
    cover_img = st.file_uploader("Upload cover image", type=["jpg","jpeg","png"], key="cover_upload")
    cover_url = st.text_input("Or paste cover image URL", placeholder="https://...", key="cover_url_input", value="")
    
    if cover_img:
        st.image(cover_img, width=150, caption="Cover preview")
        st.session_state.cover_img_temp = cover_img
        st.session_state.cover_url_temp = None
    elif cover_url:
        st.image(cover_url, width=150, caption="Cover preview")
        st.session_state.cover_url_temp = cover_url
        st.session_state.cover_img_temp = None
    else:
        st.caption("No cover selected")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear current book"):
        st.session_state.vectorstore = None
        st.session_state.book_stats = None
        st.session_state.full_text = ""
        st.session_state.current_filename = None
        st.session_state.chunks_metadata = []
        st.session_state.last_question = ""
        st.session_state.last_answer = ""
        st.session_state.last_context_docs = []
        st.session_state.rare_words = []
        st.session_state.word_definitions = {}
        st.session_state.quiz_questions = []
        st.rerun()
    
    if st.button("📤 Export library (JSON)"):
        lib = load_library()
        if lib:
            json_str = json.dumps(lib, indent=2)
            st.download_button("Download library", json_str, "book_library.json", "application/json")
        else:
            st.info("No library to export.")
    
    if st.button("🗑️ Clear entire library"):
        save_library([])
        st.success("All books removed from library.")
        st.rerun()
    
    uploaded_lib = st.file_uploader("📥 Import library (JSON)", type="json", key="lib_uploader")
    if uploaded_lib and not st.session_state.imported:
        try:
            new_lib = json.load(uploaded_lib)
            if isinstance(new_lib, list) and all(isinstance(b, dict) and "filename" in b for b in new_lib):
                save_library(new_lib)
                st.session_state.imported = True
                st.success("Library imported! Refreshing...")
                st.rerun()
            else:
                st.error("Invalid JSON structure. Expected a list of book objects.")
        except Exception as e:
            st.error(f"Invalid JSON file: {e}")
    if not uploaded_lib and st.session_state.imported:
        st.session_state.imported = False

# ---------- Main area ----------
st.title("📚 Contextual Reader")
st.markdown("*Your personal AI reading companion*")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    if st.session_state.vectorstore is None or st.session_state.current_filename != uploaded_file.name:
        with st.spinner("Processing PDF... (this runs only once per book)"):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            full_text = " ".join([d.page_content for d in documents])
            st.session_state.full_text = full_text

            chunks_meta = []
            for doc in documents:
                splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(doc.page_content)
                for s in splits:
                    chunks_meta.append((s, doc.metadata.get("page", 0)+1))
            st.session_state.chunks_metadata = chunks_meta

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            st.info(f"📄 Split into {len(chunks)} chunks")
            embeddings = get_embeddings()
            vectorstore = SimpleVectorStore(embeddings)
            vectorstore.add_documents(chunks)
            st.session_state.vectorstore = vectorstore
            st.session_state.current_filename = uploaded_file.name

            words = re.findall(r'\b[a-zA-Z]+\b', full_text.lower())
            words = [w for w in words if w.isalpha()]
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

            freq = Counter(words)
            rare = {w: c for w, c in freq.items() if w.isalpha() and w not in stop_words and 1 <= c <= 3}
            st.session_state.rare_words = sorted(rare.items(), key=lambda x: x[1])[:50]

            cover_b64 = None
            cover_url_saved = None
            if cover_img:
                img = Image.open(cover_img)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                cover_b64 = base64.b64encode(buffered.getvalue()).decode()
            elif st.session_state.cover_url_temp:
                cover_url_saved = st.session_state.cover_url_temp

            add_book_to_library(uploaded_file.name, len(words), stats, cover_b64, cover_url_saved)
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

    tabs = st.tabs(["💬 Ask", "📊 Dashboard", "📌 Saved Q&A", "📖 Vocabulary", "🧠 Quiz", "📚 Library", "✍️ Summarize", "💾 Saved Summaries & Quizzes"])

    # ---------- TAB 1: Ask ----------
    with tabs[0]:
        st.header("Ask about this book")
        question = st.text_input("Your question:", placeholder="e.g., Describe Mr. Darcy", key="ask_input")
        ask_button = st.button("Ask", key="ask_submit")
        
        if ask_button and question:
            with st.spinner("Thinking... (calls Gemini API)"):
                try:
                    docs = st.session_state.vectorstore.similarity_search(question, k=12)
                    if not docs:
                        st.warning("No relevant passages found.")
                        st.session_state.last_answer = ""
                        st.session_state.last_context_docs = []
                    else:
                        context = "\n\n".join([d.page_content for d in docs])
                        prompt = f"Answer based only on the context below. If the answer is not there, say 'I don't have enough information.'\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
                        response = llm.invoke(prompt)
                        st.session_state.last_answer = response.content
                        st.session_state.last_context_docs = docs
                        st.session_state.last_question = question
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.last_answer:
            st.success(st.session_state.last_answer)
            if st.button("💾 Save this Q&A"):
                st.session_state.saved_qa.append({
                    "question": st.session_state.last_question,
                    "answer": st.session_state.last_answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.toast("Saved!", icon="✅")
            with st.expander("📖 Source passages"):
                for i, d in enumerate(st.session_state.last_context_docs):
                    st.caption(f"Source {i+1}: {d.page_content[:500]}...")

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
            st.info("No book loaded. Upload a PDF to see dashboard.")

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
        st.subheader("Rare words from this book (frequency 1-3)")
        if st.session_state.rare_words:
            rare_df = pd.DataFrame(st.session_state.rare_words, columns=["Word", "Frequency"])
            st.dataframe(rare_df, use_container_width=True)
            selected_word = st.selectbox("Or select a rare word to see its definition", [w for w, _ in st.session_state.rare_words])
            if selected_word:
                if selected_word not in st.session_state.word_definitions:
                    with st.spinner("Fetching definition..."):
                        try:
                            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{selected_word}"
                            resp = requests.get(url)
                            if resp.status_code == 200:
                                data = resp.json()
                                definition = data[0]['meanings'][0]['definitions'][0]['definition']
                                st.session_state.word_definitions[selected_word] = definition
                            else:
                                st.session_state.word_definitions[selected_word] = "Definition not found in free dictionary."
                        except:
                            st.session_state.word_definitions[selected_word] = "Could not fetch definition."
                st.success(f"**{selected_word}**: {st.session_state.word_definitions[selected_word]}")
        else:
            st.info("No rare words found in this book.")

        st.markdown("---")
        st.subheader("Look up any word")
        custom_word = st.text_input("Enter any word", value=st.session_state.custom_word_lookup, key="custom_word_input")
        if st.button("Get definition"):
            if custom_word:
                st.session_state.custom_word_lookup = custom_word
                if custom_word not in st.session_state.word_definitions:
                    with st.spinner("Fetching definition..."):
                        try:
                            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{custom_word}"
                            resp = requests.get(url)
                            if resp.status_code == 200:
                                data = resp.json()
                                definition = data[0]['meanings'][0]['definitions'][0]['definition']
                                st.session_state.word_definitions[custom_word] = definition
                            else:
                                st.session_state.word_definitions[custom_word] = "Definition not found in free dictionary."
                        except:
                            st.session_state.word_definitions[custom_word] = "Could not fetch definition."
                st.success(f"**{custom_word}**: {st.session_state.word_definitions[custom_word]}")
            else:
                st.warning("Please enter a word.")

        st.markdown("---")
        st.subheader("Export vocabulary with definitions")
        if st.button("📤 Export rare words + definitions (CSV)"):
            export_data = []
            for word, freq in st.session_state.rare_words:
                if word not in st.session_state.word_definitions:
                    try:
                        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
                        resp = requests.get(url)
                        if resp.status_code == 200:
                            data = resp.json()
                            definition = data[0]['meanings'][0]['definitions'][0]['definition']
                            st.session_state.word_definitions[word] = definition
                        else:
                            st.session_state.word_definitions[word] = "Definition not found."
                    except:
                        st.session_state.word_definitions[word] = "Error fetching definition."
                export_data.append({
                    "Word": word,
                    "Frequency": freq,
                    "Definition": st.session_state.word_definitions.get(word, "Not available")
                })
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "vocabulary_with_definitions.csv", "text/csv")
            st.success("Export ready!")

    # ---------- TAB 5: Quiz ----------
    with tabs[4]:
        st.header("AI‑Generated Quiz")
        col1, col2 = st.columns([2,1])
        with col1:
            if st.button("🎲 Generate new quiz (5 questions)"):
                with st.spinner("Generating quiz..."):
                    try:
                        total_chunks = len(st.session_state.vectorstore.chunks)
                        sample_size = min(20, total_chunks)
                        random_indices = random.sample(range(total_chunks), sample_size)
                        sample_chunks = [st.session_state.vectorstore.chunks[i].page_content for i in random_indices]
                        context = "\n\n".join(sample_chunks)
                        prompt = f"""Generate 5 multiple-choice questions based on the following text. Return a JSON list with each question containing: "question", "options" (list of 4 strings), "correct" (0-based index of correct option). Example: [{{"question": "What is X?", "options": ["A", "B", "C", "D"], "correct": 1}}]

Text:
{context[:8000]}"""
                        response = llm.invoke(prompt)
                        text = response.content
                        start = text.find('[')
                        end = text.rfind(']')+1
                        if start != -1 and end != -1:
                            json_str = text[start:end]
                            st.session_state.quiz_questions = json.loads(json_str)
                            st.success("Quiz generated!")
                        else:
                            st.error("Could not parse quiz. Try again.")
                    except Exception as e:
                        st.error(f"Error: {e}")
        with col2:
            if st.session_state.quiz_questions:
                if st.button("💾 Save this quiz"):
                    st.session_state.saved_quizzes.append({
                        "questions": st.session_state.quiz_questions,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "book": st.session_state.current_filename
                    })
                    st.toast("Quiz saved!", icon="✅")

        if st.session_state.quiz_questions:
            with st.form(key="quiz_form"):
                user_answers = []
                for i, q in enumerate(st.session_state.quiz_questions):
                    st.markdown(f"**Q{i+1}: {q['question']}**")
                    options = q['options']
                    selected = st.radio(f"Select answer for Q{i+1}", options, key=f"quiz_radio_{i}", index=None)
                    user_answers.append(selected)
                submitted = st.form_submit_button("Submit answers")
                if submitted:
                    correct_count = 0
                    for i, (q, user_ans) in enumerate(zip(st.session_state.quiz_questions, user_answers)):
                        correct_opt = q['options'][q['correct']]
                        if user_ans == correct_opt:
                            correct_count += 1
                            st.success(f"Q{i+1}: Correct!")
                        else:
                            st.error(f"Q{i+1}: Incorrect. Correct answer: {correct_opt}")
                    st.info(f"Score: {correct_count}/{len(st.session_state.quiz_questions)}")
        else:
            st.info("Click 'Generate new quiz' to start.")

    # ---------- TAB 6: Library ----------
    with tabs[5]:
        st.header("📚 Your Library")
        library = load_library()
        if not library:
            st.info("No books in library. Upload a PDF to add it.")
        else:
            total_books = len(library)
            total_words = sum(b.get("word_count", 0) for b in library)
            finished_books = sum(1 for b in library if b.get("status") == "finished")
            reading_books = sum(1 for b in library if b.get("status") == "reading")
            completion_rate = (finished_books / total_books * 100) if total_books else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📚 Total books", total_books)
            col2.metric("📝 Total words", f"{total_words:,}")
            col3.metric("✅ Finished", finished_books)
            col4.metric("📖 Reading now", reading_books)
            st.progress(completion_rate / 100, text=f"Completion rate: {completion_rate:.1f}%")
            
            st.markdown("---")
            
            sort_by = st.selectbox("Sort by", ["Upload date (newest first)", "Word count (descending)", "Title (A-Z)", "Status", "Progress"])
            if sort_by == "Upload date (newest first)":
                library.sort(key=lambda x: x.get("upload_date", ""), reverse=True)
            elif sort_by == "Word count (descending)":
                library.sort(key=lambda x: x.get("word_count", 0), reverse=True)
            elif sort_by == "Title (A-Z)":
                library.sort(key=lambda x: x.get("filename", "").lower())
            elif sort_by == "Status":
                library.sort(key=lambda x: x.get("status", ""))
            elif sort_by == "Progress":
                library.sort(key=lambda x: x.get("progress", 0), reverse=True)
            
            for idx, book in enumerate(library):
                with st.container():
                    cols = st.columns([1, 3, 2, 1])
                    with cols[0]:
                        if book.get("cover_b64"):
                            st.image(base64.b64decode(book["cover_b64"]), width=100)
                        elif book.get("cover_url"):
                            st.image(book["cover_url"], width=100)
                        else:
                            st.image("https://via.placeholder.com/100x150?text=No+Cover", width=100)
                    with cols[1]:
                        st.markdown(f"**{book['filename']}**")
                        st.caption(f"Uploaded: {book['upload_date']}")
                        st.caption(f"Words: {book['word_count']:,} | Pages read: {book.get('progress', 0)}")
                    with cols[2]:
                        status_options = ["unread", "reading", "finished"]
                        status_idx = status_options.index(book.get("status", "unread"))
                        new_status = st.selectbox("Status", status_options, index=status_idx, key=f"status_{idx}_{book['filename']}", label_visibility="collapsed")
                        new_progress = st.number_input("Pages read", min_value=0, max_value=1000, value=book.get("progress", 0), key=f"progress_{idx}_{book['filename']}", label_visibility="collapsed")
                        if st.button("Update", key=f"update_{idx}_{book['filename']}"):
                            if update_book_status(book['filename'], new_status, new_progress):
                                st.success(f"Updated '{book['filename']}'")
                                st.rerun()
                    with cols[3]:
                        if st.button("Delete", key=f"del_{idx}_{book['filename']}"):
                            delete_book_from_library(book['filename'])
                            st.rerun()
                    st.divider()

    # ---------- TAB 7: Summarize ----------
    with tabs[6]:
        st.header("Summarize Chapters, Paragraphs, or Custom Text")
        st.markdown("**Note:** Summarization calls the Gemini API each time (spinner appears). You can save summaries for later.")
        
        if st.session_state.chunks_metadata:
            pages = sorted(set([page for _, page in st.session_state.chunks_metadata]))
            min_page = min(pages) if pages else 1
            max_page = max(pages) if pages else 1
            page_range = st.slider("Select page range from the book", min_value=min_page, max_value=max_page, value=(min_page, min(min_page+10, max_page)))
            if st.button("Summarize selected pages"):
                selected_text = []
                for text, page in st.session_state.chunks_metadata:
                    if page_range[0] <= page <= page_range[1]:
                        selected_text.append(text)
                if selected_text:
                    text_to_sum = "\n\n".join(selected_text)[:8000]
                    hash_key = hashlib.md5(text_to_sum.encode()).hexdigest()
                    if hash_key in st.session_state.summary_cache:
                        summary = st.session_state.summary_cache[hash_key]
                        st.success(summary)
                        if st.button("💾 Save this summary", key="save_summary_cached"):
                            st.session_state.saved_summaries.append({
                                "original": text_to_sum[:200] + "...",
                                "summary": summary,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "source": f"Pages {page_range[0]}-{page_range[1]}"
                            })
                            st.toast("Saved!", icon="✅")
                    else:
                        with st.spinner("Generating summary..."):
                            try:
                                prompt = f"Summarize the following text concisely:\n\n{text_to_sum}"
                                response = llm.invoke(prompt)
                                summary = response.content
                                st.session_state.summary_cache[hash_key] = summary
                                st.success(summary)
                                if st.button("💾 Save this summary", key="save_summary_new"):
                                    st.session_state.saved_summaries.append({
                                        "original": text_to_sum[:200] + "...",
                                        "summary": summary,
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "source": f"Pages {page_range[0]}-{page_range[1]}"
                                    })
                                    st.toast("Saved!", icon="✅")
                            except Exception as e:
                                st.error(f"Error: {e}")
                else:
                    st.warning("No text in that page range.")
        
        st.markdown("---")
        st.subheader("Or paste your own text")
        manual_text = st.text_area("Paste text to summarize", height=200)
        if st.button("Summarize custom text") and manual_text:
            hash_key = hashlib.md5(manual_text.encode()).hexdigest()
            if hash_key in st.session_state.summary_cache:
                summary = st.session_state.summary_cache[hash_key]
                st.success(summary)
                if st.button("💾 Save this summary", key="save_custom_cached"):
                    st.session_state.saved_summaries.append({
                        "original": manual_text[:200] + "...",
                        "summary": summary,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "Custom text"
                    })
                    st.toast("Saved!", icon="✅")
            else:
                with st.spinner("Generating summary..."):
                    try:
                        prompt = f"Summarize the following text concisely:\n\n{manual_text[:8000]}"
                        response = llm.invoke(prompt)
                        summary = response.content
                        st.session_state.summary_cache[hash_key] = summary
                        st.success(summary)
                        if st.button("💾 Save this summary", key="save_custom_new"):
                            st.session_state.saved_summaries.append({
                                "original": manual_text[:200] + "...",
                                "summary": summary,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "source": "Custom text"
                            })
                            st.toast("Saved!", icon="✅")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ---------- TAB 8: Saved Summaries & Quizzes ----------
    with tabs[7]:
        st.header("Saved Summaries & Quizzes")
        
        st.subheader("📝 Saved Summaries")
        if st.session_state.saved_summaries:
            for i, s in enumerate(st.session_state.saved_summaries):
                with st.expander(f"Summary {i+1} - {s['timestamp']} (Source: {s['source']})"):
                    st.caption(f"Original excerpt: {s['original']}")
                    st.write(f"**Summary:** {s['summary']}")
            if st.button("Export all summaries as JSON"):
                json_str = json.dumps(st.session_state.saved_summaries, indent=2)
                st.download_button("Download summaries JSON", json_str, "saved_summaries.json", "application/json")
        else:
            st.info("No saved summaries yet.")
        
        st.markdown("---")
        st.subheader("📋 Saved Quizzes")
        if st.session_state.saved_quizzes:
            for i, qz in enumerate(st.session_state.saved_quizzes):
                with st.expander(f"Quiz {i+1} - {qz['timestamp']} (Book: {qz['book']})"):
                    for j, q in enumerate(qz['questions']):
                        st.write(f"**Q{j+1}:** {q['question']}")
                        for opt in q['options']:
                            st.write(f"  - {opt}")
                        st.write(f"*Correct answer:* {q['options'][q['correct']]}")
            if st.button("Export all quizzes as JSON"):
                json_str = json.dumps(st.session_state.saved_quizzes, indent=2)
                st.download_button("Download quizzes JSON", json_str, "saved_quizzes.json", "application/json")
        else:
            st.info("No saved quizzes yet.")

else:
    st.info("👈 Upload a PDF to start.")