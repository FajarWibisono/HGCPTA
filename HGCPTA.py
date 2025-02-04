import streamlit as st
import os

# pip install streamlit langchain huggingface_hub sentence-transformers faiss-cpu

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI API & HALAMAN
# ─────────────────────────────────────────────────────────────────────────────

# Ganti GROQ_API_KEY dengan kunci Anda sendiri, misalnya di secrets.toml
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="HCTPA",
    page_icon="📓",
    layout="wide"
)

# CSS Styling
st.markdown(
    """
    <style>
        .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .user-message { background-color: #f0f2f6; }
        .bot-message { background-color: #e8f0fe; }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul Aplikasi
st.title("📓 HCTPA GUIDE")
st.markdown(
    """
    ### Selamat Datang di Asisten Pengetahuan HCTPA
    Chat Bot ini akan membantu Anda memahami lebih dalam HCTPA framework.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. STATE DAN INISIALISASI
# ─────────────────────────────────────────────────────────────────────────────
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chain' not in st.session_state:  
    st.session_state.chain = None

# ─────────────────────────────────────────────────────────────────────────────
# 3. PROMPT UNTUK MENJAMIN BAHASA INDONESIA
# ─────────────────────────────────────────────────────────────────────────────
# Prompt ini akan memaksa jawaban selalu dalam Bahasa Indonesia.
PROMPT_INDONESIA = """\
Gunakan informasi konteks berikut untuk menjawab pertanyaan pengguna dalam bahasa Indonesia yang baik dan terstruktur.
Selalu berikan jawaban terbaik yang dapat kamu berikan dalam bahasa indonesia.

Konteks: {context}
Riwayat Chat: {chat_history}
Pertanyaan: {question}

Jawaban:
"""

INDO_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=PROMPT_INDONESIA
)

# ─────────────────────────────────────────────────────────────────────────────────
# 3.1. PREPROCESS DOCUMENT (FUNGSIONALITAS UTK BAHASA INDONESIA)
# ─────────────────────────────────────────────────────────────────────────────────

def preprocess_document(text: str) -> str:
    """Preprocessing khusus untuk dokumen Bahasa Indonesia."""
    import re

    # Bersihkan karakter khusus
    text = re.sub(r'[^\w\s\.]', ' ', text)

    # Normalisasi spasi
    text = ' '.join(text.split())

    # Handling untuk singkatan umum bahasa Indonesia
    abbreviations = {
        'yg': 'yang',
        'dgn': 'dengan',
        'utk': 'untuk',
        'tsb': 'tersebut',
        'dll': 'dan lain-lain',
        'dst': 'dan seterusnya',
        'dsb': 'dan sebagainya',
        'spt': 'seperti',
        'krn': 'karena',
        'pd': 'pada',
        'dr': 'dari',
        'knp': 'kenapa',
        'HCTPA': 'Human Capital Technology & People Analytics'              
    }
    for abbr, full in abbreviations.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)

    return text

# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNGSI INISIALISASI RAG
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def initialize_rag():
    """
    Memuat dokumen PDF dari folder 'documents', memecah menjadi chunk,
    membuat FAISS vector store, dan membentuk ConversationalRetrievalChain.
    """
    try:
        # 4.1 Load Dokumen PDF
        loader = DirectoryLoader("documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        # 4.2 Split Dokumen
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=909, chunk_overlap=144)
        texts = text_splitter.split_documents(documents)

        # 4.3 Embedding Berbahasa Indonesia
        # Ganti sesuai preferensi, misal "indobenchmark/indobert-base-p1", dsb.
        embeddings = HuggingFaceEmbeddings(
            model_name="LazarusNLP/all-indo-e5-small-v4",
            model_kwargs={'device': 'cpu'}  
        )

        # 4.4 Membuat Vector Store FAISS
        vectorstore = FAISS.from_documents(texts, embeddings)

        # 4.5 Menginisialisasi LLM (ChatGroq)
        llm = ChatGroq(
            temperature=0.54,
            model_name="mixtral-8x7b-32768",
            max_tokens=1024
        )

        # 4.6 Membuat Memory untuk menyimpan riwayat percakapan
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )

        # 4.7 Membuat ConversationalRetrievalChain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                'prompt': INDO_PROMPT_TEMPLATE,  
                'output_key': 'answer'
            }
        )

        return chain

    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 5. INISIALISASI SISTEM
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.chain is None:
    with st.spinner("Memuat sistem..."):
        st.session_state.chain = initialize_rag()

# ─────────────────────────────────────────────────────────────────────────────
# 6. ANTARMUKA CHAT
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.chain:
    # 6.1 Tampilkan riwayat chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 6.2 Chat Input
    prompt = st.chat_input("✍️tuliskan pertanyaan Anda tentang HCTPA di sini")
    if prompt:
        # Tambahkan pertanyaan user ke riwayat chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 6.3 Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                try:
                    # Panggil chain
                    result = st.session_state.chain.invoke({"question": prompt})
                    # Ambil jawaban
                    answer = result.get('answer', '')
                    st.write(answer)
                    # Tambahkan ke riwayat
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# ─────────────────────────────────────────────────────────────────────────────
# 7. FOOTER & DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    ---
    **Disclaimer:**
    - Sistem ini menggunakan AI-LLM dan dapat menghasilkan jawaban yang tidak selalu akurat.
    - Ketik: LANJUTKAN JAWABANMU untuk kemungkinan mendapatkan jawaban yang lebih baik dan utuh.
    - Mohon verifikasi informasi penting dengan sumber terpercaya.
    """
)
