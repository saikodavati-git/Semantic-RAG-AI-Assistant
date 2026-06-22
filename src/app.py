from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb
import streamlit as st
from chromadb.api.models.Collection import Collection
from docx import Document
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


load_dotenv()

APP_TITLE = "SEMANTIC RAG AI ASSISTANT"
APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "semantic-rag-symbol.png"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
BATCH_SIZE = 128
SUPPORTED_FILE_TYPES = {"pdf", "docx", "txt"}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./vector_store").strip() or "./vector_store"
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads").strip() or "./uploads"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(APP_TITLE)


@dataclass(frozen=True)
class DocumentSection:


    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class PreparedChunk:
    

    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    

    chunk_id: str
    text: str
    source: str
    chunk_index: int | None
    page_number: int | None
    score: float | None




@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_chroma_collection(db_path: str) -> Collection:
    try:
        Path(db_path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Connected to ChromaDB collection '%s' at %s", COLLECTION_NAME, db_path)
        return collection
    except Exception:
        logger.exception("Failed to initialize ChromaDB")
        raise


def get_gemini_client() -> genai.Client | None:
    if not GOOGLE_API_KEY:
        return None

    return genai.Client(api_key=GOOGLE_API_KEY)




def read_pdf_sections(file_path: str) -> list[DocumentSection]:
    reader = PdfReader(file_path)
    sections: list[DocumentSection] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(DocumentSection(text=text, page_number=page_index))
    return sections


def read_docx_sections(file_path: str) -> list[DocumentSection]:
    document = Document(file_path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
    text = "\n".join(paragraphs).strip()
    return [DocumentSection(text=text)] if text else []


def read_txt_sections(file_bytes: bytes) -> list[DocumentSection]:
    try:
        text = file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="ignore").strip()
    return [DocumentSection(text=text)] if text else []


def save_uploaded_file(uploaded_file: Any) -> Path:
    upload_directory = Path(UPLOAD_DIR)
    upload_directory.mkdir(parents=True, exist_ok=True)

    safe_name = Path(uploaded_file.name).name
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("The uploaded file does not have a valid filename.")

    destination = upload_directory / safe_name
    destination.write_bytes(uploaded_file.getvalue())
    logger.info("Saved uploaded document to %s", destination)
    return destination


def extract_sections_from_upload(uploaded_file: Any) -> list[DocumentSection]:
    """Persist an upload and extract its text based on the file extension."""
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if extension not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported file type: {extension}")

    saved_path = save_uploaded_file(uploaded_file)
    if extension == "txt":
        return read_txt_sections(saved_path.read_bytes())

    if extension == "pdf":
        return read_pdf_sections(str(saved_path))
    if extension == "docx":
        return read_docx_sections(str(saved_path))

    raise ValueError(f"Unsupported file type: {extension}")


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_section(section: DocumentSection) -> list[str]:
    splitter = get_text_splitter()
    return [chunk.strip() for chunk in splitter.split_text(section.text) if chunk.strip()]


def generate_chunk_id(chunk_text: str) -> str:
    return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()


def chunk_exists(chunk_id: str, collection: Collection | None = None) -> bool:
    active_collection = collection or get_chroma_collection(CHROMA_DB_PATH)
    try:
        result = active_collection.get(ids=[chunk_id], include=[])
        return chunk_id in set(result.get("ids", []))
    except Exception:
        logger.exception("Failed to check whether chunk exists: %s", chunk_id)
        raise


def store_chunk(
    chunk_text: str,
    embedding: list[float],
    collection: Collection | None = None,
    metadata: dict[str, Any] | None = None,
    chunk_id: str | None = None,
) -> str:
    active_collection = collection or get_chroma_collection(CHROMA_DB_PATH)
    stored_chunk_id = chunk_id or generate_chunk_id(chunk_text)
    active_collection.add(
        ids=[stored_chunk_id],
        documents=[chunk_text],
        embeddings=[embedding],
        metadatas=[metadata or {"chunk_id": stored_chunk_id}],
    )
    logger.info("New chunk inserted: %s", stored_chunk_id)
    return stored_chunk_id


def existing_ids(collection: Collection, ids: list[str]) -> set[str]:
    if not ids:
        return set()

    try:
        found_ids: set[str] = set()
        unique_ids = list(dict.fromkeys(ids))
        for start in range(0, len(unique_ids), BATCH_SIZE):
            batch_ids = unique_ids[start : start + BATCH_SIZE]
            result = collection.get(ids=batch_ids, include=[])
            found_ids.update(result.get("ids", []))
        return found_ids
    except Exception:
        logger.exception("Failed to check existing chunk IDs")
        raise


def clear_collection(collection: Collection) -> None:
    logger.info("Reindex requested; deleting ChromaDB collection contents")
    try:
        all_items = collection.get(include=[])
        ids_to_delete = all_items.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
    except Exception:
        logger.exception("Failed while clearing existing ChromaDB collection")
        raise


def prepare_chunks_for_ingestion(uploaded_file: Any) -> list[PreparedChunk]:
    sections = extract_sections_from_upload(uploaded_file)
    prepared_chunks: list[PreparedChunk] = []
    chunk_index = 0

    for section in sections:
        for chunk_text in split_section(section):
            chunk_index += 1
            chunk_id = generate_chunk_id(chunk_text)
            metadata: dict[str, Any] = {
                "source": uploaded_file.name,
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
            }
            if section.page_number is not None:
                metadata["page_number"] = section.page_number

            prepared_chunks.append(
                PreparedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

    return prepared_chunks


def filter_new_chunks(
    prepared_chunks: list[PreparedChunk],
    collection: Collection,
) -> tuple[list[PreparedChunk], int]:
    known_ids = existing_ids(collection, [chunk.chunk_id for chunk in prepared_chunks])
    new_chunks: list[PreparedChunk] = []
    duplicate_count = 0

    for chunk in prepared_chunks:
        if chunk.chunk_id in known_ids:
            duplicate_count += 1
            logger.info("Duplicate chunk skipped: %s", chunk.chunk_id)
            continue

        known_ids.add(chunk.chunk_id)
        new_chunks.append(chunk)

    return new_chunks, duplicate_count


def store_new_chunks(
    new_chunks: list[PreparedChunk],
    collection: Collection,
    embedding_model: SentenceTransformer,
) -> int:

    inserted_count = 0

    for start in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[start : start + BATCH_SIZE]
        batch_texts = [chunk.text for chunk in batch]
        batch_ids = [chunk.chunk_id for chunk in batch]
        batch_metadatas = [chunk.metadata for chunk in batch]
        batch_embeddings = embedding_model.encode(
            batch_texts,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )
        inserted_count += len(batch)
        for chunk_id in batch_ids:
            logger.info("New chunk inserted: %s", chunk_id)

    return inserted_count


def run_ingestion_pipeline(uploaded_files: Iterable[Any], reindex: bool = False) -> tuple[int, int]:
    collection = get_chroma_collection(CHROMA_DB_PATH)
    embedding_model = get_embedding_model()

    if reindex:
        clear_collection(collection)

    total_chunks_processed = 0
    total_chunks_inserted = 0
    total_duplicate_chunks = 0

    for uploaded_file in uploaded_files:
        logger.info("Ingestion pipeline started for upload: %s", uploaded_file.name)
        prepared_chunks = prepare_chunks_for_ingestion(uploaded_file)
        if not prepared_chunks:
            logger.warning("No extractable text found in upload: %s", uploaded_file.name)
            continue

        total_chunks_processed += len(prepared_chunks)
        new_chunks, duplicate_count = filter_new_chunks(prepared_chunks, collection)
        total_duplicate_chunks += duplicate_count

        if not new_chunks:
            logger.info("No new chunks to index for upload: %s", uploaded_file.name)
            continue

        inserted_count = store_new_chunks(new_chunks, collection, embedding_model)
        total_chunks_inserted += inserted_count
        logger.info("Indexed %s new chunks from %s", inserted_count, uploaded_file.name)

    logger.info("Total chunks processed: %s", total_chunks_processed)
    logger.info("Total chunks inserted: %s", total_chunks_inserted)
    logger.info("Total duplicate chunks found: %s", total_duplicate_chunks)

    return total_chunks_inserted, total_duplicate_chunks


def embed_question(question: str, embedding_model: SentenceTransformer) -> list[float]:
    return embedding_model.encode([question], normalize_embeddings=True).tolist()[0]


def retrieve_relevant_chunks(
    question_embedding: list[float],
    collection: Collection,
    top_k: int = TOP_K,
) -> list[RetrievedChunk]:
    try:
        if collection.count() == 0:
            logger.info("Retrieval skipped because ChromaDB collection is empty")
            return []

        result = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        logger.info("Retrieved top %s chunks for query", top_k)
    except Exception:
        logger.exception("ChromaDB retrieval failed")
        raise

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    chunks: list[RetrievedChunk] = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        score = 1 - distance if distance is not None else None
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=document,
                source=metadata.get("source", "Unknown"),
                chunk_index=metadata.get("chunk_index") or metadata.get("chunk_number"),
                page_number=metadata.get("page_number"),
                score=score,
            )
        )
    return chunks


def format_conversation_history(messages: list[dict[str, str]], max_messages: int = 8) -> str:
    recent_messages = messages[-max_messages:]
    if not recent_messages:
        return "No prior conversation."

    lines = []
    for message in recent_messages:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = message.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks with source, page, chunk ID, and text."""
    context_blocks = []
    for chunk in chunks:
        location_parts = [f"Source: {chunk.source}", f"Chunk ID: {chunk.chunk_id}"]
        if chunk.chunk_index is not None:
            location_parts.append(f"Chunk Index: {chunk.chunk_index}")
        if chunk.page_number is not None:
            location_parts.append(f"Page: {chunk.page_number}")

        context_blocks.append(f"[{' | '.join(location_parts)}]\n{chunk.text}")

    return "\n\n".join(context_blocks)


def build_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    conversation_history: list[dict[str, str]],
) -> str:
    context = build_context_block(chunks) or "No relevant document context was retrieved."
    history = format_conversation_history(conversation_history)
    return f"""
You are a careful Retrieval-Augmented Generation assistant.
Answer the current question helpfully and accurately.


Retrieved Context:
{context}

Prior Conversation:
{history}

Current Question:
{question}

Answer:
""".strip()


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    conversation_history: list[dict[str, str]],
) -> str:
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    prompt = build_rag_prompt(question, chunks, conversation_history)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        answer = getattr(response, "text", "").strip()
        logger.info("Gemini generated a response with %s characters", len(answer))
        return answer or "I couldn't generate an answer. Please try rephrasing the question."
    except Exception:
        logger.exception("Gemini API request failed")
        raise


def run_query_pipeline(
    question: str,
    conversation_history: list[dict[str, str]],
    top_k: int = TOP_K,
) -> tuple[str, list[RetrievedChunk]]:

    collection = get_chroma_collection(CHROMA_DB_PATH)
    embedding_model = get_embedding_model()
    question_embedding = embed_question(question, embedding_model)
    chunks = retrieve_relevant_chunks(question_embedding, collection, top_k=top_k)
    answer = generate_answer(question, chunks, conversation_history)
    return answer, chunks




def initialize_session_state() -> None:
    """Initialize all Streamlit session-state keys used by the app."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_retrieved_chunks", [])
    st.session_state.setdefault("active_document_signature", None)


def apply_custom_theme() -> None:
    """Apply a clean, centered assistant layout inspired by the reference UI."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f7f8;
            color: #20242f;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 960px;
            padding-top: 2.2rem;
        }

        .assistant-hero {
            max-width: 900px;
            margin: 2.5rem auto 1.5rem auto;
            text-align: left;
        }

        .assistant-mark {
            display: block;
            width: 170px;
            height: auto;
            margin-bottom: 1.5rem;
            border-radius: 4px;
        }

        .assistant-title {
            margin: 0 0 1.35rem 0;
            font-size: clamp(2.6rem, 6vw, 4.1rem);
            line-height: 1.05;
            font-weight: 760;
            letter-spacing: 0;
            color: #3b3d46;
        }

        div[data-testid="stForm"] {
            border: 0;
            background: transparent;
            padding: 0;
        }

        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
            gap: 0.45rem;
            align-items: end;
        }

        .stTextInput input {
            min-height: 46px;
            border: 1px solid #e3e5ea;
            border-radius: 999px;
            background: #ffffff;
            color: #20242f;
            caret-color: #000000;
            padding: 0 1.1rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
        }

        div[data-testid="stBottom"] {
            z-index: 30;
            background: rgba(247, 247, 248, 0.96);
        }

        div[data-testid="stBottom"] > div {
            background: transparent;
        }

        div[data-testid="stChatInput"] {
            max-width: 760px;
            margin: 0 auto;
        }

        div[data-testid="stChatInput"] > div {
            background: #ffffff;
            border: 1px solid #e3e5ea;
            border-radius: 999px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
        }

        div[data-testid="stChatInput"] textarea {
            background: #ffffff;
            color: #20242f;
            caret-color: #000000;
        }

        div[data-testid="stChatInput"] textarea::placeholder {
            color: #7d8493;
            opacity: 1;
        }

        .stTextInput input::placeholder {
            color: #7d8493;
            opacity: 1;
        }

        .stTextInput input:focus {
            border-color: #c8ceda;
            box-shadow: 0 0 0 3px rgba(120, 133, 160, 0.14);
        }

        .stButton button {
            border-radius: 999px;
            border: 1px solid #e3e5ea;
            background: #ffffff;
            color: #20242f;
            min-height: 38px;
            box-shadow: none;
        }

        .stButton button:hover {
            border-color: #c8ceda;
            background: #fbfbfc;
            color: #20242f;
        }

        div[data-testid="stFormSubmitButton"] button {
            min-height: 46px;
            width: 54px;
            padding: 0;
            font-size: 1.4rem;
            font-weight: 700;
            color: #969cab;
        }

        .suggestion-row {
            margin-top: 0.9rem;
        }

        .chat-shell {
            max-width: 760px;
            margin: 0 auto;
            padding-bottom: 2rem;
        }

        .chat-header {
            position: sticky;
            top: 0;
            z-index: 20;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-height: 64px;
            margin-bottom: 1rem;
            padding: 0.65rem 0;
            background: rgba(247, 247, 248, 0.96);
            border-bottom: 1px solid #e1e4e8;
        }

        .chat-header img {
            width: 52px;
            height: 39px;
            object-fit: cover;
            border-radius: 4px;
        }

        .chat-header-title {
            margin: 0;
            color: #20242f;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.2;
            letter-spacing: 0;
        }

        div[data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid #e8eaf0;
            border-radius: 10px;
            color: #20242f;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #eef2f7;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: #ffffff;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] div {
            color: #20242f;
        }

        div[data-testid="stChatMessage"] code {
            color: #111827;
            background: #f1f3f6;
        }

        .inline-chat-form {
            margin-top: 1rem;
            padding-top: 0.25rem;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #eceef2;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 1.25rem;
            }

            .assistant-hero {
                margin-top: 1.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_assistant_mark() -> None:
    encoded_symbol = base64.b64encode(APP_ICON_PATH.read_bytes()).decode("ascii")
    st.markdown(
        (
            '<img class="assistant-mark" '
            f'src="data:image/png;base64,{encoded_symbol}" '
            'alt="SemanticRAG document search symbol">'
        ),
        unsafe_allow_html=True,
    )


def render_welcome_screen() -> str | None:
    st.markdown('<div class="assistant-hero">', unsafe_allow_html=True)
    render_assistant_mark()
    st.markdown('<h1 class="assistant-title">SemanticRAG AI assistant</h1>', unsafe_allow_html=True)

    submitted_question = render_inline_question_form(
        "welcome_question_form",
        "Ask a question...",
    )

    st.markdown('<div class="suggestion-row">', unsafe_allow_html=True)
    suggestions = [
        "Summarize my uploaded document"
    ]
    cols = st.columns([1.25, 1.15, 1.05, 0.85, 1.35])
    for col, suggestion in zip(cols, suggestions):
        with col:
            if st.button(suggestion, use_container_width=True):
                submitted_question = suggestion
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return submitted_question


def render_chat_header() -> None:
    encoded_symbol = base64.b64encode(APP_ICON_PATH.read_bytes()).decode("ascii")
    st.markdown(
        (
            '<div class="chat-header">'
            f'<img src="data:image/png;base64,{encoded_symbol}" '
            'alt="SemanticRAG document search symbol">'
            '<h1 class="chat-header-title">SemanticRAG AI assistant</h1>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_inline_question_form(form_key: str, placeholder: str) -> str | None:
    submitted_question = None
    st.markdown('<div class="inline-chat-form">', unsafe_allow_html=True)
    with st.form(form_key, clear_on_submit=True):
        input_col, submit_col = st.columns([12, 1])
        with input_col:
            typed_question = st.text_input(
                "Question",
                placeholder=placeholder,
                label_visibility="collapsed",
            )
        with submit_col:
            submitted = st.form_submit_button(">")
        if submitted and typed_question.strip():
            submitted_question = typed_question.strip()
    st.markdown("</div>", unsafe_allow_html=True)
    return submitted_question


def render_sidebar() -> tuple[list[Any], bool]:
    st.sidebar.header("Document Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF, DOCX, or TXT file",
        type=sorted(SUPPORTED_FILE_TYPES),
        accept_multiple_files=False,
    )

    if st.sidebar.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_retrieved_chunks = []
        st.success("Chat history cleared.")

    return [uploaded_file] if uploaded_file is not None else [], False


def render_retrieved_chunks(chunks: list[RetrievedChunk]) -> None:
    with st.sidebar.expander("Retrieved chunks", expanded=False):
        if not chunks:
            st.caption("No chunks retrieved yet.")
            return

        for index, chunk in enumerate(chunks, start=1):
            score = f"{chunk.score:.4f}" if chunk.score is not None else "N/A"
            details = [f"Source: `{chunk.source}`", f"Similarity: `{score}`"]
            if chunk.page_number is not None:
                details.append(f"Page: `{chunk.page_number}`")
            if chunk.chunk_index is not None:
                details.append(f"Chunk: `{chunk.chunk_index}`")
            st.markdown(f"**Result {index}** | " + " | ".join(details))
            st.write(chunk.text)
            st.divider()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_uploads(uploaded_files: list[Any], reindex: bool) -> None:
    if reindex and not uploaded_files:
        st.warning("Upload at least one document before reindexing.")
        return

    if not uploaded_files:
        return

    uploaded_file = uploaded_files[0]
    file_signature = (
        uploaded_file.name,
        hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
    )
    if not reindex and file_signature == st.session_state.active_document_signature:
        return

    with st.spinner():
        try:
            new_chunks, duplicates = run_ingestion_pipeline([uploaded_file], reindex=True)
            st.session_state.active_document_signature = file_signature
            st.session_state.messages = []
            st.session_state.last_retrieved_chunks = []
            logger.info(
                "Document replacement success: source=%s new=%s duplicates=%s",
                uploaded_file.name,
                new_chunks,
                duplicates,
            )
        except ValueError as exc:
            st.error(str(exc))
            logger.exception("Upload validation failed")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            logger.exception("Indexing failed")


def handle_user_question(question: str) -> None:
    """Retrieve context and answer a chat question."""
    cleaned_question = question.strip()
    if not cleaned_question:
        st.warning("Please enter a question.")
        return

    with st.chat_message("user"):
        st.markdown(cleaned_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                answer, chunks = run_query_pipeline(
                    cleaned_question,
                    st.session_state.messages,
                )
                st.session_state.last_retrieved_chunks = chunks
                st.markdown(answer)
                st.session_state.messages.append({"role": "user", "content": cleaned_question})
                st.session_state.messages.append({"role": "assistant", "content": answer})
                logger.info("Answered user question successfully")
            except RuntimeError as exc:
                message = f"Configuration error: {exc}"
                st.error(message)
                st.session_state.messages.append({"role": "user", "content": cleaned_question})
                st.session_state.messages.append({"role": "assistant", "content": message})
                logger.exception("Configuration error while answering")
            except Exception as exc:
                message = f"Sorry, I could not answer that question: {exc}"
                st.error(message)
                st.session_state.messages.append({"role": "user", "content": cleaned_question})
                st.session_state.messages.append({"role": "assistant", "content": message})
                logger.exception("Question answering failed")


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title=APP_TITLE, page_icon=str(APP_ICON_PATH), layout="wide")
    initialize_session_state()
    apply_custom_theme()

    if not GOOGLE_API_KEY:
        st.warning("GOOGLE_API_KEY is not configured. Add it to your environment or .env file.")

    uploaded_files, reindex = render_sidebar()
    handle_uploads(uploaded_files, reindex)

    if st.session_state.messages:
        st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
        render_chat_header()
        render_chat_history()
        st.markdown("</div>", unsafe_allow_html=True)
        question = st.chat_input(
            "Ask a question about your uploaded document",
            key="continuous_chat_input",
        )
    else:
        question = render_welcome_screen()

    if question:
        handle_user_question(question)
        st.rerun()

    render_retrieved_chunks(st.session_state.last_retrieved_chunks)


if __name__ == "__main__":
    main()
