"""
app.py

This is the main entry point for the Streamlit-based Document Q&A application.
It compiles the chunker, embedder, retriever, and generator modules into a unified,
highly interactive dashboard.
"""

import os
import re
import time
import streamlit as st
import numpy as np
import faiss
import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai

# Import our custom scratch-built modules
from chunker import chunk_text
from embedder import build_and_save_index
from retriever import retrieve
from generator import generate_answer
from sentence_transformers import SentenceTransformer

# Load environment variables (such as GEMINI_API_KEY) from the .env file.
# Calling load_dotenv() at the very top ensures keys are available to all imports.
load_dotenv()

# Set up the Streamlit page layout as 'wide' to accommodate the two-panel architecture.
# We also set a page title and an emoji icon for the browser tab.
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📄",
    layout="wide"
)

# -----------------------------------------------------------------------------
# PERFORMANCE: CACHED RESOURCE AND DATA FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """
    Loads the SentenceTransformer embedding model and caches it.
    
    Why:
    Loading deep learning models from disk to GPU/CPU is slow and resource-heavy.
    Caching it with @st.cache_resource ensures it is loaded exactly once for the
    entire runtime of the application, dramatically speeding up subsequent runs.
    """
    # Use a clean loading indicator that only displays during the first cold start
    with st.spinner("Loading embedding model (all-MiniLM-L6-v2)..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


@st.cache_data
def get_embeddings_cached(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of strings and caches the result.
    
    Why:
    Generating embeddings is an expensive matrix calculation. Caching this step
    with @st.cache_data means if the user processes the same document again (e.g. 
    restoring after a clear), we bypass embedding calculation entirely and load
    from memory instantly. We return a nested list of floats because custom objects/numpy
    arrays are less reliable for Streamlit hashing/caching.
    """
    model = load_embedding_model()
    # SentenceTransformer's encode returns a numpy array.
    embeddings = model.encode(texts, show_progress_bar=False)
    # Convert numpy array to list for reliable caching serialization
    return embeddings.tolist()

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
# We initialize all session state variables to prevent KeyErrors during rendering.
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = {}
if "processed" not in st.session_state:
    st.session_state.processed = False
if "sample_questions" not in st.session_state:
    st.session_state.sample_questions = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Check if Gemini API key exists
api_key = os.environ.get("GEMINI_API_KEY")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR CHAT EXPORT & SUGGESTIONS
# -----------------------------------------------------------------------------

def export_chat_history() -> str:
    """
    Builds a plain-text representation of the full conversation log.
    Includes questions, answers, confidence scores, and source chunks.
    
    Why:
    Provides a portable audit trail of the conversation for users to save locally.
    """
    lines = ["=== RAG DOCUMENT Q&A CONVERSATION LOG ===", ""]
    for i, msg in enumerate(st.session_state.messages, 1):
        role = msg["role"].upper()
        content = msg["content"]
        lines.append(f"[{i}] {role}:")
        lines.append(content)
        lines.append("")
        
        if role == "ASSISTANT" and msg.get("sources"):
            lines.append("Sources & Grounding Details:")
            for s_idx, (chunk, score) in enumerate(msg["sources"], 1):
                doc_name = chunk.get("source", "Unknown")
                score_pct = score * 100.0
                snippet = chunk.get("text", "")[:150].replace('\n', ' ')
                lines.append(f"  - Chunk {s_idx} from Doc '{doc_name}' | Similarity: {score_pct:.1f}%")
                lines.append(f"    Snippet: {snippet}...")
            lines.append("")
    return "\n".join(lines)


def generate_suggested_questions(text_excerpt: str) -> list[str]:
    """
    Calls Gemini API to generate 3 sample questions based on the document text.
    
    Why:
    Provides immediate entry points for users to explore the contents of
    their newly uploaded documents without having to read them first.
    """
    if not api_key:
        return []
    
    # We initialize a standalone Client for suggestion generation
    client = genai.Client(api_key=api_key)
    
    prompt = (
        "Generate 3 short questions a user might ask about this document. "
        "Return only the questions as a numbered list, nothing else.\n\n"
        f"Document excerpt:\n{text_excerpt}"
    )
    
    try:
        # Request content from the Gemini model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Parse the numbered lines
        lines = response.text.strip().split("\n")
        questions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Regex to strip leading number prefixes like '1. ', '2) ', etc.
            cleaned = re.sub(r'^\d+[\.\-\)]\s*', '', line)
            if cleaned:
                questions.append(cleaned)
                
        # Return at most 3 questions
        return questions[:3]
    except Exception as e:
        # Log suggestion error to console but do not break the app flow
        print(f"Error generating sample questions: {e}")
        return []

# -----------------------------------------------------------------------------
# SIDEBAR PANEL (Multi-document List and Remove controls)
# -----------------------------------------------------------------------------
# Using st.sidebar for document management lists keeps the main panels clean and spacious.
with st.sidebar:
    st.markdown("### 📚 Loaded Documents")
    if not st.session_state.loaded_docs:
        st.info("No documents are currently loaded. Upload files on the left panel to begin.")
    else:
        # Loop through loaded documents and provide a delete button for each
        for filename in list(st.session_state.loaded_docs.keys()):
            # Left sub-column for name, right sub-column for delete button
            col_name, col_btn = st.columns([0.8, 0.2])
            col_name.write(filename)
            # When the delete button is clicked:
            if col_btn.button("❌", key=f"remove_{filename}"):
                # Remove document text data from memory
                del st.session_state.loaded_docs[filename]
                # Increment the file uploader key to force Streamlit's uploader widget to reset
                st.session_state.uploader_key += 1
                # Mark system as unprocessed because the loaded corpus has changed
                st.session_state.processed = False
                st.session_state.index = None
                st.session_state.chunks = []
                st.session_state.sample_questions = []
                st.rerun()

# -----------------------------------------------------------------------------
# MAIN APP TWO-COLUMN LAYOUT
# -----------------------------------------------------------------------------
col1, col2 = st.columns([0.35, 0.65])

# =============================================================================
# LEFT COLUMN - DOCUMENT PANEL (Controls, Settings, Uploading)
# =============================================================================
with col1:
    st.header("📄 Document Panel")
    
    # 1. File uploader. Accepts both PDF and TXT.
    # Uses a dynamic key so we can clear/reset it programmatically.
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
        help="Upload up to 3 documents. Both standard text and PDFs are supported."
    )
    
    # 2. Sync uploaded files into the session state `loaded_docs`
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.loaded_docs:
                # Limit the total number of documents to 3
                if len(st.session_state.loaded_docs) >= 3:
                    st.warning("Maximum limit of 3 documents reached. Remove a document in the sidebar to add a new one.")
                    break
                
                # Show loading spinner while reading and parsing the file
                with st.spinner(f"Reading {uploaded_file.name}..."):
                    file_bytes = uploaded_file.read()
                    size_kb = len(file_bytes) / 1024.0
                    
                    if uploaded_file.name.endswith(".pdf"):
                        try:
                            # Use PyMuPDF to parse PDF bytes
                            doc = fitz.open(stream=file_bytes, filetype="pdf")
                            pages = doc.page_count
                            text_parts = []
                            # Concatenate text from each page
                            for page in doc:
                                text_parts.append(page.get_text())
                            text = "\n\n".join(text_parts)
                            doc.close()
                            
                            if not text.strip():
                                st.error(f"Could not extract text from {uploaded_file.name}. Scanned PDFs without OCR are not supported.")
                                continue
                        except Exception as e:
                            st.error(f"Failed to read PDF {uploaded_file.name}: {e}")
                            continue
                    else:
                        # Process TXT file
                        try:
                            text = file_bytes.decode("utf-8")
                            # Estimate pages by dividing text length by 3000 chars (standard page length)
                            pages = max(1, len(text) // 3000)
                        except Exception as e:
                            st.error(f"Failed to decode TXT file {uploaded_file.name}: {e}")
                            continue
                    
                    # Store file data in session state dictionary
                    st.session_state.loaded_docs[uploaded_file.name] = {
                        "text": text,
                        "size_kb": size_kb,
                        "pages": pages
                    }
                    # Reset processing state as new files were introduced
                    st.session_state.processed = False
                    st.rerun()

    # 3. If files are loaded, display their metadata in the UI
    if st.session_state.loaded_docs:
        st.markdown("---")
        st.markdown("#### Loaded Files Details")
        for name, data in st.session_state.loaded_docs.items():
            st.info(f"📁 **{name}**\n- Size: {data['size_kb']:.1f} KB\n- Pages: {data['pages']}")
            
    # 4. Advanced settings expander
    with st.expander("⚙️ Advanced settings", expanded=False):
        # Slider for choosing chunk size
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=256,
            max_value=1024,
            value=512,
            step=64,
            help="Smaller chunks fit more precise answers; larger chunks provide broader context."
        )
        
        # Slider for choosing chunk overlap
        chunk_overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=150,
            value=50,
            step=10,
            help="Overlap maintains context continuity across chunk transitions."
        )
        st.caption("ℹ️ Smaller chunks = more precise, larger = more context")
        
        # CHUNK VISUALIZER: Button to preview splitting before indexing
        if st.button("Preview chunks", key="preview_chunks_btn"):
            if not st.session_state.loaded_docs:
                st.warning("Please upload a document first to preview chunks.")
            else:
                st.write("---")
                # Preview chunk boundaries using the first document in the queue
                first_name = list(st.session_state.loaded_docs.keys())[0]
                first_text = st.session_state.loaded_docs[first_name]["text"]
                
                # Split text using our custom recursive chunker
                preview_splits = chunk_text(first_text, chunk_size, chunk_overlap)
                st.markdown(f"**Chunk Boundaries Preview (First 3 chunks of {first_name})**")
                
                for idx, chunk in enumerate(preview_splits[:3]):
                    st.markdown(f"**Chunk {idx+1} (Length: {len(chunk)} characters)**")
                    # Display the text in a clear, formatted code block to show bounds
                    st.code(chunk, language="text")
                    
    # 5. Process Document Button (Indexing)
    if st.button("Process Document", type="primary", use_container_width=True):
        if not st.session_state.loaded_docs:
            st.error("Please upload at least one document first.")
        else:
            start_time = time.time()
            all_chunks = []
            
            # Step A: Chunk all documents, tagging each chunk with its source filename
            for filename, doc_data in st.session_state.loaded_docs.items():
                chunks_str = chunk_text(doc_data["text"], chunk_size, chunk_overlap)
                for chunk_str in chunks_str:
                    all_chunks.append({
                        "text": chunk_str,
                        "source": filename
                    })
            
            if not all_chunks:
                st.error("Document is empty or has no extractable text chunks.")
            else:
                # Step B: Generate embeddings and build FAISS index
                try:
                    # Load the embedding model (utilizing cache_resource)
                    model = load_embedding_model()
                    
                    # Generate embeddings utilizing cache_data
                    texts_list = [c["text"] for c in all_chunks]
                    embeddings_list = get_embeddings_cached(texts_list)
                    
                    # Convert list back to numpy float32 matrix
                    embeddings_matrix = np.array(embeddings_list).astype('float32')
                    
                    # L2 Normalize vectors for Cosine Similarity calculations
                    faiss.normalize_L2(embeddings_matrix)
                    
                    # Create index
                    dimension = embeddings_matrix.shape[1]
                    index = faiss.IndexFlatIP(dimension)
                    index.add(embeddings_matrix)
                    
                    # Persist index and chunks in session state
                    st.session_state.index = index
                    st.session_state.chunks = all_chunks
                    st.session_state.processed = True
                    
                    # Calculate elapsed processing time
                    elapsed_time = time.time() - start_time
                    
                    st.success(
                        f"Indexed {len(all_chunks)} chunks!\n\n"
                        f"- Embedding model: `all-MiniLM-L6-v2`\n"
                        f"- Time taken: {elapsed_time:.2f} seconds"
                    )
                    
                    # Step C: Generate sample questions using the beginning of the text
                    # Concatenate first 500 characters of the documents to feed to Gemini
                    sample_text = ""
                    for filename, doc_data in st.session_state.loaded_docs.items():
                        sample_text += doc_data["text"][:300] + " "
                    
                    sample_text = sample_text[:500]
                    # Fetch suggestions and store in session state
                    st.session_state.sample_questions = generate_suggested_questions(sample_text)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error building vector index: {e}")

    # 6. Clear document button
    if st.button("Clear document", use_container_width=True):
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.loaded_docs = {}
        st.session_state.processed = False
        st.session_state.sample_questions = []
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.success("Document state cleared!")
        st.rerun()
        
    st.markdown("---")
    
    # 7. How it works guide
    with st.expander("❔ How it works", expanded=False):
        st.markdown(
            "1. **Chunking**: Text is recursively divided into small overlap paragraphs to preserve local meaning.\n"
            "2. **Embedding**: Chunks are translated into 384-dimensional semantic numbers using a Transformer model.\n"
            "3. **Indexing**: FAISS creates a flat Inner Product index (Cosine Similarity) for lightning-fast matching.\n"
            "4. **LLM Grounding**: The top-4 matched text blocks are injected into Gemini's context, restricting its answers to the source text."
        )

# =============================================================================
# RIGHT COLUMN - CHAT PANEL (Conversation history, Q&A, Sources)
# =============================================================================
with col2:
    # Top bar containing header and Clear Chat button
    chat_top_col, chat_clear_col = st.columns([0.8, 0.2])
    chat_top_col.header("💬 Document Chat")
    
    # If there are messages in history, offer a Clear Chat button
    if st.session_state.messages:
        if chat_clear_col.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
    st.markdown("---")
    
    # Render historical conversation bubbles from the session state
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        
        with st.chat_message(role):
            st.markdown(content)
            
            # If the speaker is the assistant, display the retrieved sources and confidence indicators
            if role == "assistant" and msg.get("sources"):
                sources = msg["sources"]
                
                # Expander listing the documents and chunks used
                with st.expander(f"Sources ({len(sources)} chunks used)", expanded=False):
                    for idx, (chunk, score) in enumerate(sources, 1):
                        score_pct = score * 100.0
                        source_doc = chunk.get("source", "Unknown Document")
                        chunk_text = chunk.get("text", "")
                        
                        # Apply score-based colored warnings for the badge
                        badge_label = f"Chunk {idx} | Source: {source_doc} | Similarity: {score_pct:.1f}%"
                        if score_pct > 80.0:
                            st.success(badge_label)
                        elif score_pct >= 60.0:
                            st.warning(badge_label)
                        else:
                            st.error(badge_label)
                            
                        # Show the first 200 characters of the text chunk as requested
                        snippet = chunk_text[:200]
                        if len(chunk_text) > 200:
                            snippet += "..."
                        st.write(snippet)
                        
                # Display the retrieval confidence metrics
                if "confidence" in msg:
                    confidence = msg["confidence"]
                    # Sub-layout for metrics
                    metric_col, warning_col = st.columns([0.4, 0.6])
                    metric_col.metric(label="Retrieval Confidence", value=f"{confidence:.1f}%")
                    
                    if confidence < 50.0:
                        warning_col.warning(
                            "Low confidence — the answer may not be well supported by the document."
                        )

    # 1. Clickable sample questions block (only visible if document is indexed and successfully processed)
    selected_sample = None
    if st.session_state.processed and st.session_state.sample_questions:
        st.markdown("💡 *Suggested questions based on document text:*")
        
        # Render sample questions as buttons side by side
        q_cols = st.columns(len(st.session_state.sample_questions))
        for q_idx, q_text in enumerate(st.session_state.sample_questions):
            if q_cols[q_idx].button(q_text, key=f"sample_q_{q_idx}", use_container_width=True):
                # When clicked, save the selected question to submit it
                selected_sample = q_text

    # 2. Export chat button. Display at the bottom of messages if messages exist.
    if st.session_state.messages:
        chat_log = export_chat_history()
        st.download_button(
            label="📥 Export Chat History",
            data=chat_log,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

    # 3. Main chat input widget. Always remains visible at the bottom of the panel.
    chat_input_val = st.chat_input("Ask a question about the uploaded document...")
    
    # We combine the text input value and sample button clicks.
    # If the user typed something, or clicked a suggestion, we trigger the RAG pipeline.
    active_query = chat_input_val or selected_sample
    
    if active_query:
        active_query = active_query.strip()
        # Verify the question is non-empty to prevent processing blanks
        if active_query:
            # ERROR HANDLING: Ensure document is loaded and processed first
            if not st.session_state.processed or not st.session_state.index:
                st.warning("Please upload and process a document first.")
            else:
                # Add the user's question to session messages and show it
                st.session_state.messages.append({"role": "user", "content": active_query})
                
                # Temporarily rerun to display user bubble immediately
                # (Streamlit handles chat input reruns natively, but we ensure state is captured)
                
                # A. Retrieve similarity results
                with st.spinner("Searching document..."):
                    try:
                        # Load embedding model
                        model = load_embedding_model()
                        # Run similarity retrieval
                        retrieved = retrieve(
                            query=active_query,
                            index=st.session_state.index,
                            chunks=st.session_state.chunks,
                            model=model,
                            k=4
                        )
                    except Exception as e:
                        st.error(f"Failed to retrieve document context: {e}")
                        retrieved = []
                
                # B. Generate grounded LLM response
                if retrieved:
                    with st.spinner("Generating answer..."):
                        try:
                            # Extract raw chunk dicts
                            chunk_dicts = [item[0] for item in retrieved]
                            
                            # Call Generator module (gemini-2.5-flash)
                            answer, _ = generate_answer(
                                query=active_query,
                                retrieved_chunks=chunk_dicts,
                                api_key=api_key
                            )
                            
                            # Compute retrieval confidence score (average cosine similarity)
                            avg_similarity = sum(item[1] for item in retrieved) / len(retrieved)
                            confidence_pct = max(0.0, min(100.0, avg_similarity * 100.0))
                            
                            # Store the assistant response in message history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": retrieved,
                                "confidence": confidence_pct
                            })
                            st.rerun()
                            
                        except Exception as e:
                            # ERROR HANDLING: Show error box and offer retry advice
                            st.error(
                                f"Gemini API call failed: {e}\n\n"
                                "Please check your network connection, API key limits, or try asking the question again."
                            )
                else:
                    # Handle case where FAISS returned no index matches
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I couldn't find any relevant text segments in the uploaded documents to answer your question.",
                        "sources": []
                    })
                    st.rerun()
