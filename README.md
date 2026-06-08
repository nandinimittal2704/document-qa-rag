# Scratch-Built RAG Document Q&A System

A lightweight, fully explainable Retrieval-Augmented Generation (RAG) system built in Python from scratch. This project contains zero reliance on high-level RAG frameworks like LangChain or LlamaIndex, allowing you to inspect, understand, and explain every line of code.

---

### What is RAG? (In 3 Sentences)
Retrieval-Augmented Generation (RAG) is a technique that enhances the accuracy and reliability of Large Language Models by fetching relevant facts from an external document library. When a user asks a question, the system searches a vector database to find the most semantically related passages and injects them directly into the LLM's prompt. This grounds the model's response in verified factual context, preventing hallucinations and restricting the model from using outdated or generic training knowledge.

---

## Project Structure & File Map

```text
rag-qa/
├── .env                  # Stores the Gemini API Key
├── requirements.txt      # Pinned library dependencies
├── chunker.py            # Recursive character splitting & PDF text parsing
├── embedder.py           # Text embedding generation & FAISS flat index builder
├── retriever.py          # Query embedding & top-k similarity search
├── generator.py          # Grounded prompt building & Gemini API integration
├── app.py                # Multi-document Streamlit UI with advanced analytics
└── README.md             # Project documentation & interview study guide
```

### File Explanations:
1. **`chunker.py`**: Parses plaintext and PDF documents page-by-page using PyMuPDF (`fitz`). It splits long texts using a recursive separator hierarchy (`\n\n` -> `\n` -> sentence -> word -> char) to preserve paragraphs and sentence structures intact.
2. **`embedder.py`**: Initializes the SentenceTransformer model (`all-MiniLM-L6-v2`) to compute 384-dimensional dense vectors. It normalizes vectors to unit length and indexes them using FAISS (`IndexFlatIP`) for Inner Product calculation, saving both the index and JSON metadata list to disk.
3. **`retriever.py`**: Converts the search query using the identical model and normalizes it to execute Cosine Similarity search over the FAISS index, retrieving the top `k=4` chunks.
4. **`generator.py`**: Initializes the unified `google-genai` SDK and constructs a grounded prompt structure, forcing `gemini-2.5-flash` to answer using *only* the context provided.
5. **`app.py`**: Controls the Streamlit dashboard layout, managing session states, caching models, displaying similarity percentage badges, rendering confidence scores, and exporting conversations.

---

## How to Run Locally

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.

### Step 1: Clone and Navigate to Project
```bash
# Navigate to the project directory
cd "rag-qa"
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment (Windows Powershell)
.\venv\Scripts\Activate.ps1

# Activate virtual environment (Mac/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Gemini API Key
The `.env` file is pre-configured with the API Key inside the folder. If you wish to rotate it, open `rag-qa/.env` and replace the value:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 5: Start the App
```bash
streamlit run app.py
```

---

## How to Deploy to Streamlit Cloud

1. **Push Code to GitHub**: Put your project files (excluding `venv/` and `.env`) into a public or private GitHub repository.
2. **Access Streamlit Community Cloud**: Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Deploy App**:
   - Choose your repository, branch, and set the main file path to `rag-qa/app.py`.
4. **Add Secret Environment Variables**:
   - In the deployment settings, locate the **Secrets** section.
   - Enter your Gemini API key:
     ```toml
     GEMINI_API_KEY = "your_gemini_api_key"
     ```
   - Click **Save** and deploy. Streamlit will install the requirements from `requirements.txt` and launch the web server.

---

## Screenshot Placeholder
![App Interface Mockup](file:///c:/Users/NANDINI/Desktop/Da-projects/rag%20document%20Q&A/rag-qa/screenshot_placeholder.png)
*(Run the application locally to view the beautiful layout: Left Column controls, Advanced settings, Chunk previews, Right Column chat bubbles, source scoring, and export widgets!)*

---

## What I Learned Building This (Interview Cheat Sheet)

Here are the 5 core architectural learnings gained from constructing this RAG application from scratch—highly useful for technical discussions and engineering interviews:

1. **Recursive Chunking Trade-Offs**:
   Using simple fixed-character chunking (e.g. slicing every 500 characters) frequently cuts sentences or words in half, damaging semantic context. Recursive character chunking mimics human readability by splitting on a hierarchy of logical boundaries (`\n\n` for paragraphs, `\n` for sections, `. ` for sentences, and ` ` for words). This keeps related arguments together in a single chunk, maximizing retrieval quality.

2. **Vector Space Model Matching**:
   The query vector and the document vectors must be embedded using the *exact same* model. If they are embedded using different models, their dimensions, feature weighting, and orientation will not align. Even a 384-dimensional vector from one model cannot be queried against a 384-dimensional vector of another, as they map completely different mathematical coordinate systems.

3. **Cosine Similarity vs. Euclidean Distance**:
   We used `IndexFlatIP` (Inner Product) rather than `IndexFlatL2` (Euclidean distance). L2 measures absolute straight-line distance, which is highly sensitive to text length (magnitude). By L2-normalizing our vectors to unit length prior to database insertion, the Inner Product calculation becomes mathematically identical to Cosine Similarity. Cosine similarity evaluates the angle between vectors, capturing semantic orientation and topic relevance independently of document length.

4. **Context Grounding Prompts (LLM Guardrails)**:
   Generic LLM inference draws on training sets, causing hallucinations when asked about specific company documents. To enforce grounding, we pass the retrieved text segments to the LLM and set a rigid System Instruction: `"Answer using ONLY the context below. If the answer is not in the context, say 'I don't have that information in the provided document.' Do not use any prior knowledge."` This overrides the model's pre-trained assumptions, keeping answers 100% verified.

5. **Retrieval Confidence Scoring**:
   Cosine similarity scores represent how closely the user's query aligns with the retrieved chunks. By taking the average similarity score of the top-4 chunks, we can display a "Retrieval Confidence" metric to the user. When the average similarity falls below 50%, it acts as a warning sign indicating that the uploaded documents likely do not contain the answer, signaling that the LLM response might be an empty default or low-assurance text.
