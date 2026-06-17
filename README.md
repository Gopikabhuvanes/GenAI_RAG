# 🧠 DocMind — LLM RAG Application

A fully local RAG (Retrieval-Augmented Generation) Flask app that lets you upload **PDF, Image, DOCX, and TXT** files and query them using **Mistral 7B** via the free HuggingFace Inference API.

---

## 📁 Project Structure

```
rag_app/
├── app.py                  # Flask backend + RAG logic
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Full-stack UI
├── uploads/                # Temp file storage (auto-created)
└── vector_stores/          # (reserved for persistence)
```

---

## 🚀 Quick Setup

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (for image support)
- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

### 4. Get a FREE HuggingFace API Key
1. Go to https://huggingface.co/settings/tokens
2. Create a token with **Read** access (free tier works!)

### 5. Run the app
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🧠 How It Works (RAG Pipeline)

```
Upload Files
    │
    ▼
Text Extraction (PDF→PyMuPDF, DOCX→python-docx, IMG→OCR)
    │
    ▼
Chunking (RecursiveCharacterTextSplitter, 500 chars, 50 overlap)
    │
    ▼
TF-IDF Vectorization (scikit-learn, in-memory)
    │
    ▼
Query → Cosine Similarity Retrieval → Top-K chunks
    │
    ▼
Prompt = Context + Question → Mistral 7B (HuggingFace API)
    │
    ▼
Answer displayed in Chat UI
```

---

## ✅ Supported File Types
| Type | Library | Notes |
|------|---------|-------|
| `.pdf` | PyMuPDF | Text-based PDFs |
| `.docx` | python-docx | Word documents |
| `.png/.jpg/.jpeg` | Pillow + Tesseract | OCR extraction |
| `.txt` | Built-in | Plain text |

---

## 💡 Tips
- **Free tier**: HuggingFace free inference API has rate limits. For heavy use, upgrade your plan.
- **Image quality**: Better image resolution = better OCR accuracy.
- **Multiple files**: Upload multiple files at once — they're all indexed together.
- **Session-based**: Each browser session has its own isolated vector store.

---

## 🔧 Configuration

Edit `app.py` to change:
- `chunk_size` / `chunk_overlap` — text splitting settings
- `top_k` — number of retrieved chunks per query
- Model: replace `mistralai/Mistral-7B-Instruct-v0.3` with any HuggingFace model
- `max_features` in TF-IDF vectorizer

---

## 📦 Key Dependencies
- **Flask** — web framework
- **HuggingFace Hub** — free LLM inference
- **PyMuPDF** — PDF text extraction
- **python-docx** — Word document parsing
- **Pytesseract** — OCR for images
- **LangChain** — text splitting utilities
- **scikit-learn** — TF-IDF vectorization + cosine similarity
