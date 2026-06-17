from flask import Flask, request, jsonify, render_template, session
import os
import uuid
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Document processing
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import pytesseract

# Vector store
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

# ── Load .env ──────────────────────────────────────────────
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY", "")

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vector_stores = {}


# ── Text Splitter ─────────────────────────────────────────
def split_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


# ── Extractors ────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_image(path):
    return pytesseract.image_to_string(Image.open(path))

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text(filepath, filename):
    ext = filename.rsplit(".", 1)[1].lower()
    return {
        "pdf":  extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "png":  extract_text_from_image,
        "jpg":  extract_text_from_image,
        "jpeg": extract_text_from_image,
        "txt":  extract_text_from_txt,
    }.get(ext, lambda p: "")(filepath)


# ── Vector Store ──────────────────────────────────────────
def build_vector_store(texts):
    chunks = []
    for text in texts:
        chunks.extend(split_text(text))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix, chunks

def retrieve_context(query, vectorizer, matrix, chunks, top_k=4):
    scores = cosine_similarity(vectorizer.transform([query]), matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx if scores[i] > 0.01]


# ── HuggingFace Online API ────────────────────────────────
def query_huggingface(context, question):
    if not context:
        return "No relevant information found in the documents."

    context_text = " ".join(context[:3])

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    # Single endpoint — model name goes in payload
    url = "https://router.huggingface.co/v1/chat/completions"

    models = [
        "google/gemma-2-2b-it",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct:cerebras",
        "meta-llama/Llama-3.3-70B-Instruct:sambanova",
    ]

    last_error = None

    for model in models:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions based only on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context: {context_text}\n\nQuestion: {question}"
                }
            ],
            "max_tokens": 200,
            "temperature": 0.3
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()

            elif response.status_code == 404:
                last_error = f"Model {model} not found, trying next..."
                continue

            elif response.status_code == 503:
                return "⏳ Model is loading, please retry in ~20 seconds."

            elif response.status_code == 401:
                raise ValueError("Invalid API key.")

            elif response.status_code == 429:
                raise ValueError("Rate limit hit.")

            else:
                last_error = f"HF API error {response.status_code}: {response.text}"
                continue

        except ValueError:
            raise
        except Exception as e:
            last_error = str(e)
            continue

    raise Exception(f"All models failed. Last error: {last_error}")


# ── Routes ────────────────────────────────────────────────
@app.route("/")
def index():
    session.setdefault("session_id", str(uuid.uuid4()))
    return render_template("index.html", key_set=bool(HF_API_KEY))


@app.route("/upload", methods=["POST"])
def upload():
    session_id = session.setdefault("session_id", str(uuid.uuid4()))
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    texts, names = [], []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
            file.save(path)
            text = extract_text(path, filename)
            if text.strip():
                texts.append(text)
                names.append(filename)

    if not texts:
        return jsonify({"error": "No extractable text found"}), 400

    vectorizer, matrix, chunks = build_vector_store(texts)
    vector_stores[session_id] = {"vectorizer": vectorizer, "matrix": matrix,
                                  "chunks": chunks, "files": names}

    return jsonify({"success": True, "files": names, "chunks": len(chunks),
                    "message": f"Processed {len(names)} file(s) into {len(chunks)} chunks"})


@app.route("/query", methods=["POST"])
def query():
    session_id = session.get("session_id")
    if not session_id or session_id not in vector_stores:
        return jsonify({"error": "No documents uploaded. Please upload files first."}), 400
    if not HF_API_KEY:
        return jsonify({"error": "HF_API_KEY not set in .env file"}), 500

    data = request.get_json()
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    store = vector_stores[session_id]
    ctx = retrieve_context(question, store["vectorizer"], store["matrix"], store["chunks"])
    if not ctx:
        return jsonify({"error": "No relevant content found in documents"}), 404

    try:
        answer = query_huggingface(ctx, question)
        return jsonify({"answer": answer, "sources": ctx[:2], "chunks_used": len(ctx)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        err = str(e)
        if "401" in err or "Unauthorized" in err:
            return jsonify({"error": "❌ Invalid API key. Check your .env file."}), 401
        if "429" in err:
            return jsonify({"error": "⏳ Rate limit hit. Wait a moment and retry."}), 429
        if "403" in err:
            return jsonify({"error": "❌ Access denied. Enable Inference API on your HF token."}), 403
        return jsonify({"error": f"⚠ API error: {err}"}), 500


@app.route("/clear", methods=["POST"])
def clear():
    sid = session.get("session_id")
    if sid and sid in vector_stores:
        del vector_stores[sid]
    return jsonify({"success": True})


@app.route("/status")
def status():
    return jsonify({"key_configured": bool(HF_API_KEY)})


if __name__ == "__main__":
    if not HF_API_KEY:
        print("\n⚠️  WARNING: HF_API_KEY is not set in .env!\n")
    else:
        print(f"\n✅  HuggingFace key loaded (***{HF_API_KEY[-4:]})\n")
    app.run(debug=True, port=5000)