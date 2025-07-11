from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import docx

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
index = None
documents = []

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

@app.route("/upload", methods=["POST"])
def upload_file():
    global index, documents
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if filename.endswith('.pdf'):
        text = extract_text_from_pdf(filepath)
    elif filename.endswith('.docx'):
        text = extract_text_from_docx(filepath)
    elif filename.endswith('.txt'):
        text = extract_text_from_txt(filepath)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    documents = chunks
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return jsonify({"message": "File uploaded and indexed successfully"}), 200

@app.route("/ask", methods=["POST"])
def ask_question():
    global index, documents
    data = request.get_json()
    question = data.get("question")

    if index is None:
        return jsonify({"error": "No notes uploaded"}), 400

    question_embedding = model.encode([question])
    D, I = index.search(np.array(question_embedding), k=1)
    answer = documents[I[0][0]]
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
