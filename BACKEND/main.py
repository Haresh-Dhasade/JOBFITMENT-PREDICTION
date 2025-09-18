# backend/main.py
import os
import tempfile
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
from typing import Optional

# Optional parsers
from PyPDF2 import PdfReader
from docx import Document

app = FastAPI(title="Job Fitment AI Backend", version="1.0")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)
CLF_PATH = os.path.join(MODEL_DIR, "job_clf.joblib")
EMBED_NAME_PATH = os.path.join(MODEL_DIR, "embedder_name.joblib")

# Load classifier if available
clf = None
embedder = None
if os.path.exists(CLF_PATH):
    clf = joblib.load(CLF_PATH)
if os.path.exists(EMBED_NAME_PATH):
    embed_name = joblib.load(EMBED_NAME_PATH)
    embedder = SentenceTransformer(embed_name)
else:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fallback

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception:
        return ""

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    except Exception:
        return ""

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp()
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        tmp_path = os.path.join(tmp_dir, f"resume{ext}")
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        text = ""
        if ext in [".pdf"]:
            text = extract_text_from_pdf(tmp_path)
        elif ext in [".docx"]:
            text = extract_text_from_docx(tmp_path)
        else:
            # fallback: read bytes as utf-8
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        # simple skills extraction: find lines with commas or common skill keywords
        skills = []
        lower = text.lower()
        # naive skill keywords
        skill_keywords = ["python", "java", "c++", "sql", "aws", "docker", "kubernetes", "pandas", "tensorflow", "react", "django", "flask"]
        for kw in skill_keywords:
            if kw in lower:
                skills.append(kw)

        response = {"text": text[:20000], "skills": skills}
    except Exception as e:
        response = {"error": str(e)}
    finally:
        shutil.rmtree(tmp_dir)
    return JSONResponse(content=response)

class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3

@app.post("/predict_job")
async def predict_job(req: PredictRequest):
    if clf is None:
        return JSONResponse({"error": "Classifier model not found. Please train models first."}, status_code=400)
    text = req.text
    emb = embedder.encode([text], convert_to_numpy=True)
    pred = clf.predict(emb)[0]
    output = {"prediction": pred}
    # if probability available, include top-k probabilities
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(emb)[0]
        classes = list(clf.classes_)
        # top k
        topk = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[: req.top_k or 3]
        output["top_k"] = [{"label": l, "prob": float(p)} for l, p in topk]
    return JSONResponse(content=output)

@app.post("/build_resume")
async def build_resume(
    name: str = Form(...),
    email: str = Form(...),
    summary: str = Form(""),
    skills: str = Form(""),
    education: str = Form(""),
    experiences: str = Form("")
):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}_resume.pdf")

    try:
        width, height = letter
        c = canvas.Canvas(pdf_path, pagesize=letter)
        y = height - 60
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, y, name)
        y -= 28
        c.setFont("Helvetica", 11)
        c.drawString(50, y, email)
        y -= 24

        def write_section(title, content):
            nonlocal c, y
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, title)
            y -= 18
            c.setFont("Helvetica", 10)
            # wrap naive
            lines = []
            for chunk in str(content).split("\n"):
                while len(chunk) > 90:
                    lines.append(chunk[:90])
                    chunk = chunk[90:]
                lines.append(chunk)
            for ln in lines:
                if y < 60:
                    c.showPage()
                    y = height - 60
                c.drawString(60, y, ln)
                y -= 14
            y -= 8

        write_section("Summary", summary)
        write_section("Skills", skills)
        write_section("Education", education)
        write_section("Experience", experiences)
        c.save()
        return FileResponse(pdf_path, filename="resume.pdf", media_type="application/pdf")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # will be cleaned by server or left for download; not deleting immediately to allow download
        pass