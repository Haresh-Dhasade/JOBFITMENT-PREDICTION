# frontend/app.py
import streamlit as st
import requests

st.set_page_config(page_title="Job Fitment AI", layout="wide")
st.title("Job Fitment AI — Resume, Interview & Predict")

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://backend:8000")  # default fallback

with st.sidebar:
    st.header("Navigation")
    mode = st.radio("Choose", ["Upload Resume", "Manual Skills", "Interview & Predict", "Build Resume"])


# ---------- Helper Function ----------
def safe_post(url, files=None, json=None, data=None):
    try:
        if files:
            return requests.post(url, files=files, timeout=60)
        if json:
            return requests.post(url, json=json, timeout=60)
        return requests.post(url, data=data, timeout=60)
    except Exception:
        return None


# ---------- Upload Resume ----------
if mode == "Upload Resume":
    st.subheader("Upload a resume (pdf/docx/txt)")
    uploaded = st.file_uploader("Upload file", type=["pdf", "docx", "txt"])
    if uploaded:
        files = {"file": (uploaded.name, uploaded.getvalue())}
        resp = safe_post(f"{BACKEND_URL}/upload_resume", files=files)
        if resp is None:
            st.error("Could not reach backend. Check BACKEND_URL.")
        elif resp.status_code != 200:
            st.error("Parsing failed: " + resp.text)
        else:
            parsed = resp.json()
            st.success("Parsed resume (truncated)")
            st.text_area("Extracted text (first 10k chars)", value=parsed.get("text", "")[:10000], height=300)
            skills = parsed.get("skills", [])
            st.write("Detected skills:", skills)
            chosen = st.multiselect("Choose skills for prediction", options=skills, default=skills[:5])
            if st.button("Predict from extracted content"):
                text = " ".join([parsed.get("text",""), " ".join(chosen)])
                pred = safe_post(f"{BACKEND_URL}/predict_job", json={"text": text})
                if pred is None:
                    st.error("Could not call predict endpoint")
                else:
                    st.json(pred.json())


# ---------- Manual Skills ----------
elif mode == "Manual Skills":
    st.subheader("Paste resume or skills to predict")
    text = st.text_area("Paste text here", height=300)
    if st.button("Predict"):
        resp = safe_post(f"{BACKEND_URL}/predict_job", json={"text": text})
        if resp is None:
            st.error("Error calling backend.")
        else:
            st.json(resp.json())


# ---------- Interview & Predict ----------
elif mode == "Interview & Predict":
    st.subheader("Interview Questions")
    questions = [
        "Tell me about yourself.",
        "What are your strengths and weaknesses?",
        "Why do you want to work in this role?",
        "Describe a challenging project you worked on.",
        "Where do you see yourself in 5 years?"
    ]

    answers = {}
    for i, q in enumerate(questions, 1):
        st.markdown(f"**Q{i}. {q}**")
        answers[q] = st.text_area(f"Your Answer to Q{i}", key=f"answer_{i}")

    if st.button("Predict from Answers"):
        combined_text = " ".join(answers.values())
        resp = safe_post(f"{BACKEND_URL}/predict_job", json={"text": combined_text})
        if resp:
            st.json(resp.json())
        else:
            st.error("Backend unreachable for prediction.")

    st.subheader("Record a short audio (or upload) to analyze emotion & transcribe")
    st.info("If your browser recorder is not available, upload a WAV/MP3 file.")
    audio_file = st.file_uploader("Upload audio file (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])
    if audio_file:
        files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
        resp = safe_post(f"{BACKEND_URL}/upload_audio", files=files)
        if resp is None:
            st.error("Backend unreachable")
        elif resp.status_code != 200:
            st.error("Audio analysis failed: " + resp.text)
        else:
            result = resp.json()
            st.write("Transcription:")
            st.write(result.get("transcription"))
            st.write("Emotion analysis (raw):")
            st.json(result.get("emotion"))
            if st.button("Predict using transcription"):
                txt = result.get("transcription", "")
                pred = safe_post(f"{BACKEND_URL}/predict_job", json={"text": txt})
                if pred:
                    st.json(pred.json())


# ---------- Build Resume ----------
elif mode == "Build Resume":
    st.subheader("Fill form and build a downloadable PDF resume")
    name = st.text_input("Name")
    email = st.text_input("Email")
    summary = st.text_area("Summary", height=120)
    skills = st.text_area("Skills (comma separated)")
    education = st.text_area("Education")
    experience = st.text_area("Experience")
    if st.button("Build & download"):
        data = {
            "name": name,
            "email": email,
            "summary": summary,
            "skills": skills,
            "education": education,
            "experiences": experience
        }
        resp = safe_post(f"{BACKEND_URL}/build_resume", data=data)
        if resp is None:
            st.error("Backend unreachable")
        elif resp.status_code != 200:
            st.error("Error building resume: " + resp.text)
        else:
            st.success("Resume built")
            st.download_button("Download PDF", data=resp.content, file_name="resume.pdf", mime="application/pdf")