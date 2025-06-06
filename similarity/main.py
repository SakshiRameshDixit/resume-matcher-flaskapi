from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
HF_TOKEN = os.getenv("HF_TOKEN")


app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your Hugging Face token (securely use env var in production)

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Model endpoints
BI_ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Input schema
class ResumeRequest(BaseModel):
    resume: dict
    jobDescription: dict

# Resume preprocessing
def preprocess_resume(resume_json):
    parts = [
        resume_json.get("summary", ""),
        "Skills: " + ", ".join(resume_json.get("skills", []))
    ]
    for exp in resume_json.get("experience", []):
        description = exp.get("description", "")
        if isinstance(description, list):
            description = " ".join(description)
        parts.append(f"{exp.get('title', '')} at {exp.get('company', '')}: {description}")
    return " ".join(parts)

# Utility to get bi-encoder embedding from HF API
def get_embedding(text: str):
    response = requests.post(
        f"https://api-inference.huggingface.co/embeddings/{BI_ENCODER_MODEL}",
        headers=HEADERS,
        json={"inputs": text}
    )
    return response.json()["embedding"]

# Utility to get cross-encoder score from HF API
def get_cross_score(text1: str, text2: str):
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{CROSS_ENCODER_MODEL}",
        headers=HEADERS,
        json={"inputs": [text1, text2]}
    )
    output = response.json()
    if isinstance(output, list) and isinstance(output[0], float):
        return output[0]
    return output.get("score", 0.5)  # fallback score

# Cosine similarity (manual)
def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot / (norm1 * norm2 + 1e-8)

# API endpoint
@app.post("/compare")
async def compare_similarity(req: ResumeRequest):
    resume_text = preprocess_resume(req.resume)
    job_text = req.jobDescription.get("jobDescription", "")

    # Get embeddings
    resume_embedding = get_embedding(resume_text)
    job_embedding = get_embedding(job_text)
    bi_score = cosine_similarity(resume_embedding, job_embedding)

    # Cross-encoder score
    raw_score = get_cross_score(resume_text, job_text)
    cross_score = 1 / (1 + pow(2.71828, -raw_score))  # Sigmoid manually

    # Boosted score
    final_score = (bi_score + cross_score) / 2
    boosted_score = 10 + (100 - 10) * final_score

    return {"similarityScore": round(boosted_score, 2)}
