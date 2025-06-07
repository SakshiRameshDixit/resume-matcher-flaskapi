from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv

print("Loading .env file...")
load_dotenv()  # Assumes .env is in current directory now

HF_TOKEN = os.getenv("HF_TOKEN")
print(f"HF_TOKEN loaded: {HF_TOKEN is not None and HF_TOKEN != ''}")

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API headers
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ✅ Updated models
BI_ENCODER_MODEL = "thenlper/gte-large"  # Supports inference API
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Works fine

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
    result = " ".join(parts)
    print(f"Preprocessed resume text: {result[:200]}...")
    return result

# ✅ Embedding fetch using supported endpoint
def get_embedding(text: str):
    print(f"Requesting embedding for text (first 100 chars): {text[:100]}...")
    response = requests.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{BI_ENCODER_MODEL}",
        headers=HEADERS,
        json={"inputs": text}
    )
    if response.status_code != 200:
        print(f"Embedding API error: {response.status_code} - {response.text}")
        return []
    embedding = response.json()
    if isinstance(embedding, list) and isinstance(embedding[0], list):
        print(f"Received embedding of length: {len(embedding[0])}")
        return embedding[0]
    print("Unexpected embedding format")
    return []

# Cross-encoder score via API
def get_cross_score(text1: str, text2: str):
    print(f"Requesting cross-encoder score...")
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{CROSS_ENCODER_MODEL}",
        headers=HEADERS,
        json={"inputs": [text1, text2]}
    )
    if response.status_code != 200:
        print(f"Cross-encoder API error: {response.status_code} - {response.text}")
        return 0.5
    output = response.json()
    print(f"Cross-encoder raw output: {output}")
    if isinstance(output, list) and isinstance(output[0], float):
        return output[0]
    return output.get("score", 0.5)

# Cosine similarity
def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        print("Invalid vectors for cosine similarity")
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    similarity = dot / (norm1 * norm2 + 1e-8)
    print(f"Cosine similarity: {similarity}")
    return similarity

# Endpoint
@app.post("/compare")
async def compare_similarity(req: ResumeRequest):
    print("Received /compare request")
    resume_text = preprocess_resume(req.resume)
    job_text = req.jobDescription.get("jobDescription", "")
    print(f"Job description text (first 200 chars): {job_text[:200]}")

    # Embeddings
    resume_embedding = get_embedding(resume_text)
    job_embedding = get_embedding(job_text)
    if not resume_embedding or not job_embedding:
        return {"error": "Failed to get embeddings"}

    bi_score = cosine_similarity(resume_embedding, job_embedding)

    # Cross-encoder
    raw_score = get_cross_score(resume_text, job_text)
    cross_score = 1 / (1 + pow(2.71828, -raw_score))  # Sigmoid
    print(f"Raw cross-encoder score: {raw_score}, sigmoid: {cross_score}")

    # Final score
    final_score = (bi_score + cross_score) / 2
    boosted_score = 10 + (100 - 10) * final_score
    print(f"Final boosted similarity score: {boosted_score}")

    return {"similarityScore": round(boosted_score, 2)}
