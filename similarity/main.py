from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load models
bi_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model
class ResumeRequest(BaseModel):
    resume: dict
    jobDescription: dict

# Resume preprocessing function
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

# API endpoint
@app.post("/compare")
async def compare_similarity(req: ResumeRequest):
    resume_text = preprocess_resume(req.resume)
    job_text = req.jobDescription.get("jobDescription", "")

    # Bi-encoder similarity
    resume_embedding = bi_encoder.encode(resume_text, convert_to_tensor=True)
    job_embedding = bi_encoder.encode(job_text, convert_to_tensor=True)
    bi_score = 1 - cosine(resume_embedding.cpu(), job_embedding.cpu())

    # Cross-encoder similarity
    raw_score = cross_encoder.predict([(resume_text, job_text)])[0]
    cross_score = float(F.sigmoid(torch.tensor(raw_score)))

    # Final boosted score
    final_score = (bi_score + cross_score) / 2
    boosted_score = 10 + (100 - 10) * final_score

    # Return score as native Python float
    return {"similarityScore": float(round(boosted_score, 2))}
