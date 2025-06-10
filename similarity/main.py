# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# import requests
# import os
# from dotenv import load_dotenv

# print("Loading .env file...")
# load_dotenv()  # Load .env from current directory

# HF_TOKEN = os.getenv("HF_TOKEN")
# print(f"HF_TOKEN loaded: {HF_TOKEN is not None and HF_TOKEN != ''}")

# app = FastAPI()

# # Allow CORS for all origins (adjust for production)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Set Hugging Face API authorization header
# HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# # Use a model compatible with embeddings API
# BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# class ResumeRequest(BaseModel):
#     resume: dict
#     jobDescription: dict

# def preprocess_resume(resume_json):
#     parts = [
#         resume_json.get("summary", ""),
#         "Skills: " + ", ".join(resume_json.get("skills", []))
#     ]
#     for exp in resume_json.get("experience", []):
#         description = exp.get("description", "")
#         if isinstance(description, list):
#             description = " ".join(description)
#         parts.append(f"{exp.get('title', '')} at {exp.get('company', '')}: {description}")
#     result = " ".join(parts)
#     print(f"Preprocessed resume text: {result[:200]}...")
#     return result

# def get_embedding(text: str):
#     print(f"Requesting embedding for text (first 100 chars): {text[:100]}...")
#     response = requests.post(
#         "https://api-inference.huggingface.co/pipeline/feature-extraction",
#         headers=HEADERS,
#         json={
#             "inputs": text,
#             "model": BI_ENCODER_MODEL
#         }
#     )
#     if response.status_code != 200:
#         print(f"Embedding API error: {response.status_code} - {response.text}")
#         return []
#     embedding = response.json()
#     if isinstance(embedding, list) and len(embedding) > 0:
#         token_embeddings = embedding[0]
#         avg_embedding = [sum(col) / len(col) for col in zip(*token_embeddings)]
#         print(f"Received embedding vector length: {len(avg_embedding)}")
#         return avg_embedding
#     print("Unexpected embedding format:", embedding)
#     return []

# def get_cross_score(text1: str, text2: str):
#     print(f"Requesting cross-encoder score...")
#     response = requests.post(
#         f"https://api-inference.huggingface.co/models/{CROSS_ENCODER_MODEL}",
#         headers=HEADERS,
#         json={"inputs": [text1, text2]}
#     )
#     if response.status_code != 200:
#         print(f"Cross-encoder API error: {response.status_code} - {response.text}")
#         return 0.5
#     output = response.json()
#     print(f"Cross-encoder raw output: {output}")
#     if isinstance(output, list) and isinstance(output[0], float):
#         return output[0]
#     return output.get("score", 0.5)  # fallback score

# def cosine_similarity(vec1, vec2):
#     if not vec1 or not vec2 or len(vec1) != len(vec2):
#         print("Invalid vectors for cosine similarity")
#         return 0.0
#     dot = sum(a * b for a, b in zip(vec1, vec2))
#     norm1 = sum(a * a for a in vec1) ** 0.5
#     norm2 = sum(b * b for b in vec2) ** 0.5
#     similarity = dot / (norm1 * norm2 + 1e-8)
#     print(f"Cosine similarity: {similarity}")
#     return similarity

# @app.post("/compare")
# async def compare_similarity(req: ResumeRequest):
#     print("Received /compare request")
#     resume_text = preprocess_resume(req.resume)
#     job_text = req.jobDescription.get("jobDescription", "")
#     print(f"Job description text (first 200 chars): {job_text[:200]}")

#     resume_embedding = get_embedding(resume_text)
#     job_embedding = get_embedding(job_text)

#     if not resume_embedding or not job_embedding:
#         return {"error": "Failed to get embeddings. Check if the model supports feature-extraction API."}

#     bi_score = cosine_similarity(resume_embedding, job_embedding)

#     raw_score = get_cross_score(resume_text, job_text)
#     cross_score = 1 / (1 + pow(2.71828, -raw_score))  # manual sigmoid
#     print(f"Raw cross-encoder score: {raw_score}, sigmoid: {cross_score}")

#     final_score = (bi_score + cross_score) / 2
#     boosted_score = 10 + (100 - 10) * final_score
#     print(f"Final boosted similarity score: {boosted_score}")

#     return {"similarityScore": round(boosted_score, 2)}
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