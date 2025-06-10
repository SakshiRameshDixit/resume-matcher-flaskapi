from sentence_transformers import SentenceTransformer, CrossEncoder
import os

# Create model directories
os.makedirs('./models/all-mpnet-base-v2', exist_ok=True)
os.makedirs('./models/ms-marco-MiniLM-L-6-v2', exist_ok=True)

# Download and save bi-encoder
bi_encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
bi_encoder.save('./models/all-mpnet-base-v2')

# Download and save cross-encoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
cross_encoder.save_pretrained('./models/ms-marco-MiniLM-L-6-v2')
