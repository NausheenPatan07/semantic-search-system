from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# create API
app = FastAPI()

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load dataset
df = pd.read_csv("data/news_dataset.csv")

# load embeddings
embedding_matrix = np.load("data/embeddings.npy").astype("float32")

# create FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# request structure
class QueryRequest(BaseModel):
    query: str
    k: int = 5

# API endpoint
@app.post("/search")
def search(request: QueryRequest):

    query_embedding = model.encode([request.query]).astype("float32")

    distances, indices = index.search(query_embedding, request.k)

    results = []

    for i in indices[0]:
        results.append({
            "category": df.iloc[i]["category"],
            "text": df.iloc[i]["clean_text"][:200]
        })

    return {"results": results}