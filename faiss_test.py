import tools.rag_pipeline.faiss_utils as faiss_utils
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)
IDX_PATH_1 = os.path.abspath(os.path.join("data", "knowledge_base_1"))
idx, _ = faiss_utils.create_load_index(MODEL, IDX_PATH_1)

print(f"Index has {idx.ntotal} embeddings.")

chunks = [
    "This is an examination",
    "This is a cat.",
    "This is a finite closed communicating class.",
    "This is the president of india.",
    "This is an activist."
]

idx, _ = faiss_utils.add_embeddings(chunks, MODEL, idx, IDX_PATH_1)
print(f"Index now has {idx.ntotal} embeddings.")

query = "Who is into politics?"

search_results = faiss_utils.search_index(query, MODEL, idx, 5, IDX_PATH_1)

for result in search_results:
    print(f"Text: {result['text']}, Score: {result['score']}")

# faiss_main.clear_index(IDX_PATH_1)
