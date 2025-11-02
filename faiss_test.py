import src.tools.rag_pipeline.faiss_main as faiss_main
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)

idx, _ = faiss_main.create_load_index(MODEL)

print(f"Index has {idx.ntotal} embeddings.")

chunks = [
    "This is an examination",
    "This is a cat.",
    "This is a finite closed communicating class.",
    "This is the president of india."
]

idx, _ = faiss_main.add_embeddings(chunks, MODEL, idx)
print(f"Index now has {idx.ntotal} embeddings.")

query = "Who is into politics?"

search_results = faiss_main.search_index(query, MODEL, idx, 5)

for result in search_results:
    print(f"Text: {result['text']}, Score: {result['score']}")

faiss_main.clear_index()
