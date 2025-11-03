import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import json

INDEX_PATH = os.path.abspath(os.path.join("data", "knowledge_base"))
EMBEDDING_SUBPATH = os.path.join("embeddings", "index.faiss")
METADATA_SUBPATH = os.path.join("metadata", "metadata.json")

def load_index(path: str = INDEX_PATH):
    idx_path = os.path.join(path, EMBEDDING_SUBPATH)
    if os.path.exists(idx_path):
        index = faiss.read_index(idx_path)
        with open(os.path.join(path, METADATA_SUBPATH), 'r') as f:
            metadata = json.load(f)

        return index, metadata
    else:
        raise FileNotFoundError(f"No index found at {idx_path}")

def create_load_index(embedding_model: SentenceTransformer ,idx_path: str = INDEX_PATH):
    idx_embed_path = os.path.join(idx_path, EMBEDDING_SUBPATH)
    if os.path.exists(idx_embed_path):
        print(f"Index already exists at {idx_embed_path}")
        return load_index()

    else:
        embedding_dim = np.array(embedding_model.encode(["Hello world"]))
        print(embedding_dim.shape)
        embedding_dim = embedding_dim.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        faiss.write_index(index, idx_embed_path)

        metadata = []
        with open(os.path.join(idx_path, METADATA_SUBPATH), 'w') as f:
            json.dump(metadata, f)
            f.flush()
        return index, metadata
    
def add_embeddings(texts:list, embedding_model:SentenceTransformer, index: faiss.IndexFlatIP, path: str = INDEX_PATH):
    idx_path = os.path.join(path, EMBEDDING_SUBPATH)
    old_count = index.ntotal
    embeddings = embedding_model.encode(texts)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, idx_path)
    new_count = index.ntotal
    print(f"Added {new_count - old_count} embeddings to the index.")

    metadata_path = os.path.join(path, METADATA_SUBPATH)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    embeddings_list = embeddings.tolist()
    for i in range(len(texts)):
        metadata.append({"text": texts[i], "embedding": embeddings_list[i]})

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
        f.flush()

    return index, metadata

def search_index(query: str, embedding_model: SentenceTransformer, index: faiss.IndexFlatIP, top_k: int = 5, idx_path: str = INDEX_PATH):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), top_k)

    with open(os.path.join(idx_path, METADATA_SUBPATH), 'r') as f:
        metadata = json.load(f)

    print(f"Search results indices: {I}, distances: {D}")
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({"text": metadata[idx]['text'], "score": float(score)})
    # results = [{"text": metadata[i]["text"], "score": float(D[0][j])} for j, i in enumerate(I[0])]
    return results
    
def clear_index(path: str = INDEX_PATH):
    idx_path = os.path.join(path, EMBEDDING_SUBPATH)
    metadata_path = os.path.join(path, METADATA_SUBPATH)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
        print(f"Metadata at {metadata_path} has been deleted.")
    if os.path.exists(idx_path):
        os.remove(idx_path)
        print(f"Index at {idx_path} has been deleted.")
    else:
        print(f"No index found at {path} to delete.")



