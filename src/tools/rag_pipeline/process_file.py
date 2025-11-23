from document_chunking import load_chunk_document
from faiss_utils import create_load_index, add_embeddings, clear_index, search_index
from sentence_transformers import SentenceTransformer
import os
import numpy as np

def process_file_add_to_index(embedding_model: SentenceTransformer, idx_num: int = 1):
    # Load and chunk the document
    if idx_num == 1:
        knowledge_base_path = os.path.abspath(os.path.join("data", "knowledge_base_1"))
    else:
        knowledge_base_path = os.path.abspath(os.path.join("data", "knowledge_base_2"))

    file_dir = os.path.join(knowledge_base_path, "documents")

    for file_name in os.listdir(file_dir):
        # print(file_name)
        if file_name.endswith('.pdf') or file_name.endswith('.txt'):
            file_path = os.path.join(file_dir, file_name)

            chunks = load_chunk_document(file_path)
            print(f"Loaded and chunked document: {file_path} into {len(chunks)} chunks.")

            index, metadata = create_load_index(embedding_model, idx_num=idx_num)
            print("FAISS index is ready.")

            # print(index)

            add_embeddings(chunks, embedding_model, idx_num=idx_num)
            print("Embeddings added to the index.")
            


if __name__ == "__main__":
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    file_path_pdf = "data/knowledge_base_1/documents/Leveraging_Content_and_Acoustic_Representations_for_Speech_Emotion_Recognition.pdf"
    file_path_txt = "data/knowledge_base_1/documents/LAPSES.txt"

    print("Processing PDF file...")
    process_file_add_to_index(embedding_model, idx_num=1)

    print("Processing TXT file...")
    process_file_add_to_index(embedding_model, idx_num=2)

    res = search_index("What is speech emotion recognition?", embedding_model, top_k=3, idx_num=1)

    print("Search results from index 1:")
    print(res[0].keys())
    for r in res:
        print(r)

    clear_index(idx_num=1)
    clear_index(idx_num=2)