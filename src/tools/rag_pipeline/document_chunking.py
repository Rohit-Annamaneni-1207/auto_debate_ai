# import pytesseract
# from PIL import Image
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_extract_chunk_pdf(pdf_path:str, chunk_size:int=500, chunk_overlap:int=100):
    loader = PyPDFLoader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = loader.load_and_split(text_splitter)
    chunks = [text.page_content for text in texts]
    return chunks

def load_extract_chunk_txt(txt_path:str, chunk_size:int=500, chunk_overlap:int=100):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def load_chunk_document(file_path:str):
    if file_path.lower().endswith('.pdf'):
        chunks = load_extract_chunk_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        chunks = load_extract_chunk_txt(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or TXT file.")
    return chunks

# print(load_chunk_document("data/knowledge_base_1/documents/Leveraging_Content_and_Acoustic_Representations_for_Speech_Emotion_Recognition.pdf"))
# print(load_chunk_document("data/knowledge_base_1/documents/LAPSES.txt"))

if __name__ == "__main__":
    pdf_chunks = load_chunk_document("data/knowledge_base_1/documents/Leveraging_Content_and_Acoustic_Representations_for_Speech_Emotion_Recognition.pdf")
    print(f"PDF Chunks: {len(pdf_chunks)}")
    txt_chunks = load_chunk_document("data/knowledge_base_1/documents/LAPSES.txt")
    print(f"TXT Chunks: {len(txt_chunks)}")