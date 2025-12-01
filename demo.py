import os
from dotenv import load_dotenv
from src.agents.orchestrator_agent import OrchestratorAgent
from src.tools.rag_pipeline.faiss_utils import clear_index, search_index
from src.tools.rag_pipeline.process_file import process_file_add_to_index
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Load environment variables
load_dotenv()

def main():
    # Check for API key
    clear_index(idx_num=1)
    clear_index(idx_num=2)
    process_file_add_to_index(embedding_model=EMBEDDING_MODEL, idx_num=1)
    process_file_add_to_index(embedding_model=EMBEDDING_MODEL, idx_num=2)
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        # For demo purposes, we might need to mock or ask user, but let's assume it's there or fail gracefully
        return

    orchestrator = OrchestratorAgent()
    
    # problem = "Find the number of rectangles that can be formed inside a fixed regular dodecagon (12-gon) where each side of the rectangle lies on either a side or a diagonal of the dodecagon."

    problem = "What is an acoustic representation?"

    search_results = search_index(problem, EMBEDDING_MODEL, top_k=5, idx_num=1)
    search_results.extend(search_index(problem, EMBEDDING_MODEL, top_k=5, idx_num=2))

    search_results = sorted(search_results, key=lambda x: x['score'], reverse=True)

    num_retrieved = 5
    retrieved_text = "\n".join([result['text'] for result in search_results[:num_retrieved]])
    problem = problem + "\n\nRetrieved Text for context: " + retrieved_text

    print(f"Problem: {problem}\n")
    
    print("Running Orchestrator...")
    result = orchestrator.invoke({"problem": problem})
    
    print("\n--- Final Answer ---")
    print(result["final_answer"])
    
    print("\n--- Debug Info ---")
    print(f"Worker 1 Response: {result.get('worker1_response')}...")
    print(f"Worker 2 Response: {result.get('worker2_response')}...")
    print(f"Worker 1 Critique: {result.get('worker1_critique')}...")
    print(f"Worker 2 Critique: {result.get('worker2_critique')}...")

if __name__ == "__main__":
    main()
