import os
from dotenv import load_dotenv
from src.agents.orchestrator_agent import OrchestratorAgent

# Load environment variables
load_dotenv()

def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        # For demo purposes, we might need to mock or ask user, but let's assume it's there or fail gracefully
        return

    orchestrator = OrchestratorAgent()
    
    problem = "Find the number of rectangles that can be formed inside a fixed regular dodecagon (12-gon) where each side of the rectangle lies on either a side or a diagonal of the dodecagon."
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
