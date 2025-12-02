from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from src.tools.rag_pipeline.faiss_utils import search_index
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()


class DebateState(TypedDict):
    topic: str
    current_round: int
    max_rounds: int
    proponent_arguments: List[str]
    opponent_arguments: List[str]
    proponent_last_argument: str
    opponent_last_argument: str
    proponent_context: str  # Retrieved RAG context for proponent (index 1)
    opponent_context: str   # Retrieved RAG context for opponent (index 2)
    debate_history: List[dict]
    final_summary: str


class DebateAgent:
    """Individual debate agent that can argue for or against a topic."""
    
    def __init__(self, stance: str, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """
        Initialize a debate agent.
        
        Args:
            stance: Either "for" or "against"
            model_name: The LLM model to use
        """
        self.stance = stance
        base_url = os.getenv("OPENAI_API_BASE")
        self.llm = ChatOpenAI(model=model_name, temperature=0.8, base_url=base_url)
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow for this agent."""
        workflow = StateGraph(dict)
        
        workflow.add_node("generate_argument", self.generate_argument)
        workflow.set_entry_point("generate_argument")
        workflow.add_edge("generate_argument", END)
        
        return workflow.compile()
    
    def _construct_system_prompt(self, topic: str) -> str:
        """Create system prompt based on stance."""
        if self.stance == "for":
            return f"""You are a skilled debater arguing IN FAVOR of: "{topic}"

Your role:
- Present strong, logical arguments supporting this position
- Use evidence, examples, and reasoning
- Respond to opposing arguments with counterpoints
- Be persuasive and articulate while remaining respectful
- Build upon your previous arguments
- Keep responses concise (2-3 paragraphs) but impactful"""
        else:
            return f"""You are a skilled debater arguing AGAINST: "{topic}"

Your role:
- Present strong, logical arguments opposing this position
- Use evidence, examples, and reasoning
- Respond to supporting arguments with counterpoints
- Be persuasive and articulate while remaining respectful
- Build upon your previous arguments
- Keep responses concise (2-3 paragraphs) but impactful"""
    
    def generate_argument(self, state: dict):
        """Generate an argument based on current debate state."""
        topic = state["topic"]
        round_num = state["round_num"]
        opponent_argument = state.get("opponent_argument", None)
        previous_arguments = state.get("previous_arguments", [])
        retrieved_context = state.get("retrieved_context", "")  # Pre-retrieved context from initial RAG search
        
        # Build context from previous arguments
        context = ""
        if previous_arguments:
            context = "\n\nYour previous arguments:\n" + "\n".join(
                [f"Round {i+1}: {arg}" for i, arg in enumerate(previous_arguments)]
            )
        
        # Create prompt based on round
        if round_num == 1 and not opponent_argument:
            prompt = f"This is Round {round_num}. Present your opening argument.{context}"
        elif opponent_argument:
            prompt = f"""This is Round {round_num}.

Your opponent just argued:
{opponent_argument}

Please respond with your counterargument.{context}"""
        else:
            prompt = f"This is Round {round_num}. Continue building your case.{context}"
        
        # Add retrieved context to prompt (only if available)
        if retrieved_context:
            full_prompt = f"""{prompt}

Retrieved knowledge from your knowledge base:
{retrieved_context}

Use the retrieved information to support your arguments where relevant."""
        else:
            full_prompt = prompt
        
        # Create messages with system prompt
        system_prompt = self._construct_system_prompt(topic)
        messages = [
            HumanMessage(content=f"{system_prompt}\n\n{full_prompt}")
        ]
        
        # Generate response
        response = self.llm.invoke(messages)
        argument = response.content
        
        return {"argument": argument}
    
    def invoke(self, inputs):
        """Invoke the agent graph."""
        return self.graph.invoke(inputs)


class DebateModerator:
    """Orchestrates a debate between two agents using LangGraph."""
    
    def __init__(self, topic: str, embedding_model: SentenceTransformer, num_rounds: int = 3, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """
        Initialize the debate moderator.
        
        Args:
            topic: The debate topic
            embedding_model: SentenceTransformer model for RAG search
            num_rounds: Number of debate rounds
            model_name: LLM model to use
        """
        self.topic = topic
        self.num_rounds = num_rounds
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        # Create debate agents (without RAG parameters - context provided via state)
        self.proponent = DebateAgent(stance="for", model_name=model_name)
        self.opponent = DebateAgent(stance="against", model_name=model_name)
        
        # Create judge LLM for final summary
        base_url = os.getenv("OPENAI_API_BASE")
        self.judge_llm = ChatOpenAI(model=model_name, temperature=0.7, base_url=base_url)
        
        # Build the debate workflow
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow for the debate."""
        workflow = StateGraph(DebateState)
        
        # Add nodes
        # workflow.add_node("initial_rag_search", self.initial_rag_search) #FIND BACK
        workflow.add_node("proponent_argues", self.proponent_argues)
        workflow.add_node("opponent_argues", self.opponent_argues)
        workflow.add_node("increment_round", self.increment_round)
        workflow.add_node("generate_summary", self.generate_summary)
        
        # Set entry point to initial RAG search
        workflow.set_entry_point("proponent_argues")
        
        # Add edges
        # workflow.add_edge("initial_rag_search", "proponent_argues")
        workflow.add_edge("proponent_argues", "opponent_argues")
        workflow.add_edge("opponent_argues", "increment_round")
        
        # Conditional edge to check if debate should continue
        workflow.add_conditional_edges(
            "increment_round",
            self.should_continue,
            {
                "continue": "proponent_argues",
                "summarize": "generate_summary"
            }
        )
        
        workflow.add_edge("generate_summary", END)
        
        return workflow.compile()
    
    def initial_rag_search(self, state: DebateState):
        """Perform initial RAG search once for both agents."""
        topic = state["topic"]
        
        print("Performing initial RAG search...")
        
        # Search index 1 for proponent (FOR)
        proponent_results = search_index(topic, self.embedding_model, top_k=5, idx_num=1)
        proponent_context = "\n".join([result['text'] for result in proponent_results])
        
        # Search index 2 for opponent (AGAINST)
        opponent_results = search_index(topic, self.embedding_model, top_k=5, idx_num=2)
        opponent_context = "\n".join([result['text'] for result in opponent_results])
        
        print("RAG search completed.\n")
        
        return {
            "proponent_context": proponent_context,
            "opponent_context": opponent_context
        }
    
    def proponent_argues(self, state: DebateState):
        """Proponent presents their argument."""
        # Get opponent's last argument if available
        opponent_arg = state.get("opponent_last_argument", None)
        
        # Invoke proponent agent with pre-retrieved context
        result = self.proponent.invoke({
            "topic": state["topic"],
            "round_num": state["current_round"],
            "opponent_argument": opponent_arg,
            "previous_arguments": state.get("proponent_arguments", []),
            "retrieved_context": state.get("proponent_context", "")
        })
        
        argument = result["argument"]
        
        # Update state
        proponent_args = state.get("proponent_arguments", [])
        proponent_args.append(argument)
        
        debate_history = state.get("debate_history", [])
        debate_history.append({
            "round": state["current_round"],
            "speaker": "PROPONENT",
            "argument": argument
        })
        
        print(f"\n{'=' * 80}")
        print(f"ROUND {state['current_round']} - PROPONENT (FOR)")
        print('=' * 80)
        print(argument)
        
        return {
            "proponent_arguments": proponent_args,
            "proponent_last_argument": argument,
            "debate_history": debate_history
        }
    
    def opponent_argues(self, state: DebateState):
        """Opponent presents their counter-argument."""
        # Get proponent's last argument
        proponent_arg = state["proponent_last_argument"]
        
        # Invoke opponent agent with pre-retrieved context
        result = self.opponent.invoke({
            "topic": state["topic"],
            "round_num": state["current_round"],
            "opponent_argument": proponent_arg,
            "previous_arguments": state.get("opponent_arguments", []),
            "retrieved_context": state.get("opponent_context", "")
        })
        
        argument = result["argument"]
        
        # Update state
        opponent_args = state.get("opponent_arguments", [])
        opponent_args.append(argument)
        
        debate_history = state.get("debate_history", [])
        debate_history.append({
            "round": state["current_round"],
            "speaker": "OPPONENT",
            "argument": argument
        })
        
        print(f"\n{'=' * 80}")
        print(f"ROUND {state['current_round']} - OPPONENT (AGAINST)")
        print('=' * 80)
        print(argument)
        
        return {
            "opponent_arguments": opponent_args,
            "opponent_last_argument": argument,
            "debate_history": debate_history
        }
    
    def increment_round(self, state: DebateState):
        """Increment the round counter."""
        return {"current_round": state["current_round"] + 1}
    
    def should_continue(self, state: DebateState):
        """Check if debate should continue or move to summary."""
        if state["current_round"] > state["max_rounds"]:
            return "summarize"
        return "continue"
    
    def generate_summary(self, state: DebateState):
        """Generate a summary and judgment of the debate."""
        # Construct debate transcript
        transcript = f"Topic: {state['topic']}\n\n"
        for entry in state["debate_history"]:
            transcript += f"Round {entry['round']} - {entry['speaker']}:\n{entry['argument']}\n\n"
        
        prompt = f"""You are an impartial debate judge. Analyze this debate and provide:
1. A brief summary of key arguments from both sides
2. Assessment of which side presented stronger arguments and why
3. Overall quality of the debate

Debate Transcript:
{transcript}

Provide your analysis:"""
        
        response = self.judge_llm.invoke([HumanMessage(content=prompt)])
        summary = response.content
        
        print(f"\n{'=' * 80}")
        print("JUDGE'S ANALYSIS")
        print('=' * 80)
        print(summary)
        
        return {"final_summary": summary}
    
    def invoke(self, inputs):
        """Run the debate."""
        return self.graph.invoke(inputs)


def main():
    """Main function to run a debate."""
    # Check for API key
    if not os.getenv("OPENAI_API_BASE"):
        print("Error: OPENAI_API_BASE not found in environment variables.")
        print("Please set OPENAI_API_BASE in your .env file.")
        return
    
    # Initialize embedding model for RAG
    print("Initializing embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Debate topic
    topic = "Artificial Intelligence will ultimately benefit humanity more than harm it"
    
    print("=" * 80)
    print(f"DEBATE TOPIC: {topic}")
    print("=" * 80)
    print("\nNote: Proponent uses knowledge base index 1, Opponent uses knowledge base index 2")
    print()
    
    # Create moderator with embedding model
    moderator = DebateModerator(
        topic=topic,
        embedding_model=embedding_model,
        num_rounds=3,
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    
    # Run debate
    result = moderator.invoke({
        "topic": topic,
        "current_round": 1,
        "max_rounds": 3,
        "proponent_arguments": [],
        "opponent_arguments": [],
        "proponent_last_argument": "",
        "opponent_last_argument": "",
        "proponent_context": "",  # Will be populated by initial_rag_search
        "opponent_context": "",   # Will be populated by initial_rag_search
        "debate_history": [],
        "final_summary": ""
    })
    
    print("\n" + "=" * 80)
    print("DEBATE CONCLUDED")
    print("=" * 80)


if __name__ == "__main__":
    main()
