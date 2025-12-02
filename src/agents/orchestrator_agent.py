from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from .worker_agent import WorkerAgent
import os

class OrchestratorState(TypedDict):
    problem: str
    worker1_response: str
    worker2_response: str
    worker1_critique: str
    worker2_critique: str
    final_answer: str
    messages: List[BaseMessage]
    iteration: int
    worker1_history: List[Dict[str, str]]  # List of {"type": "solve/critique/refine", "content": str}
    worker2_history: List[Dict[str, str]]  # List of {"type": "solve/critique/refine", "content": str}

class OrchestratorAgent:
    def __init__(self):
        self.worker1 = WorkerAgent(name="Worker1")
        self.worker2 = WorkerAgent(name="Worker2")
        base_url = os.getenv("OPENAI_API_BASE")
        self.llm = ChatOpenAI(model="meta-llama/Meta-Llama-3.1-8B-Instruct", temperature=0.5, base_url=base_url)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(OrchestratorState)

        workflow.add_node("initial_solve", self.initial_solve)
        workflow.add_node("critique_peer", self.critique_peer)
        workflow.add_node("refine_response", self.refine_response)
        workflow.add_node("synthesize", self.synthesize)

        workflow.set_entry_point("initial_solve")
        workflow.add_edge("initial_solve", "critique_peer")
        workflow.add_edge("critique_peer", "refine_response")
        
        workflow.add_conditional_edges(
            "refine_response",
            self.check_loop,
            {
                "continue": "critique_peer",
                "done": "synthesize"
            }
        )
        
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def initial_solve(self, state: OrchestratorState):
        problem = state["problem"]
        
        # Worker 1 solve
        w1_result = self.worker1.invoke({
            "messages": [HumanMessage(content=problem)],
            "task": "solve"
        })
        w1_response = w1_result["messages"][-1].content

        # Worker 2 solve
        w2_result = self.worker2.invoke({
            "messages": [HumanMessage(content=problem)],
            "task": "solve"
        })
        w2_response = w2_result["messages"][-1].content

        # Initialize history
        w1_history = state.get("worker1_history", [])
        w2_history = state.get("worker2_history", [])
        
        w1_history.append({"type": "solve", "iteration": 0, "content": w1_response})
        w2_history.append({"type": "solve", "iteration": 0, "content": w2_response})

        return {
            "worker1_response": w1_response,
            "worker2_response": w2_response,
            "worker1_history": w1_history,
            "worker2_history": w2_history,
            "iteration": 0
        }

    def critique_peer(self, state: OrchestratorState):
        w1_response = state["worker1_response"]
        w2_response = state["worker2_response"]
        problem = state["problem"]
        iteration = state["iteration"]

        # Worker 1 critiques Worker 2's response
        w1_critique_result = self.worker1.invoke({
            "messages": [HumanMessage(content=problem)], # Context
            "task": "critique",
            "original_response": w2_response
        })
        w1_critique = w1_critique_result["messages"][-1].content

        # Worker 2 critiques Worker 1's response
        w2_critique_result = self.worker2.invoke({
            "messages": [HumanMessage(content=problem)], # Context
            "task": "critique",
            "original_response": w1_response
        })
        w2_critique = w2_critique_result["messages"][-1].content

        # Track critiques in history
        w1_history = state.get("worker1_history", [])
        w2_history = state.get("worker2_history", [])
        
        w1_history.append({"type": "critique", "iteration": iteration, "content": w1_critique, "target": "Worker 2"})
        w2_history.append({"type": "critique", "iteration": iteration, "content": w2_critique, "target": "Worker 1"})

        return {
            "worker1_critique": w1_critique,
            "worker2_critique": w2_critique,
            "worker1_history": w1_history,
            "worker2_history": w2_history
        }

    def refine_response(self, state: OrchestratorState):
        w1_response = state["worker1_response"]
        w2_critique = state["worker2_critique"] # Critique OF Worker 1 (by Worker 2)
        
        w2_response = state["worker2_response"]
        w1_critique = state["worker1_critique"] # Critique OF Worker 2 (by Worker 1)
        
        problem = state["problem"]
        iteration = state["iteration"]

        # Worker 1 refines
        w1_refine_result = self.worker1.invoke({
            "messages": [HumanMessage(content=problem)],
            "task": "refine",
            "original_response": w1_response,
            "critique": w2_critique
        })
        w1_new_response = w1_refine_result["messages"][-1].content

        # Worker 2 refines
        w2_refine_result = self.worker2.invoke({
            "messages": [HumanMessage(content=problem)],
            "task": "refine",
            "original_response": w2_response,
            "critique": w1_critique
        })
        w2_new_response = w2_refine_result["messages"][-1].content

        # Track refinements in history
        w1_history = state.get("worker1_history", [])
        w2_history = state.get("worker2_history", [])
        
        w1_history.append({"type": "refine", "iteration": iteration + 1, "content": w1_new_response})
        w2_history.append({"type": "refine", "iteration": iteration + 1, "content": w2_new_response})

        return {
            "worker1_response": w1_new_response,
            "worker2_response": w2_new_response,
            "worker1_history": w1_history,
            "worker2_history": w2_history,
            "iteration": state["iteration"] + 1
        }

    def check_loop(self, state: OrchestratorState):
        if state["iteration"] < 1:
            return "continue"
        return "done"

    def synthesize(self, state: OrchestratorState):
        problem = state["problem"]
        w1_response = state["worker1_response"]
        w2_response = state["worker2_response"]
        # We use the latest critiques available, though they might be from the previous iteration's response
        # Actually, after refinement, we don't have critiques for the NEW responses yet.
        # But for synthesis, we can just use the final refined responses.
        
        prompt = f"""
        Problem: {problem}

        Worker 1 Final Response: {w1_response}
        Worker 2 Final Response: {w2_response}

        Based on the above refined responses, please synthesize a final, high-quality answer.
        """

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"final_answer": response.content}

    def invoke(self, inputs):
        return self.graph.invoke(inputs)
