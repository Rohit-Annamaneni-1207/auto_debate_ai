from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os

class WorkerState(TypedDict):
    messages: List[BaseMessage]
    task: str # "solve", "critique", or "refine"
    original_response: str # For critique/refine task
    critique: str # For refine task

class WorkerAgent:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", name: str = "Worker"):
        self.name = name
        base_url = os.getenv("OPENAI_API_BASE")
        self.llm = ChatOpenAI(model=model_name, temperature=0.7, base_url=base_url)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(WorkerState)

        workflow.add_node("process_task", self.process_task)
        workflow.set_entry_point("process_task")
        workflow.add_edge("process_task", END)

        return workflow.compile()

    def process_task(self, state: WorkerState):
        task = state.get("task")
        messages = state.get("messages", [])
        
        if task == "solve":
            # Just respond to the last message which should be the problem
            response = self.llm.invoke(messages)
            return {"messages": [response]}
        
        elif task == "critique":
            original_response = state.get("original_response")
            # Construct a critique prompt
            critique_prompt = f"""Please critique the following response. 
            Focus ONLY on identifying logical flaws or calculation errors. 
            If there are no such errors, explicitly state that the response is correct. 
            Do not offer vague suggestions or style improvements.

            Response to critique:
            {original_response}"""
            
            # Add to messages or just invoke
            # We preserve history but add the critique request
            messages_with_prompt = messages + [HumanMessage(content=critique_prompt)]
            response = self.llm.invoke(messages_with_prompt)
            return {"messages": [response]}

        elif task == "refine":
            original_response = state.get("original_response")
            critique = state.get("critique")
            refine_prompt = f"Here is your original response:\n{original_response}\n\nHere is a critique:\n{critique}\n\nPlease refine your response based on the critique."
            
            messages_with_prompt = messages + [HumanMessage(content=refine_prompt)]
            response = self.llm.invoke(messages_with_prompt)
            return {"messages": [response]}
        
        return {"messages": [AIMessage(content="Unknown task")]}

    def invoke(self, inputs):
        return self.graph.invoke(inputs)
