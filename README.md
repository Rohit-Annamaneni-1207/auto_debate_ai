# Auto Debate AI

A  multi-llm system that facilitates intelligent debates and collaborative problem-solving using LangGraph, RAG, and LLMs.

## Overview

We built a multi-agent framework that implements two primary workflows:

1. **Problem solving system**: A collaborative problem-solving approach where multiple worker agents solve problems supported by RAG, critique each other's solutions, refine their responses, and orchestrator synthesizes a final answer.

2. **Debate System**: A structured debate format where two agents (Proponent and Opponent) argue for and against a given topic, each supported by separate knowledge bases through RAG.

Both systems use LangGraph for workflow orchestration and FAISS for vector search and indexing.
## Architecture

### Core Components

```
auto_debate_ai/
├── src/
│   ├── agents/
│   │   ├── orchestrator_agent.py    # Multi-agent orchestrator for problem-solving
│   │   └── worker_agent.py          # Individual worker agents (solve, critique, refine)
│   └── tools/
│       ├── rag_pipeline/
│       │   ├── document_chunking.py # Document processing and chunking
│       │   ├── faiss_utils.py       # FAISS vector index operations
│       │   └── process_file.py      # File processing pipeline
│       └── web_crawl/
│           └── tavilly_utils.py     # Web search utilities (currently not in use because of poor websearch results)
├── data/
│   ├── knowledge_base_1/            # Documents for Proponent/General KB
│   └── knowledge_base_2/            # Documents for Opponent/Alternate KB
├── app.py                           # Streamlit web interface
├── debate.py                        # Debate moderator and agents
└── demo.py                          # CLI demo for orchestrator
```

### Agent Workflows
#### Debate Workflow
workflow flowchart<br />
![Flowchart](/workflow_images/GenAI_Flowchart.drawio.png)
workflow UML<br />
![UML](/workflow_images/GenAI_Flowchart_2.drawio.png)
#### Orchestrator Workflow
Initial response<br />
![Initial response](/workflow_images/orchestrator_part_one.jpg)
Peer critique<br />
![Peer critique](/workflow_images/orchestrator_part_two.jpg)
Refine response and loop decision<br />
![Refine response](/workflow_images/orchestrator_part_three.jpg)

## Features

- **Multi-Agent Collaboration**: Worker agents collaborate through iterative critique and refinement
- **Workflow to ensure quality of output**: The implemented workflow, proves useful to ensure correctness of response in multi-step agentic workflows prone to cascading errors.
- **Dual Knowledge Bases**: Separate FAISS indexes for different perspectives/knowledge sources
- **RAG Integration**: Retrieval-Augmented Generation for contextually grounded responses
- **Structured Debates**: Formal debate rounds with proponent and opponent agents
- **Web Interface**: Full-featured Streamlit UI for easy interaction
- **Flexible LLM Backend**: Compatible with OpenAI-style API endpoints (including local models)

## Prerequisites

- Python 3.8+
- OpenAI-compatible API endpoint (We have used a vllm server with openai compatible api use llama-3.1-8B)
- OpenAI API key

## Installation

1. **Clone the repository and go to project directory**

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies in requirements.txt**

4. **Configure environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=empty
OPENAI_API_BASE=http://0.0.0.0:8000/v1
```

## Usage

### Streamlit Web Interface (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

The UI provides three main tabs:

1. **Indexing**: Upload documents to Knowledge Base 1 or 2, process document button will process them into FAISS indexes, clear the existing index and metadata, button to delete existing files.
2. **Orchestrator**: Run collaborative problem-solving with RAG-enhanced context
3. **Debate**: Start structured debates with separate knowledge bases for each side

#### Orchestrator Example

```python
from src.agents.orchestrator_agent import OrchestratorAgent
from src.tools.rag_pipeline.faiss_utils import search_index
from sentence_transformers import SentenceTransformer

# Initialize
orchestrator = OrchestratorAgent()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Search knowledge base
problem = "What is an acoustic representation?"
search_results = search_index(problem, embedding_model, top_k=5, idx_num=1)
retrieved_text = "\n".join([r['text'] for r in search_results])

# Augment problem with context
augmented_problem = f"{problem}\n\nRetrieved Text for context: {retrieved_text}"

# Run orchestrator
result = orchestrator.invoke({"problem": augmented_problem})
print(result["final_answer"])
```

#### Debate Example

```python
from debate import DebateModerator
from sentence_transformers import SentenceTransformer

# Initialize
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
topic = "Artificial Intelligence will benefit humanity more than harm it"

moderator = DebateModerator(
    topic=topic,
    embedding_model=embedding_model,
    num_rounds=3
)

# Initial state
initial_state = {
    "topic": topic,
    "current_round": 1,
    "max_rounds": 3,
    "proponent_arguments": [],
    "opponent_arguments": [],
    "proponent_last_argument": "",
    "opponent_last_argument": "",
    "proponent_context": "",
    "opponent_context": "",
    "debate_history": [],
    "final_summary": ""
}

# Run debate
result = moderator.invoke(initial_state)
print(result["final_summary"])
```

## Knowledge Base Setup

### Adding Documents

**Via Streamlit UI**: Use the "Indexing" tab to upload PDF or TXT files

### Processing Documents

Documents must be processed to create embeddings and FAISS indexes:

```python
from src.tools.rag_pipeline.process_file import process_file_add_to_index
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Process KB1
process_file_add_to_index(embedding_model=embedding_model, idx_num=1)

# Process KB2
process_file_add_to_index(embedding_model=embedding_model, idx_num=2)
```

### Clearing Indexes

```python
from src.tools.rag_pipeline.faiss_utils import clear_index

clear_index(idx_num=1)  # Clear Knowledge Base 1
clear_index(idx_num=2)  # Clear Knowledge Base 2
```

### Model Configuration

The default model is `meta-llama/Meta-Llama-3.1-8B-Instruct`. To use a different model:

For problem solving: The temperature is set to 0.5 by default.
For debate: The temperature is set to 0.8 for agents and 0.7 for judge.

### RAG Parameters

- **top_k**: Number of retrieved documents (default: 5)
- **embedding_model**: SentenceTransformer model (default: 'all-MiniLM-L6-v2')
- **chunk_size**: Document chunk size

### Orchestrator Parameters

- Number of critique-refine cycles (Hard coded to 1 because of compute restrictions)
- LLM temperature for synthesis (default: 0.5)

### Debate Parameters

- **num_rounds**: Number of debate rounds (default: 3)
- **stance**: "for" or "against" for debate agents
- **temperature**: default: 0.8 for agents, 0.7 for judge

### Document Processing Pipeline

```
Upload Document (PDF/TXT)
    ↓
[Load & Parse] (document_chunking.py)
    ↓
[Chunk Text] (RecursiveCharacterTextSplitter)
    ↓
[Generate Embeddings] (SentenceTransformer)
    ↓
[Add to FAISS Index] (faiss_utils.py)
    ↓
[Store Metadata] (metadata.json)
```


```
User Query
    ↓
[Embed Query] (SentenceTransformer)
    ↓
[Search FAISS Index] (top_k results)
    ↓
[Retrieve Chunk Metadata]
    ↓
[Return Ranked Results] (score, text, source)
```

## Dependencies

- **langgraph**: Agent orchestration
- **langchain**: LLM utilities
- **langchain-openai**: OpenAI integration
- **langchain-community**: pypdf
- **langchain-text-splitters**: RecursiveCharacterTextSplitter
- **faiss**: Vector similarity search
- **sentence-transformers**: Text embedding model
- **streamlit**: Web interface
- **python-dotenv**: Set environment vars
- **tavily-python**: Web search integration (Currently not in use because of poor search results)

