# Auto Debate AI

A sophisticated multi-agent AI system that facilitates intelligent debates and collaborative problem-solving using LangGraph, RAG (Retrieval-Augmented Generation), and Large Language Models.

## 🌟 Overview

Auto Debate AI is a multi-agent framework that implements two primary workflows:

1. **Orchestrator System**: A collaborative problem-solving approach where multiple worker agents solve problems, critique each other's solutions, refine their responses, and synthesize a final answer.

2. **Debate System**: A structured debate format where two agents (Proponent and Opponent) argue for and against a given topic, each backed by separate knowledge bases through RAG.

Both systems leverage LangGraph for workflow orchestration and use FAISS-based vector search for knowledge retrieval.

## 🏗️ Architecture

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
│           └── tavilly_utils.py     # Web search utilities
├── data/
│   ├── knowledge_base_1/            # Documents for Proponent/General KB
│   └── knowledge_base_2/            # Documents for Opponent/Alternate KB
├── app.py                           # Streamlit web interface
├── debate.py                        # Debate moderator and agents
└── demo.py                          # CLI demo for orchestrator
```

### Agent Workflows

#### Orchestrator Workflow
```
Problem Input
    ↓
[Initial Solve] (Worker 1 & Worker 2)
    ↓
[Critique Peer] (Cross-critique)
    ↓
[Refine Response] (Based on critiques)
    ↓
[Check Loop] (Iterate or continue)
    ↓
[Synthesize] (Final answer)
```

#### Debate Workflow
```
Topic + RAG Context
    ↓
[Proponent Arguments] (KB1)
    ↓
[Opponent Arguments] (KB2)
    ↓
[Increment Round]
    ↓
[Check Continue]
    ↓
[Generate Summary] (Judge's analysis)
```

## 🚀 Features

- **Multi-Agent Collaboration**: Worker agents collaborate through iterative critique and refinement
- **Dual Knowledge Bases**: Separate FAISS indexes for different perspectives/knowledge sources
- **RAG Integration**: Retrieval-Augmented Generation for contextually grounded responses
- **Structured Debates**: Formal debate rounds with proponent and opponent agents
- **Web Interface**: Full-featured Streamlit UI for easy interaction
- **Flexible LLM Backend**: Compatible with OpenAI-style API endpoints (including local models)

## 📋 Prerequisites

- Python 3.8+
- OpenAI-compatible API endpoint (OpenAI, local LLM servers, etc.)
- Required API keys (configured in `.env`)

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd auto_debate_ai
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # Or your custom endpoint
TAVILLY_API_KEY=your_tavily_key_here  # Optional, for web search
```

## 🎯 Usage

### Streamlit Web Interface (Recommended)

Launch the interactive web application:

```bash
streamlit run app.py
```

The UI provides four main tabs:

1. **Indexing**: Upload documents (PDF/TXT) to Knowledge Base 1 or 2, process them into FAISS indexes
2. **Orchestrator**: Run collaborative problem-solving with RAG-enhanced context
3. **Debate**: Start structured debates with separate knowledge bases for each side
4. **Diagnostics**: View environment status, file counts, and index information

### Command-Line Usage

#### Orchestrator Demo

```bash
python demo.py
```

This runs the orchestrator workflow on a predefined problem with RAG context retrieval.

#### Debate Demo

```bash
python debate.py
```

This initiates a debate on a predefined topic with configurable rounds.

### Programmatic Usage

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

## 📚 Knowledge Base Setup

### Adding Documents

1. **Via Streamlit UI**: Use the "Indexing" tab to upload PDF or TXT files
2. **Manually**: Place files in `data/knowledge_base_1/documents/` or `data/knowledge_base_2/documents/`

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

## 🔧 Configuration

### Model Configuration

The default model is `meta-llama/Meta-Llama-3.1-8B-Instruct`. To use a different model:

```python
# In orchestrator_agent.py
self.llm = ChatOpenAI(
    model="your-model-name",
    temperature=0.5,
    base_url=os.getenv("OPENAI_API_BASE")
)

# In debate.py
moderator = DebateModerator(
    topic=topic,
    embedding_model=embedding_model,
    num_rounds=3,
    model_name="your-model-name"
)
```

### RAG Parameters

- **top_k**: Number of retrieved documents (default: 5)
- **embedding_model**: SentenceTransformer model (default: 'all-MiniLM-L6-v2')
- **chunk_size**: Document chunk size (configured in document_chunking.py)

### Orchestrator Parameters

- **max_iterations**: Number of critique-refine cycles (default: 1 in check_loop)
- **temperature**: LLM temperature for synthesis (default: 0.5)

### Debate Parameters

- **num_rounds**: Number of debate rounds (default: 3)
- **stance**: "for" or "against" for debate agents
- **temperature**: LLM creativity (default: 0.8 for agents, 0.7 for judge)

## 📊 Data Flow

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

### Query-Time RAG Flow

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

## 🛠️ Key Dependencies

- **langgraph**: Workflow orchestration for multi-agent systems
- **langchain**: LLM framework and utilities
- **langchain-openai**: OpenAI integration
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Text embeddings
- **streamlit**: Web interface
- **python-dotenv**: Environment configuration
- **tavily-python**: Web search integration (optional)

## 🔍 How It Works

### Orchestrator System

1. **Initial Solve**: Two worker agents independently solve the problem
2. **Cross-Critique**: Each worker critiques the other's solution
3. **Refinement**: Workers refine their solutions based on received critiques
4. **Iteration**: Process repeats for configured iterations
5. **Synthesis**: Orchestrator synthesizes final answer from refined responses

### Debate System

1. **Setup**: Topic is set, RAG retrieves relevant context from separate KBs
2. **Rounds**: Proponent and Opponent alternate arguments for N rounds
3. **Context**: Each agent uses their designated knowledge base (KB1 or KB2)
4. **History**: All arguments are tracked in debate_history
5. **Judgment**: Judge LLM provides final analysis and summary

### RAG Integration

- Documents are chunked and embedded using SentenceTransformers
- FAISS provides efficient similarity search
- Query embeddings find most relevant document chunks
- Retrieved context augments agent prompts
- Separate indexes enable perspective-based knowledge separation

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the project root
cd auto_debate_ai
python -m streamlit run app.py
```

**Missing API Keys**
- Check `.env` file exists and contains valid keys
- Verify `OPENAI_API_BASE` points to correct endpoint

**Empty RAG Results**
- Ensure documents are uploaded to correct KB directory
- Process documents using "Process KB" buttons in UI
- Check index files exist in `data/knowledge_base_*/embeddings/`

**LLM Connection Errors**
- Verify API endpoint is accessible
- Check API key validity
- Confirm model name matches available models at endpoint

## 📝 License

[Specify your license here]

## 👥 Contributing

[Add contribution guidelines if applicable]

## 🙏 Acknowledgments

Built with:
- LangGraph for multi-agent orchestration
- LangChain for LLM integration
- FAISS for efficient vector search
- Streamlit for interactive UI
- SentenceTransformers for embeddings

## 📧 Contact

[Your contact information]

---

**Note**: This project demonstrates advanced multi-agent AI patterns including collaborative problem-solving, structured debates, and knowledge-grounded reasoning through RAG. It's designed for research, education, and experimentation with multi-agent systems.
