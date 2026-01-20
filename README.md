# Agentic AI Project

A personal agentic AI system built from scratch to learn RAG and agentic workflows using AWS Bedrock and Weaviate Cloud.

## Overview

This project implements:
- **ReAct Agent Pattern**: Reasoning + Acting loop for autonomous task completion
- **RAG Pipeline**: Retrieval-Augmented Generation with Weaviate Cloud for persistence
- **Multi-Provider LLM Architecture**: Abstract base class supporting multiple providers
- **Document Processing**: PDF and Word document parsing with intelligent chunking

The code is heavily documented for learning purposes - every function explains what it does and why.

## Project Structure

```
agentic-ai/
├── pyproject.toml              # Project config and dependencies
├── .python-version             # Python version (3.10)
├── .env.example                # Credentials template
├── README.md
├── src/
│   └── agentic_ai/
│       ├── __init__.py
│       ├── agents/
│       │   ├── __init__.py
│       │   └── base_agent.py          # Core ReAct agent loop
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── embeddings.py          # Bedrock Titan embeddings
│       │   ├── vector_store.py        # Weaviate Cloud + In-memory stores
│       │   └── retriever.py           # Semantic search
│       ├── tools/
│       │   ├── __init__.py
│       │   └── document_parser.py     # PDF/DOCX parsing
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── llm_client.py          # Abstract LLM client base class
│       │   ├── bedrock_client.py      # AWS Bedrock implementation
│       │   └── azure_client.py        # Azure OpenAI (placeholder)
│       └── utils/
│           ├── __init__.py
│           └── config.py              # Configuration management
├── examples/
│   ├── simple_rag_example.py      # RAG with Weaviate persistence
│   └── agent_example.py           # ReAct agent demo
├── tests/
│   └── test_agent.py
└── data/
    ├── regulations/               # Sample regulation docs
    └── procedures/                # Sample procedure docs
```

## Setup

### Install uv (if not already installed)

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Create virtual environment and install dependencies

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Install dev dependencies

```bash
uv pip install -e ".[dev]"
```

### Configure credentials

```bash
cp .env.example .env
# Edit .env with your credentials
```

Your `.env` file should contain:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# Weaviate Cloud Configuration
WEAVIATE_URL=https://your-cluster-id.weaviate.network
WEAVIATE_API_KEY=your_weaviate_api_key
```

### Setting up Weaviate Cloud

1. Create an account at [Weaviate Cloud Console](https://console.weaviate.cloud/)
2. Create a new cluster (free tier available)
3. Copy the cluster URL and API key to your `.env` file

## Usage

### Test RAG Pipeline with Weaviate

```bash
python examples/simple_rag_example.py
```

This example demonstrates persistent storage:
1. Connects to Weaviate Cloud
2. Creates a collection (on first run)
3. Adds documents with embeddings (on first run)
4. Queries for relevant documents
5. **Data persists across runs!** - Run again to see it skip document creation

### Test Agent

```bash
python examples/agent_example.py
```

This example demonstrates the ReAct pattern:
1. **THINK**: Agent reasons about what to do
2. **ACT**: Agent calls a tool (calculator, weather, etc.)
3. **OBSERVE**: Agent sees the result
4. **REPEAT**: Until task is complete

## Key Concepts

### Multi-Provider LLM Architecture

The system uses an abstract `LLMClient` base class that allows swapping LLM providers:

```python
from agentic_ai.llm import BedrockClient, LLMClient

# All clients share the same interface
client: LLMClient = BedrockClient()

# Standardized response format
response = client.generate([
    {"role": "user", "content": "Hello!"}
])
print(response["content"])  # Same format regardless of provider

# With tool calling
response = client.generate_with_tools(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    tools=[calculator_tool]
)
for tool_call in response["tool_calls"]:
    print(f"Tool: {tool_call['name']}, Input: {tool_call['input']}")
```

To add a new provider, implement the `LLMClient` abstract methods:
- `generate()` - Basic text generation
- `generate_with_tools()` - Tool/function calling
- `get_embeddings()` - Text embeddings
- `stream()` - Streaming responses (optional)

### RAG with Weaviate Cloud

Data persists in Weaviate Cloud - no local database files needed:

```python
from agentic_ai.rag import VectorStore, Retriever
from agentic_ai.rag.embeddings import get_embeddings_batch

# Connect to Weaviate Cloud
with VectorStore() as store:
    # Create collection (if doesn't exist)
    store.create_collection("Regulations", "Regulatory documents")

    # Add documents (they persist!)
    store.add_documents(
        collection_name="Regulations",
        documents=["Text 1", "Text 2"],
        embeddings=get_embeddings_batch(["Text 1", "Text 2"]),
        metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}]
    )

    # Retrieve documents
    retriever = Retriever(store, collection_name="Regulations")
    results = retriever.retrieve("your query", top_k=5)
```

Multi-collection support for different document types:
```python
from agentic_ai.rag import MultiCollectionRetriever

retriever = MultiCollectionRetriever(
    store,
    collections=["Regulations", "Procedures"]
)
results = retriever.retrieve("safety requirements")  # Searches both
```

### ReAct Agent Pattern

The ReAct pattern combines reasoning and acting in a loop:

```python
from agentic_ai.agents import BaseAgent, Tool

# Define a tool
def calculator(expression: str) -> str:
    return str(eval(expression))

calc_tool = Tool(
    name="calculator",
    description="Evaluates mathematical expressions",
    input_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        },
        "required": ["expression"]
    },
    function=calculator
)

# Create and run agent
agent = BaseAgent(tools=[calc_tool])
result = agent.run("What is 15 * 7 + 23?")
```

## Architecture

### Why Build From Scratch?

This project intentionally avoids high-level abstractions to:
- Understand exactly how LLM APIs work
- See the raw request/response formats
- Learn the agent loop at a fundamental level
- Have full control over the implementation

### Multi-Provider Design

The abstract `LLMClient` class enables:
- **Provider Independence**: Agents don't need to change when switching providers
- **Standardized Responses**: Same format from all providers
- **Easy Extension**: Implement new providers by subclassing

```
┌─────────────┐
│   Agents    │  ← Use LLMClient interface
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LLMClient  │  ← Abstract base class
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
┌─────┐ ┌─────┐
│Bedrock│ │Azure│  ← Concrete implementations
└─────┘ └─────┘
```

### Vector Store Architecture

```
┌─────────────────────────────────────────┐
│            Retriever                     │
│  (Embeds queries, formats results)       │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌───────────────────┐
│  VectorStore  │    │ InMemoryVectorStore│
│  (Weaviate)   │    │   (for testing)    │
└───────┬───────┘    └───────────────────┘
        │
        ▼
┌───────────────┐
│ Weaviate Cloud│  ← Persistent storage
└───────────────┘
```

## Development

### Run tests

```bash
pytest tests/
```

### Format code

```bash
black src/ examples/ tests/
ruff check src/ examples/ tests/
```

## Next Steps

After testing the foundation, we'll build:
1. **Compliance Agent**: Check procedures against regulations
2. **Research Agent**: Generate reports from proposals

## Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Claude API Documentation](https://docs.anthropic.com/claude/docs)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [RAG Overview](https://arxiv.org/abs/2005.11401)
