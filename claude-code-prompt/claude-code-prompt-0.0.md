I want to build a personal agentic AI system from scratch to learn RAG and agentic workflows. I'm using AWS Bedrock API with Python and uv for package management.

Project Goals:
1. Build core agentic AI from scratch (not using LangChain's agent abstractions)
2. Two use cases: 
   - Compliance checker: verify procedure documents against regulations
   - Research agent: generate reports from proposals
3. Learn by implementing: RAG, tool calling, multi-step reasoning

Tech Stack Decisions:
- Python 3.10+
- uv for package management (modern, fast alternative to pip)
- AWS Bedrock (boto3 for direct API calls to Claude)
- Build from scratch: LLM calls, agent loops, RAG retrieval logic
- Use selectively: LangChain document loaders & text splitters only
- LangGraph: for complex multi-step workflows (later phase)
- ChromaDB or simple vector store for embeddings
- AWS Bedrock Titan for embeddings

Project Structure I Want:
```
agentic-ai-project/
├── pyproject.toml              # Project config and dependencies
├── .python-version             # Python version (3.10)
├── uv.lock                     # Auto-generated lock file
├── .env.example                # AWS credentials template
├── README.md
├── src/
│   └── agentic_ai/
│       ├── init.py
│       ├── agents/
│       │   ├── init.py
│       │   ├── base_agent.py          # Core agent loop (ReAct pattern)
│       │   ├── compliance_agent.py    # Compliance checker
│       │   └── research_agent.py      # Research agent
│       ├── rag/
│       │   ├── init.py
│       │   ├── embeddings.py          # Bedrock embedding calls
│       │   ├── vector_store.py        # Simple vector DB
│       │   └── retriever.py           # Search logic
│       ├── tools/
│       │   ├── init.py
│       │   ├── document_parser.py     # Uses LangChain loaders
│       │   └── web_search.py          # Tool for research agent
│       ├── llm/
│       │   ├── init.py
│       │   └── bedrock_client.py      # Direct boto3 Bedrock calls
│       └── utils/
│           ├── init.py
│           └── config.py               # AWS config
├── examples/
│   ├── simple_rag_example.py      # Test RAG pipeline
│   └── agent_example.py           # Test agent loop
├── tests/
│   ├── init.py
│   └── test_agent.py
└── data/
├── regulations/               # Sample regulation docs
└── procedures/                # Sample procedure docs
```

Please create:

1. **Project Configuration (`pyproject.toml`)**:
````toml
   [project]
   name = "agentic-ai"
   version = "0.1.0"
   description = "Personal agentic AI system with RAG and AWS Bedrock"
   requires-python = ">=3.10"
   dependencies = [
       "boto3>=1.34.0",
       "langchain-community>=0.0.20",
       "pypdf>=3.17.0",
       "python-docx>=1.1.0",
       "numpy>=1.26.0",
       "python-dotenv>=1.0.0",
   ]

   [project.optional-dependencies]
   dev = [
       "pytest>=7.4.0",
       "black>=23.0.0",
       "ruff>=0.1.0",
   ]

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"
````

2. **Core Agent Loop** (`agents/base_agent.py`):
   - Implement ReAct pattern (Reasoning + Acting)
   - Direct boto3 Bedrock API calls for Claude
   - Tool calling loop with proper message formatting
   - Max iteration control
   - Clean, well-commented code showing the agent loop clearly

3. **RAG Pipeline** (`rag/` folder):
   - `embeddings.py`: Function to get embeddings from Bedrock Titan
   - `vector_store.py`: Simple in-memory vector store with cosine similarity search
   - `retriever.py`: Retrieve top-k relevant documents

4. **Bedrock Client** (`llm/bedrock_client.py`):
   - Wrapper for Bedrock API calls
   - Support for Claude 3.5 Sonnet (model ID: anthropic.claude-3-5-sonnet-20241022-v2:0)
   - Handle streaming responses (optional)
   - Tool use support with proper message formatting
   - Proper error handling

5. **Document Parser** (`tools/document_parser.py`):
   - Use LangChain's PyPDFLoader and Docx2txtLoader
   - Use RecursiveCharacterTextSplitter for intelligent chunking
   - Pipeline: load → split → return chunks with metadata
   - Clean interface that hides LangChain complexity

6. **Configuration Files**:
   - `.python-version`: Contains "3.10"
   - `.env.example`: Template for AWS credentials
 AWS_REGION=us-east-1
 AWS_ACCESS_KEY_ID=your_access_key
 AWS_SECRET_ACCESS_KEY=your_secret_key
 BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
 EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1
   - `utils/config.py`: Load config from environment variables

7. **Simple Examples**:
   - `examples/simple_rag_example.py`: Load a doc, embed it, query it
   - `examples/agent_example.py`: Run agent with a simple tool (e.g., calculator)

8. **README.md** with setup instructions:
````markdown
   # Agentic AI Project

   ## Setup

   ### Install uv (if not already installed)
```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
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

   ### Configure AWS credentials
```bash
   cp .env.example .env
   # Edit .env with your AWS credentials
```

   ## Usage

   ### Test RAG Pipeline
```bash
   python examples/simple_rag_example.py
```

   ### Test Agent
```bash
   python examples/agent_example.py
```
````

Key Requirements:
- Use boto3 directly for all Bedrock calls (no LangChain LLM wrappers)
- Show the raw API calls and message formatting clearly
- Add extensive comments explaining what each part does
- Keep it simple and educational - this is for learning
- Make it easy to see the agent's reasoning and tool calls (add logging/print statements)
- Use proper Python package structure with __init__.py files
- Follow modern Python best practices (type hints, docstrings)
- Make the code modular and easy to extend

Implementation Priority:
1. Start with `llm/bedrock_client.py` - foundation for everything
2. Then `agents/base_agent.py` - core agent loop
3. Then `rag/vector_store.py` and `rag/embeddings.py` - RAG basics
4. Then `tools/document_parser.py` - document processing
5. Finally `examples/` - tie it all together

Make sure each component is well-documented with:
- Module-level docstrings explaining purpose
- Function docstrings with parameters and return types
- Inline comments for complex logic
- Example usage in docstrings

I want to understand exactly how the agent loop and RAG work at a low level, so prioritize clarity over cleverness.

After you create the scaffold, I'll test it and then we can build the compliance_agent.py and research_agent.py on top of this foundation.