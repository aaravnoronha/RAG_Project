# RAG Agent System - Building AI Agents on Language Models

A comprehensive framework for building Retrieval-Augmented Generation (RAG) agents with multi-agent orchestration capabilities. This system enables you to create sophisticated AI agents that can access knowledge bases, use tools, and collaborate to solve complex problems.

## 🚀 Features

### Core Capabilities
- **RAG Implementation**: Complete retrieval-augmented generation pipeline
- **Multi-Agent Orchestration**: Coordinate multiple specialized agents
- **Tool Integration**: Extensible tool system for web search, code execution, calculations
- **Vector Storage**: Efficient similarity search using FAISS
- **Document Processing**: Support for PDF, DOCX, Markdown, and more
- **Memory Management**: Short-term and long-term memory for conversations
- **Async Operations**: Full async/await support for scalability

### Agent Types
- **Research Agents**: Web research and information gathering
- **Analyst Agents**: Data analysis and pattern recognition  
- **Coding Agents**: Code review, generation, and execution
- **QA Agents**: Document-based question answering
- **Coordinator Agents**: Task planning and workflow orchestration

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-agent-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
# or
export ANTHROPIC_API_KEY="your-api-key"
```

## 🎯 Quick Start

### Basic RAG Agent

```python
import asyncio
from rag_agent_system import AgentFactory

async def main():
    # Create a research agent
    agent = AgentFactory.create_research_agent("your-api-key")
    
    # Add documents to knowledge base
    agent.add_documents(["document.pdf", "data.txt"])
    
    # Ask questions
    response = await agent.process_query("What does the document say about AI?")
    print(response)

asyncio.run(main())
```

### Multi-Agent System

```python
from agent_orchestrator import AgentOrchestrator, SpecializedAgent

# Create orchestrator
orchestrator = AgentOrchestrator()

# Register specialized agents
orchestrator.register_agent(research_agent)
orchestrator.register_agent(analyst_agent)

# Execute complex workflow
results = await orchestrator.hierarchical_problem_solving(
    "Design a recommendation system"
)
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────┐
│              Agent Orchestrator              │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │Research │  │ Analyst │  │  Coder  │   │
│  │  Agent  │  │  Agent  │  │  Agent  │   │
│  └────┬────┘  └────┬────┘  └────┬────┘   │
│       └────────────┴────────────┘         │
│                    │                       │
├────────────────────┴───────────────────────┤
│              RAG Pipeline                   │
├─────────────────────────────────────────────┤
│  Document     →  Chunking  →  Embeddings   │
│  Processing                                 │
│                                             │
│  Vector Store ←  Retrieval ←  Query        │
│                                             │
│  Context      →  LLM       →  Response     │
│  Injection                                  │
└─────────────────────────────────────────────┘
```

### Key Classes

- **`RAGAgent`**: Base agent with RAG capabilities
- **`SpecializedAgent`**: Extended agent with role-specific abilities
- **`AgentOrchestrator`**: Manages multi-agent collaboration
- **`VectorStore`**: Handles embeddings and similarity search
- **`DocumentProcessor`**: Processes and chunks documents
- **`Tool`**: Base class for agent tools

## 📚 Usage Examples

### 1. Customer Support Bot

```python
from examples import CustomerSupportAgent

# Initialize with knowledge base
support = CustomerSupportAgent(api_key, "knowledge_base/")

# Handle customer queries
response = await support.handle_customer_query(
    "How do I return a product?"
)
```

### 2. Code Review Agent

```python
from examples import CodeReviewAgent

reviewer = CodeReviewAgent(api_key)
review = await reviewer.review_code(code_string, "python")

print(review['review'])
print(review['recommendations'])
```

### 3. Research Assistant

```python
from examples import ResearchAssistant

researcher = ResearchAssistant(api_key)
results = await researcher.conduct_research(
    topic="quantum computing applications",
    depth="comprehensive"
)
```

### 4. Document Q&A System

```python
from examples import DocumentQASystem

qa = DocumentQASystem(api_key)
qa.add_document("research_paper.pdf")

answer = await qa.ask_question(
    "What are the main findings?"
)
```

## 🛠️ Configuration

Edit `config.yaml` to customize:

- LLM settings (provider, model, parameters)
- Vector store configuration
- Document processing options
- Agent capabilities
- Tool configurations
- System prompts

## 🔧 Extending the System

### Adding Custom Tools

```python
from rag_agent_system import Tool

class DatabaseTool(Tool):
    @property
    def name(self):
        return "database_query"
    
    @property
    def description(self):
        return "Query a SQL database"
    
    async def execute(self, query: str):
        # Implementation
        return results
```

### Creating Custom Agents

```python
class DataAnalystAgent(SpecializedAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add specialized capabilities
    
    async def analyze_data(self, data):
        # Custom analysis logic
        return insights
```

## 🚦 Workflow Examples

### Sequential Workflow
```python
workflow = [
    WorkflowStep("research", AgentRole.RESEARCHER, ...),
    WorkflowStep("analyze", AgentRole.ANALYST, ...),
    WorkflowStep("report", AgentRole.SYNTHESIZER, ...)
]
```

### Parallel Execution
```python
tasks = [
    Task("Research topic A"),
    Task("Research topic B"),
    Task("Research topic C")
]
results = await orchestrator.parallel_execution(tasks)
```

### Consensus Decision
```python
consensus = await orchestrator.consensus_decision(
    "What technology stack should we use?",
    min_agents=3
)
```

## 📊 Performance Optimization

### Vector Store
- Use GPU-accelerated FAISS for large datasets
- Implement hierarchical indexing for millions of documents
- Consider cloud solutions (Pinecone, Weaviate) for production

### LLM Optimization
- Batch processing for multiple queries
- Caching for repeated questions
- Model selection based on task complexity

### Memory Management
- Implement conversation compression
- Use Redis for distributed memory storage
- Periodic memory consolidation

## 🔒 Security Considerations

- Store API keys in environment variables
- Implement input validation and sanitization
- Use sandboxed environments for code execution
- Add rate limiting for API calls
- Encrypt sensitive data in vector stores

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=rag_agent_system tests/
```

## 📈 Monitoring

The system includes built-in logging and monitoring:

- Performance metrics (response time, token usage)
- Error tracking with Sentry integration
- Prometheus metrics export
- Detailed activity logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **API key errors**: Set environment variables correctly
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. **Memory issues with large documents**: Adjust chunk size in config
   ```yaml
   document_processing:
     chunk_size: 500  # Smaller chunks
   ```

4. **Slow retrieval**: Optimize vector store settings
   ```python
   vector_store = VectorStore(
       index_type="ivf",  # Use IVF for large datasets
       nprobe=10
   )
   ```

## 📞 Support

- Documentation: See `/docs` folder
- Issues: GitHub Issues
- Discussions: GitHub Discussions

## 🎯 Roadmap

- [ ] Graph-based knowledge representation
- [ ] Multi-modal support (images, audio)
- [ ] Distributed agent execution
- [ ] AutoML for agent optimization
- [ ] Natural language workflow definition
- [ ] Real-time collaboration features
- [ ] Advanced reasoning chains
- [ ] Integration with more LLM providers

---
