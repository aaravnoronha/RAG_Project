"""
Practical RAG Agent Examples
Demonstrates real-world usage of the RAG agent system
"""

import os
import asyncio
from pathlib import Path
import json
from typing import List, Dict, Any
from datetime import datetime

# Import our RAG system components
from rag_agent_system import (
    RAGAgent,
    DocumentProcessor,
    VectorStore,
    OpenAIProvider,
    AnthropicProvider,
    WebSearchTool,
    CalculatorTool,
    CodeExecutorTool,
    AgentFactory,
    Message
)

from agent_orchestrator import (
    AgentOrchestrator,
    SpecializedAgent,
    AgentRole,
    AgentCapability,
    Task,
    WorkflowStep
)


# ============= Example 1: Customer Support Bot =============

class CustomerSupportAgent:
    """RAG-powered customer support agent with knowledge base"""
    
    def __init__(self, api_key: str, knowledge_base_dir: str):
        self.llm = OpenAIProvider(api_key)
        self.vector_store = VectorStore()
        self.processor = DocumentProcessor()
        
        # Custom system prompt for customer support
        system_prompt = """You are a helpful customer support agent.
        Use the knowledge base to answer customer questions accurately.
        Always be polite, professional, and solution-oriented.
        If you cannot find an answer in the knowledge base, offer to escalate to a human agent."""
        
        self.agent = RAGAgent(
            llm_provider=self.llm,
            vector_store=self.vector_store,
            tools=[],
            system_prompt=system_prompt
        )
        
        # Load knowledge base
        self._load_knowledge_base(knowledge_base_dir)
    
    def _load_knowledge_base(self, kb_dir: str):
        """Load all documents from knowledge base directory"""
        kb_path = Path(kb_dir)
        if kb_path.exists():
            for file_path in kb_path.glob("**/*"):
                if file_path.suffix in ['.txt', '.md', '.pdf', '.docx']:
                    print(f"Loading {file_path}...")
                    doc = self.processor.process_document(str(file_path))
                    self.vector_store.add_documents(doc.chunks)
    
    async def handle_customer_query(self, query: str) -> str:
        """Process a customer support query"""
        response = await self.agent.process_query(query, use_rag=True)
        return response
    
    async def interactive_support(self):
        """Interactive customer support session"""
        print("Customer Support Bot Active")
        print("Type 'exit' to end session")
        print("-" * 50)
        
        while True:
            query = input("\nCustomer: ").strip()
            if query.lower() == 'exit':
                print("Thank you for using our support service!")
                break
            
            response = await self.handle_customer_query(query)
            print(f"\nSupport Agent: {response}")


# ============= Example 2: Code Review Agent =============

class CodeReviewAgent:
    """Agent that reviews code and provides feedback"""
    
    def __init__(self, api_key: str):
        self.llm = OpenAIProvider(api_key, model="gpt-4")
        self.vector_store = VectorStore()
        
        system_prompt = """You are an expert code reviewer.
        Analyze code for:
        - Bugs and potential issues
        - Performance optimizations
        - Security vulnerabilities
        - Code style and best practices
        - Documentation quality
        Provide constructive feedback with specific suggestions."""
        
        self.agent = RAGAgent(
            llm_provider=self.llm,
            vector_store=self.vector_store,
            tools=[CodeExecutorTool()],
            system_prompt=system_prompt
        )
    
    async def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Review code and provide structured feedback"""
        
        review_query = f"""
        Review this {language} code:
        
        ```{language}
        {code}
        ```
        
        Provide detailed feedback in these categories:
        1. Correctness & Bugs
        2. Performance
        3. Security
        4. Code Quality & Style
        5. Documentation
        
        Rate each category 1-10 and provide specific recommendations.
        """
        
        response = await self.agent.process_query(review_query)
        
        # Parse and structure the response
        review_result = {
            'timestamp': datetime.now().isoformat(),
            'language': language,
            'code_length': len(code),
            'review': response,
            'recommendations': self._extract_recommendations(response)
        }
        
        return review_result
    
    def _extract_recommendations(self, review_text: str) -> List[str]:
        """Extract actionable recommendations from review text"""
        # Simple extraction - in production, use more sophisticated parsing
        recommendations = []
        lines = review_text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['should', 'consider', 'recommend', 'suggest']):
                recommendations.append(line.strip())
        return recommendations
    
    async def review_file(self, file_path: str) -> Dict[str, Any]:
        """Review a code file"""
        with open(file_path, 'r') as f:
            code = f.read()
        
        language = Path(file_path).suffix[1:]  # Get extension without dot
        return await self.review_code(code, language)


# ============= Example 3: Research Assistant =============

class ResearchAssistant:
    """Multi-agent research system for comprehensive analysis"""
    
    def __init__(self, api_key: str):
        self.orchestrator = AgentOrchestrator()
        self.api_key = api_key
        self._setup_agents()
    
    def _setup_agents(self):
        """Setup specialized research agents"""
        
        # Data Collector Agent
        collector = SpecializedAgent(
            agent_id="collector-001",
            role=AgentRole.RESEARCHER,
            capabilities=[
                AgentCapability(
                    name="web_research",
                    description="Collect information from web sources",
                    required_tools=["web_search"]
                )
            ],
            llm_provider=OpenAIProvider(self.api_key),
            vector_store=VectorStore(),
            tools=[WebSearchTool()],
            system_prompt="You collect and organize research data from various sources."
        )
        
        # Fact Checker Agent
        fact_checker = SpecializedAgent(
            agent_id="factchecker-001",
            role=AgentRole.VALIDATOR,
            capabilities=[
                AgentCapability(
                    name="fact_verification",
                    description="Verify facts and check sources",
                    required_tools=["web_search"]
                )
            ],
            llm_provider=OpenAIProvider(self.api_key),
            vector_store=VectorStore(),
            tools=[WebSearchTool()],
            system_prompt="You verify facts and validate information accuracy."
        )
        
        # Analyst Agent
        analyst = SpecializedAgent(
            agent_id="analyst-001",
            role=AgentRole.ANALYST,
            capabilities=[
                AgentCapability(
                    name="data_analysis",
                    description="Analyze and interpret research data",
                    required_tools=["calculator"]
                )
            ],
            llm_provider=OpenAIProvider(self.api_key),
            vector_store=VectorStore(),
            tools=[CalculatorTool()],
            system_prompt="You analyze data and identify patterns and insights."
        )
        
        # Writer Agent
        writer = SpecializedAgent(
            agent_id="writer-001",
            role=AgentRole.SYNTHESIZER,
            capabilities=[
                AgentCapability(
                    name="report_writing",
                    description="Write comprehensive research reports",
                    required_tools=[]
                )
            ],
            llm_provider=OpenAIProvider(self.api_key),
            vector_store=VectorStore(),
            system_prompt="You write clear, well-structured research reports."
        )
        
        # Register all agents
        self.orchestrator.register_agent(collector)
        self.orchestrator.register_agent(fact_checker)
        self.orchestrator.register_agent(analyst)
        self.orchestrator.register_agent(writer)
    
    async def conduct_research(self, topic: str, depth: str = "standard") -> Dict[str, Any]:
        """Conduct comprehensive research on a topic"""
        
        # Define workflow based on depth
        if depth == "quick":
            workflow = [
                WorkflowStep(
                    name="collect_data",
                    agent_role=AgentRole.RESEARCHER,
                    action=lambda x: x,
                    inputs=["input"],
                    outputs=["raw_data"]
                ),
                WorkflowStep(
                    name="write_summary",
                    agent_role=AgentRole.SYNTHESIZER,
                    action=lambda x: x,
                    inputs=["raw_data"],
                    outputs=["summary"]
                )
            ]
        else:  # standard or deep
            workflow = [
                WorkflowStep(
                    name="initial_research",
                    agent_role=AgentRole.RESEARCHER,
                    action=lambda x: x,
                    inputs=["input"],
                    outputs=["raw_data"]
                ),
                WorkflowStep(
                    name="fact_checking",
                    agent_role=AgentRole.VALIDATOR,
                    action=lambda x: x,
                    inputs=["raw_data"],
                    outputs=["verified_data"]
                ),
                WorkflowStep(
                    name="analysis",
                    agent_role=AgentRole.ANALYST,
                    action=lambda x: x,
                    inputs=["verified_data"],
                    outputs=["insights"]
                ),
                WorkflowStep(
                    name="report_writing",
                    agent_role=AgentRole.SYNTHESIZER,
                    action=lambda x: x,
                    inputs=["verified_data", "insights"],
                    outputs=["final_report"]
                )
            ]
        
        # Execute research workflow
        results = await self.orchestrator.execute_workflow(workflow, topic)
        
        return {
            'topic': topic,
            'depth': depth,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }


# ============= Example 4: Document Q&A System =============

class DocumentQASystem:
    """Interactive Q&A system for document collections"""
    
    def __init__(self, api_key: str):
        self.llm = OpenAIProvider(api_key)
        self.vector_store = VectorStore()
        self.processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        
        system_prompt = """You are a document Q&A assistant.
        Answer questions based solely on the provided document context.
        Always cite the specific parts of documents you're referencing.
        If the answer is not in the documents, clearly state that."""
        
        self.agent = RAGAgent(
            llm_provider=self.llm,
            vector_store=self.vector_store,
            system_prompt=system_prompt
        )
        
        self.document_metadata = {}
    
    def add_document(self, file_path: str, metadata: Dict[str, Any] = None):
        """Add a document to the Q&A system"""
        print(f"Processing {file_path}...")
        
        doc = self.processor.process_document(file_path, metadata)
        self.vector_store.add_documents(doc.chunks)
        
        self.document_metadata[doc.id] = {
            'path': file_path,
            'title': metadata.get('title', Path(file_path).stem) if metadata else Path(file_path).stem,
            'chunks': len(doc.chunks),
            'added_at': datetime.now().isoformat()
        }
        
        print(f"Added {len(doc.chunks)} chunks from {file_path}")
    
    async def ask_question(self, question: str, num_sources: int = 3) -> Dict[str, Any]:
        """Ask a question about the documents"""
        
        # Search for relevant chunks
        search_results = self.vector_store.search(question, k=num_sources)
        
        # Get answer from agent
        answer = await self.agent.process_query(question, use_rag=True)
        
        # Format response with sources
        sources = []
        for chunk, score in search_results:
            doc_meta = self.document_metadata.get(chunk.document_id, {})
            sources.append({
                'document': doc_meta.get('title', 'Unknown'),
                'relevance_score': float(score),
                'excerpt': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            })
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the system"""
        return list(self.document_metadata.values())


# ============= Main Demo Function =============

async def main():
    """Run demonstrations of the RAG agent system"""
    
    # Note: You need to set your API key
    API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    print("=" * 60)
    print("RAG Agent System Demonstrations")
    print("=" * 60)
    
    while True:
        print("\nSelect a demo:")
        print("1. Customer Support Bot")
        print("2. Code Review Agent")
        print("3. Research Assistant")
        print("4. Document Q&A System")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("\n--- Customer Support Bot Demo ---")
            # Create a sample knowledge base directory
            kb_dir = "/home/claude/knowledge_base"
            os.makedirs(kb_dir, exist_ok=True)
            
            # Create sample FAQ document
            with open(f"{kb_dir}/faq.txt", "w") as f:
                f.write("""
                Q: What are your business hours?
                A: We are open Monday-Friday 9AM-5PM EST.
                
                Q: How do I return a product?
                A: You can return products within 30 days with receipt.
                
                Q: Do you offer international shipping?
                A: Yes, we ship to over 50 countries worldwide.
                """)
            
            support_agent = CustomerSupportAgent(API_KEY, kb_dir)
            await support_agent.interactive_support()
            
        elif choice == "2":
            print("\n--- Code Review Agent Demo ---")
            reviewer = CodeReviewAgent(API_KEY)
            
            # Sample code to review
            sample_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    avg = total / len(numbers)
    return avg

# Usage
result = calculate_average([1, 2, 3, 4, 5])
print(result)
            """
            
            print("Reviewing code...")
            review = await reviewer.review_code(sample_code, "python")
            print(f"\nReview Results:\n{review['review']}")
            
        elif choice == "3":
            print("\n--- Research Assistant Demo ---")
            researcher = ResearchAssistant(API_KEY)
            
            topic = input("Enter research topic: ").strip()
            if topic:
                print(f"Conducting research on: {topic}")
                research_results = await researcher.conduct_research(topic, "quick")
                print(f"\nResearch Results:\n{json.dumps(research_results, indent=2)}")
            
        elif choice == "4":
            print("\n--- Document Q&A System Demo ---")
            qa_system = DocumentQASystem(API_KEY)
            
            # Create a sample document
            sample_doc = "/home/claude/sample_doc.txt"
            with open(sample_doc, "w") as f:
                f.write("""
                Introduction to RAG Systems
                
                Retrieval-Augmented Generation (RAG) combines the benefits of 
                retrieval-based and generation-based approaches. RAG systems 
                use vector databases to store and retrieve relevant information,
                which is then used to augment the language model's responses.
                
                Key components include:
                - Document processing and chunking
                - Embedding generation
                - Vector storage and similarity search
                - Context injection into LLM prompts
                
                Benefits of RAG:
                - Reduced hallucination
                - Ability to cite sources
                - Easy knowledge updates
                - Better performance on domain-specific tasks
                """)
            
            qa_system.add_document(sample_doc, {"title": "RAG Systems Guide"})
            
            while True:
                question = input("\nAsk a question (or 'back' to return): ").strip()
                if question.lower() == 'back':
                    break
                
                result = await qa_system.ask_question(question)
                print(f"\nAnswer: {result['answer']}")
                print(f"\nSources:")
                for source in result['sources']:
                    print(f"- {source['document']}: {source['excerpt']}")
            
        elif choice == "5":
            print("\nExiting demos. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    asyncio.run(main())
