"""
Advanced Agent Orchestrator - Multi-Agent Collaboration System
Enables multiple specialized agents to work together on complex tasks
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime
from collections import deque
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import yaml

from rag_agent_system import (
    RAGAgent, Message, AgentMemory, 
    LLMProvider, VectorStore, Tool
)


# ============= Agent Roles and Types =============

class AgentRole(Enum):
    """Defines different agent roles in the system"""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CODER = "coder"
    REVIEWER = "reviewer"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"


@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    required_tools: List[str]
    confidence_threshold: float = 0.8
    max_iterations: int = 5


@dataclass
class Task:
    """Represents a task in the system"""
    id: str
    description: str
    type: str
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class WorkflowStep:
    """Represents a step in a workflow"""
    name: str
    agent_role: AgentRole
    action: Callable
    inputs: List[str]
    outputs: List[str]
    conditions: Dict[str, Any] = field(default_factory=dict)


# ============= Specialized Agents =============

class SpecializedAgent(RAGAgent):
    """Extended RAG Agent with specialization capabilities"""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: List[AgentCapability],
        llm_provider: LLMProvider,
        vector_store: VectorStore,
        tools: List[Tool] = None,
        system_prompt: str = None
    ):
        super().__init__(llm_provider, vector_store, tools, system_prompt)
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.task_queue = deque()
        self.completed_tasks = []
        self.collaboration_history = []
    
    async def can_handle_task(self, task: Task) -> float:
        """Evaluate if this agent can handle a specific task"""
        # Use LLM to evaluate capability match
        evaluation_prompt = f"""
        Evaluate if an agent with role '{self.role.value}' and capabilities:
        {[cap.name for cap in self.capabilities]}
        
        Can handle this task:
        {task.description}
        
        Return a confidence score between 0 and 1.
        """
        
        response = await self.llm.generate([
            Message(role="system", content="You are an agent capability evaluator."),
            Message(role="user", content=evaluation_prompt)
        ])
        
        try:
            # Parse confidence from response
            confidence = float(response.strip())
            return min(max(confidence, 0), 1)
        except:
            return 0.5  # Default confidence
    
    async def execute_task(self, task: Task) -> Any:
        """Execute a specific task"""
        self.task_queue.append(task)
        task.status = "in_progress"
        task.assigned_agent = self.agent_id
        
        try:
            # Process the task using the base RAG capabilities
            result = await self.process_query(
                task.description,
                use_rag=True,
                use_tools=True
            )
            
            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()
            self.completed_tasks.append(task)
            
            return result
            
        except Exception as e:
            task.status = "failed"
            task.result = str(e)
            return None
        
        finally:
            self.task_queue.remove(task)
    
    async def collaborate_with(self, other_agent: 'SpecializedAgent', message: str) -> str:
        """Collaborate with another agent"""
        collaboration_entry = {
            'from': self.agent_id,
            'to': other_agent.agent_id,
            'message': message,
            'timestamp': datetime.now()
        }
        
        self.collaboration_history.append(collaboration_entry)
        other_agent.collaboration_history.append(collaboration_entry)
        
        # Other agent processes the message
        response = await other_agent.process_query(
            f"Collaboration request from {self.role.value}: {message}"
        )
        
        return response
    
    async def review_output(self, content: str, criteria: List[str]) -> Dict[str, Any]:
        """Review and validate output based on criteria"""
        review_prompt = f"""
        Review the following content based on these criteria:
        {', '.join(criteria)}
        
        Content to review:
        {content}
        
        Provide a structured review with:
        1. Score for each criterion (0-10)
        2. Overall assessment
        3. Suggestions for improvement
        """
        
        response = await self.process_query(review_prompt)
        
        return {
            'review': response,
            'reviewer': self.agent_id,
            'timestamp': datetime.now()
        }


# ============= Agent Orchestrator =============

class AgentOrchestrator:
    """Orchestrates multiple agents working together"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.agents: Dict[str, SpecializedAgent] = {}
        self.task_queue: deque = deque()
        self.workflow_graph = nx.DiGraph()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.config = self._load_config(config_path) if config_path else {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def register_agent(self, agent: SpecializedAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.agent_id] = agent
        print(f"Registered agent: {agent.agent_id} with role: {agent.role.value}")
    
    async def assign_task(self, task: Task) -> Optional[str]:
        """Assign a task to the most suitable agent"""
        best_agent = None
        best_confidence = 0
        
        # Evaluate all agents
        for agent_id, agent in self.agents.items():
            confidence = await agent.can_handle_task(task)
            if confidence > best_confidence:
                best_confidence = confidence
                best_agent = agent
        
        if best_agent and best_confidence > 0.5:
            task.assigned_agent = best_agent.agent_id
            return best_agent.agent_id
        
        return None
    
    async def execute_workflow(self, workflow: List[WorkflowStep], initial_input: Any) -> Any:
        """Execute a multi-step workflow"""
        context = {'input': initial_input, 'outputs': {}}
        
        for step in workflow:
            # Find agent for this step
            agent = None
            for ag in self.agents.values():
                if ag.role == step.agent_role:
                    agent = ag
                    break
            
            if not agent:
                raise ValueError(f"No agent found for role: {step.agent_role}")
            
            # Prepare inputs for this step
            step_inputs = {}
            for input_name in step.inputs:
                if input_name in context['outputs']:
                    step_inputs[input_name] = context['outputs'][input_name]
                elif input_name == 'input':
                    step_inputs[input_name] = context['input']
            
            # Execute the step
            task = Task(
                id=str(uuid.uuid4()),
                description=f"Execute {step.name} with inputs: {step_inputs}",
                type=step.name
            )
            
            result = await agent.execute_task(task)
            
            # Store outputs
            for output_name in step.outputs:
                context['outputs'][output_name] = result
        
        return context['outputs']
    
    async def parallel_execution(self, tasks: List[Task]) -> List[Any]:
        """Execute multiple tasks in parallel"""
        # Group tasks by dependencies
        dependency_graph = nx.DiGraph()
        
        for task in tasks:
            dependency_graph.add_node(task.id)
            for dep in task.dependencies:
                dependency_graph.add_edge(dep, task.id)
        
        # Execute in topological order with parallelization
        results = {}
        
        for layer in nx.topological_generations(dependency_graph):
            # Execute all tasks in this layer in parallel
            layer_tasks = [t for t in tasks if t.id in layer]
            
            futures = []
            for task in layer_tasks:
                agent_id = await self.assign_task(task)
                if agent_id:
                    agent = self.agents[agent_id]
                    futures.append(agent.execute_task(task))
            
            # Wait for all tasks in this layer to complete
            if futures:
                layer_results = await asyncio.gather(*futures)
                for task, result in zip(layer_tasks, layer_results):
                    results[task.id] = result
        
        return [results.get(t.id) for t in tasks]
    
    async def consensus_decision(self, question: str, min_agents: int = 3) -> Dict[str, Any]:
        """Get consensus from multiple agents"""
        responses = {}
        
        # Get responses from multiple agents
        agent_subset = list(self.agents.values())[:min_agents]
        
        for agent in agent_subset:
            response = await agent.process_query(question)
            responses[agent.agent_id] = response
        
        # Synthesize consensus
        synthesis_prompt = f"""
        Multiple agents have provided answers to: {question}
        
        Responses:
        {json.dumps(responses, indent=2)}
        
        Synthesize a consensus answer that:
        1. Identifies common agreements
        2. Notes any disagreements
        3. Provides a balanced final answer
        """
        
        # Use coordinator or first available agent for synthesis
        synthesizer = None
        for agent in self.agents.values():
            if agent.role == AgentRole.COORDINATOR:
                synthesizer = agent
                break
        
        if not synthesizer:
            synthesizer = list(self.agents.values())[0]
        
        consensus = await synthesizer.process_query(synthesis_prompt)
        
        return {
            'question': question,
            'individual_responses': responses,
            'consensus': consensus,
            'timestamp': datetime.now()
        }
    
    async def hierarchical_problem_solving(self, problem: str) -> Dict[str, Any]:
        """Solve complex problems using hierarchical decomposition"""
        
        # Step 1: Problem decomposition
        decomposer = None
        for agent in self.agents.values():
            if agent.role in [AgentRole.COORDINATOR, AgentRole.ANALYST]:
                decomposer = agent
                break
        
        if not decomposer:
            raise ValueError("No suitable agent for problem decomposition")
        
        decomposition_prompt = f"""
        Decompose this problem into sub-tasks:
        {problem}
        
        Provide a list of specific sub-tasks that can be solved independently.
        """
        
        decomposition = await decomposer.process_query(decomposition_prompt)
        
        # Parse sub-tasks (simplified - in production, use structured output)
        sub_tasks = [
            Task(
                id=str(uuid.uuid4()),
                description=line.strip(),
                type="sub_problem"
            )
            for line in decomposition.split('\n')
            if line.strip() and not line.startswith('#')
        ][:5]  # Limit to 5 sub-tasks for demo
        
        # Step 2: Solve sub-tasks
        sub_results = await self.parallel_execution(sub_tasks)
        
        # Step 3: Synthesize solution
        synthesis_prompt = f"""
        Original problem: {problem}
        
        Sub-task results:
        {json.dumps([{'task': t.description, 'result': r} 
                    for t, r in zip(sub_tasks, sub_results)], indent=2)}
        
        Synthesize these results into a comprehensive solution.
        """
        
        final_solution = await decomposer.process_query(synthesis_prompt)
        
        return {
            'problem': problem,
            'decomposition': decomposition,
            'sub_results': sub_results,
            'solution': final_solution
        }
    
    def visualize_workflow(self, workflow: List[WorkflowStep]) -> str:
        """Generate a visual representation of the workflow"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        G = nx.DiGraph()
        
        # Add nodes and edges
        for i, step in enumerate(workflow):
            G.add_node(step.name, role=step.agent_role.value)
            if i > 0:
                G.add_edge(workflow[i-1].name, step.name)
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes with colors based on role
        role_colors = {
            AgentRole.COORDINATOR: 'lightblue',
            AgentRole.RESEARCHER: 'lightgreen',
            AgentRole.ANALYST: 'lightyellow',
            AgentRole.CODER: 'lightcoral',
            AgentRole.REVIEWER: 'lightgray',
            AgentRole.EXECUTOR: 'lightpink',
            AgentRole.VALIDATOR: 'lightsalmon',
            AgentRole.SYNTHESIZER: 'lightcyan'
        }
        
        node_colors = [role_colors.get(G.nodes[node].get('role', ''), 'white') 
                      for node in G.nodes()]
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors,
                node_size=2000, font_size=10, font_weight='bold',
                arrows=True, edge_color='gray')
        
        plt.title("Agent Workflow Visualization")
        plt.savefig('/home/claude/workflow_visualization.png')
        plt.close()
        
        return "/home/claude/workflow_visualization.png"


# ============= Example Usage =============

async def demonstrate_multi_agent_system():
    """Demonstration of the multi-agent system"""
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    
    # Create mock LLM provider and vector store (you'd use real ones)
    class MockLLMProvider(LLMProvider):
        async def generate(self, messages: List[Message], **kwargs) -> str:
            # Simulate LLM response
            return f"Simulated response to: {messages[-1].content if messages else 'empty'}"
        
        async def generate_with_tools(self, messages: List[Message], tools: List[Tool], **kwargs):
            return await self.generate(messages), None
    
    llm = MockLLMProvider()
    vector_store = VectorStore()
    
    # Create specialized agents
    research_agent = SpecializedAgent(
        agent_id="research-001",
        role=AgentRole.RESEARCHER,
        capabilities=[
            AgentCapability(
                name="web_research",
                description="Research topics on the web",
                required_tools=["web_search"]
            ),
            AgentCapability(
                name="document_analysis",
                description="Analyze documents and papers",
                required_tools=[]
            )
        ],
        llm_provider=llm,
        vector_store=vector_store,
        system_prompt="You are a research specialist."
    )
    
    analyst_agent = SpecializedAgent(
        agent_id="analyst-001",
        role=AgentRole.ANALYST,
        capabilities=[
            AgentCapability(
                name="data_analysis",
                description="Analyze data and provide insights",
                required_tools=["calculator"]
            ),
            AgentCapability(
                name="pattern_recognition",
                description="Identify patterns and trends",
                required_tools=[]
            )
        ],
        llm_provider=llm,
        vector_store=vector_store,
        system_prompt="You are a data analyst."
    )
    
    coordinator_agent = SpecializedAgent(
        agent_id="coordinator-001",
        role=AgentRole.COORDINATOR,
        capabilities=[
            AgentCapability(
                name="task_planning",
                description="Plan and coordinate tasks",
                required_tools=[]
            ),
            AgentCapability(
                name="synthesis",
                description="Synthesize information from multiple sources",
                required_tools=[]
            )
        ],
        llm_provider=llm,
        vector_store=vector_store,
        system_prompt="You are a project coordinator."
    )
    
    # Register agents
    orchestrator.register_agent(research_agent)
    orchestrator.register_agent(analyst_agent)
    orchestrator.register_agent(coordinator_agent)
    
    # Example 1: Simple task assignment
    print("\n=== Example 1: Task Assignment ===")
    task = Task(
        description="Research the latest trends in artificial intelligence",
        type="research"
    )
    
    assigned = await orchestrator.assign_task(task)
    print(f"Task assigned to: {assigned}")
    
    # Example 2: Workflow execution
    print("\n=== Example 2: Workflow Execution ===")
    workflow = [
        WorkflowStep(
            name="research_phase",
            agent_role=AgentRole.RESEARCHER,
            action=lambda x: x,
            inputs=["input"],
            outputs=["research_results"]
        ),
        WorkflowStep(
            name="analysis_phase",
            agent_role=AgentRole.ANALYST,
            action=lambda x: x,
            inputs=["research_results"],
            outputs=["analysis_results"]
        ),
        WorkflowStep(
            name="synthesis_phase",
            agent_role=AgentRole.COORDINATOR,
            action=lambda x: x,
            inputs=["research_results", "analysis_results"],
            outputs=["final_report"]
        )
    ]
    
    results = await orchestrator.execute_workflow(
        workflow,
        "Analyze the impact of LLMs on software development"
    )
    print(f"Workflow results: {results}")
    
    # Example 3: Consensus decision
    print("\n=== Example 3: Consensus Decision ===")
    consensus = await orchestrator.consensus_decision(
        "What is the best approach to implement a chatbot?"
    )
    print(f"Consensus reached: {consensus['consensus']}")
    
    # Example 4: Hierarchical problem solving
    print("\n=== Example 4: Hierarchical Problem Solving ===")
    solution = await orchestrator.hierarchical_problem_solving(
        "Design a recommendation system for an e-commerce platform"
    )
    print(f"Solution: {solution['solution']}")
    
    # Visualize workflow
    viz_path = orchestrator.visualize_workflow(workflow)
    print(f"\nWorkflow visualization saved to: {viz_path}")


if __name__ == "__main__":
    asyncio.run(demonstrate_multi_agent_system())
