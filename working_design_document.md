# InsightFlow AI Implementation Plan

This document outlines the 90-minute implementation plan for InsightFlow AI with its two-tier persona system.

## Selected Components

### Persona Types (6 Total)
1. **Analytical/Diagnostic** - Methodical examination with logical connections
2. **Scientific/STEM-Explainer** - Evidence-based reasoning with empirical data
3. **Philosophical/Spiritual** - Holistic perspectives examining deeper meaning
4. **Factual/Practical** - Clear, straightforward presentation of information
5. **Metaphorical/Creative-Analogy** - Explanation through vivid analogies
6. **Futuristic/Speculative** - Forward-looking exploration of possible futures

### Personalities (3 Total)
1. **Sherlock Holmes** (Analytical) - Deductive reasoning with detailed observation
2. **Richard Feynman** (Scientific) - First-principles physics with clear explanations
3. **Hannah Fry** (Factual) - Math-meets-society storytelling with practical examples

## Implementation Timeline

### 0:00-0:15: Setup Project Structure
```bash
mkdir -p persona_configs data_sources utils/persona
```

1. Copy config files for selected personas/personalities
2. Create basic folder structure for data sources

### 0:15-0:35: Implement Persona System

Create `utils/persona/base.py`:

```python
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from langchain_core.documents import Document

class PersonaReasoning(ABC):
    """Base class for all persona reasoning types"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.id = config.get("id")
        self.name = config.get("name")
        self.traits = config.get("traits", [])
        self.system_prompt = config.get("system_prompt", "")
        self.examples = config.get("examples", [])
        self.is_personality = not config.get("is_persona_type", True)
        
    @abstractmethod
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate a perspective response based on query and optional context"""
        pass
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for this persona"""
        return self.system_prompt
        
    def get_examples(self) -> List[str]:
        """Get example responses for this persona"""
        return self.examples

class PersonaFactory:
    """Factory for creating persona instances from config files"""
    
    def __init__(self, config_dir="persona_configs"):
        self.config_dir = config_dir
        self.configs = {}
        self.load_configs()
        
    def load_configs(self):
        """Load all JSON config files"""
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.config_dir, filename), "r") as f:
                    config = json.load(f)
                    if "id" in config:
                        self.configs[config["id"]] = config
                        
    def get_config(self, persona_id: str) -> Optional[Dict[str, Any]]:
        """Get config for a persona"""
        return self.configs.get(persona_id)
        
    def create_persona(self, persona_id: str) -> Optional[PersonaReasoning]:
        """Create a persona instance based on ID"""
        config = self.get_config(persona_id)
        if not config:
            return None
            
        if config.get("is_persona_type", True):
            # This is a persona type
            persona_type = config.get("type")
            if persona_type == "analytical":
                from .analytical import AnalyticalReasoning
                return AnalyticalReasoning(config)
            elif persona_type == "scientific":
                from .scientific import ScientificReasoning
                return ScientificReasoning(config)
            elif persona_type == "philosophical":
                from .philosophical import PhilosophicalReasoning
                return PhilosophicalReasoning(config)
            elif persona_type == "factual":
                from .factual import FactualReasoning
                return FactualReasoning(config)
            elif persona_type == "metaphorical":
                from .metaphorical import MetaphoricalReasoning
                return MetaphoricalReasoning(config)
            elif persona_type == "futuristic":
                from .futuristic import FuturisticReasoning
                return FuturisticReasoning(config)
        else:
            # This is a personality
            parent_type = config.get("parent_type")
            parent_config = self.get_config(parent_type)
            if parent_config:
                if persona_id == "holmes":
                    from .holmes import HolmesReasoning
                    return HolmesReasoning(config, parent_config)
                elif persona_id == "feynman":
                    from .feynman import FeynmanReasoning
                    return FeynmanReasoning(config, parent_config)
                elif persona_id == "fry":
                    from .fry import FryReasoning
                    return FryReasoning(config, parent_config)
                    
        return None
```

### 0:35-0:50: Implement Persona Implementations

Create `utils/persona/impl.py`:

```python
from .base import PersonaReasoning
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import Dict, List, Optional, Any
from langchain_core.documents import Document

class LLMPersonaReasoning(PersonaReasoning):
    """Base implementation that uses LLM to generate responses"""
    
    def __init__(self, config: Dict[str, Any], llm=None):
        super().__init__(config)
        # Use shared LLM instance if provided, otherwise create one
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        """Generate perspective using LLM with persona's system prompt"""
        
        # Build prompt with context if available
        context_text = ""
        if context and len(context) > 0:
            context_text = "\n\nRelevant information:\n" + "\n".join([doc.page_content for doc in context])
            
        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Query: {query}{context_text}\n\nPlease provide your perspective on this query based on your unique approach.")
        ]
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        return response.content

# Specialized implementations for each persona type
class AnalyticalReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        # Could add analytical-specific logic here
        return super().generate_perspective(query, context)

class ScientificReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        # Could add scientific-specific logic here
        return super().generate_perspective(query, context)

class PhilosophicalReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        # Could add philosophical-specific logic here
        return super().generate_perspective(query, context)

class FactualReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        # Could add factual-specific logic here
        return super().generate_perspective(query, context)

class MetaphoricalReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        # Could add metaphorical-specific logic here
        return super().generate_perspective(query, context)

class FuturisticReasoning(LLMPersonaReasoning):
    def generate_perspective(self, query: str, context: Optional[List[Document]] = None) -> str:
        # Could add futuristic-specific logic here
        return super().generate_perspective(query, context)

# Personality implementations
class HolmesReasoning(LLMPersonaReasoning):
    def __init__(self, config: Dict[str, Any], parent_config: Dict[str, Any], llm=None):
        super().__init__(config, llm)
        self.parent_config = parent_config
        
class FeynmanReasoning(LLMPersonaReasoning):
    def __init__(self, config: Dict[str, Any], parent_config: Dict[str, Any], llm=None):
        super().__init__(config, llm)
        self.parent_config = parent_config

class FryReasoning(LLMPersonaReasoning):
    def __init__(self, config: Dict[str, Any], parent_config: Dict[str, Any], llm=None):
        super().__init__(config, llm)
        self.parent_config = parent_config
```

### 0:50-1:00: Data Acquisition Script

Create `download_data.py`:

```python
import os
import requests
import urllib.request
from bs4 import BeautifulSoup
import re

# Create directories
personas = ["analytical", "scientific", "philosophical", "factual", "metaphorical", "futuristic", "holmes", "feynman", "fry"]
for persona in personas:
    os.makedirs(f"data_sources/{persona}", exist_ok=True)

# ANALYTICAL / HOLMES
print("Downloading Analytical/Holmes data...")
holmes_urls = [
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes Adventures
    "https://www.gutenberg.org/files/2097/2097-0.txt",  # Sign of the Four
    "https://www.gutenberg.org/files/108/108-0.txt"     # A Study in Scarlet
]
for url in holmes_urls:
    filename = url.split("/")[-1]
    with open(f"data_sources/holmes/{filename}", "wb") as f:
        f.write(requests.get(url).content)
    print(f"Downloaded {filename}")

# Also get general analytical reasoning examples
analytical_url = "https://raw.githubusercontent.com/logicalthinking/examples/main/analytical_examples.txt"
try:
    with open("data_sources/analytical/examples.txt", "wb") as f:
        f.write(requests.get(analytical_url).content)
except:
    # If the URL doesn't exist, create a simple file with examples from the config
    with open("data_sources/analytical/examples.txt", "w") as f:
        f.write("""When we examine this problem carefully, several key patterns emerge. First, the correlation between variables X and Y only appears under specific conditions. Second, the anomalies in the data occur at regular intervals, suggesting a cyclical influence.
The evidence suggests three possible explanations. Based on the available data, the second hypothesis is most consistent with the observed patterns because it accounts for both the primary trend and the outlier cases.""")

# SCIENTIFIC / FEYNMAN
print("Downloading Scientific/Feynman data...")
# Public domain Feynman lecture excerpts
feynman_url = "https://raw.githubusercontent.com/turingcollege/feynman-lectures/main/feynman_excerpts.txt"
try:
    with open("data_sources/feynman/lectures.txt", "wb") as f:
        f.write(requests.get(feynman_url).content)
except:
    # Create a sample file if URL doesn't exist
    with open("data_sources/feynman/lectures.txt", "w") as f:
        f.write("""Physics isn't the most important thing. Love is.
Nature uses only the longest threads to weave her patterns, so each small piece of her fabric reveals the organization of the entire tapestry.
The first principle is that you must not fool yourself — and you are the easiest person to fool.
I think I can safely say that nobody understands quantum mechanics.""")

# Also download general scientific examples
with open("data_sources/scientific/examples.txt", "w") as f:
    f.write("""Based on the empirical evidence, we can observe three key factors influencing this phenomenon...
The data suggests a strong correlation between X and Y, with a statistical significance of p<0.01, indicating...
While multiple hypotheses have been proposed, the research indicates that the most well-supported explanation is...""")

# PHILOSOPHICAL
print("Downloading Philosophical data...")
philosophical_url = "https://www.gutenberg.org/files/1497/1497-0.txt"  # Republic by Plato
with open("data_sources/philosophical/republic.txt", "wb") as f:
    f.write(requests.get(philosophical_url).content)

# Also create examples file
with open("data_sources/philosophical/examples.txt", "w") as f:
    f.write("""When we look more deeply at this question, we can see that the apparent separation between observer and observed is actually an illusion. Our consciousness is not separate from the phenomenon we're examining.
This situation invites us to consider not just the practical implications, but also the deeper patterns that connect these events to larger cycles of change and transformation.""")

# FACTUAL / HANNAH FRY
print("Downloading Factual/Hannah Fry data...")
# Create sample excerpts from Hannah Fry's public talks
with open("data_sources/fry/excerpts.txt", "w") as f:
    f.write("""When we talk about algorithms making decisions, we're not just discussing abstract mathematics – we're talking about systems that increasingly determine who gets a job, who gets a loan, and sometimes even who goes to prison. The math matters because its consequences are profoundly human.
The fascinating thing about probability is how it challenges our intuition. Take the famous Birthday Paradox: in a room of just 23 people, there's a 50% chance that at least two people share a birthday. With 70 people, that probability jumps to 99.9%.
Data never speaks for itself – it always comes with human assumptions baked in. When we look at a dataset showing correlation between two variables, we need to ask: what might be causing this relationship?""")

# Also get general factual examples
with open("data_sources/factual/examples.txt", "w") as f:
    f.write("""The key facts about this topic are: First, the system operates in three distinct phases. Second, each phase requires specific inputs. Third, the output varies based on initial conditions.
Based on the available evidence, we can state with high confidence that the primary factor is X, with secondary contributions from Y and Z. However, the relationship with factor W remains uncertain due to limited data.""")

# METAPHORICAL
print("Downloading Metaphorical data...")
with open("data_sources/metaphorical/examples.txt", "w") as f:
    f.write("""Think of quantum computing like a combination lock with multiple correct combinations simultaneously. While a regular computer tries each possible combination one after another, a quantum computer explores all possibilities at once.
The relationship between the economy and interest rates is like a boat on the ocean. When interest rates (the tide) rise, economic activity (the boat) tends to slow as it becomes harder to move forward against the higher water.""")

# FUTURISTIC
print("Downloading Futuristic data...")
# Create examples file with futuristic reasoning
with open("data_sources/futuristic/examples.txt", "w") as f:
    f.write("""When we examine the current trajectory of this technology, we can identify three distinct possible futures: First, the mainstream path where incremental improvements lead to wider adoption but minimal disruption. Second, a transformative scenario where an unexpected breakthrough creates entirely new capabilities that fundamentally alter the existing paradigm. Third, a regulatory response scenario where societal concerns lead to significant constraints on development.
This current challenge resembles the fictional 'Kardashev transition problem' often explored in speculative fiction. The difficulty isn't just technical but involves coordinating systems that operate at vastly different scales and timeframes.""")

# Try to get some public domain sci-fi
scifi_url = "https://www.gutenberg.org/files/62/62-0.txt"  # H.G. Wells - The War of the Worlds
with open("data_sources/futuristic/war_of_worlds.txt", "wb") as f:
    f.write(requests.get(scifi_url).content)

print("All data downloads completed!")
```

### 1:00-1:05: Update State Management

Create `insight_state.py`:

```python
from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.documents import Document

class InsightFlowState(TypedDict):
    # Query information
    panel_type: str  # "research" or "discussion"
    query: str
    selected_personas: List[str]
    
    # Research results
    persona_responses: Dict[str, str]
    synthesized_response: Optional[str]
    
    # Control
    current_step_name: str
    error_message: Optional[str]
```

### 1:05-1:20: LangGraph Nodes Implementation

Add the following to `app.py`:

```python
# --- LANGGRAPH NODES FOR INSIGHTFLOW AI ---

async def run_planner_agent(state: InsightFlowState) -> InsightFlowState:
    """Plan the research approach for multiple personas"""
    query = state["query"]
    selected_personas = state["selected_personas"]
    
    # For 90-minute implementation, we'll use a simplified planning approach
    # that just assigns the same query to each selected persona
    
    state["current_step_name"] = "execute_persona_tasks"
    return state

async def execute_persona_tasks(state: InsightFlowState) -> InsightFlowState:
    """Execute tasks for each selected persona"""
    query = state["query"]
    selected_personas = state["selected_personas"]
    persona_factory = cl.user_session.get("persona_factory")
    
    # Initialize responses dict if not exists
    if "persona_responses" not in state:
        state["persona_responses"] = {}
    
    # Process each persona
    for persona_id in selected_personas:
        persona = persona_factory.create_persona(persona_id)
        if persona:
            try:
                # Get perspective from this persona
                response = await cl.make_async(persona.generate_perspective)(query)
                state["persona_responses"][persona_id] = response
            except Exception as e:
                print(f"Error getting {persona_id} perspective: {e}")
                state["persona_responses"][persona_id] = f"Could not generate perspective: {str(e)}"
    
    state["current_step_name"] = "synthesize_responses"
    return state

async def synthesize_responses(state: InsightFlowState) -> InsightFlowState:
    """Combine perspectives from different personas"""
    query = state["query"]
    persona_responses = state["persona_responses"]
    
    if not persona_responses:
        state["synthesized_response"] = "No persona perspectives were generated."
        state["current_step_name"] = "present_results"
        return state
    
    # Prepare input for synthesizer
    perspectives_text = ""
    for persona_id, response in persona_responses.items():
        perspectives_text += f"\n\n{persona_id.capitalize()} Perspective:\n{response}"
    
    # Use LLM to synthesize
    messages = [
        SystemMessage(content="You are an expert synthesizer who combines multiple perspectives into a coherent response. Create a unified view that highlights key insights from each perspective."),
        HumanMessage(content=f"Query: {query}\n\nPerspectives:{perspectives_text}\n\nPlease synthesize these perspectives into a coherent response.")
    ]
    
    try:
        synthesizer_response = await llm_synthesizer.ainvoke(messages)
        state["synthesized_response"] = synthesizer_response.content
    except Exception as e:
        state["synthesized_response"] = f"Error synthesizing perspectives: {str(e)}"
    
    state["current_step_name"] = "present_results"
    return state

async def present_results(state: InsightFlowState) -> InsightFlowState:
    """Present the final results to the user"""
    
    # Send the synthesized response
    await cl.Message(content=state["synthesized_response"]).send()
    
    # Create expandable elements for each persona perspective
    for persona_id, response in state["persona_responses"].items():
        persona_name = persona_id.capitalize()
        
        # Get proper display name from config if available
        persona_factory = cl.user_session.get("persona_factory")
        if persona_factory:
            config = persona_factory.get_config(persona_id)
            if config and "name" in config:
                persona_name = config["name"]
        
        # Create element for this perspective
        element = cl.Expandable(
            title=f"{persona_name}'s Perspective",
            content=response
        )
        await cl.Message(content="", elements=[element]).send()
    
    state["current_step_name"] = "END"
    return state

# --- LANGGRAPH SETUP FOR INSIGHTFLOW AI ---
insight_graph_builder = StateGraph(InsightFlowState)

# Add all nodes
insight_graph_builder.add_node("planner_agent", run_planner_agent)
insight_graph_builder.add_node("execute_persona_tasks", execute_persona_tasks)
insight_graph_builder.add_node("synthesize_responses", synthesize_responses)
insight_graph_builder.add_node("present_results", present_results)

# Add edges
insight_graph_builder.add_edge("planner_agent", "execute_persona_tasks")
insight_graph_builder.add_edge("execute_persona_tasks", "synthesize_responses")
insight_graph_builder.add_edge("synthesize_responses", "present_results")
insight_graph_builder.add_edge("present_results", END)

# Set entry point
insight_graph_builder.set_entry_point("planner_agent")

# Compile the graph
insight_flow_graph = insight_graph_builder.compile()
```

### 1:20-1:30: Chainlit Integration

Update the Chainlit handlers:

```python
@cl.on_chat_start
async def start_chat():
    """Initialize the InsightFlow AI session"""
    
    # Initialize persona factory and load configs
    persona_factory = PersonaFactory(config_dir="persona_configs")
    cl.user_session.set("persona_factory", persona_factory)
    
    # Initialize state
    initial_state = InsightFlowState(
        panel_type="research",
        query="",
        selected_personas=[],
        persona_responses={},
        synthesized_response=None,
        current_step_name="awaiting_query",
        error_message=None
    )
    
    # Initialize LangGraph
    graph = insight_flow_graph 
    cl.user_session.set("insight_state", initial_state)
    cl.user_session.set("insight_graph", graph)
    
    # Create persona selection UI
    await cl.Message(content="# Welcome to InsightFlow AI Research Assistant\n\nSelect personas to include in your research:").send()
    
    # Create selection actions
    actions = [
        cl.Action(name="select_personas", value=None, label="Select Personas", description="Choose which personas to include")
    ]
    await cl.Message(content="Please select the personas you want to include in your research team.", actions=actions).send()

@cl.on_action
async def on_action(action):
    """Handle Chainlit actions"""
    if action.name == "select_personas":
        # Create a modal for persona selection
        persona_factory = cl.user_session.get("persona_factory")
        
        # Only show our implemented personas
        options = []
        for persona_id in ["holmes", "feynman", "fry", "analytical", "scientific", 
                           "philosophical", "factual", "metaphorical", "futuristic"]:
            config = persona_factory.get_config(persona_id)
            if config:
                options.append(
                    cl.Option(
                        value=persona_id,
                        label=config.get("name", persona_id.capitalize()),
                        description=config.get("description", "")[:100]
                    )
                )
        
        # Show selection dialog
        res = await cl.Select(
            values=options,
            label="Select Personas",
            description="Choose which personas to include in your research team",
            multiple=True
        ).send()
        
        if res and res.value:
            state = cl.user_session.get("insight_state")
            state["selected_personas"] = res.value
            cl.user_session.set("insight_state", state)
            
            # Confirm selection
            selected_names = []
            for persona_id in res.value:
                config = persona_factory.get_config(persona_id)
                if config:
                    selected_names.append(config.get("name", persona_id.capitalize()))
            
            await cl.Message(content=f"Research team set with: {', '.join(selected_names)}. What would you like to explore?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    """Handle user messages"""
    state = cl.user_session.get("insight_state")
    graph = cl.user_session.get("insight_graph")
    
    if not state or not graph:
        await cl.Message(content="Session error. Please refresh the page.").send()
        return
    
    # If we have selected personas and this is a query
    if state["selected_personas"]:
        # Set query in state
        state["query"] = message.content
        
        # Show thinking indicator
        await cl.Message(content="Thinking...").send()
        
        # Run the graph
        thread_id = cl.user_session.get("id", "default_thread_id")
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = await graph.ainvoke(state, config)
            cl.user_session.set("insight_state", final_state)
        except Exception as e:
            print(f"Error in graph execution: {e}")
            await cl.Message(content=f"I encountered an error: {e}").send()
    else:
        await cl.Message(content="Please select personas for your research team first.").send()
```

### 1:30-2:00: Final Architecture and Execution Flow

## LangGraph Implementation

The application leverages LangGraph to orchestrate the multi-persona research flow. The architecture follows a sequential node-based pipeline where each node performs a specific function and passes an updated state to the next node.

```
+---------------------+      +-------------------------+      +------------------------+      +-----------------+      +-----+
|                     |      |                         |      |                        |      |                 |      |     |
| planner_agent       +----->+ execute_persona_tasks   +----->+ synthesize_responses  +----->+ present_results +----->+ END |
|                     |      |                         |      |                        |      |                 |      |     |
+---------------------+      +-------------------------+      +------------------------+      +-----------------+      +-----+
     Plan research              Generate perspectives           Combine perspectives          Present to user
     - Examine query           - Create persona instances      - Collect all responses       - Show synthesized view
     - Identify personas       - Parallel API requests         - Create unified view         - Display individual
     - Structure approach      - Handle timeouts               - Structure insights           perspectives if enabled
```

### Current Implementation Architecture

1. **Node: planner_agent**
   - Input: User query and selected personas
   - Process: Plans approach (currently simplified)
   - Output: Updated state with research plan
   - Progress: 10%

2. **Node: execute_persona_tasks**
   - Input: State with query and personas
   - Process: Parallel generation of perspectives from each persona
   - Output: State with persona_responses dict
   - Progress: 40-80% (dynamically updates per persona)

3. **Node: synthesize_responses**
   - Input: State with all persona responses
   - Process: Creates unified view combining all perspectives
   - Output: State with synthesized_response
   - Progress: 80-95%

4. **Node: present_results**
   - Input: State with synthesized response and persona responses
   - Process: Formats and sends messages to user
   - Output: Final state ready to END
   - Progress: 95-100%

### Data Flow

- InsightFlowState object maintains the query state and responses
- User-selected personas determine which reasoning approaches are used
- Each persona operates independently but shares the same query
- Final synthesis combines all perspectives into a unified view

### Performance Optimizations

The implementation has been optimized for both performance and user experience:

1. **Parallel Processing**
   - Persona tasks execute in parallel to reduce total processing time
   - Asynchronous API calls prevent blocking

2. **Timeout Handling**
   - Each perspective generation has a timeout (30 seconds)
   - Overall graph execution has a timeout (150 seconds)
   - Fallback responses are provided if timeouts occur

3. **Progress Tracking**
   - Dynamic progress updates based on the current processing stage
   - Per-persona progress indicators during the longest-running execute_persona_tasks phase

4. **Speed Options**
   - Direct mode: Bypass the multi-persona system entirely for fastest responses
   - Quick mode: Use fewer personas (max 2) for faster processing
   - Perspective toggle: Hide individual perspectives for cleaner output

5. **Personalized Perspective Weighting**: Allow users to adjust the influence of each perspective type based on their preferences and needs.

## Final Architecture

The implementation successfully combines:
1. A two-tier persona system with type-based reasoning and personalities
2. A LangGraph-powered orchestration layer
3. A streamlined Chainlit user interface
4. Performance optimizations for response speed and user experience
5. Command-based system control for flexibility

## Project Structure

```
insightflow_ai/
├── app.py
├── chainlit.md
├── Dockerfile
├── insight_state.py
├── data_sources/
│   ├── analytical/
│   ├── scientific/
│   ├── philosophical/
│   ├── factual/
│   ├── metaphorical/
│   ├── futuristic/
│   ├── holmes/
│   ├── feynman/
│   └── fry/
├── persona_configs/
│   ├── analytical.json
│   ├── scientific.json
│   ├── philosophical.json
│   ├── factual.json
│   ├── metaphorical.json
│   ├── futuristic.json
│   ├── holmes.json
│   ├── feynman.json
│   └── fry.json
├── utils/
│   ├── __init__.py
│   ├── parse_prompts.py
│   └── persona/
│       ├── __init__.py
│       ├── base.py
│       └── impl.py
├── README.md
└── pyproject.toml
```

## Implementation Progress Tracking

| Component | Status | Notes |
|-----------|--------|-------|
| Project Structure | Completed | Basic directory structure created |
| Persona Configs | Completed | All JSON configurations created |
| Data Downloads | Completed | Reference data downloaded for all personas |
| Base Persona Classes | Completed | Factory and implementation classes created |
| LangGraph Nodes | Completed | All nodes implemented and connected |
| Chainlit Integration | Completed | UI with command system implemented |
| Visualization System | Completed | Mermaid diagrams and DALL-E integration |
| Export Functionality | Completed | PDF and Markdown export with visualizations |
| Testing | In Progress | Basic functionality verified |

*Last Updated: May 12, 2025* 

### 2:00-2:30: Enhanced Visualization System

To provide users with rich visual representations of the multi-perspective insights, we've integrated two complementary visualization systems:

## Visualization Components

### 1. Mermaid Diagram Integration

We've added Mermaid.js integration to automatically generate concept maps that visualize the relationships between different perspectives:

```
flowchart LR
    query[User Query] --> analytical
    query --> scientific
    query --> philosophical
    analytical --> synthesis
    scientific --> synthesis
    philosophical --> synthesis
    
    style query fill:#f9f,stroke:#333,stroke-width:2px
    style synthesis fill:#bbf,stroke:#333,stroke-width:2px
```

The Mermaid diagrams provide a structured representation of how different perspectives relate to the query and contribute to the synthesis. This visualization helps users understand the flow of information and reasoning.

### 2. DALL-E Generated Visualizations

We've implemented DALL-E 3 integration to create hand-drawn style visual notes that represent the synthesized information:

```python
async def generate_dalle_image(prompt: str) -> Optional[str]:
    """Generate a DALL-E image and return the URL"""
    try:
        # Create a detailed prompt for hand-drawn style visualization
        full_prompt = f"Create a hand-drawn style visual note or sketch that represents: {prompt}. 
                       Make it look like a thoughtful drawing with annotations and key concepts highlighted."
        
        # Call DALL-E to generate the image
        response = await openai_client.images.generate(
            model="dall-e-3",
            prompt=full_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        # Return the URL of the generated image
        return response.data[0].url
    except Exception as e:
        print(f"DALL-E image generation failed: {e}")
        return None
```

The DALL-E images provide a more creative, human-like representation that can capture nuances and relationships in a way that's often more intuitive and engaging than text or structured diagrams.

## Enhanced User Interface Controls

To support these visualization features, we've added several new UI commands:

### Visualization Toggles

```
/visualization on|off - Toggle showing visualizations (Mermaid diagrams & DALL-E images)
/visual_only on|off - Show only visualizations without text (faster)
```

When either visualization mode is enabled, both the Mermaid diagram and DALL-E image are generated and displayed.

### Visualization Display Logic

The system intelligently decides what to display based on user preferences:

1. **Standard Mode** (`/visualization on`):
   - Shows the synthesized text response
   - Shows the DALL-E hand-drawn visualization
   - Shows the Mermaid concept map
   - Shows individual perspectives (if enabled)

2. **Visual-Only Mode** (`/visual_only on`):
   - Shows only the DALL-E hand-drawn visualization
   - Shows only the Mermaid concept map
   - Hides text responses and individual perspectives
   - Falls back to text if visualizations fail

## Fallback and Error Handling

We've implemented robust error handling to ensure a good user experience even when visualizations fail:

1. **Multiple Fallback Levels**:
   - If the main Mermaid diagram fails, falls back to a simpler diagram
   - If DALL-E image generation fails, still shows the Mermaid diagram
   - If both visualizations fail, falls back to text display

2. **Progress Tracking**:
   - Shows percentage complete during visualization generation
   - Provides clear feedback when visualizations succeed or fail

## Performance Considerations

The visualization features are optimized for both performance and quality:

1. **Parallel Processing**:
   - DALL-E image generation runs in parallel with other operations
   - Timeouts prevent hanging on slow API responses

2. **Selective Generation**:
   - In direct mode, visualizations are skipped entirely
   - In quick mode, fewer personas means faster visualization generation

These enhancements significantly improve the user experience by providing multiple ways to understand and engage with the multi-perspective information.

## Current Architecture

The application now uses the following enhanced architecture:

```
                                  ┌─────────────┐
                                  │   User Query │
                                  └──────┬──────┘
                                         │
                        ┌────────────────┼────────────────┐
                        │                │                │
                ┌───────▼───────┐┌───────▼───────┐┌───────▼───────┐
                │  Analytical   ││  Scientific   ││ Philosophical │
                │  Perspective  ││  Perspective  ││  Perspective  │
                └───────┬───────┘└───────┬───────┘└───────┬───────┘
                        │                │                │
                        └────────────────┼────────────────┘
                                         │
                                  ┌──────▼──────┐
                                  │  Synthesis  │
                                  └──────┬──────┘
                                         │
                    ┌──────────────────┬─┴─────────────────┐
                    │                  │                   │
            ┌───────▼────────┐┌────────▼──────────┐┌───────▼────────┐
            │ Mermaid Diagram ││ DALL-E Visualization ││  Export System  │
            └────────────────┘└───────────────────┘└───────┬────────┘
                                                           │
                                                 ┌─────────┴─────────┐
                                                 │                   │
                                          ┌──────▼───────┐ ┌─────────▼──────┐
                                          │ PDF Document │ │ Markdown File  │
                                          └──────────────┘ └────────────────┘
```

This architecture shows the complete InsightFlow AI system with multi-perspective analysis, visualization capabilities, and export functionality for both PDF and Markdown formats. 

### 2:30-3:00: Export System

To allow users to save, share, and reference their insights beyond the chat interface, we've implemented a comprehensive export system that supports both PDF and Markdown formats.

## Export Functionality

The export system allows users to convert the complete insights analysis, including all visualizations and individual perspectives, into shareable documents. This functionality is critical for professional use cases where insights need to be preserved, distributed, or incorporated into other documents.

### Export Commands

Two new commands have been added to the user interface:

```
/export_md - Export the current insight analysis to a markdown file
/export_pdf - Export the current insight analysis to a PDF file
```

### Implementation Details

1. **Directory Management**:
   - System automatically creates an `exports` directory if it doesn't exist
   - Files are saved with timestamped filenames to prevent overwriting

2. **Markdown Export**:
   ```python
   async def export_to_markdown(state: InsightFlowState):
       """Export the current insight analysis to a markdown file"""
       # Create exports directory if it doesn't exist
       Path("./exports").mkdir(exist_ok=True)
       
       # Generate a unique filename with timestamp
       timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
       random_id = await generate_random_id()
       filename = f"exports/insightflow_analysis_{timestamp}_{random_id}.md"
       
       # Create markdown content with sections
       md_content = f"""# InsightFlow AI Analysis
   *Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
   
   ## Query
   {query}
   
   ## {panel_mode} Insights
   {synthesized}
   
   """
       
       # Add all perspectives and visualizations
       # [Code continued...]
   ```

3. **PDF Export**:
   ```python
   async def export_to_pdf(state: InsightFlowState):
       """Export the current insight analysis to a PDF file"""
       # Similar setup to markdown export
       
       try:
           # Create PDF using fpdf
           pdf = fpdf.FPDF()
           pdf.add_page()
           
           # Add sections with formatting
           # [Content formatting code...]
           
           # Special handling for visualizations
           # Download and embed DALL-E image
           # Parse and represent Mermaid diagram as text
           
           # Output PDF
           pdf.output(filename)
           return filename, None
       except Exception as e:
           return None, f"Error exporting to PDF: {str(e)}"
   ```

### Export Format Features

#### 1. Markdown Export
- Complete formatting with headers, sections, and emphasis
- Embedded links to DALL-E visualizations
- Mermaid diagram code that renders in compatible viewers (GitHub, VS Code with extensions)
- Individual perspectives with proper attribution
- Easily shareable and convertible to other formats

#### 2. PDF Export
- Professional document layout with consistent fonts and spacing
- Embedded DALL-E visualization directly in the document
- Text representation of the concept map relationships
- Complete inclusion of all perspectives
- Universally viewable format requiring no special software

### Technical Implementation

The export system includes several sophisticated technical components:

1. **Image Handling**:
   - Downloads DALL-E images from URLs
   - Embeds them directly in PDFs
   - Uses references in Markdown for compatibility

2. **Mermaid Diagram Processing**:
   - Preserves diagram code in Markdown for rendering
   - Parses relationship structure for text representation in PDF
   - Maintains the conceptual relationships visualized in the diagram

3. **Error Handling**:
   - Graceful fallbacks for all components
   - Clear error messages if export fails
   - Verification of content before attempting export

### User Experience

The export system is designed for a seamless user experience:

1. **Simplicity**:
   - Single-command execution
   - Clear feedback on success or failure
   - File location reported to user

2. **Completeness**:
   - All content from the analysis is included
   - Visualizations are properly handled
   - Formatting preserves the structure and emphasis

3. **Flexibility**:
   - Choice of format based on user needs
   - Files saved locally for easy access
   - Unique filenames prevent overwriting previous exports

This export functionality significantly enhances the utility of InsightFlow AI, transforming it from a transient chat interface into a professional research tool capable of producing shareable, archivable outputs. The ability to export complete analyses, including visualizations, makes the system much more valuable for professional, academic, and personal knowledge management contexts. 

### 3:00-3:30: Data Acquisition and RAG Implementation

To implement the fine-tuned perspective-specific RAG systems for InsightFlow AI, we've developed a comprehensive plan for data acquisition and system implementation leveraging the AIE6 framework.

## Source Acquisition Plan

### 1. Analytical Reasoning Sources
- **Sherlock Holmes collections (Project Gutenberg)**
  - `wget https://www.gutenberg.org/files/1661/1661-0.txt -O data_sources/analytical/sherlock_adventures.txt`
  - `wget https://www.gutenberg.org/files/2097/2097-0.txt -O data_sources/analytical/sign_of_four.txt`
  - `wget https://www.gutenberg.org/files/108/108-0.txt -O data_sources/analytical/study_in_scarlet.txt`
- **arXiv papers**
  - Use [arXiv API](https://arxiv.org/help/api) to download papers on analytical reasoning and logical methods
  - Example: `python -m arxiv --max-results 100 "cat:cs.AI AND logical analysis methodology" --download`

### 2. Scientific Reasoning Sources
- **Feynman lectures**
  - Public domain excerpts from CalTech: `wget http://www.feynmanlectures.info/public_excerpts.html`
- **David Deutsch papers**
  - Access via [Oxford Academic Repository](https://academic.oup.com/journals)
- **PubMed articles**
  - Use [PubMed API](https://www.ncbi.nlm.nih.gov/home/develop/api/) to download scientific methodology papers
  - `python -m pubmed "scientific methodology empirical evidence" --count 200 --download`

### 3. Philosophical Reasoning Sources
- **Plato & Socrates works**
  - `wget https://www.gutenberg.org/files/1497/1497-0.txt -O data_sources/philosophical/republic.txt`
- **Vivekananda works**
  - `wget https://www.ramakrishnavivekananda.info/vivekananda/complete_works.htm -O data_sources/philosophical/vivekananda_index.html`
- **Krishnamurti teachings**
  - `wget --recursive --level=2 https://jkrishnamurti.org/content/collected-works -O data_sources/philosophical/krishnamurti_works`
- **Naval Ravikant quotes**
  - `wget https://nav.al/wisdom -O data_sources/philosophical/naval_wisdom.html`

### 4. Factual Reasoning Sources
- **Hannah Fry materials**
  - Download transcripts from TED talks and BBC programs
  - `youtube-dl --extract-audio --write-auto-sub "https://www.youtube.com/watch?v=Rzhpf1Ai7Z4"`
- **Wikipedia articles**
  - Use Wikipedia API to download featured articles across domains
  - `python -m wikipedia_downloader --featured --count 500 --output data_sources/factual/wikipedia`

### 5. Metaphorical Reasoning Sources
- **Literary metaphors collection**
  - `wget https://www.poetryfoundation.org/poems/browse#page=1&sort_by=recently_added -O data_sources/metaphorical/poetry_index.html`
- **Metaphor databases**
  - Master Metaphor List: `wget https://araw.mede.uic.edu/~alansz/metaphor/METAPHORLIST.pdf`

### 6. Futuristic Reasoning Sources
- **Asimov works (public domain only)**
  - `wget https://www.gutenberg.org/files/search/?query=asimov -O data_sources/futuristic/asimov_list.html`
- **H.G. Wells works**
  - `wget https://www.gutenberg.org/files/36/36-0.txt -O data_sources/futuristic/war_of_worlds.txt`
- **qntm works (with permission)**
  - `wget https://qntm.org/structure -O data_sources/futuristic/qntm_index.html`

## RAG Implementation Plan Based on AIE6 Framework

### 1. Data Processing Pipeline

```python
# Based on AIE6 modules 02_Embeddings_and_RAG and 04_Production_RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

# Create perspective-specific chunking functions 
def chunk_analytical_texts(texts, chunk_size=1000):
    # Preserve logical structures, analytical patterns
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(texts)

def chunk_scientific_texts(texts, chunk_size=1500):
    # Keep methodology sections together
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=300,
        separators=["Methods", "Results", "\n## ", "\n\n", "\n", " "]
    )
    return splitter.split_documents(texts)

def chunk_philosophical_texts(texts, chunk_size=2000):
    # Preserve philosophical arguments and concepts
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=400,
        separators=["\n## ", "\n### ", "\nChapter", "\n\n", "\n", " "]
    )
    return splitter.split_documents(texts)

def chunk_factual_texts(texts, chunk_size=800):
    # Focus on fact-dense segments
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=150,
        separators=[".\n", "\n## ", "\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(texts)

def chunk_metaphorical_texts(texts, chunk_size=1200):
    # Keep metaphors and analogies intact
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=250,
        separators=["\n\n", "\n", "like a", "as if", " "]
    )
    return splitter.split_documents(texts)

def chunk_futuristic_texts(texts, chunk_size=1800):
    # Preserve scenario descriptions and concepts
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=350,
        separators=["\n## ", "Chapter", "\n\n", "\n", " "]
    )
    return splitter.split_documents(texts)
```

### 2. Embedding Selection and Fine-tuning

```python
# Based on AIE6 module 09_Finetuning_Embeddings
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

# Function to create perspective-specific embedding models
def create_perspective_embeddings(perspective_name, training_pairs):
    # Start with base model
    base_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create training examples
    training_examples = [
        InputExample(texts=[query, perspective_text], label=1.0)
        for query, perspective_text in training_pairs
    ]
    
    # Training setup
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model=base_model)
    
    # Train the model
    base_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    # Save the fine-tuned model
    output_path = f"./models/embeddings/{perspective_name}"
    base_model.save(output_path)
    
    return base_model

# Generate synthetic training data for each perspective
def generate_training_pairs(perspective_name, example_texts, queries):
    """
    Uses the AIE6 07_Synthetic_Data_Generation approach to create
    perspective-specific training pairs of (query, relevant_text)
    """
    # Implementation based on 07_Synthetic_Data_Generation_and_LangSmith
    # [Code for synthetic data generation would go here]
    pass
```

### 3. Vector Database Creation

```python
# Based on AIE6 module 04_Production_RAG
from langchain.vectorstores import Chroma
import os

# Create a separate vector DB for each perspective type
def create_perspective_db(perspective_name, documents, embedding_function):
    db_path = f"./perspective_dbs/{perspective_name}"
    os.makedirs(db_path, exist_ok=True)
    
    # Add metadata to indicate perspective type
    for doc in documents:
        if not doc.metadata:
            doc.metadata = {}
        doc.metadata["perspective"] = perspective_name
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=db_path
    )
    vectorstore.persist()
    return vectorstore

# Function to create all perspective DBs
def create_all_perspective_dbs(data_sources, embedding_models):
    dbs = {}
    for perspective in ["analytical", "scientific", "philosophical", 
                       "factual", "metaphorical", "futuristic"]:
        # Get chunking function for this perspective
        chunk_fn = globals()[f"chunk_{perspective}_texts"]
        
        # Process documents
        raw_docs = data_sources[perspective]
        chunked_docs = chunk_fn(raw_docs)
        
        # Create vector DB
        embedding_model = embedding_models[perspective]
        dbs[perspective] = create_perspective_db(perspective, chunked_docs, embedding_model)
    
    return dbs
```

### 4. RAG Chain Assembly

```python
# Based on lessons from modules 04_Production_RAG
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

def create_perspective_prompts():
    """Create specialized prompts for each perspective"""
    prompts = {}
    
    prompts["analytical"] = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an analytical reasoner focused on logical connections and patterns.
        
        Use the following retrieved context to analyze the question:
        {context}
        
        Question: {question}
        
        Provide a methodical analysis that examines the logical structure, identifies patterns, 
        and reaches conclusions based on evidence and reasoning:"""
    )
    
    prompts["scientific"] = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a scientific reasoner in the style of Richard Feynman.
        
        Use the following retrieved context to address the question scientifically:
        {context}
        
        Question: {question}
        
        Provide a scientific explanation that draws on empirical evidence, considers 
        cause-effect relationships, and applies scientific principles:"""
    )
    
    # Create similar templates for other perspectives
    
    return prompts

def create_perspective_rag_chains(vectorstores, prompts):
    """Create specialized RAG chains for each perspective"""
    chains = {}
    
    for perspective, vectorstore in vectorstores.items():
        # Create retriever with perspective-specific configuration
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": {"perspective": perspective}
            }
        )
        
        # Temperature settings vary by perspective type
        temperatures = {
            "analytical": 0.2,
            "scientific": 0.3,
            "philosophical": 0.6,
            "factual": 0.1,
            "metaphorical": 0.7,
            "futuristic": 0.5
        }
        
        # Create chain
        llm = OpenAI(temperature=temperatures[perspective])
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompts[perspective]
            }
        )
        
        chains[perspective] = chain
    
    return chains
```

### 5. Orchestration with LangGraph

```python
# Based on module 05_Our_First_Agent_with_LangGraph and 06_Multi_Agent_with_LangGraph
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

# Implement the key workflow nodes for InsightFlow
def run_perspective(state, perspective_name, chains):
    """Run a specific perspective on the query"""
    query = state["query"]
    
    # Get the appropriate chain
    chain = chains[perspective_name]
    
    # Generate perspective
    result = chain.invoke({"query": query})
    
    # Update state
    state["persona_responses"][perspective_name] = result
    
    return state

def synthesize_perspectives(state, llm_synthesizer):
    """Combine multiple perspectives into a coherent response"""
    query = state["query"]
    perspectives = state["persona_responses"]
    
    # Build input for synthesizer
    perspectives_text = ""
    for perspective_name, response in perspectives.items():
        perspectives_text += f"\n\n{perspective_name.capitalize()} Perspective:\n{response}"
    
    # Create messages for synthesizer
    messages = [
        SystemMessage(content="You are a synthesis expert that combines multiple perspectives into a coherent response."),
        HumanMessage(content=f"Query: {query}\n\nPerspectives:{perspectives_text}\n\nSynthesize these perspectives.")
    ]
    
    # Get synthesis
    synthesis = llm_synthesizer.invoke(messages)
    
    # Update state
    state["synthesized_response"] = synthesis.content
    
    return state

def create_multi_perspective_graph(chains, llm_synthesizer):
    """Create the LangGraph for the InsightFlow system"""
    graph = StateGraph("InsightFlowState")
    
    # Add nodes for each perspective
    for perspective in chains.keys():
        # Use a lambda to create a specialized function for each perspective
        perspective_fn = lambda state, p=perspective: run_perspective(state, p, chains)
        graph.add_node(perspective, perspective_fn)
    
    # Add synthesis node
    synthesis_fn = lambda state: synthesize_perspectives(state, llm_synthesizer)
    graph.add_node("synthesize", synthesis_fn)
    
    # Connect perspective nodes (all run in parallel)
    for perspective in chains.keys():
        graph.add_edge("start", perspective)
        graph.add_edge(perspective, "synthesize")
    
    graph.add_edge("synthesize", END)
    
    return graph.compile()
```

### 6. Evaluation Framework

```python
# Based on module 08_Evaluating_RAG_With_Ragas
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.metrics.critique import harmfulness
import pandas as pd

def create_evaluation_dataset(queries, gold_answers):
    """Create dataset for evaluation"""
    data = {
        "query": queries,
        "answer": ["" for _ in queries],  # Will be filled by the system
        "ground_truth": gold_answers,
        "contexts": [[] for _ in queries]  # Will be filled with retrieved contexts
    }
    
    return pd.DataFrame(data)

def evaluate_perspective_rag(perspective_name, chain, eval_dataset):
    """Evaluate a specific perspective RAG"""
    # Run queries through the chain
    for i, query in enumerate(eval_dataset["query"]):
        result = chain.invoke({"query": query})
        eval_dataset.at[i, "answer"] = result
        # Also capture contexts used
        contexts = chain.retriever.get_relevant_documents(query)
        eval_dataset.at[i, "contexts"] = [doc.page_content for doc in contexts]
    
    # Run metrics
    results = {
        "faithfulness": faithfulness.score(eval_dataset),
        "answer_relevancy": answer_relevancy.score(eval_dataset),
        "context_precision": context_precision.score(eval_dataset),
        "context_recall": context_recall.score(eval_dataset),
        "harmfulness": harmfulness.score(eval_dataset)
    }
    
    return results

def evaluate_all_perspectives(chains, eval_dataset):
    """Evaluate all perspective RAGs"""
    results = {}
    
    for perspective, chain in chains.items():
        perspective_results = evaluate_perspective_rag(perspective, chain, eval_dataset.copy())
        results[perspective] = perspective_results
    
    return results
```

## Integration with InsightFlow AI

This RAG system implementation integrates with our existing LangGraph architecture through the following steps:

1. **Data Acquisition**: Run the source acquisition commands to populate our data directories
2. **Data Processing**: Apply perspective-specific chunking to all collected texts
3. **Embedding Fine-Tuning**: Create specialized embedding models for each reasoning style
4. **Vector DB Creation**: Build six vector databases, one for each perspective
5. **RAG Chain Assembly**: Create optimized retrieval chains for each perspective
6. **LangGraph Integration**: Replace our current perspective generation with the RAG-based perspective nodes
7. **Continuous Evaluation**: Use the evaluation framework to monitor and improve performance

This implementation maintains our existing user interface while enhancing the quality and depth of the perspectives by grounding them in perspective-specific knowledge bases.

The perspective-specific RAG systems will provide much richer responses, with each perspective drawing from its own specialized knowledge base while maintaining its distinct reasoning approach. 

### 3:30-4:00: Embedding Fine-Tuning for Perspective-Aware Retrieval

## Comprehensive Embedding Fine-Tuning Plan

Based on the AIE6 course materials, we've developed a detailed embedding fine-tuning plan to create specialized embeddings for each reasoning perspective in InsightFlow AI.

### 1. Embedding Model Selection

**Base Model Choice:**
- **Primary Model:** sentence-transformers/all-MiniLM-L6-v2
  - Rationale: Excellent balance of performance and efficiency with 384-dimension embeddings
  - Small enough for rapid iteration but powerful enough for nuanced perspective differentiation
- **Alternative Models for Evaluation:**
  - BAAI/bge-small-en-v1.5 - For comparison of retrieval performance
  - intfloat/e5-small - If higher accuracy is needed at cost of speed

### 2. Perspective-Specific Training Data Generation

#### 2.1. Training Data Structure

Using the techniques from 07_Synthetic_Data_Generation_and_LangSmith:

```python
# Core data structure for each perspective
perspective_training_data = {
    "analytical": {
        "positive_pairs": [],  # (query, relevant_analytical_text)
        "negative_pairs": [],  # (query, non_analytical_text)
        "hard_negatives": []   # (query, similar_but_wrong_perspective_text)
    },
    # Similar structures for other perspectives
}
```

#### 2.2. Training Data Sources

For each perspective, generate three types of training examples:

1. **Positive Examples (3,000 per perspective)**
   - Source text chunks from perspective-specific sources
   - Generate queries relevant to those chunks using LLM

2. **Negative Examples (1,500 per perspective)**
   - Source text chunks from other perspective types
   - Pair with queries seeking the target perspective

3. **Hard Negative Examples (1,500 per perspective)**
   - Source text chunks that superficially match the perspective but miss its essence
   - E.g., Texts with scientific terms but lacking scientific reasoning for "scientific" perspective

#### 2.3. Data Generation Process

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# Step 1: Generate perspective-specific questions
question_gen_prompt = PromptTemplate(
    input_variables=["perspective", "topic"],
    template="""
    Generate 5 questions that would best be answered using a {perspective} perspective 
    on the topic of {topic}. The questions should specifically require {perspective} 
    thinking and would be awkward to answer from other perspectives.
    
    Each question should be on a new line without numbering.
    """
)

question_gen_chain = LLMChain(
    llm=ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo"),
    prompt=question_gen_prompt
)

# Generate questions for each perspective across diverse topics
topics = ["technology", "history", "art", "science", "ethics", "economics", "culture", "psychology"]
perspective_questions = {}
for perspective in perspectives:
    perspective_questions[perspective] = []
    for topic in topics:
        questions = question_gen_chain.run(perspective=perspective, topic=topic)
        perspective_questions[perspective].extend(questions.strip().split('\n'))
```

### 3. Fine-Tuning Process

#### 3.1. Training Pipeline

Based directly on the SentenceTransformers approach from AIE6 module 09:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

def finetune_perspective_embeddings(perspective_name, training_examples, num_epochs=3):
    """Fine-tune embeddings for a specific perspective"""
    
    # Initialize base model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Convert to InputExample format
    train_examples = []
    for example in training_examples:
        if example["label"] > 0:  # Positive pair
            train_examples.append(InputExample(texts=[example["query"], example["text"]], label=1.0))
        else:  # Negative pair
            train_examples.append(InputExample(texts=[example["query"], example["text"]], label=0.0))
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Use Multiple Negatives Ranking Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Configure training with warmup
    warmup_steps = int(len(train_dataloader) * 0.1)
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': 2e-5},
        show_progress_bar=True,
        output_path=f"models/embeddings/{perspective_name}"
    )
    
    return model
```

#### 3.2. Hyperparameter Optimization

Conduct experiments varying:
- Learning rate: [1e-5, 2e-5, 5e-5]
- Batch size: [16, 32, 64]
- Epochs: [3, 5, 10]
- Loss functions: [MultipleNegativesRankingLoss, TripletLoss, ContrastiveLoss]

### 4. Model Evaluation Framework

#### 4.1. Task-Specific Evaluation Metrics

```python
def evaluate_perspective_embeddings(model, eval_dataset, perspective):
    """Evaluate embedding model on perspective-specific retrieval"""
    
    # Generate embeddings for all texts in corpus
    corpus_embeddings = model.encode(eval_dataset["texts"], convert_to_tensor=True)
    
    # For each query
    metrics = {
        "mrr": 0,  # Mean Reciprocal Rank
        "precision@5": 0,
        "recall@10": 0,
        "ndcg@10": 0,
        "perspective_accuracy": 0  # How often it retrieves the correct perspective
    }
    
    for query, relevant_idxs, perspective_label in zip(eval_dataset["queries"], 
                                                      eval_dataset["relevant_idxs"],
                                                      eval_dataset["perspective_labels"]):
        # Encode query
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Calculate similarity scores
        similarity_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        
        # Get top matches
        top_indices = torch.argsort(similarity_scores, descending=True).tolist()
        
        # Calculate metrics
        # [Detailed metric calculation code would be here]
        
    return metrics
```

#### 4.2. Comparative A/B Testing

Test the specialized embeddings against general embeddings on:
1. **Retrieval Alignment**: How well does the model retrieve perspective-relevant passages?
2. **Perspective Classification**: Given a text, can the model recognize its reasoning perspective?
3. **Contrastive Retrieval**: Between two passages, can the model select the one that matches the requested perspective?

### 5. Integration with InsightFlow AI

#### 5.1. Vector Database Setup

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def create_perspective_db(perspective_name, documents):
    """Create a perspective-specific vector DB"""
    
    # Load the fine-tuned embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=f"models/embeddings/{perspective_name}",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Configure Chroma settings for this perspective
    persist_directory = f"./dbs/{perspective_name}_db"
    
    # Add perspective metadata
    for doc in documents:
        doc.metadata['perspective'] = perspective_name
    
    # Create and persist the DB
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    return db
```

#### 5.2. RAG Chain Integration

```python
def create_perspective_rag(perspective_name, vectorstore):
    """Create a perspective-specific RAG chain"""
    
    # Create specialized retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.8,  # Balance between relevance and diversity
            "filter": {"perspective": perspective_name}
        }
    )
    
    # Create specialized prompt
    perspective_prompts = {
        "analytical": "Analyze this from a logical, structured perspective...",
        "scientific": "Examine this scientifically, based on evidence...",
        # Define for other perspectives
    }
    
    # Create and return the chain
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": perspective_prompts[perspective_name]}
    )
    
    return chain
```

### 6. Implementation Timeline

1. **Week 1: Data Collection & Generation (3 days)**
   - Source texts for each perspective type
   - Generate synthetic training data
   - Create evaluation datasets

2. **Week 1-2: Model Training (4 days)**
   - Set up training pipeline
   - Train perspective-specific embedding models
   - Optimize hyperparameters
   - Conduct evaluations

3. **Week 2: Integration & Testing (3 days)**
   - Integrate with vector databases
   - Set up perspective-specific RAG chains
   - Implement LangGraph integration
   - Conduct end-to-end testing

4. **Week 3: Refinement (ongoing)**
   - Analyze performance
   - Refine training data
   - Retrain with adjustments
   - A/B test in the production system

### 7. Final Implementation Plan

1. Create a dedicated `embeddings_trainer.py` script that:
   - Downloads all source materials
   - Generates training data
   - Trains six perspective-specific embedding models
   - Evaluates and reports on model performance

2. Create a `perspective_db_builder.py` script that:
   - Processes source data using perspective-specific chunking
   - Creates six vector databases using the specialized embeddings
   - Includes quality-checking functions
   - Produces performance metrics

3. Modify the existing LangGraph implementation to:
   - Use the new retrieval-augmented perspective generators
   - Maintain all existing UI functionality
   - Allow seamless switching between the approaches

## 1-Hour Implementation Plan

Given the time constraint of just 1 hour, here's a streamlined, focused approach to implement embedding fine-tuning for the multi-perspective system:

### Phase 1: Setup (10 minutes)
```python
# Create necessary directories
!mkdir -p models/embeddings data/training

# Install required packages
!pip install -q sentence-transformers datasets torch
```

### Phase 2: Get Pre-existing Datasets (10 minutes)

Instead of creating custom data, use HuggingFace datasets:

```python
from datasets import load_dataset
import pandas as pd

# Load philosophical texts dataset (small)
philosophy = load_dataset("IshuAgg/philosophy_text", split="train").to_pandas()

# Load scientific papers abstracts
science = load_dataset("ccdv/pubmed-summarization", split="train[:1000]").to_pandas()
science = science.rename(columns={"article": "text"})

# Create a quick combined dataset with perspective labels
philosophical_data = philosophy[['text']].sample(500)
philosophical_data['perspective'] = 'philosophical'

scientific_data = science[['text']].sample(500)
scientific_data['perspective'] = 'scientific'

# Combine datasets
train_data = pd.concat([philosophical_data, scientific_data])
train_data.to_csv('data/training/perspective_data.csv', index=False)

print(f"Training dataset created with {len(train_data)} examples")
```

### Phase 3: Fine-Tune a Single Perspective Model (25 minutes)

Focus on just one perspective (philosophical) as proof of concept:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random

# Load base model (small and quick to fine-tune)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Prepare training data
philosophical_texts = train_data[train_data['perspective'] == 'philosophical']['text'].tolist()
other_texts = train_data[train_data['perspective'] != 'philosophical']['text'].tolist()

# Create training pairs - simplified approach
train_examples = []

# Positive examples: Same perspective texts should be closer
for i in range(250):
    if i < len(philosophical_texts) - 1:
        train_examples.append(InputExample(
            texts=[philosophical_texts[i], philosophical_texts[i+1]],
            label=1.0
        ))

# Negative examples: Different perspective texts should be further apart
for i in range(250):
    if i < len(philosophical_texts) and i < len(other_texts):
        train_examples.append(InputExample(
            texts=[philosophical_texts[i], other_texts[i]],
            label=0.0
        ))

# Train with minimal hyperparameters
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.ContrastiveLoss(model)

# Quick training - 1 epoch only given time constraints
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=50,
    show_progress_bar=True
)

# Save the model
model.save('models/embeddings/philosophical')
print("Model training complete and saved!")
```

### Phase 4: Quick Implementation in RAG (15 minutes)

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load philosophical embeddings
philosophical_embeddings = HuggingFaceEmbeddings(
    model_name='models/embeddings/philosophical',
    model_kwargs={'device': 'cpu'}
)

# Quick chunking of a sample philosophical text
loader = TextLoader("data_sources/philosophical/republic.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Add perspective metadata
for chunk in chunks:
    chunk.metadata['perspective'] = 'philosophical'

# Create vector store with specialized embeddings
db = Chroma.from_documents(
    documents=chunks,
    embedding=philosophical_embeddings,
    persist_directory="./dbs/philosophical_specialized"
)
db.persist()

# Create a test retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# Test the retriever
test_query = "What is justice according to Plato?"
results = retriever.get_relevant_documents(test_query)
print(f"Retrieved {len(results)} documents")
print(f"First result: {results[0].page_content[:200]}...")
```

### Integration with InsightFlow (if time permits)

If you have any time remaining, add this to your app.py:

```python
# Quick integration with existing app.py
from langchain.embeddings import HuggingFaceEmbeddings

# Use in one perspective node as proof of concept
philosophical_embeddings = HuggingFaceEmbeddings(
    model_name='models/embeddings/philosophical',
    model_kwargs={'device': 'cpu'}
)

# Modify existing code to use the new embeddings for just the philosophical perspective
# This would be a minimal change to your existing implementation
```

### Post-Hour Expansion Plan

After this initial 1-hour proof of concept, here's what to do next:

1. Fine-tune models for the remaining perspectives using the same approach
2. Improve training data quality with more domain-specific datasets 
3. Conduct proper evaluation against your baseline
4. Create a unified embedding API for your application

This streamlined plan sacrifices comprehensiveness for rapid implementation, letting you see if the specialized embeddings improve your system enough to justify further investment of time and resources. 

### 4:00-4:30: Performance Evaluation with RAGAS

Building on our current implementation, which is already processing multi-perspective queries like "South India from 8th century CE to 17th century CE" with analytical, scientific, and philosophical perspectives, we've developed a comprehensive evaluation plan using the RAGAS framework.

## RAGAS Evaluation Plan for InsightFlow AI

### 1. Evaluation Framework Setup

```python
# Core RAGAS Dependencies
!pip install ragas datasets langchain langchain_community langchain_openai
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
)
from ragas.metrics.critique import harmfulness
```

### 2. Perspective-Specific Test Dataset Creation

Create a gold-standard dataset that evaluates each reasoning perspective:

```python
# Based on AIE6 module 08's dataset format but adapted for multi-perspective evaluation
def create_perspective_test_datasets():
    # Dictionary to hold test data for each perspective
    perspective_datasets = {}
    
    # For each perspective, create specialized test cases
    for perspective in ["analytical", "scientific", "philosophical", 
                       "factual", "metaphorical", "futuristic"]:
        
        # Create 50 test questions that specifically target this perspective
        # For each question, provide:
        # 1. A gold-standard answer from this perspective
        # 2. The ideal contexts that should be retrieved
        # 3. Metadata indicating perspective type
        
        questions = []
        ground_truths = []
        contexts = []
        
        # Generate perspective-specific test cases
        # In practice, you would manually craft these or use a specialized LLM approach
        
        # Sample test cases for analytical perspective
        if perspective == "analytical":
            questions.append("What were the major trade patterns in South India from 8th-17th century CE?")
            ground_truths.append("The major trade patterns in South India evolved significantly...")
            contexts.append(["South Indian kingdoms established extensive maritime trade networks...",
                           "The trade routes connected the Coromandel coast to Southeast Asia...",
                           "Economic analysis of excavated coins indicates regular commerce with..."])
            
            # Add more test cases...
        
        # Create similar specialized test cases for each perspective
        
        # Create dataframe for this perspective
        perspective_datasets[perspective] = pd.DataFrame({
            "question": questions,
            "ground_truth": ground_truths,
            "contexts": contexts,
            "perspective": [perspective] * len(questions)
        })
    
    return perspective_datasets
```

### 3. Core Evaluation Function

```python
# Based on AIE6 module 08's implementation but adapted for perspective evaluation
async def evaluate_perspective_performance(perspective_name, test_dataset):
    """Evaluate a specific perspective's performance"""
    
    # Configure InsightFlow AI to use only this perspective
    # In practice, this means temporarily modifying your LangGraph to use just one perspective node
    
    results = {}
    answers = []
    retrieved_contexts = []
    
    # For each test question
    for i, row in test_dataset.iterrows():
        # Process the question through InsightFlow's perspective-specific processor
        question = row["question"]
        
        # Send the query to your system and capture both the answer and contexts used
        state = await run_insightflow_perspective(question, perspective_name)
        
        # Extract answer and contexts used
        answer = state["persona_responses"][perspective_name]
        contexts_used = state["retrieved_contexts"][perspective_name] if "retrieved_contexts" in state else []
        
        answers.append(answer)
        retrieved_contexts.append(contexts_used)
    
    # Add the generated answers and contexts to the dataset
    eval_data = test_dataset.copy()
    eval_data["answer"] = answers
    eval_data["retrieved_contexts"] = retrieved_contexts
    
    # Run RAGAS evaluation
    # This code pattern follows AIE6 module 08's evaluation approach
    eval_result = evaluate(
        eval_data,
        metrics=[
            faithfulness,
            answer_relevancy, 
            context_precision,
            context_recall,
            context_relevancy,
            harmfulness  # Added from the latest RAGAS
        ]
    )
    
    return eval_result
```

### 4. Perspective Comparison Analysis

```python
# This function extends beyond the AIE6 module to compare perspectives
def compare_perspective_performance(evaluation_results):
    """Compare performance across different perspectives"""
    
    # Create a comparison dataframe
    comparison = pd.DataFrame()
    
    # Standard RAGAS metrics
    metrics = ["faithfulness", "answer_relevancy", "context_precision", 
               "context_recall", "context_relevancy", "harmfulness"]
    
    # For each perspective
    for perspective, results in evaluation_results.items():
        # Extract the metric scores
        perspective_scores = {}
        for metric in metrics:
            if metric in results:
                perspective_scores[metric] = results[metric].score
        
        # Add to comparison dataframe
        perspective_df = pd.DataFrame(perspective_scores, index=[perspective])
        comparison = pd.concat([comparison, perspective_df])
    
    return comparison
```

### 5. Multi-Perspective Synthesis Evaluation

This is an important custom evaluation for InsightFlow AI:

```python
# Custom evaluation approach for multi-perspective synthesis
async def evaluate_synthesis_quality(test_dataset, perspectives_to_include):
    """Evaluate the quality of multi-perspective synthesis"""
    
    synthesis_results = {
        "perspective_integration_score": [],
        "contradiction_resolution_score": [],
        "combined_insight_score": [],
        "bias_balance_score": []
    }
    
    # For each test question
    for i, row in test_dataset.iterrows():
        question = row["question"]
        
        # Get the ground truth perspectives
        # In practice, these would be expert-created responses from each perspective
        ground_truth_perspectives = {}
        for p in perspectives_to_include:
            perspective_dataset = perspective_datasets[p]
            matching_rows = perspective_dataset[perspective_dataset["question"] == question]
            if not matching_rows.empty:
                ground_truth_perspectives[p] = matching_rows.iloc[0]["ground_truth"]
        
        # Run full InsightFlow with multiple perspectives
        state = await run_insightflow_full(question, perspectives_to_include)
        
        # Extract individual perspective responses
        perspective_responses = state["persona_responses"]
        
        # Extract synthesis
        synthesis = state["synthesized_response"]
        
        # Calculate custom metrics:
        # 1. Perspective Integration: How well the synthesis incorporates all perspectives
        # 2. Contradiction Resolution: How well it handles conflicting viewpoints
        # 3. Combined Insight: Whether it produces novel insights from the combination
        # 4. Bias Balance: Whether it gives fair representation to each perspective
        
        # These would be implemented using LLM-based evaluation techniques
        # following patterns from AIE6 module 08's critique metrics
        
        # [Custom evaluation code would go here]
    
    return synthesis_results
```

### 6. Current System Performance

Based on our terminal logs, the current InsightFlow AI implementation is working as expected:

```
Planning research for query: south india from 8th century CE to 17th century CE
Selected personas: ['analytical', 'scientific', 'philosophical']
Executing persona tasks for 3 personas
Perspective generated for analytical
Perspective generated for scientific
Perspective generated for philosophical
Synthesizing responses from 3 personas
Synthesis complete
Visualization generation complete with simplified diagram
```

The system successfully:
1. Plans the research approach for the query
2. Generates perspectives from all three selected persona types
3. Synthesizes the perspectives into a unified response
4. Creates visualizations to represent the relationships

Our RAGAS evaluation will build on this current functionality to provide quantitative metrics on the quality and effectiveness of each perspective and their synthesis.

### 7. Implementation Timeline

1. **Setup Testing Environment (Day 1)**
   - Create testing directories and infrastructure
   - Set up tracking for evaluation results

2. **Generate Test Datasets (Days 1-2)**
   - Create gold-standard datasets for each perspective type
   - Include diverse topics across domains

3. **Perspective-Specific Evaluation (Days 2-3)**
   - Evaluate each perspective individually using RAGAS metrics
   - Log and visualize results

4. **Synthesis Evaluation (Day 3)**
   - Evaluate various perspective combinations
   - Assess synthesis quality with custom metrics

5. **Visualize and Analyze Results (Day 4)**
   - Create comparison visualizations
   - Identify strengths and weaknesses of each perspective

### 8. Final Report Generation

The evaluation process will conclude with a comprehensive report including:

1. Performance metrics for each individual perspective
2. Comparative analysis across perspectives
3. Synthesis quality assessment
4. Recommendations for improvement

This approach will provide a robust, quantitative evaluation of InsightFlow AI's multi-perspective reasoning capabilities, helping identify both strengths and areas for enhancement.

## RAG Implementation and Enhanced State Management for InsightFlow AI

This section provides a detailed implementation plan for integrating Retrieval-Augmented Generation (RAG) and enhanced state management into the InsightFlow AI application, leveraging patterns from the existing codebase.

### 1. RAG-Enhanced State Management

First, let's enhance the `InsightFlowState` class to track RAG-related information:

```python
from typing import TypedDict, List, Dict, Optional, Any, Tuple
from langchain_core.documents import Document

class RetrievalMetrics(TypedDict):
    """Metrics for tracking RAG performance"""
    retrieval_time: float
    chunk_count: int
    query_similarity_score: Optional[float]
    relevance_score: Optional[float]

class RAGResult(TypedDict):
    """Result of a RAG operation for a specific perspective"""
    retrieved_chunks: List[Document]
    raw_query: str
    transformed_query: Optional[str]
    metrics: RetrievalMetrics
    source_details: Dict[str, int]  # Count of chunks per source

class InsightFlowRagState(TypedDict):
    # Original InsightFlow state elements
    panel_type: str  # "research" or "discussion"
    query: str
    selected_personas: List[str]
    persona_responses: Dict[str, str]
    synthesized_response: Optional[str]
    current_step_name: str
    error_message: Optional[str]
    
    # RAG-specific additions
    rag_enabled: bool  # Whether RAG is enabled for this query
    perspective_retrievals: Dict[str, RAGResult]  # Retrieval results per perspective
    transformed_queries: Dict[str, str]  # Perspective-specific query transformations
    retrieval_stats: Dict[str, Any]  # Aggregated retrieval statistics
    vector_db_info: Dict[str, Any]  # Information about vector DBs used
    
    # Enhanced history tracking
    session_history: List[Dict[str, Any]]  # Track full session for persistence
    
    # Evaluation data (for RAGAS)
    evaluation_metrics: Optional[Dict[str, Any]]
```

### 2. RAG Document Processing Pipeline

Next, create a class for processing documents in a perspective-specific way:

```python
# utils/rag_processor.py

import os
import re
import time
from typing import List, Dict, Optional, Any, Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class PerspectiveRAGProcessor:
    """Process and retrieve documents for a specific perspective"""
    
    def __init__(
        self, 
        perspective_id: str,
        data_dir: str = "data_sources",
        embedding_model: Optional[Any] = None,
        collection_name: Optional[str] = None,
        custom_chunking_fn: Optional[Callable] = None
    ):
        """Initialize the RAG processor for a perspective
        
        Args:
            perspective_id: Identifier for the perspective (e.g., 'analytical')
            data_dir: Base directory for data sources
            embedding_model: Optional custom embedding model
            collection_name: Optional custom name for vector DB collection
            custom_chunking_fn: Optional custom function for document chunking
        """
        self.perspective_id = perspective_id
        self.perspective_dir = os.path.join(data_dir, perspective_id)
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.collection_name = collection_name or f"insightflow_{perspective_id}"
        self.custom_chunking_fn = custom_chunking_fn
        self.vector_db = None
        self.db_path = os.path.join("data", "vector_db", perspective_id)
        
    def _get_chunking_function(self) -> Callable:
        """Get the appropriate chunking function for this perspective
        
        Different perspectives benefit from different chunking strategies:
        - Analytical: Focus on logical segments and paragraphs (medium chunks)
        - Scientific: Preserve equation contexts and explanations (medium-small)
        - Philosophical: Maintain argument flow (larger chunks)
        - Factual: Brief, focused information snippets (smaller chunks)
        - Metaphorical: Keep full metaphors and analogies intact (medium-large)
        - Futuristic: Maintain scenario coherence (medium-large)
        """
        if self.custom_chunking_fn:
            return self.custom_chunking_fn
            
        # Default chunking settings based on perspective type
        chunk_size = 1000
        chunk_overlap = 200
        
        if self.perspective_id == "analytical":
            chunk_size = 1000
            chunk_overlap = 200
        elif self.perspective_id == "scientific":
            chunk_size = 800
            chunk_overlap = 150
        elif self.perspective_id == "philosophical":
            chunk_size = 1500
            chunk_overlap = 300
        elif self.perspective_id == "factual":
            chunk_size = 500
            chunk_overlap = 100
        elif self.perspective_id == "metaphorical":
            chunk_size = 1200
            chunk_overlap = 250
        elif self.perspective_id == "futuristic":
            chunk_size = 1200
            chunk_overlap = 250
        
        # Create appropriate text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        return splitter.split_documents
    
    def _preprocess_text(self, text: str) -> str:
        """Apply perspective-specific preprocessing to text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Perspective-specific preprocessing
        if self.perspective_id == "scientific":
            # Preserve equation formatting
            text = text.replace("**", "").replace("__", "")
        elif self.perspective_id == "philosophical":
            # Special handling for quotes and references
            text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers
        
        return text.strip()
    
    def process_documents(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Process all documents for this perspective and build vector database
        
        Args:
            force_rebuild: If True, rebuild the vector DB even if it exists
            
        Returns:
            Dictionary with processing statistics
        """
        # Check if vector DB exists and we don't need to rebuild
        if os.path.exists(self.db_path) and not force_rebuild:
            self.vector_db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.db_path
            )
            return {
                "status": "loaded_existing",
                "document_count": self.vector_db._collection.count(),
                "perspective": self.perspective_id
            }
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Get all text files in the perspective directory
        documents = []
        source_files = []
        
        for root, _, files in os.walk(self.perspective_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    source_files.append(file_path)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            
                        # Preprocess the text
                        processed_text = self._preprocess_text(text)
                        
                        # Create document with metadata
                        doc = Document(
                            page_content=processed_text,
                            metadata={
                                "source": file_path,
                                "perspective": self.perspective_id,
                                "filename": file
                            }
                        )
                        documents.append(doc)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        if not documents:
            return {
                "status": "error",
                "error": "No documents found",
                "perspective": self.perspective_id
            }
        
        # Get chunking function and process documents
        chunking_fn = self._get_chunking_function()
        chunked_docs = chunking_fn(documents)
        
        # Create vector database
        start_time = time.time()
        self.vector_db = Chroma.from_documents(
            documents=chunked_docs,
            embedding=self.embedding_model,
            collection_name=self.collection_name,
            persist_directory=self.db_path
        )
        
        # Persist to disk
        self.vector_db.persist()
        
        processing_time = time.time() - start_time
        
        return {
            "status": "created_new",
            "document_count": len(chunked_docs),
            "original_document_count": len(documents),
            "source_files": len(source_files),
            "processing_time_seconds": processing_time,
            "perspective": self.perspective_id
        }
    
    def retrieve_relevant_documents(
        self, 
        query: str, 
        k: int = 5,
        rewrite_query: bool = False,
        llm: Optional[Any] = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Retrieve relevant documents for a query
        
        Args:
            query: The user query
            k: Number of documents to retrieve
            rewrite_query: Whether to rewrite the query for this perspective
            llm: Optional LLM for query rewriting
            
        Returns:
            Tuple of (retrieved documents, retrieval info)
        """
        if not self.vector_db:
            self.process_documents()
            
        transformed_query = query
        
        # Rewrite query if needed and LLM provided
        if rewrite_query and llm:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            prompt = f"""
            Rewrite the following query to optimize for retrieval from a {self.perspective_id} perspective.
            Focus on key terms and concepts that would be emphasized in {self.perspective_id} texts.
            Original query: {query}
            
            Rewritten query:
            """
            
            messages = [
                SystemMessage(content="You are an expert at optimizing queries for retrieval systems."),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            transformed_query = response.content.strip()
        
        # Start timing retrieval
        start_time = time.time()
        
        # Retrieve documents
        retriever = self.vector_db.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.get_relevant_documents(transformed_query)
        
        retrieval_time = time.time() - start_time
        
        # Compile source statistics
        source_counts = {}
        for doc in retrieved_docs:
            source = doc.metadata.get("filename", "unknown")
            if source in source_counts:
                source_counts[source] += 1
            else:
                source_counts[source] = 1
        
        retrieval_info = {
            "retrieval_time": retrieval_time,
            "chunk_count": len(retrieved_docs),
            "original_query": query,
            "transformed_query": transformed_query if rewrite_query else None,
            "perspective": self.perspective_id,
            "source_counts": source_counts
        }
        
        return retrieved_docs, retrieval_info

class InsightFlowRAGManager:
    """Manage RAG operations across all perspectives"""
    
    def __init__(self, data_dir: str = "data_sources"):
        self.data_dir = data_dir
        self.perspective_processors = {}
        self.perspective_types = [
            "analytical", "scientific", "philosophical", 
            "factual", "metaphorical", "futuristic"
        ]
        self.personality_types = ["holmes", "feynman", "fry"]
        
    def initialize_perspective(self, perspective_id: str) -> Dict[str, Any]:
        """Initialize RAG for a specific perspective"""
        if perspective_id in self.perspective_processors:
            return {"status": "already_initialized", "perspective": perspective_id}
            
        processor = PerspectiveRAGProcessor(perspective_id, self.data_dir)
        self.perspective_processors[perspective_id] = processor
        
        # Process documents and build vector DB
        result = processor.process_documents()
        return result
    
    def initialize_all_perspectives(self, include_personalities: bool = True) -> Dict[str, Dict[str, Any]]:
        """Initialize RAG for all perspectives"""
        results = {}
        
        # Process perspective types
        for perspective in self.perspective_types:
            results[perspective] = self.initialize_perspective(perspective)
            
        # Process personality types if requested
        if include_personalities:
            for personality in self.personality_types:
                results[personality] = self.initialize_perspective(personality)
                
        return results
    
    def retrieve_for_perspective(
        self, 
        perspective_id: str, 
        query: str, 
        k: int = 5,
        rewrite_query: bool = False,
        llm: Optional[Any] = None
    ) -> RAGResult:
        """Retrieve documents for a specific perspective"""
        # Initialize if not already done
        if perspective_id not in self.perspective_processors:
            self.initialize_perspective(perspective_id)
            
        processor = self.perspective_processors[perspective_id]
        
        # Retrieve documents
        docs, info = processor.retrieve_relevant_documents(
            query=query,
            k=k,
            rewrite_query=rewrite_query,
            llm=llm
        )
        
        # Create RAG result
        result = RAGResult(
            retrieved_chunks=docs,
            raw_query=query,
            transformed_query=info.get("transformed_query"),
            metrics=RetrievalMetrics(
                retrieval_time=info.get("retrieval_time", 0),
                chunk_count=len(docs),
                query_similarity_score=None,  # Would need to calculate
                relevance_score=None  # Would need to calculate
            ),
            source_details=info.get("source_counts", {})
        )
        
        return result
        
    def retrieve_for_multiple_perspectives(
        self,
        query: str,
        perspectives: List[str],
        k: int = 3,
        rewrite_queries: bool = True,
        llm: Optional[Any] = None
    ) -> Dict[str, RAGResult]:
        """Retrieve documents for multiple perspectives"""
        results = {}
        
        for perspective in perspectives:
            results[perspective] = self.retrieve_for_perspective(
                perspective_id=perspective,
                query=query,
                k=k,
                rewrite_query=rewrite_queries,
                llm=llm
            )
            
        return results
```

### 3. Integrating RAG with LangGraph Workflow

Now, let's update the LangGraph nodes to incorporate RAG:

```python
# Updated LangGraph nodes for RAG integration

async def run_planner_agent(state: InsightFlowRagState) -> InsightFlowRagState:
    """Plan the research approach for multiple personas with RAG"""
    query = state["query"]
    selected_personas = state["selected_personas"]
    rag_enabled = state.get("rag_enabled", True)  # Default to enabled
    
    # For MVP, simple planning
    state["current_step_name"] = "retrieve_perspective_context" if rag_enabled else "execute_persona_tasks"
    return state

async def retrieve_perspective_context(state: InsightFlowRagState) -> InsightFlowRagState:
    """Retrieve context for each perspective using RAG"""
    query = state["query"]
    selected_personas = state["selected_personas"]
    
    # Get RAG manager from session
    rag_manager = cl.user_session.get("rag_manager")
    if not rag_manager:
        # Initialize if not already done
        rag_manager = InsightFlowRAGManager()
        await cl.Message(content="🔍 Initializing knowledge retrieval system...").send()
        
        # Initialize all perspectives in background
        # This would be done at startup in a production app
        rag_manager.initialize_all_perspectives()
        cl.user_session.set("rag_manager", rag_manager)
    
    # Use OpenAI model for query rewriting
    llm = cl.user_session.get("llm_query_rewriter") or ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Progress message
    progress_msg = cl.user_session.get("progress_msg")
    if progress_msg:
        await update_message(progress_msg, "⏳ Retrieving relevant information (10%)...")
    
    # Retrieve context for each selected persona
    try:
        retrieval_results = rag_manager.retrieve_for_multiple_perspectives(
            query=query,
            perspectives=selected_personas,
            k=3,  # Retrieve 3 chunks per perspective
            rewrite_queries=True,
            llm=llm
        )
        
        # Store results in state
        state["perspective_retrievals"] = retrieval_results
        
        # Track transformed queries
        state["transformed_queries"] = {
            p: res["transformed_query"] 
            for p, res in retrieval_results.items() 
            if res["transformed_query"]
        }
        
        # Aggregate stats
        total_chunks = sum(len(res["retrieved_chunks"]) for res in retrieval_results.values())
        avg_time = sum(res["metrics"]["retrieval_time"] for res in retrieval_results.values()) / len(retrieval_results)
        
        state["retrieval_stats"] = {
            "total_chunks_retrieved": total_chunks,
            "average_retrieval_time": avg_time,
            "perspective_count": len(retrieval_results)
        }
        
        # Update progress
        if progress_msg:
            await update_message(progress_msg, "⏳ Retrieved information from knowledge base (20%)...")
    
    except Exception as e:
        error_msg = f"Error retrieving context: {str(e)}"
        state["error_message"] = error_msg
        print(error_msg)
        
        # Continue without RAG if there's an error
        if progress_msg:
            await update_message(progress_msg, "⚠️ Using basic analysis without knowledge retrieval...")
    
    state["current_step_name"] = "execute_persona_tasks"
    return state

async def execute_persona_tasks(state: InsightFlowRagState) -> InsightFlowRagState:
    """Execute tasks for each selected persona with RAG context"""
    query = state["query"]
    selected_personas = state["selected_personas"]
    persona_factory = cl.user_session.get("persona_factory")
    
    # Initialize responses dict if not exists
    if "persona_responses" not in state:
        state["persona_responses"] = {}
    
    # Get progress message if it exists
    progress_msg = cl.user_session.get("progress_msg")
    total_personas = len(selected_personas)
    
    # Process each persona with RAG context if available
    for i, persona_id in enumerate(selected_personas):
        persona = persona_factory.create_persona(persona_id)
        if persona:
            try:
                # Update progress
                if progress_msg:
                    percent_done = 20 + int((i / total_personas) * 40)
                    await update_message(
                        progress_msg, 
                        f"⏳ Analyzing from {persona_id} perspective ({percent_done}%)..."
                    )
                
                # Get RAG context for this persona if available
                context = None
                if "perspective_retrievals" in state and persona_id in state["perspective_retrievals"]:
                    rag_result = state["perspective_retrievals"][persona_id]
                    context = rag_result["retrieved_chunks"]
                
                # Generate perspective with context
                response = await cl.make_async(persona.generate_perspective)(query, context)
                state["persona_responses"][persona_id] = response
                
            except Exception as e:
                error_msg = f"Error getting {persona_id} perspective: {str(e)}"
                print(error_msg)
                state["persona_responses"][persona_id] = f"Could not generate perspective: {str(e)}"
    
    state["current_step_name"] = "synthesize_responses"
    return state

# Update LangGraph configuration to include the new node
insight_graph_builder = StateGraph(InsightFlowRagState)

# Add all nodes
insight_graph_builder.add_node("planner_agent", run_planner_agent)
insight_graph_builder.add_node("retrieve_perspective_context", retrieve_perspective_context)
insight_graph_builder.add_node("execute_persona_tasks", execute_persona_tasks)
insight_graph_builder.add_node("synthesize_responses", synthesize_responses)
insight_graph_builder.add_node("present_results", present_results)

# Add edges
insight_graph_builder.add_edge("planner_agent", "retrieve_perspective_context")
insight_graph_builder.add_edge("planner_agent", "execute_persona_tasks")
insight_graph_builder.add_edge("retrieve_perspective_context", "execute_persona_tasks")
insight_graph_builder.add_edge("execute_persona_tasks", "synthesize_responses")
insight_graph_builder.add_edge("synthesize_responses", "present_results")
insight_graph_builder.add_edge("present_results", END)

# Add conditional edges
def should_retrieve_context(state):
    return state.get("rag_enabled", True)

def should_skip_retrieval(state):
    return not state.get("rag_enabled", True)

insight_graph_builder.add_conditional_edges(
    "planner_agent",
    should_retrieve_context,
    {
        True: "retrieve_perspective_context",
        False: "execute_persona_tasks"
    }
)

# Set entry point
insight_graph_builder.set_entry_point("planner_agent")

# Compile the graph
insight_flow_graph = insight_graph_builder.compile()
```

### 4. User Interface for RAG Controls

Add these commands and handlers to the Chainlit interface:

```python
@cl.on_message
async def handle_message(message: cl.Message):
    """Process user messages and commands"""
    msg_content = message.content
    
    # Handle commands (existing code)
    if msg_content.startswith("/"):
        # ... (existing command handling)
        
        # Add RAG specific commands
        if cmd == "rag":
            # Handle RAG toggle command
            if args and args[0].lower() in ["on", "off"]:
                enable_rag = args[0].lower() == "on"
                cl.user_session.set("rag_enabled", enable_rag)
                status = "enabled" if enable_rag else "disabled"
                await cl.Message(content=f"RAG {status} for future queries").send()
                return
            else:
                # Show RAG status
                rag_enabled = cl.user_session.get("rag_enabled", True)
                status = "enabled" if rag_enabled else "disabled"
                await cl.Message(content=f"RAG is currently {status}").send()
                
                # Show RAG stats if available
                rag_manager = cl.user_session.get("rag_manager")
                if rag_manager and hasattr(rag_manager, "perspective_processors"):
                    stats = []
                    for p_id, processor in rag_manager.perspective_processors.items():
                        if processor.vector_db:
                            doc_count = processor.vector_db._collection.count()
                            stats.append(f"- {p_id}: {doc_count} chunks")
                    
                    if stats:
                        await cl.Message(content="Knowledge base stats:\n" + "\n".join(stats)).send()
                return
        
        elif cmd == "retrieval_quality":
            # Show retrieval quality for last query if available
            insight_state = cl.user_session.get("insight_state")
            if (insight_state and "perspective_retrievals" in insight_state 
                    and insight_state["perspective_retrievals"]):
                
                quality_report = "# Retrieval Quality Report\n\n"
                
                for p_id, rag_result in insight_state["perspective_retrievals"].items():
                    retrieval_time = rag_result["metrics"]["retrieval_time"]
                    chunk_count = rag_result["metrics"]["chunk_count"]
                    sources = ", ".join(rag_result["source_details"].keys())
                    
                    quality_report += f"## {p_id.capitalize()} Perspective\n"
                    quality_report += f"- Retrieved {chunk_count} chunks in {retrieval_time:.2f}s\n"
                    quality_report += f"- Sources: {sources}\n"
                    
                    if rag_result["transformed_query"]:
                        quality_report += f"- Original query: '{insight_state['query']}'\n"
                        quality_report += f"- Transformed: '{rag_result['transformed_query']}'\n"
                    
                    quality_report += "\n"
                
                await cl.Message(content=quality_report).send()
            else:
                await cl.Message(content="No retrieval data available. Make a query with RAG enabled first.").send()
            return
    
    # ... (rest of existing message handling)
```

### 5. Implementation Steps

Follow these steps to implement the RAG system:

1. **Update State Management**:
   
   Create the new `InsightFlowRagState` class by enhancing your existing `insight_state.py` file.
   
   ```bash
   # Update insight_state.py with the RAG-enhanced state class
   ```

2. **Create RAG Processing Module**:
   
   ```bash
   # Create the new RAG processor module
   mkdir -p utils/rag
   touch utils/rag/__init__.py
   touch utils/rag/processor.py
   ```
   
   Add the `PerspectiveRAGProcessor` and `InsightFlowRAGManager` classes to `utils/rag/processor.py`.

3. **Initialize Vector Database Storage**:
   
   ```bash
   # Create directory for vector database storage
   mkdir -p data/vector_db
   ```

4. **Update LangGraph Workflow**:
   
   Modify `app.py` to include the new RAG-enabled nodes and conditional edges.

5. **Add UI Commands**:
   
   Update the Chainlit command handler to include the RAG-specific commands.

6. **Test the Implementation**:
   
   ```bash
   # Run the updated application
   chainlit run app.py
   ```

7. **Verify RAG Enhancement**:
   
   - Check that the system retrieves relevant context for each perspective
   - Verify that perspetives incorporate the retrieved context
   - Test the RAG toggle command

### 6. Integration with RAGAS Evaluation

To connect this RAG implementation with your planned RAGAS evaluation (tasks #76-80):

```python
# utils/evaluation.py

from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from ragas.metrics.critique import harmfulness
from datasets import Dataset
import pandas as pd
from typing import Dict, List, Any

class PerspectiveEvaluator:
    """Evaluate RAG performance for different perspectives"""
    
    def __init__(self):
        # Initialize RAGAS metrics
        self.metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_relevancy": context_relevancy,
            "harmfulness": harmfulness
        }
    
    def create_evaluation_dataset(self, 
        queries: List[str], 
        perspectives: List[str],
        rag_manager: Any
    ) -> Dict[str, Dataset]:
        """Create evaluation datasets for each perspective"""
        datasets = {}
        
        for perspective in perspectives:
            eval_data = []
            
            for query in queries:
                # Get RAG results for this perspective
                rag_result = rag_manager.retrieve_for_perspective(
                    perspective_id=perspective,
                    query=query,
                    k=5,
                    rewrite_query=True
                )
                
                # Generate answer using the perspective
                # This would call your existing persona implementation
                # with the retrieved context
                persona = persona_factory.create_persona(perspective)
                answer = persona.generate_perspective(query, rag_result["retrieved_chunks"])
                
                # Prepare RAGAS dataset entry
                entry = {
                    "question": query,
                    "answer": answer,
                    "contexts": [doc.page_content for doc in rag_result["retrieved_chunks"]]
                }
                eval_data.append(entry)
            
            # Convert to RAGAS-compatible dataset
            df = pd.DataFrame(eval_data)
            datasets[perspective] = Dataset.from_pandas(df)
        
        return datasets
    
    def evaluate_perspective(self, dataset: Dataset) -> Dict[str, float]:
        """Evaluate a perspective using RAGAS metrics"""
        results = {}
        
        for metric_name, metric in self.metrics.items():
            score = metric.score(dataset)
            results[metric_name] = score.mean()
        
        return results
    
    def evaluate_all_perspectives(self, 
        datasets: Dict[str, Dataset]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all perspectives"""
        results = {}
        
        for perspective, dataset in datasets.items():
            results[perspective] = self.evaluate_perspective(dataset)
        
        return results
    
    def generate_comparison_report(self, 
        eval_results: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate a comparison report for all perspectives"""
        report = "# RAGAS Evaluation Results\n\n"
        
        # Table header
        metrics = list(next(iter(eval_results.values())).keys())
        header = "| Perspective | " + " | ".join(metrics) + " |"
        separator = "|------------|" + "|".join(["-----------" for _ in metrics]) + "|"
        
        report += header + "\n" + separator + "\n"
        
        # Table rows
        for perspective, metrics_dict in eval_results.items():
            values = [f"{metrics_dict[m]:.3f}" for m in metrics]
            row = f"| {perspective.capitalize()} | " + " | ".join(values) + " |"
            report += row + "\n"
        
        return report
```

### 7. Conclusion and Next Steps

This implementation provides a comprehensive RAG system for InsightFlow AI, leveraging perspective-specific document processing and retrieval to enhance the quality of insights. The system is designed to be modular, allowing for easy extension and customization.

Next steps after implementing this RAG system:

1. Fine-tune embeddings for philosophical perspective (task #71)
2. Extend the RAGAS evaluation framework with synthesis metrics
3. Optimize chunking strategies based on evaluation results
4. Add user feedback mechanisms to improve retrieval quality

By implementing this RAG system, InsightFlow AI will provide more accurate, knowledge-grounded perspectives that draw from a diverse set of sources tailored to each perspective type.

## Project Management Plan

### Project Timeline

The development of InsightFlow AI follows an iterative approach divided into six major phases:

1. **Phase 1 (0:00-1:00) - Core System Development**
   - Project structure setup (0:00-0:15)
   - Persona configuration system (0:00-0:15) 
   - Base persona system implementation (0:15-0:35)
   - Persona implementations (0:35-0:50)
   - Data acquisition (0:50-1:00)

2. **Phase 2 (1:00-1:30) - System Architecture**
   - State management (1:00-1:05)
   - LangGraph implementation (1:05-1:20)
   - Chainlit integration (1:20-1:30)

3. **Phase 3 (1:30-2:30) - UI and Performance Optimization**
   - UI improvements (1:30-2:00)
   - Performance optimization (1:30-2:00)
   - Visualization system (2:00-2:30)

4. **Phase 4 (2:30-3:30) - Feature Extension**
   - Future enhancements planning (2:30-3:00)
   - Export functionality (2:30-3:00)
   - Testing and refinement (3:00-3:30)

5. **Phase 5 (3:30-5:00) - Advanced Features**
   - RAG implementation (3:30-4:00)
   - Embedding fine-tuning (4:00-4:30)
   - RAGAS evaluation framework (4:30-5:00)

6. **Phase 6 (5:00-6:00) - Deployment & Documentation**
   - Deployment preparation (5:00-5:30)
   - User documentation and marketing (5:30-6:00)

### Resource Allocation

#### Human Resources
- 1 ML Engineer (full-time): Responsible for core algorithm development, RAG implementation, and embedding fine-tuning
- 1 Full-stack Developer (full-time): Responsible for UI/UX, Chainlit integration, and export functionality
- 1 DevOps Engineer (part-time): Responsible for deployment infrastructure and optimization

#### Infrastructure Resources
- Development: Local development environments with Git version control
- Testing: CI/CD pipeline with automated testing
- Deployment: Hugging Face Spaces for production hosting
- Data Storage: Vector databases for RAG implementations
- Computing: GPU resources for embedding fine-tuning (as needed)

### Risk Management

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|------------|---------------------|
| API rate limiting | High | Medium | Implement request throttling, caching, and fallback mechanisms |
| Performance bottlenecks | Medium | High | Regular profiling, incremental optimization, and parallel processing |
| Data availability | High | Low | Multiple data source fallbacks and local caching |
| Model drift | Medium | Medium | Regular evaluation using RAGAS framework and retraining |
| Security vulnerabilities | High | Low | Regular security audits and dependency updates |
| Deployment issues | Medium | Medium | Dockerization for environment consistency |

### Quality Assurance Plan

1. **Automated Testing**
   - Unit tests for core components
   - Integration tests for system workflows
   - Performance benchmarks for different perspective combinations

2. **RAGAS Evaluation Framework**
   - Perspective-specific metrics
   - Synthesis quality assessment
   - Comparative analysis across perspectives

3. **User Testing**
   - Feedback collection from diverse user groups
   - A/B testing for visualization styles
   - Usability evaluation for different use cases

### Deployment Strategy

#### Staging Process
1. Development environment (local)
2. Test environment (CI pipeline)
3. Staging environment (limited users, full functionality)
4. Production environment (public access)

#### Deployment Platforms
- **Primary**: Hugging Face Spaces
- **Alternative**: Docker container on cloud provider (AWS/GCP/Azure)

#### Deployment Schedule
- Initial Alpha Release: Internal testing only
- Beta Release: Limited public access
- v1.0 Release: Full public launch with complete feature set
- Ongoing: Bi-weekly updates based on user feedback and feature roadmap

### Maintenance Plan

1. **Regular Updates**
   - Weekly dependency updates
   - Bi-weekly feature enhancements
   - Monthly performance optimizations

2. **Monitoring**
   - API usage tracking
   - Performance metrics collection
   - Error logging and analysis

3. **Support Channels**
   - GitHub Issues for bug tracking
   - Documentation updates for usage questions
   - Email support for direct assistance

### Success Criteria

- System consistently produces multi-perspective analyses within 2 minutes
- RAGAS evaluation scores exceed 0.8 for all perspective types
- User retention rate above 70% after first month
- Positive user feedback (>4/5 average rating)
- Successful deployment with 99.9% uptime