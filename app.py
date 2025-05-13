# InsightFlow AI - Main Application
import chainlit as cl
from insight_state import InsightFlowState # Import InsightFlowState
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI # Import AsyncOpenAI for DALL-E
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage # Added for prompts
from utils.persona import PersonaFactory # Import PersonaFactory
from utils.visualization_utils import generate_dalle_image, generate_mermaid_code # <--- UPDATED IMPORT
import asyncio # For asyncio.gather in execute_persona_tasks
from langchain_core.callbacks.base import AsyncCallbackHandler # <--- ADDED for progress
from typing import Any, Dict, List, Optional, Union # <--- ADDED for callbacks
from langchain_core.outputs import LLMResult # <--- ADDED for callbacks

# --- GLOBAL CONFIGURATION STATE ---
_configurations_initialized = False
llm_planner = None
llm_synthesizer = None
llm_direct = None
llm_analytical = None
llm_scientific = None
llm_philosophical = None
llm_factual = None
llm_metaphorical = None
llm_futuristic = None
llm_mermaid_generator = None # <--- ADDED
openai_async_client = None # For DALL-E
PERSONA_LLM_MAP = {}

QUICK_MODE_PERSONAS = ["analytical", "factual"] # Default personas for Quick Mode

PERSONA_TEAMS = {
    "creative_synthesis": {
        "name": "🎨 Creative Synthesis Team",
        "description": "Generates novel ideas and artistic interpretations.",
        "members": ["metaphorical", "futuristic", "philosophical"]
    },
    "data_driven_analysis": {
        "name": "📊 Data-Driven Analysis Squad",
        "description": "Focuses on factual accuracy and logical deduction.",
        "members": ["analytical", "factual", "scientific"]
    },
    "balanced_overview": {
        "name": "⚖️ Balanced Overview Group",
        "description": "Provides a well-rounded perspective.",
        "members": ["analytical", "philosophical", "factual"] # Example
    }
}

def initialize_configurations():
    """Loads environment variables and initializes LLM configurations."""
    global _configurations_initialized
    global llm_planner, llm_synthesizer, llm_direct, llm_analytical, llm_scientific
    global llm_philosophical, llm_factual, llm_metaphorical, llm_futuristic
    global llm_mermaid_generator # <--- ADDED
    global openai_async_client # Add new client to globals
    global PERSONA_LLM_MAP

    if _configurations_initialized:
        return

    print("Initializing configurations: Loading .env and setting up LLMs...")
    load_dotenv()

    # LLM CONFIGURATIONS
    llm_planner = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    llm_synthesizer = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    llm_direct = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    llm_analytical = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    llm_scientific = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    llm_philosophical = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    llm_factual = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    llm_metaphorical = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
    llm_futuristic = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
    llm_mermaid_generator = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) # <--- ADDED INITIALIZATION

    # Initialize OpenAI client for DALL-E etc.
    openai_async_client = AsyncOpenAI()

    # Mapping persona IDs to their specific LLM instances
    PERSONA_LLM_MAP.update({
        "analytical": llm_analytical,
        "scientific": llm_scientific,
        "philosophical": llm_philosophical,
        "factual": llm_factual,
        "metaphorical": llm_metaphorical,
        "futuristic": llm_futuristic,
    })
    
    _configurations_initialized = True
    print("Configurations initialized.")

# Load environment variables first
# load_dotenv() # Moved to initialize_configurations

# --- LLM CONFIGURATIONS ---
# Configurations based on tests/test_llm_config.py
# llm_planner = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) # Moved
# llm_synthesizer = ChatOpenAI(model="gpt-4o-mini", temperature=0.4) # Moved
# llm_direct = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3) # Moved
# llm_analytical = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2) # Moved
# llm_scientific = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3) # Moved
# llm_philosophical = ChatOpenAI(model="gpt-4o-mini", temperature=0.5) # Moved
# llm_factual = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) # Moved
# llm_metaphorical = ChatOpenAI(model="gpt-4o-mini", temperature=0.6) # Moved
# llm_futuristic = ChatOpenAI(model="gpt-4o-mini", temperature=0.6) # Moved

# Mapping persona IDs to their specific LLM instances
# PERSONA_LLM_MAP = { # Moved and will be populated in initialize_configurations
#     "analytical": llm_analytical,
#     "scientific": llm_scientific,
#     "philosophical": llm_philosophical,
#     "factual": llm_factual,
#     "metaphorical": llm_metaphorical,
#     "futuristic": llm_futuristic,
    # Add other personas here if they have dedicated LLMs or share one from above
# }

# --- SYSTEM PROMPTS (from original app.py) ---
DIRECT_SYSPROMPT = """You are a highly intelligent AI assistant that provides clear, direct, and helpful answers.
Your responses should be accurate, concise, and well-reasoned."""

SYNTHESIZER_SYSTEM_PROMPT_TEMPLATE = """You are a master synthesizer AI. Your task is to integrate the following diverse perspectives into a single, coherent, and insightful response. Ensure that the final synthesis is well-structured, easy to understand, and accurately reflects the nuances of each provided viewpoint. Do not simply list the perspectives; weave them together.

Perspectives:
{formatted_perspectives}

Synthesized Response:"""

# --- LANGGRAPH NODE FUNCTIONS (DUMMIES FOR NOW) ---
async def run_planner_agent(state: InsightFlowState) -> InsightFlowState:
    print(f"Node: run_planner_agent, Query: {state.get('query')}")
    # For now, planner just passes to execute_persona_tasks
    # It could eventually decide *which* personas to run based on query
    state["current_step_name"] = "execute_persona_tasks"
    return state

async def execute_persona_tasks(state: InsightFlowState) -> InsightFlowState:
    print(f"Node: execute_persona_tasks")
    persona_factory: PersonaFactory = cl.user_session.get("persona_factory")
    if not persona_factory:
        state["error_message"] = "PersonaFactory not found in session."
        state["current_step_name"] = "error_presenting" # Or a dedicated error state
        return state

    query = state.get("query")
    selected_persona_ids = state.get("selected_personas", [])
    
    if not query:
        state["error_message"] = "Query not found in state for execute_persona_tasks."
        state["current_step_name"] = "error_presenting"
        return state

    tasks = []
    valid_persona_ids_for_results = [] # Keep track of personas for which tasks were created
    for persona_id in selected_persona_ids:
        # Get the appropriate LLM for this persona_id
        persona_llm = PERSONA_LLM_MAP.get(persona_id.lower())
        if not persona_llm:
            print(f"Warning: LLM not found for persona {persona_id}. Skipping.")
            continue # Skip this persona if no LLM is mapped

        persona = persona_factory.create_persona(persona_id, persona_llm) # Pass the specific LLM
        if persona:
            tasks.append(persona.generate_perspective(query))
            valid_persona_ids_for_results.append(persona_id) # Add to list for result mapping
        else:
            print(f"Warning: Persona instance {persona_id} could not be created even with an LLM.")
    
    state["persona_responses"] = {} # Initialize/clear previous responses
    if tasks:
        try:
            # Timeout logic can be added here if needed for asyncio.gather
            persona_results = await asyncio.gather(*tasks)
            # Store responses keyed by the valid persona_ids used for tasks
            for i, persona_id in enumerate(valid_persona_ids_for_results):
                state["persona_responses"][persona_id] = persona_results[i]
        except Exception as e:
            print(f"Error during persona perspective generation: {e}")
            state["error_message"] = f"Error generating perspectives: {str(e)[:100]}"
            # Optionally, populate partial results if some tasks succeeded before error
            # For now, just reports error. Individual task errors could be handled in PersonaReasoning too.

    state["current_step_name"] = "synthesize_responses"
    return state

async def synthesize_responses(state: InsightFlowState) -> InsightFlowState:
    print(f"Node: synthesize_responses")
    persona_responses = state.get("persona_responses", {})
    
    if not persona_responses:
        print("No persona responses to synthesize.")
        state["synthesized_response"] = "No perspectives were available to synthesize."
        state["current_step_name"] = "generate_visualization"
        return state

    formatted_perspectives_list = []
    for persona_id, response_text in persona_responses.items():
        formatted_perspectives_list.append(f"- Perspective from {persona_id}: {response_text}")
    
    formatted_perspectives_string = "\n".join(formatted_perspectives_list)
    
    final_prompt_content = SYNTHESIZER_SYSTEM_PROMPT_TEMPLATE.format(
        formatted_perspectives=formatted_perspectives_string
    )
    
    messages = [
        SystemMessage(content=final_prompt_content)
    ]
    
    try:
        # Ensure llm_synthesizer is available (initialized by initialize_configurations)
        if llm_synthesizer is None:
            print("Error: llm_synthesizer is not initialized.")
            state["error_message"] = "Synthesizer LLM not available."
            state["synthesized_response"] = "Synthesis failed due to internal error."
            state["current_step_name"] = "error_presenting" # Or a suitable error state
            return state

        ai_response = await llm_synthesizer.ainvoke(messages)
        synthesized_text = ai_response.content
        state["synthesized_response"] = synthesized_text
        print(f"Synthesized response: {synthesized_text[:200]}...") # Log snippet
    except Exception as e:
        print(f"Error during synthesis: {e}")
        state["error_message"] = f"Synthesis error: {str(e)[:100]}"
        state["synthesized_response"] = "Synthesis failed."
        # Optionally, decide if we proceed to visualization or an error state
        # For now, let's assume we still try to visualize if there's a partial/failed synthesis

    state["current_step_name"] = "generate_visualization"
    return state

async def generate_visualization(state: InsightFlowState) -> InsightFlowState:
    print(f"Node: generate_visualization")
    synthesized_response = state.get("synthesized_response")
    image_url = None
    mermaid_code_output = None # Changed variable name for clarity

    # DALL-E Image Generation (existing logic)
    if synthesized_response and openai_async_client:
        dalle_prompt = f"A hand-drawn style visual note or sketch representing the key concepts of: {synthesized_response}"
        if len(dalle_prompt) > 4000: 
            dalle_prompt = dalle_prompt[:3997] + "..."
        
        print(f"Attempting DALL-E image generation for: {dalle_prompt[:100]}...")
        image_url = await generate_dalle_image(prompt=dalle_prompt, client=openai_async_client)
        if image_url:
            state["visualization_image_url"] = image_url
            print(f"DALL-E Image URL: {image_url}")
        else:
            print("DALL-E image generation failed or returned no URL.")
            state["visualization_image_url"] = None 
    elif not synthesized_response:
        print("No synthesized response available to generate DALL-E image.")
        state["visualization_image_url"] = None
    elif not openai_async_client:
        print("OpenAI async client not initialized, skipping DALL-E generation.")
        state["visualization_image_url"] = None
    
    # Mermaid Code Generation
    if synthesized_response and llm_mermaid_generator: # Check if both are available
        print(f"Attempting Mermaid code generation for: {synthesized_response[:100]}...")
        mermaid_code_output = await generate_mermaid_code(synthesized_response, llm_mermaid_generator)
        if mermaid_code_output:
            state["visualization_code"] = mermaid_code_output
            print(f"Mermaid code generated: {mermaid_code_output[:100]}...")
        else:
            print("Mermaid code generation failed or returned no code.")
            state["visualization_code"] = None # Ensure it's None if failed
    else:
        print("Skipping Mermaid code generation due to missing response or LLM.")
        state["visualization_code"] = None

    state["current_step_name"] = "present_results"
    return state

async def present_results(state: InsightFlowState) -> InsightFlowState:
    print(f"Node: present_results")

    # Get user display preferences from session
    # Defaults are True if not set, matching on_chat_start initialization
    show_visualization = cl.user_session.get("show_visualization")
    show_perspectives = cl.user_session.get("show_perspectives")

    # 1. Send Synthesized Response (always)
    synthesized_response = state.get("synthesized_response")
    if synthesized_response:
        await cl.Message(content=synthesized_response).send()
    else:
        # Fallback if no synthesis, though graph should ideally always produce one or an error message
        await cl.Message(content="No synthesized response was generated.").send()

    # 2. Send Visualizations (if enabled and available)
    if show_visualization:
        # DALL-E Image
        image_url = state.get("visualization_image_url")
        if image_url:
            image_element = cl.Image(
                url=image_url, 
                name="dalle_visualization", 
                display="inline", 
                size="large" # As per test expectation
            )
            await cl.Message(content="", elements=[image_element]).send() # Send image with empty content or a title
        
        # Mermaid Diagram
        mermaid_code = state.get("visualization_code")
        if mermaid_code:
            mermaid_element = cl.Text(
                content=mermaid_code, 
                mime_type="text/mermaid",  # Corrected: MimeType should be mime_type
                name="generated_diagram", 
                display="inline" # Or "side", "page"
            )
            await cl.Message(content="", elements=[mermaid_element]).send() # Send Mermaid diagram

    # 3. Send Persona Perspectives (if enabled and available)
    if show_perspectives:
        persona_responses = state.get("persona_responses")
        if persona_responses:
            for persona_id, response_text in persona_responses.items():
                if response_text: # Ensure there is text to send
                    perspective_content = f"**Perspective from {persona_id}:**\n{response_text}"
                    await cl.Message(content=perspective_content).send()
    
    state["current_step_name"] = "results_presented"
    # The actual end of a query processing cycle is managed by the graph edge to END
    return state

# --- LANGGRAPH SETUP ---
insight_graph_builder = StateGraph(InsightFlowState)

# Add nodes
insight_graph_builder.add_node("planner_agent", run_planner_agent)
insight_graph_builder.add_node("execute_persona_tasks", execute_persona_tasks)
insight_graph_builder.add_node("synthesize_responses", synthesize_responses)
insight_graph_builder.add_node("generate_visualization", generate_visualization)
insight_graph_builder.add_node("present_results", present_results)

# Set entry point
insight_graph_builder.set_entry_point("planner_agent")

# Add edges
insight_graph_builder.add_edge("planner_agent", "execute_persona_tasks")
insight_graph_builder.add_edge("execute_persona_tasks", "synthesize_responses")
insight_graph_builder.add_edge("synthesize_responses", "generate_visualization")
insight_graph_builder.add_edge("generate_visualization", "present_results")
insight_graph_builder.add_edge("present_results", END)

# Compile the graph
insight_flow_graph = insight_graph_builder.compile()

print("LangGraph setup complete.")

# --- CUSTOM CALLBACK HANDLER FOR PROGRESS UPDATES --- #
class InsightFlowCallbackHandler(AsyncCallbackHandler):
    def __init__(self, progress_message: cl.Message):
        self.progress_message = progress_message
        self.step_counter = 0

    async def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        self.step_counter += 1
        chain_name = serialized.get("name", serialized.get("id", ["Unknown Chain"]))[-1]
        # For LangGraph, top-level graph name might be available, or individual node names.
        # Let's try to get a meaningful name.
        # The `id` field in `serialized` often looks like ["LangGraph", "__start__", "planner_agent"]
        # We can try to extract the last meaningful part.
        if isinstance(chain_name, list): # If id is a list
            chain_name = chain_name[-1]
        
        update_text = f"⏳ Step {self.step_counter}: Running {chain_name}..."
        if self.progress_message:
            await self.progress_message.stream_token(f"\n{update_text}") # Stream as new line
            # await self.progress_message.update() # Not strictly needed after every stream_token if we update at end

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        llm_name = serialized.get("name", serialized.get("id",["Unknown LLM"]))[-1]
        update_text = f"Calling LLM: {llm_name}..."
        if self.progress_message:
            await self.progress_message.stream_token(f"\n   L {update_text}")

    # We can add on_agent_action, on_tool_start for more granular updates if nodes use agents/tools
    # For now, on_chain_start (which LangGraph nodes trigger) and on_llm_start should give good visibility.

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        chain_name = kwargs.get("name", "Unknown chain/node") # LangGraph provides name in kwargs for on_chain_end
        if self.progress_message:
            await self.progress_message.stream_token(f"\nStep completed: {chain_name}.")

# This allows the test to import these names from app if needed
__all__ = [
    "InsightFlowState", "on_chat_start", "on_message", 
    "invoke_direct_llm", "invoke_langgraph",
    "run_planner_agent", "execute_persona_tasks", "synthesize_responses",
    "generate_visualization", "present_results",
    "StateGraph", # Expose StateGraph if tests patch app.StateGraph directly
    "initialize_configurations" # Expose for testing
]

@cl.on_chat_start
async def on_chat_start():
    """Initializes session state and sends a welcome message."""
    # Ensure configurations are loaded and LLMs are initialized
    if not _configurations_initialized:
        initialize_configurations()

    # Initialize PersonaFactory and store it
    persona_factory = PersonaFactory() # Using default config_dir
    cl.user_session.set("persona_factory", persona_factory)

    initial_state = InsightFlowState(
        panel_type="research", # Default panel type
        query="",
        selected_personas=["analytical", "scientific", "philosophical"], # Default personas
        persona_responses={},
        synthesized_response=None,
        visualization_code=None,
        visualization_image_url=None,
        current_step_name="awaiting_query",
        error_message=None
    )
    cl.user_session.set("insight_flow_state", initial_state)

    # Set default UI options
    cl.user_session.set("direct_mode", False)
    cl.user_session.set("show_perspectives", True)
    cl.user_session.set("show_visualization", True)
    cl.user_session.set("quick_mode", False)

    # Send welcome message
    await cl.Message(content="Welcome to InsightFlow AI!").send()

    # Send persona selection action
    persona_selection_action = cl.Action(
        name="select_personas", 
        label="Select/Update Personas", 
        description="Choose which personas to engage for the analysis.",
        payload={"value": "trigger_selection"}
    )
    await cl.Message(
        content="Configure your research team:", 
        actions=[persona_selection_action]
    ).send()

    # Placeholder for progress message, initialized to None
    cl.user_session.set("progress_msg", None)

@cl.action_callback("select_personas")
async def select_personas_action(action: cl.Action):
    """Handles the 'select_personas' action to display persona selection UI including team actions."""
    persona_factory: PersonaFactory = cl.user_session.get("persona_factory")
    insight_flow_state: InsightFlowState = cl.user_session.get("insight_flow_state")

    if not persona_factory or not insight_flow_state:
        await cl.Message(content="Error: Session data missing. Cannot display selection.").send()
        return

    available_personas = persona_factory.get_available_personas()
    currently_selected_ids = insight_flow_state.get("selected_personas", [])

    if not available_personas:
        await cl.Message(content="No personas are available for selection.").send()
        return

    # --- Create Team Action Buttons ---
    team_actions = []
    for team_id, team_info in PERSONA_TEAMS.items():
        team_actions.append(cl.Action(
            name="handle_team_selection", # A new callback will handle this
            label=team_info["name"],
            description=team_info["description"],
            payload={"team_id": team_id} # Pass team_id in payload
        ))

    # --- Create Individual Persona Select Toggles ---
    select_elements = []
    for p_data in available_personas:
        select_elements.append(
            cl.Select(
                id=p_data["id"], 
                label=p_data["name"],
                initial_value=p_data["id"] in currently_selected_ids
            )
        )
    
    # --- Create Update Button for Individual Selections ---
    update_individual_action = cl.Action(
        name="submit_persona_selection", 
        label="Update Individual Personas", # Changed label for clarity
        payload={}
    )

    # Combine team actions and the update action for individual selections
    all_actions = team_actions + [update_individual_action]

    await cl.Message(
        content="Select a predefined team or adjust individual personas below. Click \"Update Individual Personas\" to apply manual changes.",
        elements=select_elements,
        actions=all_actions
    ).send()

@cl.action_callback("handle_team_selection")
async def handle_team_selection_action(action: cl.Action):
    """Handles selection of a predefined persona team, updates state, and refreshes the selection UI."""
    team_id = action.payload.get("team_id")
    if not team_id:
        await cl.Message(content="Error: Team ID missing in action payload.").send()
        return

    team_info = PERSONA_TEAMS.get(team_id)
    if not team_info:
        await cl.Message(content=f"Error: Team {team_id} not found.").send()
        return

    insight_flow_state: InsightFlowState = cl.user_session.get("insight_flow_state")
    if not insight_flow_state:
        await cl.Message(content="Error: Session state not found. Cannot update team selection.").send()
        return

    # Update selected_personas with the team's members
    updated_state = insight_flow_state.copy()
    updated_state["selected_personas"] = list(team_info["members"]) # Ensure it's a new list
    cl.user_session.set("insight_flow_state", updated_state)

    # Refresh the persona selection UI to reflect the new team selection
    # We can call select_personas_action again. It doesn't use its action param for rendering.
    await select_personas_action(action=None) # Pass None or a dummy action if needed by its signature

@cl.action_callback("submit_persona_selection")
async def submit_persona_selection_action(action: cl.Action, values: Dict[str, bool]):
    """Handles submission of persona selections, updates state, and confirms."""
    insight_flow_state: InsightFlowState = cl.user_session.get("insight_flow_state")

    if not insight_flow_state:
        await cl.Message(content="Error: Session state not found. Cannot update persona selection.").send()
        return

    # Update selected_personas based on the submitted values
    newly_selected_personas = [persona_id for persona_id, is_selected in values.items() if is_selected]
    
    # Create a mutable copy of the state to update
    updated_state = insight_flow_state.copy()
    updated_state["selected_personas"] = newly_selected_personas
    
    cl.user_session.set("insight_flow_state", updated_state)
    
    await cl.Message(content="Persona selection updated!").send()

# Placeholder for direct LLM invocation logic
async def invoke_direct_llm(query: str):
    print(f"invoke_direct_llm called with query: {query}")
    
    messages = [
        SystemMessage(content=DIRECT_SYSPROMPT),
        HumanMessage(content=query)
    ]
    
    response_message = cl.Message(content="")
    await response_message.send()

    async for chunk in llm_direct.astream(messages):
        if chunk.content:
            await response_message.stream_token(chunk.content)
    
    await response_message.update() # Finalize the streamed message
    return "Direct response streamed" # Test expects a return, actual content is streamed

# Placeholder for LangGraph invocation logic
async def invoke_langgraph(query: str, initial_state: InsightFlowState):
    print(f"invoke_langgraph called with query: {query}")
    
    # Create and send initial progress message
    progress_msg = cl.Message(content="") # Start with empty content, stream to it
    await progress_msg.send() # Send it first to get a message ID for updates
    await progress_msg.stream_token("⏳ Initializing InsightFlow process...")
    cl.user_session.set("progress_msg", progress_msg) # Store for potential updates outside callbacks

    # Setup callback handler
    callback_handler = InsightFlowCallbackHandler(progress_message=progress_msg)

    current_state = initial_state.copy() # Work with a copy
    current_state["query"] = query
    current_state["current_step_name"] = "planner_agent" # Reset step for new invocation

    # Check for Quick Mode and adjust personas if needed
    quick_mode_active = cl.user_session.get("quick_mode", False) # Default to False if not set
    if quick_mode_active:
        print("Quick Mode is ON. Using predefined quick mode personas.")
        current_state["selected_personas"] = list(QUICK_MODE_PERSONAS) # Ensure it's a new list copy
    # If quick_mode is OFF, selected_personas from initial_state (set by on_chat_start or commands) will be used.

    # Prepare config for LangGraph invocation (e.g., for session/thread ID)
    # In a Chainlit context, cl.user_session.get("id") can give a thread_id
    thread_id = cl.user_session.get("id", "default_thread_id") # Get Chainlit thread_id or a default
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [callback_handler] # Add our callback handler
    }

    # progress_msg = cl.Message(content="⏳ Processing with InsightFlow (0%)...") # OLD TODO
    # await progress_msg.send()
    # cl.user_session.set("progress_msg", progress_msg)

    final_state = await insight_flow_graph.ainvoke(current_state, config=config)
    
    # Final progress update
    if progress_msg:
        await progress_msg.stream_token("\n✨ InsightFlow processing complete!")
        await progress_msg.update() # Ensure all streamed tokens are sent

    # The present_results node should handle sending messages. 
    # invoke_langgraph will return the final state which on_message saves.
    return final_state

@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming user messages and routes based on direct_mode."""
    direct_mode = cl.user_session.get("direct_mode")
    msg_content_lower = message.content.lower().strip()

    # Command handling for /direct
    if msg_content_lower == "/direct on":
        cl.user_session.set("direct_mode", True)
        await cl.Message(content="Direct mode ENABLED.").send()
        return # Command processed, no further action
    elif msg_content_lower == "/direct off":
        cl.user_session.set("direct_mode", False)
        await cl.Message(content="Direct mode DISABLED.").send()
        return # Command processed, no further action
    
    # Command handling for /show perspectives
    elif msg_content_lower == "/show perspectives on":
        cl.user_session.set("show_perspectives", True)
        await cl.Message(content="Show perspectives ENABLED.").send()
        return
    elif msg_content_lower == "/show perspectives off":
        cl.user_session.set("show_perspectives", False)
        await cl.Message(content="Show perspectives DISABLED.").send()
        return

    # Command handling for /show visualization
    elif msg_content_lower == "/show visualization on":
        cl.user_session.set("show_visualization", True)
        await cl.Message(content="Show visualization ENABLED.").send()
        return
    elif msg_content_lower == "/show visualization off":
        cl.user_session.set("show_visualization", False)
        await cl.Message(content="Show visualization DISABLED.").send()
        return

    # Command handling for /quick_mode
    elif msg_content_lower == "/quick_mode on":
        cl.user_session.set("quick_mode", True)
        await cl.Message(content="Quick mode ENABLED.").send()
        return
    elif msg_content_lower == "/quick_mode off":
        cl.user_session.set("quick_mode", False)
        await cl.Message(content="Quick mode DISABLED.").send()
        return

    # If not a /direct command, proceed with existing direct_mode check for LLM calls
    if direct_mode:
        await invoke_direct_llm(message.content)
    else:
        insight_flow_state = cl.user_session.get("insight_flow_state")
        if not insight_flow_state:
            # Fallback if state isn't somehow initialized (should not happen with on_chat_start)
            await cl.Message(content="Error: Session state not found. Please restart the chat.").send()
            return
        updated_state = await invoke_langgraph(message.content, insight_flow_state)
        cl.user_session.set("insight_flow_state", updated_state) # Save updated state

print("app.py initialized with LLMs, on_chat_start, and on_message defined")