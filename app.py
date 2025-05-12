from typing import TypedDict, List, Dict, Optional, Any
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
import chainlit as cl
import os
import asyncio
import base64
import requests
import time
import datetime
import random
import string
import fpdf
from pathlib import Path

# Re-enable the Tavily search tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
# from langchain_core.language_models import FakeListLLM  # Add FakeListLLM for testing
from langgraph.graph import StateGraph, END
from openai import OpenAI, AsyncOpenAI

# Import InsightFlow components
from insight_state import InsightFlowState
from utils.persona import PersonaFactory, PersonaReasoning

# Load environment variables
load_dotenv()

# Initialize OpenAI client for DALL-E
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- INITIALIZE CORE COMPONENTS ---

# Re-enable search tool initialization
tavily_tool = TavilySearchResults(max_results=3)

# Initialize LLMs with optimized settings for speed
llm_planner = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, request_timeout=20)
llm_analytical = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, request_timeout=20)
llm_scientific = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, request_timeout=20)
llm_philosophical = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4, request_timeout=20)
llm_factual = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, request_timeout=20)
llm_metaphorical = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, request_timeout=20)
llm_futuristic = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, request_timeout=20)
llm_synthesizer = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, request_timeout=20)

# Direct mode LLM with slightly higher quality
llm_direct = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, request_timeout=25)

# --- SYSTEM PROMPTS ---

PLANNER_SYSPROMPT = """You are an expert planner agent that coordinates research across multiple personas.
Given a user query, your task is to create a research plan with specific sub-tasks for each selected persona.
Break down complex queries into specific tasks that leverage each persona's unique perspective.
"""

SYNTHESIZER_SYSPROMPT = """You are a synthesis expert that combines multiple perspectives into a coherent response.
Given different persona perspectives on the same query, create a unified response that:
1. Highlights unique insights from each perspective
2. Notes areas of agreement and divergence
3. Organizes information logically for the user
Present the final response in a cohesive format that integrates all perspectives.
"""

DIRECT_SYSPROMPT = """You are a highly intelligent AI assistant that provides clear, direct, and helpful answers.
Your responses should be accurate, concise, and well-reasoned.
"""

# --- LANGGRAPH NODES FOR INSIGHTFLOW AI ---

async def run_planner_agent(state: InsightFlowState) -> InsightFlowState:
    """Plan the research approach for multiple personas"""
    query = state["query"]
    selected_personas = state["selected_personas"]
    
    # For the MVP implementation, we'll use a simplified planning approach
    # that just assigns the same query to each selected persona
    # In a full implementation, the planner would create custom tasks for each persona
    
    print(f"Planning research for query: {query}")
    print(f"Selected personas: {selected_personas}")
    
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
    
    print(f"Executing persona tasks for {len(selected_personas)} personas")
    
    # Get progress message if it exists
    progress_msg = cl.user_session.get("progress_msg")
    total_personas = len(selected_personas)
    
    # Process each persona with timeout safety
    # Using asyncio.gather to run multiple persona tasks in parallel for speed
    persona_tasks = []
    
    # First, create all personas and tasks
    for persona_id in selected_personas:
        persona = persona_factory.create_persona(persona_id)
        if persona:
            # Add progress message for user feedback
            await cl.Message(content=f"Generating insights from {persona_id} perspective...").send()
            # Create task to run in parallel
            task = generate_perspective_with_timeout(persona, query)
            persona_tasks.append((persona_id, task))
    
    # Run all perspective generations in parallel
    completed = 0
    for persona_id, task in persona_tasks:
        try:
            # Update dynamic progress if progress message exists
            if progress_msg:
                percent_done = 40 + int((completed / total_personas) * 40)
                await update_message(
                    progress_msg, 
                    f"⏳ Generating perspective from {persona_id} ({percent_done}%)..."
                )
            
            response = await task
            state["persona_responses"][persona_id] = response
            print(f"Perspective generated for {persona_id}")
            
            # Increment completed count
            completed += 1
            
        except Exception as e:
            print(f"Error getting {persona_id} perspective: {e}")
            state["persona_responses"][persona_id] = f"Could not generate perspective: {str(e)}"
            
            # Still increment completed count
            completed += 1
    
    state["current_step_name"] = "synthesize_responses"
    return state

async def generate_perspective_with_timeout(persona, query):
    """Generate a perspective with timeout handling"""
    try:
        # Set a timeout for each perspective generation
        response = await asyncio.wait_for(
            cl.make_async(persona.generate_perspective)(query),
            timeout=30  # 30-second timeout (reduced for speed)
        )
        return response
    except asyncio.TimeoutError:
        # Handle timeout by providing a simplified response
        return f"The perspective generation timed out. This may be due to high API traffic or complexity of the query."
    except Exception as e:
        # Handle other errors
        return f"Error generating perspective: {str(e)}"

async def synthesize_responses(state: InsightFlowState) -> InsightFlowState:
    """Combine perspectives from different personas"""
    query = state["query"]
    persona_responses = state["persona_responses"]
    
    if not persona_responses:
        state["synthesized_response"] = "No persona perspectives were generated."
        state["current_step_name"] = "present_results"
        return state

    print(f"Synthesizing responses from {len(persona_responses)} personas")
    
    # Add progress message for user feedback
    await cl.Message(content="Synthesizing insights from all perspectives...").send()
    
    # Prepare input for synthesizer
    perspectives_text = ""
    for persona_id, response in persona_responses.items():
        perspectives_text += f"\n\n{persona_id.capitalize()} Perspective:\n{response}"
    
    # Use LLM to synthesize with timeout
    messages = [
        SystemMessage(content=SYNTHESIZER_SYSPROMPT),
        HumanMessage(content=f"Query: {query}\n\nPerspectives:{perspectives_text}\n\nPlease synthesize these perspectives into a coherent response.")
    ]
    
    try:
        # Set a timeout for the synthesis
        synthesizer_response = await asyncio.wait_for(
            llm_synthesizer.ainvoke(messages),
            timeout=30  # 30-second timeout (reduced for speed)
        )
        state["synthesized_response"] = synthesizer_response.content
        print("Synthesis complete")
    except asyncio.TimeoutError:
        # Handle timeout for synthesis
        state["synthesized_response"] = "The synthesis of perspectives timed out. Here are the individual perspectives instead."
        print("Synthesis timed out")
    except Exception as e:
        print(f"Error synthesizing perspectives: {e}")
        state["synthesized_response"] = f"Error synthesizing perspectives: {str(e)}"
    
    state["current_step_name"] = "generate_visualization"
    return state

async def generate_dalle_image(prompt: str) -> Optional[str]:
    """Generate a DALL-E image and return the URL"""
    try:
        # Create a detailed prompt for hand-drawn style visualization
        full_prompt = f"Create a hand-drawn style visual note or sketch that represents: {prompt}. Make it look like a thoughtful drawing with annotations and key concepts highlighted. Include multiple perspectives connected together in a coherent visualization. Style: thoughtful hand-drawn sketch, notebook style with labels."
        
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

async def generate_visualization(state: InsightFlowState) -> InsightFlowState:
    """Generate a Mermaid diagram from the multiple perspectives"""
    # Get progress message if available and update it
    progress_msg = cl.user_session.get("progress_msg")
    if progress_msg:
        await update_message(progress_msg, "⏳ Generating visual representation (90%)...")
    
    # Skip if no synthesized response or no personas
    if not state.get("synthesized_response") or not state.get("persona_responses"):
        state["current_step_name"] = "present_results"
        return state
    
    # Get visualization settings
    show_visualization = cl.user_session.get("show_visualization", True)
    visual_only_mode = cl.user_session.get("visual_only_mode", False)
    
    # Determine if we should generate visualizations (either mode is on)
    should_visualize = show_visualization or visual_only_mode
    
    # Generate mermaid diagram if visualizations are enabled
    if should_visualize:
        try:
            # Create the absolute simplest Mermaid diagram possible
            query = state.get("query", "Query")
            query_short = query[:20] + "..." if len(query) > 20 else query
            
            # Generate the most basic diagram structure
            mermaid_text = f"""graph TD
        Q["{query_short}"]
        S["Synthesized View"]"""
            
            # Add each persona with a simple connection
            for i, persona in enumerate(state.get("persona_responses", {}).keys()):
                persona_short = persona.capitalize()
                node_id = f"P{i+1}"
                mermaid_text += f"""
        {node_id}["{persona_short}"]
        Q --> {node_id}
        {node_id} --> S"""
            
            # Store the simplified mermaid code
            state["visualization_code"] = mermaid_text
            print("Visualization generation complete with simplified diagram")
            
        except Exception as e:
            print(f"Error generating visualization: {e}")
            state["visualization_code"] = None
    
        # Generate DALL-E image if visualizations are enabled
        try:
            # Update progress message
            if progress_msg:
                await update_message(progress_msg, "⏳ Generating hand-drawn visualization (92%)...")
            
            # Create a prompt from the synthesized response
            image_prompt = state.get("synthesized_response", "")
            if len(image_prompt) > 500:
                image_prompt = image_prompt[:500]  # Limit prompt length
            
            # Add the query for context
            image_prompt = f"Query: {state.get('query', '')}\n\nSynthesis: {image_prompt}"
            
            # Generate the image
            image_url = await generate_dalle_image(image_prompt)
            state["visualization_image_url"] = image_url
            print("DALL-E visualization generated successfully")
        except Exception as e:
            print(f"Error generating DALL-E image: {e}")
            state["visualization_image_url"] = None
    
    state["current_step_name"] = "present_results"
    return state

async def present_results(state: InsightFlowState) -> InsightFlowState:
    """Present the final results to the user"""
    synthesized_response = state.get("synthesized_response", "No synthesized response available.")
    
    print("Presenting results to user")
    
    # Ensure progress is at 100% before showing results
    progress_msg = cl.user_session.get("progress_msg")
    if progress_msg:
        await update_message(progress_msg, "✅ Process complete (100%)")
    
    # Get visualization settings
    visual_only_mode = cl.user_session.get("visual_only_mode", False)
    show_visualization = cl.user_session.get("show_visualization", True)
    
    # Check if either visualization mode is enabled
    visualization_enabled = visual_only_mode or show_visualization
    
    # Determine panel mode
    panel_mode = "Research Assistant" if state["panel_type"] == "research" else "Multi-Persona Discussion"
    
    # Check if we have visualizations available
    has_mermaid = state.get("visualization_code") is not None
    has_dalle_image = state.get("visualization_image_url") is not None
    has_any_visualization = has_mermaid or has_dalle_image
    
    # Send text response if we're not in visual-only mode OR if no visualizations are available
    if not visual_only_mode or (visual_only_mode and not has_any_visualization):
        panel_indicator = f"**{panel_mode} Insights:**\n\n"
        # In visual-only mode with no visualizations, add an explanation
        if visual_only_mode and not has_any_visualization:
            panel_indicator = f"**{panel_mode} Insights (No visualizations available):**\n\n"
        await cl.Message(content=panel_indicator + synthesized_response).send()
    
    # Display DALL-E generated image if available and visualizations are enabled
    if has_dalle_image and visualization_enabled:
        try:
            # Add a title for the image
            if visual_only_mode:
                image_title = f"**Hand-drawn Visualization of {panel_mode} Insights:**"
            else:
                image_title = "**Hand-drawn Visualization:**"
            
            # Send the title
            await cl.Message(content=image_title).send()
            
            # Send the image URL as markdown
            image_url = state["visualization_image_url"]
            image_markdown = f"![DALL-E Visualization]({image_url})"
            await cl.Message(content=image_markdown).send()
            
        except Exception as e:
            print(f"Error displaying DALL-E image: {e}")
            # If in visual-only mode and image fails but we have no other visualization or text shown
            if visual_only_mode and not has_mermaid and state.get("text_fallback_shown", False) is not True:
                panel_indicator = f"**{panel_mode} Insights (Image generation failed):**\n\n"
                await cl.Message(content=panel_indicator + synthesized_response).send()
                state["text_fallback_shown"] = True
    
    # Display Mermaid diagram if available and visualizations are enabled
    if has_mermaid and visualization_enabled:
        try:
            # Add a brief summary in visual-only mode
            if visual_only_mode:
                diagram_title = f"**Concept Map of {panel_mode} Insights:**"
            else:
                diagram_title = "**Concept Map:**"
            
            # First send a title message
            await cl.Message(content=diagram_title).send()
            
            # Try to render the mermaid diagram
            try:
                # Ensure the diagram is extremely simple and valid
                mermaid_code = state['visualization_code']
                
                # Fallback to a guaranteed working diagram if rendering fails
                if not mermaid_code or len(mermaid_code) < 10:
                    mermaid_code = """graph TD
    A[Query] --> B[Analysis]
    B --> C[Result]"""
                
                # Create the mermaid block with proper syntax
                # Each line needs to be separate without extra indentation
                mermaid_block = "```mermaid\n"
                for line in mermaid_code.split('\n'):
                    mermaid_block += line.strip() + "\n"
                mermaid_block += "```"
                
                # Send the diagram as its own message
                await cl.Message(content=mermaid_block).send()
            except Exception as diagram_err:
                print(f"Error rendering diagram: {diagram_err}")
                # Try an ultra-simple fallback diagram
                ultra_simple = """```mermaid
graph TD
    A[Start] --> B[End]
```"""
                await cl.Message(content=ultra_simple).send()
            
            # Send the footer only if we have visualizations
            if has_any_visualization:
                await cl.Message(content="_Visualizations represent the key relationships between concepts from different perspectives._").send()
            
        except Exception as e:
            print(f"Error displaying visualization: {e}")
            # If in visual-only mode and visualization fails but no image shown yet and no text shown yet
            if visual_only_mode and not has_dalle_image and state.get("text_fallback_shown", False) is not True:
                panel_indicator = f"**{panel_mode} Insights (Visualization failed):**\n\n"
                await cl.Message(content=panel_indicator + synthesized_response).send()
                # Mark that we showed the fallback text to avoid duplicates
                state["text_fallback_shown"] = True
    
    # Check if user wants to see individual perspectives (not in visual-only mode)
    if cl.user_session.get("show_perspectives", True) and not visual_only_mode:
        # Show individual perspectives as separate messages instead of expandable elements
        for persona_id, response in state["persona_responses"].items():
            persona_name = persona_id.capitalize()
            
            # Get proper display name from config if available
            persona_factory = cl.user_session.get("persona_factory")
            if persona_factory:
                config = persona_factory.get_config(persona_id)
                if config and "name" in config:
                    persona_name = config["name"]
            
            # Just send the perspective as a message with a header
            perspective_message = f"**{persona_name}'s Perspective:**\n\n{response}"
            await cl.Message(content=perspective_message).send()
    
    state["current_step_name"] = "END"
    return state

# --- LANGGRAPH SETUP FOR INSIGHTFLOW AI ---
# Now define the graph with the functions we've defined above
insight_graph_builder = StateGraph(InsightFlowState)

# Add all nodes
insight_graph_builder.add_node("planner_agent", run_planner_agent)
insight_graph_builder.add_node("execute_persona_tasks", execute_persona_tasks)
insight_graph_builder.add_node("synthesize_responses", synthesize_responses)
insight_graph_builder.add_node("generate_visualization", generate_visualization)
insight_graph_builder.add_node("present_results", present_results)

# Add edges
insight_graph_builder.add_edge("planner_agent", "execute_persona_tasks")
insight_graph_builder.add_edge("execute_persona_tasks", "synthesize_responses")
insight_graph_builder.add_edge("synthesize_responses", "generate_visualization")
insight_graph_builder.add_edge("generate_visualization", "present_results")
insight_graph_builder.add_edge("present_results", END)

# Set entry point
insight_graph_builder.set_entry_point("planner_agent")

# Compile the graph
insight_flow_graph = insight_graph_builder.compile()
print("InsightFlow graph compiled successfully")

# --- DIRECT QUERY FUNCTION ---
async def direct_query(query: str):
    """Process a direct query without using multiple personas"""
    messages = [
        SystemMessage(content=DIRECT_SYSPROMPT),
        HumanMessage(content=query)
    ]
    
    try:
        # Direct query to LLM with streaming
        async for chunk in llm_direct.astream(messages):
            if chunk.content:
                # Yield chunk for streaming UI updates
                yield chunk.content
    except Exception as e:
        error_msg = f"Error processing direct query: {str(e)}"
        yield error_msg

# Helper function to display help information
async def display_help():
    """Display all available commands"""
    help_text = """
# InsightFlow AI Commands

**Persona Management:**
- `/add persona_name` - Add a persona to your research team (e.g., `/add factual`)
- `/remove persona_name` - Remove a persona from your team (e.g., `/remove philosophical`)
- `/list` - Show all available personas
- `/team` - Show your current team and settings

**Speed and Mode Options:**
- `/direct on|off` - Toggle direct LLM mode (bypasses multi-persona system)
- `/quick on|off` - Toggle quick mode (uses fewer personas)
- `/perspectives on|off` - Toggle showing individual perspectives
- `/visualization on|off` - Toggle showing visualizations (Mermaid diagrams & DALL-E images)
- `/visual_only on|off` - Show only visualizations without text (faster)

**Export Options:**
- `/export_md` - Export the current insight analysis to a markdown file
- `/export_pdf` - Export the current insight analysis to a PDF file

**System Commands:**
- `/help` - Show this help message

**Available Personas:**
- analytical - Logical problem-solving
- scientific - Evidence-based reasoning
- philosophical - Meaning and implications
- factual - Practical information 
- metaphorical - Creative analogies
- futuristic - Forward-looking possibilities
"""
    await cl.Message(content=help_text).send()

# Export functions
async def generate_random_id(length=8):
    """Generate a random ID for export filenames"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

async def export_to_markdown(state: InsightFlowState):
    """Export the current insight analysis to a markdown file"""
    if not state.get("synthesized_response"):
        return None, "No analysis available to export. Please run a query first."
    
    # Create exports directory if it doesn't exist
    Path("./exports").mkdir(exist_ok=True)
    
    # Generate a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = await generate_random_id()
    filename = f"exports/insightflow_analysis_{timestamp}_{random_id}.md"
    
    # Prepare content
    query = state.get("query", "No query specified")
    synthesized = state.get("synthesized_response", "No synthesized response")
    panel_mode = "Research Assistant" if state["panel_type"] == "research" else "Multi-Persona Discussion"
    
    # Create markdown content
    md_content = f"""# InsightFlow AI Analysis
*Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Query
{query}

## {panel_mode} Insights
{synthesized}

"""
    
    # Add perspectives if available
    if state.get("persona_responses"):
        md_content += "## Individual Perspectives\n\n"
        for persona_id, response in state["persona_responses"].items():
            persona_name = persona_id.capitalize()
            md_content += f"### {persona_name}'s Perspective\n{response}\n\n"
    
    # Add visualization section header
    md_content += "## Visualizations\n\n"
    
    # Add DALL-E image if available
    if state.get("visualization_image_url"):
        md_content += f"### Hand-drawn Visual Representation\n\n"
        md_content += f"![InsightFlow Visualization]({state['visualization_image_url']})\n\n"
    
    # Add visualization if available
    if state.get("visualization_code"):
        md_content += "### Concept Map\n\n```mermaid\n"
        for line in state["visualization_code"].split('\n'):
            md_content += line.strip() + "\n"
        md_content += "```\n\n"
        md_content += "*Note: The mermaid diagram will render in applications that support mermaid syntax, like GitHub or VS Code with appropriate extensions.*\n\n"
    
    # Add footer
    md_content += "---\n*Generated by InsightFlow AI*"
    
    # Write to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(md_content)
        return filename, None
    except Exception as e:
        return None, f"Error exporting to markdown: {str(e)}"

async def export_to_pdf(state: InsightFlowState):
    """Export the current insight analysis to a PDF file"""
    if not state.get("synthesized_response"):
        return None, "No analysis available to export. Please run a query first."
    
    # Create exports directory if it doesn't exist
    Path("./exports").mkdir(exist_ok=True)
    
    # Generate a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = await generate_random_id()
    filename = f"exports/insightflow_analysis_{timestamp}_{random_id}.pdf"
    
    try:
        # Create PDF
        pdf = fpdf.FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'InsightFlow AI Analysis', 0, 1, 'C')
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        pdf.ln(10)
        
        # Add query
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Query:', 0, 1)
        pdf.set_font('Arial', '', 11)
        query = state.get("query", "No query specified")
        pdf.multi_cell(0, 10, query)
        pdf.ln(5)
        
        # Add synthesized insights
        panel_mode = "Research Assistant" if state["panel_type"] == "research" else "Multi-Persona Discussion"
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f'{panel_mode} Insights:', 0, 1)
        pdf.set_font('Arial', '', 11)
        synthesized = state.get("synthesized_response", "No synthesized response")
        pdf.multi_cell(0, 10, synthesized)
        pdf.ln(10)
        
        # Add perspectives if available
        if state.get("persona_responses"):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Individual Perspectives:', 0, 1)
            pdf.ln(5)
            
            for persona_id, response in state["persona_responses"].items():
                persona_name = persona_id.capitalize()
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 10, f"{persona_name}'s Perspective:", 0, 1)
                pdf.set_font('Arial', '', 11)
                pdf.multi_cell(0, 10, response)
                pdf.ln(5)
        
        # Add visualizations section
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Visualizations', 0, 1, 'C')
        pdf.ln(5)
        
        # Add DALL-E image if available
        if state.get("visualization_image_url"):
            try:
                # Add header for the visualization
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'Hand-drawn Visual Representation:', 0, 1)
                pdf.ln(5)
                
                # Download the image
                image_url = state.get("visualization_image_url")
                image_path = f"exports/temp_image_{timestamp}_{random_id}.jpg"
                
                # Download the image using requests
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(image_path, 'wb') as img_file:
                        for chunk in response.iter_content(1024):
                            img_file.write(chunk)
                    
                    # Add the image to PDF with proper sizing
                    pdf.image(image_path, x=10, y=None, w=190)
                    pdf.ln(5)
                    
                    # Remove the temporary image
                    os.remove(image_path)
                else:
                    pdf.multi_cell(0, 10, "Could not download the visualization image.")
            except Exception as img_error:
                pdf.multi_cell(0, 10, f"Error including visualization image: {str(img_error)}")
        
        # Add mermaid diagram if available
        if state.get("visualization_code"):
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Concept Map Structure:', 0, 1)
            pdf.ln(5)
            
            # Extract relationships from the mermaid code
            mermaid_code = state.get("visualization_code", "")
            pdf.set_font('Arial', 'I', 10)
            pdf.multi_cell(0, 10, "Below is a text representation of the concept relationships:")
            pdf.ln(5)
            
            # Add a text representation of the diagram
            try:
                # Parse the mermaid code to extract relationships
                relationships = []
                for line in mermaid_code.split('\n'):
                    line = line.strip()
                    if '-->' in line:
                        parts = line.split('-->')
                        if len(parts) == 2:
                            source = parts[0].strip()
                            target = parts[1].strip()
                            relationships.append(f"• {source} connects to {target}")
                
                if relationships:
                    pdf.set_font('Arial', '', 10)
                    for rel in relationships:
                        pdf.multi_cell(0, 8, rel)
                else:
                    # Add a simplified representation of the concept map
                    pdf.multi_cell(0, 10, "The concept map shows relationships between the query and multiple perspectives, leading to a synthesized view.")
            except Exception as diagram_error:
                pdf.multi_cell(0, 10, f"Error parsing concept map: {str(diagram_error)}")
                pdf.multi_cell(0, 10, "The concept map shows the relationships between different perspectives on the topic.")
        
        # Add footer
        pdf.set_y(-15)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, 'Generated by InsightFlow AI', 0, 0, 'C')
        
        # Output PDF
        pdf.output(filename)
        return filename, None
    except Exception as e:
        return None, f"Error exporting to PDF: {str(e)}"

# --- CHAINLIT INTEGRATION ---
# Super simplified version with command-based persona selection

@cl.on_chat_start
async def start_chat():
    """Initialize the InsightFlow AI session"""
    print("InsightFlow AI chat started: Initializing session...")
    
    # Initialize persona factory and load configs
    persona_factory = PersonaFactory(config_dir="persona_configs")
    cl.user_session.set("persona_factory", persona_factory)
    
    # Initialize state with default personas
    initial_state = InsightFlowState(
        panel_type="research",
        query="",
        selected_personas=["analytical", "scientific", "philosophical"],
        persona_responses={},
        synthesized_response=None,
        current_step_name="awaiting_query",
        error_message=None
    )
    
    # Initialize LangGraph
    cl.user_session.set("insight_state", initial_state)
    cl.user_session.set("insight_graph", insight_flow_graph)
    
    # Set default options
    cl.user_session.set("direct_mode", False)  # Default to InsightFlow mode
    cl.user_session.set("show_perspectives", True)  # Default to showing all perspectives
    cl.user_session.set("quick_mode", False)  # Default to normal speed
    cl.user_session.set("show_visualization", True)  # Default to showing visualizations
    cl.user_session.set("visual_only_mode", False)  # Default to showing both text and visuals
    
    # Welcome message with command instructions
    welcome_message = """
# Welcome to InsightFlow AI

This assistant provides multiple perspectives on your questions using specialized personas.

**Your current research team:**
- Analytical reasoning
- Scientific reasoning
- Philosophical reasoning

Type `/help` to see all available commands.
"""
    await cl.Message(content=welcome_message).send()
    
    # Display help initially
    await display_help()

# Update function for Chainlit 2.5.5 compatibility
async def update_message(message, new_content):
    """Update a message in a way that's compatible with Chainlit 2.5.5"""
    try:
        # First try the direct content update method (newer versions)
        await message.update(content=new_content)
    except TypeError:
        # Fall back to older method for Chainlit 2.5.5
        message.content = new_content
        await message.update()

@cl.on_message
async def handle_message(message: cl.Message):
    """Handle user messages"""
    state = cl.user_session.get("insight_state")
    graph = cl.user_session.get("insight_graph")
    
    if not state or not graph:
        await cl.Message(content="Session error. Please refresh the page.").send()
        return
    
    # Check for commands to change personas or settings
    msg_content = message.content.strip()
    
    # Handle commands
    if msg_content.startswith('/'):
        parts = msg_content.split()
        command = parts[0].lower()
        
        if command == '/help':
            # Show help text
            await display_help()
            return
            
        elif command == '/list':
            # List available personas
            persona_list = """
**Available personas:**
- analytical - Logical problem-solving
- scientific - Evidence-based reasoning
- philosophical - Meaning and implications
- factual - Practical information 
- metaphorical - Creative analogies
- futuristic - Forward-looking possibilities
"""
            await cl.Message(content=persona_list).send()
            return
            
        elif command == '/team':
            # Show current team
            team_list = ", ".join([p.capitalize() for p in state["selected_personas"]])
            direct_mode = "ON" if cl.user_session.get("direct_mode", False) else "OFF"
            quick_mode = "ON" if cl.user_session.get("quick_mode", False) else "OFF"
            show_perspectives = "ON" if cl.user_session.get("show_perspectives", True) else "OFF"
            show_visualization = "ON" if cl.user_session.get("show_visualization", True) else "OFF"
            visual_only_mode = "ON" if cl.user_session.get("visual_only_mode", False) else "OFF"
            
            status = f"""
**Your current settings:**
- Research team: {team_list}
- Direct mode: {direct_mode}
- Quick mode: {quick_mode}
- Show perspectives: {show_perspectives}
- Show visualizations: {show_visualization}
- Visual-only mode: {visual_only_mode} (Mermaid diagrams & DALL-E images)
"""
            await cl.Message(content=status).send()
            return
            
        elif command == '/add' and len(parts) > 1:
            # Add persona
            persona_id = parts[1].lower()
            persona_factory = cl.user_session.get("persona_factory")
            
            if persona_factory and persona_factory.get_config(persona_id):
                if persona_id not in state["selected_personas"]:
                    state["selected_personas"].append(persona_id)
                    cl.user_session.set("insight_state", state)
                    await cl.Message(content=f"Added {persona_id} to your research team.").send()
                else:
                    await cl.Message(content=f"{persona_id} is already in your research team.").send()
            else:
                await cl.Message(content=f"Unknown persona: {persona_id}. Use /list to see available personas.").send()
            return
            
        elif command == '/remove' and len(parts) > 1:
            # Remove persona
            persona_id = parts[1].lower()
            
            if persona_id in state["selected_personas"]:
                if len(state["selected_personas"]) > 1:  # Don't remove the last persona
                    state["selected_personas"].remove(persona_id)
                    cl.user_session.set("insight_state", state)
                    await cl.Message(content=f"Removed {persona_id} from your research team.").send()
                else:
                    await cl.Message(content="Cannot remove the last persona. You need at least one for analysis.").send()
            else:
                await cl.Message(content=f"{persona_id} is not in your research team.").send()
            return
            
        elif command == '/direct' and len(parts) > 1:
            # Toggle direct mode
            setting = parts[1].lower()
            if setting in ['on', 'true', '1', 'yes']:
                cl.user_session.set("direct_mode", True)
                await cl.Message(content="Direct mode enabled. Bypassing InsightFlow for faster responses.").send()
            elif setting in ['off', 'false', '0', 'no']:
                cl.user_session.set("direct_mode", False)
                await cl.Message(content="Direct mode disabled. Using full InsightFlow system.").send()
            else:
                await cl.Message(content="Invalid option. Use `/direct on` or `/direct off`.").send()
            return
            
        elif command == '/perspectives' and len(parts) > 1:
            # Toggle showing perspectives
            setting = parts[1].lower()
            if setting in ['on', 'true', '1', 'yes']:
                cl.user_session.set("show_perspectives", True)
                await cl.Message(content="Individual perspectives will be shown.").send()
            elif setting in ['off', 'false', '0', 'no']:
                cl.user_session.set("show_perspectives", False)
                await cl.Message(content="Individual perspectives will be hidden for concise output.").send()
            else:
                await cl.Message(content="Invalid option. Use `/perspectives on` or `/perspectives off`.").send()
            return
            
        elif command == '/quick' and len(parts) > 1:
            # Toggle quick mode
            setting = parts[1].lower()
            if setting in ['on', 'true', '1', 'yes']:
                cl.user_session.set("quick_mode", True)
                if len(state["selected_personas"]) > 2:
                    # In quick mode, use max 2 personas
                    state["selected_personas"] = state["selected_personas"][:2]
                    cl.user_session.set("insight_state", state)
                await cl.Message(content="Quick mode enabled. Using fewer personas for faster responses.").send()
            elif setting in ['off', 'false', '0', 'no']:
                cl.user_session.set("quick_mode", False)
                await cl.Message(content="Quick mode disabled. Using your full research team.").send()
            else:
                await cl.Message(content="Invalid option. Use `/quick on` or `/quick off`.").send()
            return
            
        elif command == '/visualization' and len(parts) > 1:
            # Toggle showing Mermaid diagrams
            setting = parts[1].lower()
            if setting in ['on', 'true', '1', 'yes']:
                cl.user_session.set("show_visualization", True)
                await cl.Message(content="Visual diagrams will be shown to represent insights.").send()
            elif setting in ['off', 'false', '0', 'no']:
                cl.user_session.set("show_visualization", False)
                await cl.Message(content="Visual diagrams will be hidden.").send()
            else:
                await cl.Message(content="Invalid option. Use `/visualization on` or `/visualization off`.").send()
            return
            
        elif command == '/visual_only' and len(parts) > 1:
            # Toggle visual-only mode
            setting = parts[1].lower()
            if setting in ['on', 'true', '1', 'yes']:
                # When enabling visual-only mode, turn off other display options
                cl.user_session.set("visual_only_mode", True)
                cl.user_session.set("show_visualization", True)  # Ensure visualization is on
                cl.user_session.set("show_perspectives", False)  # Turn off perspective display
                await cl.Message(content="Visual-only mode enabled. Only visualizations (Mermaid diagrams & DALL-E images) will be shown. Individual perspectives have been disabled.").send()
            elif setting in ['off', 'false', '0', 'no']:
                cl.user_session.set("visual_only_mode", False)
                cl.user_session.set("show_perspectives", True)  # Restore default when turning off
                await cl.Message(content="Visual-only mode disabled. Both text and visualizations will be shown.").send()
            else:
                await cl.Message(content="Invalid option. Use `/visual_only on` or `/visual_only off`.").send()
            return
            
        elif command == '/export_md':
            # Export to markdown
            state = cl.user_session.get("insight_state")
            if not state:
                await cl.Message(content="No analysis data available. Run a query first.").send()
                return
            
            await cl.Message(content="Exporting analysis to markdown...").send()
            filename, error = await export_to_markdown(state)
            
            if error:
                await cl.Message(content=f"Error: {error}").send()
            else:
                await cl.Message(content=f"Analysis exported to: `{filename}`").send()
            return
            
        elif command == '/export_pdf':
            # Export to PDF
            state = cl.user_session.get("insight_state")
            if not state:
                await cl.Message(content="No analysis data available. Run a query first.").send()
                return
            
            await cl.Message(content="Exporting analysis to PDF...").send()
            filename, error = await export_to_pdf(state)
            
            if error:
                await cl.Message(content=f"Error: {error}").send()
            else:
                await cl.Message(content=f"Analysis exported to: `{filename}`").send()
            return
    
    # Process query (either direct or through InsightFlow)
    # Create streaming message for results
    answer_msg = cl.Message(content="")
    await answer_msg.send()
    
    # Create progress message
    progress_msg = cl.Message(content="⏳ Processing your query (0%)...")
    await progress_msg.send()
    
    try:
        # Check if direct mode is enabled
        if cl.user_session.get("direct_mode", False):
            # Direct mode with streaming - bypass InsightFlow
            await update_message(progress_msg, "⏳ Processing in direct mode (20%)...")
            
            # Stream response directly
            full_response = ""
            async for chunk in direct_query(msg_content):
                full_response += chunk
                # Update the message with the new chunk
                await update_message(answer_msg, f"**Direct Answer:**\n\n{full_response}")
                
            # Complete the progress
            await update_message(progress_msg, "✅ Processing complete (100%)")
            return
        
        # Apply quick mode if enabled
        if cl.user_session.get("quick_mode", False) and len(state["selected_personas"]) > 2:
            # Temporarily use just 2 personas for speed
            original_personas = state["selected_personas"].copy()
            state["selected_personas"] = state["selected_personas"][:2]
            await update_message(progress_msg, f"⏳ Using quick mode with personas: {', '.join(state['selected_personas'])} (10%)...")
        
        # Standard InsightFlow processing
        # Set query in state
        state["query"] = msg_content
        
        # Setup for progress tracking
        cl.user_session.set("progress_msg", progress_msg)
        cl.user_session.set("progress_steps", {
            "planner_agent": 10,
            "execute_persona_tasks": 40, 
            "synthesize_responses": 80,
            "generate_visualization": 90,
            "present_results": 95,
            "END": 100
        })
        
        # Hook into state changes for progress
        async def state_monitor():
            """Monitor state changes to update progress"""
            last_step = None
            while True:
                current_step = state.get("current_step_name")
                if current_step != last_step:
                    progress_steps = cl.user_session.get("progress_steps", {})
                    if current_step in progress_steps:
                        progress = progress_steps[current_step]
                        status_messages = {
                            "planner_agent": "Planning research approach",
                            "execute_persona_tasks": "Generating persona perspectives", 
                            "synthesize_responses": "Synthesizing perspectives",
                            "generate_visualization": "Generating visual representation",
                            "present_results": "Finalizing results",
                            "END": "Complete"
                        }
                        status = status_messages.get(current_step, current_step)
                        await update_message(progress_msg, f"⏳ {status} ({progress}%)...")
                    last_step = current_step
                
                # Check if we're done
                if current_step == "END":
                    await update_message(progress_msg, f"✅ Process complete (100%)")
                    break
                    
                # Wait before checking again
                await asyncio.sleep(0.5)
        
        # Start the monitor in the background
        asyncio.create_task(state_monitor())
        
        # Run the graph with timeout protection
        thread_id = cl.user_session.get("id", "default_thread_id")
        config = {"configurable": {"thread_id": thread_id}}
        
        # Set an overall timeout for the entire graph execution
        final_state = await asyncio.wait_for(
            graph.ainvoke(state, config), 
            timeout=150  # 2.5 minute timeout
        )
        cl.user_session.set("insight_state", final_state)
        
        # Update the answer message with the response
        panel_mode = "Research Assistant" if final_state["panel_type"] == "research" else "Multi-Persona Discussion"
        panel_indicator = f"**{panel_mode} Insights:**\n\n"
        await update_message(answer_msg, panel_indicator + final_state.get("synthesized_response", "No response generated."))
        
        # Show individual perspectives if enabled
        if cl.user_session.get("show_perspectives", True):
            for persona_id, response in final_state["persona_responses"].items():
                persona_name = persona_id.capitalize()
                
                # Get proper display name from config if available
                persona_factory = cl.user_session.get("persona_factory")
                if persona_factory:
                    config = persona_factory.get_config(persona_id)
                    if config and "name" in config:
                        persona_name = config["name"]
                
                # Send perspective as a message
                perspective_message = f"**{persona_name}'s Perspective:**\n\n{response}"
                await cl.Message(content=perspective_message).send()
        
        # Restore original personas if in quick mode
        if cl.user_session.get("quick_mode", False) and 'original_personas' in locals():
            state["selected_personas"] = original_personas
            cl.user_session.set("insight_state", state)
        
    except asyncio.TimeoutError:
        print("Overall graph execution timed out")
        await update_message(answer_msg, "The analysis took too long and timed out. Try using `/direct on` or `/quick on` for faster responses.")
        await update_message(progress_msg, "❌ Process timed out")
    except Exception as e:
        print(f"Error in query processing: {e}")
        await update_message(answer_msg, f"I encountered an error: {e}")
        await update_message(progress_msg, f"❌ Error: {str(e)}")

print("InsightFlow AI setup complete. Ready to start.")