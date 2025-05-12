---
title: InsightFlow AI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
app_port: 7860
short_description: Multi-perspective research assistant with visualization capabilities
---

# InsightFlow AI: Multi-Perspective Research Assistant

InsightFlow AI is an advanced research assistant that analyzes topics from multiple perspectives, providing a comprehensive and nuanced understanding of complex subjects.

![InsightFlow AI](https://huggingface.co/datasets/suhas/InsightFlow-AI-demo/resolve/main/insightflow_banner.png)

## Features

### Multiple Perspective Analysis
- **Analytical**: Logical examination with methodical connections and patterns
- **Scientific**: Evidence-based reasoning grounded in empirical data
- **Philosophical**: Holistic exploration of deeper meaning and implications
- **Factual**: Straightforward presentation of verified information
- **Metaphorical**: Creative explanations through vivid analogies
- **Futuristic**: Forward-looking exploration of potential developments

### Personality Perspectives
- **Sherlock Holmes**: Deductive reasoning with detailed observation
- **Richard Feynman**: First-principles physics with clear explanations
- **Hannah Fry**: Math-meets-society storytelling with practical examples

### Visualization Capabilities
- **Concept Maps**: Automatically generated Mermaid diagrams showing relationships
- **Visual Notes**: DALL-E generated hand-drawn style visualizations of key insights
- **Visual-Only Mode**: Option to focus on visual representations for faster comprehension

### Export Options
- **Markdown Export**: Save analyses as formatted markdown with embedded visualizations
- **PDF Export**: Generate professionally formatted PDF documents

## How to Use

1. **Select Personas**: Use the `/add [persona_name]` command to build your research team
2. **Ask Your Question**: Type any research question or topic to analyze
3. **Review Insights**: Explore the synthesized view and individual perspectives
4. **Export Results**: Use `/export_md` or `/export_pdf` to save your analysis

## Commands

```
# Persona Management
/add [persona_name]    - Add a perspective to your research team
/remove [persona_name] - Remove a perspective from your team
/list                  - Show all available perspectives
/team                  - Show your current team and settings

# Visualization Options
/visualization on|off  - Toggle visualizations (Mermaid & DALL-E)
/visual_only on|off    - Show only visualizations without text

# Export Options
/export_md             - Export to markdown file
/export_pdf            - Export to PDF file

# Mode Options
/direct on|off         - Toggle direct LLM mode (bypasses multi-persona)
/perspectives on|off   - Toggle showing individual perspectives
```

## Example Topics

- Historical events from multiple perspectives
- Scientific concepts with philosophical implications
- Societal issues that benefit from diverse viewpoints
- Future trends analyzed from different angles
- Complex problems requiring multi-faceted analysis

## Technical Details

Built with Python using:
- LangGraph for orchestration
- OpenAI APIs for reasoning and visualization
- Chainlit for the user interface
- Custom persona system for perspective management

## Try These Examples

- "The impact of artificial intelligence on society"
- "Climate change adaptation strategies"
- "Consciousness and its relationship to the brain"
- "The future of work in the next 20 years"
- "Ancient Greek philosophy and its relevance today"

## Feedback and Support

For questions, feedback, or support, please open an issue on the [GitHub repository](https://github.com/suhas/InsightFlow-AI) or comment on this Space.
