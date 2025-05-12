"""
Base classes for the persona system.
"""

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
        if not os.path.exists(self.config_dir):
            print(f"Warning: Config directory {self.config_dir} not found")
            return
            
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.config_dir, filename), "r") as f:
                        config = json.load(f)
                        if "id" in config:
                            self.configs[config["id"]] = config
                except Exception as e:
                    print(f"Error loading config file {filename}: {e}")
                        
    def get_config(self, persona_id: str) -> Optional[Dict[str, Any]]:
        """Get config for a persona"""
        return self.configs.get(persona_id)
    
    def get_available_personas(self) -> List[Dict[str, Any]]:
        """Get list of all available personas with basic info"""
        result = []
        for persona_id, config in self.configs.items():
            result.append({
                "id": persona_id,
                "name": config.get("name", persona_id.capitalize()),
                "description": config.get("description", ""),
                "is_persona_type": config.get("is_persona_type", True),
                "parent_type": config.get("parent_type", "")
            })
        return result
        
    def create_persona(self, persona_id: str) -> Optional[PersonaReasoning]:
        """Create a persona instance based on ID"""
        config = self.get_config(persona_id)
        if not config:
            return None
            
        # Lazily import implementations to avoid circular imports
        try:
            if config.get("is_persona_type", True):
                # This is a persona type
                persona_type = config.get("type")
                if persona_type == "analytical":
                    from .impl import AnalyticalReasoning
                    return AnalyticalReasoning(config)
                elif persona_type == "scientific":
                    from .impl import ScientificReasoning
                    return ScientificReasoning(config)
                elif persona_type == "philosophical":
                    from .impl import PhilosophicalReasoning
                    return PhilosophicalReasoning(config)
                elif persona_type == "factual":
                    from .impl import FactualReasoning
                    return FactualReasoning(config)
                elif persona_type == "metaphorical":
                    from .impl import MetaphoricalReasoning
                    return MetaphoricalReasoning(config)
                elif persona_type == "futuristic":
                    from .impl import FuturisticReasoning
                    return FuturisticReasoning(config)
            else:
                # This is a personality
                parent_type = config.get("parent_type")
                parent_config = self.get_config(parent_type)
                if parent_config:
                    if persona_id == "holmes":
                        from .impl import HolmesReasoning
                        return HolmesReasoning(config, parent_config)
                    elif persona_id == "feynman":
                        from .impl import FeynmanReasoning
                        return FeynmanReasoning(config, parent_config)
                    elif persona_id == "fry":
                        from .impl import FryReasoning
                        return FryReasoning(config, parent_config)
        except Exception as e:
            print(f"Error creating persona {persona_id}: {e}")
                    
        return None 