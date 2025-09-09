import json
import logging
import re
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
import os
import time
from .prompts import prompt_pronouns, prompt_coreference, prompt_ner

logger = logging.getLogger(__name__)


class LLMProcessor:
    """Processes Wikipedia data using Google GenAI with Vertex AI backend"""
    
    def __init__(self, 
                 project_id: str = "cultural-heritage-gemini",
                 location: str = "us-central1",
                 model_name: str = "gemini-2.5-flash-lite"):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize the client with Vertex AI backend
        self.http_options = types.HttpOptions(api_version="v1")
        self.client = genai.Client(
            vertexai=True, 
            project=project_id, 
            location=location,
            http_options=self.http_options
        )
        
        # Generation config with appropriate token limits
        self.base_config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.8,
            max_output_tokens=8192,  # Maximum allowed by API (8193 exclusive)
            response_mime_type="application/json",
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
            ]
        )
        
        # Special config for NER that returns plain text
        self.ner_config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.8,
            max_output_tokens=8192,
            response_mime_type="text/plain",  # Force plain text output
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
            ]
        )
        

        
        logger.info(f"Initialized GenAI client with Vertex AI backend for project {project_id}")
    
    def perform_ner(self, text: str, labels: List[str]) -> str:
        """
        Perform Named Entity Recognition on the given text using the specified labels.
        
        Args:
            text: The input text to process
            labels: List of entity labels to use for annotation
            
        Returns:
            The annotated text with entities in markdown format [entity](LABEL)
        """
        try:
            # Format the labels for the prompt
            labels_str = ", ".join(labels)
            
            # Create the prompt
            prompt = prompt_ner.format(labels=labels_str, text=text)
            
            # Generate content using text config
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.ner_config  # Use text-only config
            )
            
            # Extract the text from response
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    result = candidate.content.parts[0].text.strip()
                    
                    # Handle potential JSON responses despite text/plain setting
                    if result.startswith('[') and result.endswith(']'):
                        # If we still get JSON, try to convert it back to text
                        try:
                            import json
                            entities = json.loads(result)
                            logger.warning("Model returned JSON despite text/plain setting, converting...")
                            # This is a fallback - the prompt should prevent this
                            return text  # Return original text as fallback
                        except:
                            pass
                    
                    # Clean up any quote wrapping
                    if result.startswith('"') and result.endswith('"'):
                        result = result[1:-1]
                    
                    return result
            
            logger.warning("No valid response received from the model")
            return text  # Return original text if no response
            
        except Exception as e:
            logger.error(f"Error in NER processing: {str(e)}")
            return text  # Return original text on error