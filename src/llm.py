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
                 model_name: str = "gemini-2.5-flash"):
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
        
        # Define the grounding tool for Google Search (following Vertex AI docs)
        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
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
        
        # Special config for NER that returns JSON (simplified without grounding first)
        self.ner_config = types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.8,
            max_output_tokens=8192,
            response_mime_type="application/json",  # JSON for structured entity data
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
            ]
        )
        
        # Config with grounding (following Vertex AI documentation) - relaxed safety settings
        self.ner_config_with_grounding = types.GenerateContentConfig(
            temperature=1.0,  # Documentation recommends 1.0 for ideal grounding results
            tools=[self.grounding_tool],  # Enable grounding for entity resolution
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
            ]
        )
        

        
        logger.info(f"Initialized GenAI client with Vertex AI backend for project {project_id}")
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the structured text response into JSON format - handles both text and JSON"""
        import re
        import json
        
        # Initialize result structure
        result = {
            "annotated_text": "",
            "entities": []
        }
        
        # First, try to parse as JSON (fallback case)
        try:
            json_data = json.loads(response_text)
            if isinstance(json_data, dict):
                if "annotated_text" in json_data:
                    result["annotated_text"] = json_data["annotated_text"]
                
                # Handle different JSON structures
                entities_key = None
                if "entities" in json_data:
                    entities_key = "entities"
                elif "entities_found" in json_data:
                    entities_key = "entities_found"
                
                if entities_key and json_data[entities_key]:
                    for entity_data in json_data[entities_key]:
                        entity = {}
                        
                        # Map different field names
                        entity["text"] = entity_data.get("text") or entity_data.get("entity", "")
                        entity["label"] = entity_data.get("label", "")
                        
                        # Handle position
                        if "start_pos" in entity_data and "end_pos" in entity_data:
                            entity["start_pos"] = entity_data["start_pos"]
                            entity["end_pos"] = entity_data["end_pos"]
                        elif "position" in entity_data:
                            pos = entity_data["position"]
                            if isinstance(pos, str) and "-" in pos:
                                start, end = pos.split("-")
                                entity["start_pos"] = int(start)
                                entity["end_pos"] = int(end)
                        
                        # Handle Wikidata ID
                        wikidata_id = entity_data.get("wikidata_id")
                        if wikidata_id and wikidata_id != "null" and wikidata_id is not None:
                            entity["wikidata_id"] = str(wikidata_id)
                        else:
                            entity["wikidata_id"] = ""
                        
                        entity["description"] = entity_data.get("description") or ""
                        entity["confidence"] = float(entity_data.get("confidence", 0.8))
                        
                        if entity["text"] and entity["label"]:
                            result["entities"].append(entity)
                
                return result
        except json.JSONDecodeError:
            pass  # Continue with text parsing
        
        # Parse as structured text format
        # Extract annotated text
        annotated_match = re.search(r'ANNOTATED TEXT:\s*\n(.+?)(?=\n\nENTITIES FOUND:|$)', response_text, re.DOTALL)
        if annotated_match:
            result["annotated_text"] = annotated_match.group(1).strip()
        
        # Extract entities section
        entities_match = re.search(r'ENTITIES FOUND:\s*\n(.+)', response_text, re.DOTALL)
        if entities_match:
            entities_text = entities_match.group(1)
            
            # Parse individual entities
            # Look for entity blocks starting with "- Entity:"
            entity_blocks = re.split(r'\n(?=- Entity:)', entities_text)
            
            for block in entity_blocks:
                if not block.strip():
                    continue
                    
                entity = {}
                
                # Extract entity fields using regex - handle multiline format
                entity_match = re.search(r'- Entity:\s*(.+?)(?=\n|$)', block, re.MULTILINE)
                label_match = re.search(r'Label:\s*(.+?)(?=\n|$)', block, re.MULTILINE)
                position_match = re.search(r'Position:\s*(\d+)-(\d+)', block)
                wikidata_match = re.search(r'Wikidata ID:\s*(.+?)(?=\n|$)', block, re.MULTILINE)
                description_match = re.search(r'Description:\s*(.+?)(?=\n|$)', block, re.MULTILINE)
                confidence_match = re.search(r'Confidence:\s*([0-9.]+)', block)
                
                if entity_match and label_match:
                    entity["text"] = entity_match.group(1).strip()
                    entity["label"] = label_match.group(1).strip()
                    
                    if position_match:
                        entity["start_pos"] = int(position_match.group(1))
                        entity["end_pos"] = int(position_match.group(2))
                    
                    if wikidata_match:
                        wikidata_id = wikidata_match.group(1).strip()
                        # Clean up common variations and validate format
                        if wikidata_id and wikidata_id not in ["N/A", "None", "NONE", "null"]:
                            # Basic validation: should be Q followed by numbers
                            if wikidata_id.startswith("Q") and wikidata_id[1:].isdigit():
                                entity["wikidata_id"] = wikidata_id
                            else:
                                logger.warning(f"Invalid Wikidata ID format: {wikidata_id}")
                                entity["wikidata_id"] = ""
                        else:
                            entity["wikidata_id"] = ""
                    
                    if description_match:
                        entity["description"] = description_match.group(1).strip()
                    
                    if confidence_match:
                        entity["confidence"] = float(confidence_match.group(1))
                    else:
                        entity["confidence"] = 0.8  # Default confidence
                    
                    result["entities"].append(entity)
        
        return result
    
    def _validate_wikidata_id(self, entity_text: str, wikidata_id: str) -> bool:
        """Basic validation of Wikidata ID format and reasonableness"""
        if not wikidata_id or not wikidata_id.startswith("Q"):
            return False
        
        # Check if it's a reasonable number (not too high)
        try:
            qnum = int(wikidata_id[1:])
            # Most real Wikidata entities are below Q100000000
            if qnum > 100000000:
                logger.warning(f"Suspicious high Wikidata ID {wikidata_id} for entity '{entity_text}'")
                return False
            return True
        except ValueError:
            return False
    
    def perform_ner(self, text: str, labels: List[str], use_grounding: bool = True) -> Dict[str, Any]:
        """
        Perform Named Entity Recognition on the given text using the specified labels.
        
        Args:
            text: The input text to process
            labels: List of entity labels to use for annotation
            use_grounding: Whether to use grounding tools (fallback to no grounding if fails)
            
        Returns:
            Dictionary containing annotated text and detailed entity information with Wikidata IDs
        """
        try:
            # Format the labels for the prompt
            labels_str = ", ".join(labels)
            
            # Create the prompt
            prompt = prompt_ner.format(labels=labels_str, text=text)
            logger.info(f"Generated prompt: {prompt[:200]}...")
            
            # Choose config based on grounding preference
            config_to_use = self.ner_config_with_grounding if use_grounding else self.ner_config
            logger.info(f"Using grounding: {use_grounding}")
            
            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config_to_use
            )
            
            logger.info(f"Response received: {response}")
            
            # Extract the response with grounding metadata
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                # Check for grounding metadata
                grounding_info = {}
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    grounding_info = {
                        "search_queries": getattr(candidate.grounding_metadata, 'web_search_queries', []),
                        "grounding_sources": getattr(candidate.grounding_metadata, 'sources', [])
                    }
                    logger.info(f"Grounding metadata found: {grounding_info}")
                
                if candidate.content and candidate.content.parts:
                    result = candidate.content.parts[0].text.strip()
                    logger.info(f"Raw response text: {result}")
                    
                    if result:  # Check if we have actual content
                        try:
                            # Parse the structured text response
                            parsed_result = self._parse_text_response(result)
                            
                            # Add grounding metadata to the result
                            if grounding_info:
                                parsed_result["grounding_metadata"] = grounding_info
                            
                            # Validate the expected structure
                            if "annotated_text" in parsed_result and "entities" in parsed_result:
                                # Validate Wikidata IDs
                                validated_entities = []
                                for entity in parsed_result["entities"]:
                                    if entity.get("wikidata_id"):
                                        if self._validate_wikidata_id(entity["text"], entity["wikidata_id"]):
                                            validated_entities.append(entity)
                                        else:
                                            logger.warning(f"Invalid Wikidata ID {entity['wikidata_id']} for {entity['text']}, removing")
                                            entity["wikidata_id"] = ""
                                            validated_entities.append(entity)
                                    else:
                                        validated_entities.append(entity)
                                
                                parsed_result["entities"] = validated_entities
                                logger.info(f"Successfully parsed NER result with {len(parsed_result['entities'])} entities")
                                return parsed_result
                            else:
                                logger.warning(f"Response missing expected fields. Keys found: {list(parsed_result.keys())}")
                                
                        except Exception as e:
                            logger.warning(f"Failed to parse text response: {e}")
                            logger.warning(f"Raw response that failed to parse: {result}")
                    else:
                        logger.warning("Empty response content received")
            
            # Check if we have grounding metadata but no content (common with grounding failures)
            logger.warning("No valid response content received from the model")
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    logger.warning(f"Finish reason: {candidate.finish_reason}")
            
            # If grounding was used and failed, try without grounding
            if use_grounding:
                logger.warning("Grounding failed, retrying without grounding...")
                return self.perform_ner(text, labels, use_grounding=False)
            
            # Final fallback: return basic structure
            logger.warning("Using fallback response structure")
            return {
                "annotated_text": text,
                "entities": []
            }
            
        except Exception as e:
            logger.error(f"Error in NER processing: {str(e)}")
            
            # If grounding was used and caused exception, try without grounding
            if use_grounding:
                logger.warning("Exception with grounding, retrying without grounding...")
                return self.perform_ner(text, labels, use_grounding=False)
            
            return {
                "annotated_text": text,
                "entities": []
            }
    
    def redo_single_entity(self, entity_text: str, entity_label: str, context_text: str = "") -> Dict[str, Any]:
        """Re-process a single entity to find better Wikidata ID and description"""
        try:
            # Create a focused prompt for single entity resolution
            focused_prompt = f"""
You are an expert entity resolution system with access to grounding tools. Your task is to find the most accurate Wikidata ID for a specific entity.

CRITICAL INSTRUCTIONS:
1. ONLY use Wikidata IDs that you find through grounding searches
2. Use grounding tools to search for "[entity_text] {entity_label.lower()} wikidata"
3. If multiple candidates exist, choose the most relevant one based on context
4. Return NONE if no clear match is found

Entity to resolve: "{entity_text}"
Entity type: {entity_label}
Context: {context_text if context_text else "No additional context"}

Search using grounding tools and return the result in this format:

ENTITY RESOLUTION:
- Entity: {entity_text}
- Label: {entity_label}
- Wikidata ID: [Q-number from grounding or NONE]
- Description: [description from grounding]
- Confidence: [0.0-1.0]

Use grounding tools to search for accurate information about this specific entity.
"""
            
            logger.info(f"Re-processing entity: {entity_text} ({entity_label})")
            
            # Generate content using grounding
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=focused_prompt,
                config=self.ner_config_with_grounding
            )
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                if candidate.content and candidate.content.parts:
                    result = candidate.content.parts[0].text.strip()
                    logger.info(f"Raw redo response: {result}")
                    
                    # Parse the single entity response
                    entity_data = self._parse_single_entity_response(result)
                    if entity_data:
                        return {
                            "success": True,
                            "entity": entity_data
                        }
            
            return {
                "success": False,
                "error": "No valid response from model"
            }
            
        except Exception as e:
            logger.error(f"Error in single entity redo: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_single_entity_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the response for a single entity resolution"""
        import re
        
        entity = {}
        
        # Extract fields using regex
        entity_match = re.search(r'- Entity:\s*(.+?)(?=\n|$)', response_text, re.MULTILINE)
        label_match = re.search(r'- Label:\s*(.+?)(?=\n|$)', response_text, re.MULTILINE)
        wikidata_match = re.search(r'- Wikidata ID:\s*(.+?)(?=\n|$)', response_text, re.MULTILINE)
        description_match = re.search(r'- Description:\s*(.+?)(?=\n|$)', response_text, re.MULTILINE)
        confidence_match = re.search(r'- Confidence:\s*([0-9.]+)', response_text)
        
        if entity_match and label_match:
            entity["text"] = entity_match.group(1).strip()
            entity["label"] = label_match.group(1).strip()
            
            if wikidata_match:
                wikidata_id = wikidata_match.group(1).strip()
                if wikidata_id and wikidata_id not in ["N/A", "None", "NONE", "null"]:
                    if wikidata_id.startswith("Q") and wikidata_id[1:].isdigit():
                        entity["wikidata_id"] = wikidata_id
                    else:
                        entity["wikidata_id"] = ""
                else:
                    entity["wikidata_id"] = ""
            
            if description_match:
                entity["description"] = description_match.group(1).strip()
            
            if confidence_match:
                entity["confidence"] = float(confidence_match.group(1))
            else:
                entity["confidence"] = 0.5
                
            return entity
        
        return None
    
    def simple_wikidata_lookup(self, entity_text: str, entity_label: str) -> Dict[str, Any]:
        """Simple function to look up Wikidata ID for an entity using LLM"""
        try:
            # Create a simple prompt for Wikidata lookup
            lookup_prompt = f"""
You are a Wikidata expert. Your task is to find the most accurate Wikidata ID for an entity.

Entity: "{entity_text}"
Type: {entity_label}

Instructions:
1. Use grounding tools to search for "{entity_text} {entity_label.lower()} wikidata"
2. Find the most relevant Wikidata ID (Q-number)
3. Respond with ONLY the Wikidata ID (e.g., Q12345) or "NONE" if not found

Search for the entity and return the Wikidata ID:
"""
            
            logger.info(f"Looking up Wikidata ID for: {entity_text} ({entity_label})")
            
            # Generate content using grounding
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=lookup_prompt,
                config=self.ner_config_with_grounding
            )
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                
                if candidate.content and candidate.content.parts:
                    result = candidate.content.parts[0].text.strip()
                    logger.info(f"Raw lookup response: {result}")
                    
                    # Extract Wikidata ID from response
                    import re
                    wikidata_match = re.search(r'Q\d+', result)
                    if wikidata_match:
                        wikidata_id = wikidata_match.group(0)
                        if self._validate_wikidata_id(entity_text, wikidata_id):
                            return {
                                "success": True,
                                "wikidata_id": wikidata_id,
                                "description": f"Found via lookup"
                            }
                    
                    return {
                        "success": False,
                        "error": "No valid Wikidata ID found"
                    }
            
            return {
                "success": False,
                "error": "No response from model"
            }
            
        except Exception as e:
            logger.error(f"Error in simple Wikidata lookup: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }