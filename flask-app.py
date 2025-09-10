from flask import Flask, render_template, request, jsonify, send_from_directory
import re
import difflib
import requests
import json
from typing import List, Tuple, Dict, Any
from src.llm import LLMProcessor
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the LLM processor
llm_processor = LLMProcessor()

def parse_labels_input(labels_text: str) -> List[str]:
    """Parse the labels input text into a list of labels."""
    if not labels_text.strip():
        return []
    
    # Split by comma and clean up
    labels = [label.strip().upper() for label in labels_text.split(',')]
    return [label for label in labels if label]

def strip_markdown_annotations(text: str) -> str:
    """Remove markdown annotations [entity](LABEL) from text, keeping only the entity."""
    # Pattern to match [entity](LABEL)
    pattern = r'\[([^\]]+)\]\([^)]+\)'
    return re.sub(pattern, r'\1', text)

def extract_entities(annotated_text: str) -> List[Tuple[str, str]]:
    """Extract entities and their labels from annotated text."""
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    return re.findall(pattern, annotated_text)

def get_wikidata_coordinates(wikidata_id: str) -> Dict[str, Any]:
    """Get geographic coordinates for a Wikidata entity."""
    try:
        # SPARQL query to get coordinates
        sparql_url = "https://query.wikidata.org/sparql"
        
        query = f"""
        SELECT ?item ?itemLabel ?coord WHERE {{
          VALUES ?item {{ wd:{wikidata_id} }}
          ?item wdt:P625 ?coord.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        
        headers = {
            'User-Agent': 'Flask-NER-Demo/1.0 (https://github.com/example/ner-demo; contact@example.com)',
            'Accept': 'application/sparql-results+json'
        }
        
        response = requests.get(sparql_url, params={'query': query}, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('results', {}).get('bindings'):
            binding = data['results']['bindings'][0]
            coord_value = binding.get('coord', {}).get('value', '')
            
            # Parse coordinate string like "Point(longitude latitude)"
            import re
            coord_match = re.search(r'Point\(([+-]?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\)', coord_value)
            if coord_match:
                longitude = float(coord_match.group(1))
                latitude = float(coord_match.group(2))
                
                return {
                    'wikidata_id': wikidata_id,
                    'label': binding.get('itemLabel', {}).get('value', ''),
                    'latitude': latitude,
                    'longitude': longitude,
                    'coordinate_string': coord_value
                }
        
        return {}
        
    except Exception as e:
        logger.error(f"Error fetching coordinates for {wikidata_id}: {e}")
        return {}

def search_wikidata(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search Wikidata for entities matching the query."""
    try:
        # Wikidata search API
        url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'search': query,
            'language': 'en',
            'format': 'json',
            'limit': limit
        }
        
        # Add User-Agent header to avoid 403 errors
        headers = {
            'User-Agent': 'Flask-NER-Demo/1.0 (https://github.com/example/ner-demo; contact@example.com)'
        }
        
        logger.info(f"Making Wikidata API request: {url} with params: {params}")
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Wikidata API response: {data}")
        
        results = []
        search_results = data.get('search', [])
        
        for item in search_results:
            result_item = {
                'id': item.get('id', ''),
                'label': item.get('label', ''),
                'description': item.get('description', ''),
                'url': f"https://www.wikidata.org/wiki/{item.get('id', '')}"
            }
            results.append(result_item)
        
        logger.info(f"Processed {len(results)} Wikidata results")
        return results
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error searching Wikidata: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error from Wikidata response: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching Wikidata: {e}")
        return []

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

@app.route('/api/process', methods=['POST'])
def process_ner():
    """Process the text for NER and return results."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        labels_text = data.get('labels', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text to process.'}), 400
        
        if not labels_text:
            return jsonify({'error': 'Please enter at least one entity label.'}), 400
        
        # Parse labels
        labels = parse_labels_input(labels_text)
        
        if not labels:
            return jsonify({'error': 'Please enter valid entity labels separated by commas.'}), 400
        
        # Perform NER
        logger.info(f"Processing text with labels: {labels}")
        ner_result = llm_processor.perform_ner(text, labels)
        
        # Return the complete result
        return jsonify({
            'success': True,
            'result': ner_result,
            'original_text': text
        })
        
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/wikidata/search', methods=['POST'])
def search_wikidata_api():
    """Search Wikidata for entities."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        query = data.get('query', '').strip()
        logger.info(f"Wikidata search request for query: '{query}'")
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if len(query) < 2:
            return jsonify({'results': []})
        
        results = search_wikidata(query)
        logger.info(f"Found {len(results)} results for query '{query}'")
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error in Wikidata search: {e}")
        return jsonify({'error': f'Failed to search Wikidata: {str(e)}'}), 500

@app.route('/api/entity/update', methods=['POST'])
def update_entity():
    """Update an entity's information."""
    try:
        data = request.get_json()
        entity_id = data.get('entity_id')
        updates = data.get('updates', {})
        
        # In a real application, you would save this to a database
        # For now, we'll just return success
        logger.info(f"Updating entity {entity_id} with: {updates}")
        
        return jsonify({
            'success': True,
            'message': 'Entity updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating entity: {e}")
        return jsonify({'error': 'Failed to update entity'}), 500

@app.route('/api/entities/coordinates', methods=['POST'])
def get_entities_coordinates():
    """Get coordinates for entities with Wikidata IDs."""
    try:
        data = request.get_json()
        entities = data.get('entities', [])
        
        geolocated_entities = []
        
        for entity in entities:
            wikidata_id = entity.get('wikidata_id', '').strip()
            if wikidata_id and wikidata_id.startswith('Q'):
                logger.info(f"Fetching coordinates for {wikidata_id}")
                coordinates = get_wikidata_coordinates(wikidata_id)
                
                if coordinates:
                    # Merge entity data with coordinates
                    geolocated_entity = {
                        **entity,
                        'latitude': coordinates['latitude'],
                        'longitude': coordinates['longitude'],
                        'coordinate_string': coordinates['coordinate_string']
                    }
                    geolocated_entities.append(geolocated_entity)
                    logger.info(f"Found coordinates for {entity['text']}: {coordinates['latitude']}, {coordinates['longitude']}")
                else:
                    logger.info(f"No coordinates found for {wikidata_id}")
        
        return jsonify({
            'success': True,
            'geolocated_entities': geolocated_entities,
            'total_entities': len(entities),
            'geolocated_count': len(geolocated_entities)
        })
        
    except Exception as e:
        logger.error(f"Error getting entity coordinates: {e}")
        return jsonify({'error': f'Failed to get coordinates: {str(e)}'}), 500

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
