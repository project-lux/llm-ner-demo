import gradio as gr
import re
import difflib
from typing import List, Tuple, Dict, Any
from src.llm import LLMProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM processor
llm_processor = LLMProcessor()

# Global state for current entities
current_ner_result = {"entities": [], "annotated_text": ""}

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

def create_entity_visualization(ner_result: Dict[str, Any]) -> str:
    """Create a visual representation of entities with color coding and Wikidata information."""
    if not ner_result or "entities" not in ner_result:
        return "No entities found."
    
    annotated_text = ner_result.get("annotated_text", "")
    entities = ner_result.get("entities", [])
    
    # Create a color map for different entity types
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ]
    
    unique_labels = list(set(entity.get("label", "") for entity in entities))
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # Replace annotations with enriched HTML spans
    result = annotated_text
    for entity in entities:
        entity_text = entity.get("text", "")
        label = entity.get("label", "")
        wikidata_id = entity.get("wikidata_id", "")
        description = entity.get("description", "")
        confidence = entity.get("confidence", 0.0)
        
        color = color_map.get(label, "#CCCCCC")
        
        # Create tooltip with entity information
        tooltip_content = f"Type: {label}"
        if wikidata_id:
            tooltip_content += f"\\nWikidata: {wikidata_id}"
        if description:
            tooltip_content += f"\\nDescription: {description}"
        tooltip_content += f"\\nConfidence: {confidence:.2f}"
        
        # Create Wikidata link if available
        wikidata_link = ""
        if wikidata_id:
            wikidata_link = f' <a href="https://www.wikidata.org/wiki/{wikidata_id}" target="_blank" style="text-decoration: none; color: #0645ad;">🔗</a>'
        
        # Confidence indicator
        confidence_color = "#00AA00" if confidence > 0.8 else "#FFAA00" if confidence > 0.6 else "#FF6B6B"
        confidence_indicator = f'<span style="color: {confidence_color}; font-size: 0.8em;">●</span>'
        
        original_pattern = f"[{re.escape(entity_text)}]({re.escape(label)})"
        replacement = f'''<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold; cursor: help;" 
                          title="{tooltip_content}">
                          {entity_text} 
                          <sub style="font-size: 0.7em;">({label})</sub>
                          {confidence_indicator}
                          {wikidata_link}
                        </span>'''
        
        result = result.replace(f"[{entity_text}]({label})", replacement, 1)
    
    return result

def create_entity_table(ner_result: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Create an HTML table for entity validation and editing."""
    if not ner_result or "entities" not in ner_result:
        return "<p>No entities to validate.</p>", []
    
    entities = ner_result.get("entities", [])
    if not entities:
        return "<p>No entities found.</p>", []
    
    # Create dropdown choices for entity selection
    entity_choices = [f"{entity['text']} ({entity['label']})" for entity in entities]
    
    # Create HTML table
    table_html = """
    <style>
        .entity-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin: 10px 0;
        }
        .entity-table th, .entity-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .entity-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .entity-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .confidence-high { color: #00AA00; font-weight: bold; }
        .confidence-medium { color: #FFAA00; font-weight: bold; }
        .confidence-low { color: #FF6B6B; font-weight: bold; }
        .wikidata-link { 
            color: #0645ad; 
            text-decoration: none; 
            font-weight: bold;
        }
        .wikidata-link:hover { text-decoration: underline; }
    </style>
    <table class="entity-table">
        <thead>
            <tr>
                <th>Entity</th>
                <th>Type</th>
                <th>Position</th>
                <th>Confidence</th>
                <th>Wikidata ID</th>
                <th>Description</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for i, entity in enumerate(entities):
        entity_text = entity.get("text", "")
        label = entity.get("label", "")
        start_pos = entity.get("start_pos", 0)
        end_pos = entity.get("end_pos", 0)
        wikidata_id = entity.get("wikidata_id", "")
        description = entity.get("description", "")
        confidence = entity.get("confidence", 0.0)
        
        # Escape quotes for JavaScript (do this first)
        entity_text_js = entity_text.replace("'", "\\'").replace('"', '\\"')
        label_js = label.replace("'", "\\'").replace('"', '\\"')
        wikidata_id_js = wikidata_id.replace("'", "\\'").replace('"', '\\"') if wikidata_id else ""
        
        # Confidence styling
        if confidence > 0.8:
            confidence_class = "confidence-high"
        elif confidence > 0.6:
            confidence_class = "confidence-medium"
        else:
            confidence_class = "confidence-low"
        
        # Editable Wikidata cell
        wikidata_value = wikidata_id if wikidata_id else ""
        if wikidata_id:
            wikidata_cell = f'''
                <div style="display: flex; align-items: center; gap: 5px;">
                    <input type="text" value="{wikidata_id}" 
                           onchange="updateWikidata(this, '{entity_text_js}', '{label_js}')"
                           style="width: 80px; padding: 3px; border: 1px solid #ddd; border-radius: 3px; font-family: monospace; font-size: 0.9em;"
                           title="Click to edit Wikidata ID">
                    <a href="https://www.wikidata.org/wiki/{wikidata_id}" target="_blank" style="color: #1976d2; text-decoration: none; font-size: 0.8em;" title="View on Wikidata">🔗</a>
                </div>
            '''
        else:
            wikidata_cell = f'''
                <input type="text" value="" placeholder="Q12345"
                       onchange="updateWikidata(this, '{entity_text_js}', '{label_js}')"
                       style="width: 80px; padding: 3px; border: 1px solid #ddd; border-radius: 3px; font-family: monospace; font-size: 0.9em;"
                       title="Enter Wikidata ID (e.g., Q123)">
            '''
        
        # Truncate long descriptions
        if len(description) > 60:
            description = description[:57] + "..."
        
        table_html += f"""
            <tr>
                <td><strong>{entity_text}</strong></td>
                <td>{label}</td>
                <td>{start_pos}-{end_pos}</td>
                <td class="{confidence_class}">{confidence:.2f}</td>
                <td>{wikidata_cell}</td>
                <td>{description}</td>
                <td>
                    <button onclick="refreshWikidata('{entity_text_js}', '{label_js}')" 
                            style="background: #4CAF50; color: white; border: none; padding: 4px 8px; border-radius: 3px; cursor: pointer; font-size: 0.8em;" 
                            title="Search for better Wikidata ID">🔄</button>
                </td>
            </tr>
        """
    
    table_html += """
        </tbody>
    </table>
    <p style="font-size: 0.9em; color: #666;">
        <strong>Legend:</strong> 
        <span class="confidence-high">●</span> High confidence (>0.8) |
        <span class="confidence-medium">●</span> Medium confidence (0.6-0.8) |
        <span class="confidence-low">●</span> Low confidence (<0.6)<br>
        <strong>Actions:</strong> 🔄 Refresh Wikidata ID | Edit Q-codes directly in table
    </p>
    
    """
    
    return table_html, []

def create_text_diff(original: str, annotated: str) -> str:
    """Create a side-by-side diff view of original and processed text."""
    # Strip annotations for comparison
    processed = strip_markdown_annotations(annotated)
    
    # Create diff
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        processed.splitlines(keepends=True),
        fromfile='Original Text',
        tofile='Processed Text',
        lineterm=''
    )
    
    diff_text = ''.join(diff)
    
    if not diff_text.strip():
        return "No differences found between original and processed text (annotations removed)."
    
    return f"```diff\n{diff_text}\n```"

def process_ner(text: str, labels_text: str) -> Tuple[str, str, str, str]:
    """Process the text for NER and return all outputs."""
    global current_ner_result
    
    if not text.strip():
        return "Please enter some text to process.", "", "", ""
    
    if not labels_text.strip():
        return "Please enter at least one entity label.", "", "", ""
    
    try:
        # Parse labels
        labels = parse_labels_input(labels_text)
        
        if not labels:
            return "Please enter valid entity labels separated by commas.", "", "", ""
        
        # Perform NER
        logger.info(f"Processing text with labels: {labels}")
        ner_result = llm_processor.perform_ner(text, labels)
        print(ner_result)
        
        # Store globally for entity actions
        current_ner_result = ner_result.copy()
        current_ner_result["original_text"] = text
        
        # Extract annotated text
        annotated_text = ner_result.get("annotated_text", text)
        
        # Create visualizations
        visualization = create_entity_visualization(ner_result)
        entity_table, _ = create_entity_table(ner_result)
        diff_output = create_text_diff(text, annotated_text)
        
        return annotated_text, visualization, entity_table, diff_output
        
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        logger.error(error_msg)
        return error_msg, "", "", ""

def update_example_labels():
    """Provide example labels for common NER tasks."""
    return "PERSON, LOCATION, ORGANIZATION, DATE, MONEY"

def handle_redo_entity(selected_entity: str) -> Tuple[str, str, str, List[str]]:
    """Redo resolution for a specific entity"""
    global current_ner_result
    
    if not selected_entity or not current_ner_result.get("entities"):
        return "", "", "❌ No entity selected or no current results", []
    
    try:
        # Parse entity from selection (format: "EntityText (LABEL)")
        entity_text = selected_entity.split(" (")[0]
        entity_label = selected_entity.split(" (")[1].rstrip(")")
        
        logger.info(f"Redoing entity resolution for: {entity_text} ({entity_label})")
        
        # Use the LLM processor to redo the entity
        result = llm_processor.redo_single_entity(
            entity_text, 
            entity_label, 
            current_ner_result.get("original_text", "")
        )
        
        if result.get("success"):
            new_entity = result["entity"]
            
            # Update the entity in the current result
            updated_entities = []
            for entity in current_ner_result["entities"]:
                if entity["text"] == entity_text and entity["label"] == entity_label:
                    # Update with new resolution
                    updated_entity = entity.copy()
                    updated_entity.update({
                        "wikidata_id": new_entity.get("wikidata_id", ""),
                        "description": new_entity.get("description", ""),
                        "confidence": new_entity.get("confidence", 0.5)
                    })
                    updated_entities.append(updated_entity)
                else:
                    updated_entities.append(entity)
            
            # Update global state
            current_ner_result["entities"] = updated_entities
            
            # Regenerate outputs
            visualization = create_entity_visualization(current_ner_result)
            table_html, entity_choices = create_entity_table(current_ner_result)
            
            wikidata_id = new_entity.get("wikidata_id", "No ID found")
            status = f"✅ Successfully updated {entity_text}: {wikidata_id}"
            
            return visualization, table_html, status, entity_choices
        else:
            error_msg = result.get("error", "Unknown error")
            return "", "", f"❌ Failed to redo {entity_text}: {error_msg}", []
            
    except Exception as e:
        logger.error(f"Error in handle_redo_entity: {str(e)}")
        return "", "", f"❌ Error: {str(e)}", []

def handle_action_trigger(action_json: str) -> Tuple[str, str, str]:
    """Handle JavaScript-triggered actions from table buttons"""
    global current_ner_result
    
    if not action_json or not action_json.strip():
        return "", "", ""
    
    try:
        import json
        action = json.loads(action_json)
        action_type = action.get("type")
        entity_text = action.get("entity")
        entity_label = action.get("label")
        
        if action_type == "redo":
            # Handle redo action
            result = handle_redo_entity(f"{entity_text} ({entity_label})")
            return result[1], result[2], ""  # table, status, clear trigger
            
        elif action_type == "edit":
            # Handle edit action
            new_wikidata_id = action.get("wikidataId", "")
            result = handle_edit_entity(f"{entity_text} ({entity_label})", new_wikidata_id)
            return result[1], result[2], ""  # table, status, clear trigger
            
    except Exception as e:
        logger.error(f"Error handling action trigger: {str(e)}")
        return "", f"❌ Error: {str(e)}", ""
    
    return "", "", ""

def handle_edit_entity(selected_entity: str, new_wikidata_id: str) -> Tuple[str, str, str, List[str]]:
    """Edit the Wikidata ID for a specific entity"""
    global current_ner_result
    
    if not selected_entity or not current_ner_result.get("entities"):
        return "", "", "❌ No entity selected or no current results", []
    
    try:
        # Parse entity from selection
        entity_text = selected_entity.split(" (")[0]
        entity_label = selected_entity.split(" (")[1].rstrip(")")
        
        logger.info(f"Editing Wikidata ID for: {entity_text} ({entity_label}) -> {new_wikidata_id}")
        
        # Validate Wikidata ID format
        if new_wikidata_id and not (new_wikidata_id.startswith("Q") and new_wikidata_id[1:].isdigit()):
            return "", "", "❌ Invalid Wikidata ID format. Use Q followed by numbers (e.g., Q12345)", []
        
        # Update the entity in the current result
        updated_entities = []
        updated = False
        
        for entity in current_ner_result["entities"]:
            if entity["text"] == entity_text and entity["label"] == entity_label:
                # Update with new Wikidata ID
                updated_entity = entity.copy()
                updated_entity["wikidata_id"] = new_wikidata_id
                if new_wikidata_id:
                    updated_entity["description"] = f"Manually set to {new_wikidata_id}"
                    updated_entity["confidence"] = 1.0  # Manual edit gets high confidence
                else:
                    updated_entity["description"] = "Manually cleared"
                    updated_entity["confidence"] = 0.0
                updated_entities.append(updated_entity)
                updated = True
            else:
                updated_entities.append(entity)
        
        if updated:
            # Update global state
            current_ner_result["entities"] = updated_entities
            
            # Regenerate outputs
            visualization = create_entity_visualization(current_ner_result)
            table_html, entity_choices = create_entity_table(current_ner_result)
            
            if new_wikidata_id:
                wikidata_link = f"https://www.wikidata.org/wiki/{new_wikidata_id}"
                status = f"✅ Updated {entity_text} → {new_wikidata_id} ([View]({wikidata_link}))"
            else:
                status = f"✅ Cleared Wikidata ID for {entity_text}"
                
            return visualization, table_html, status, entity_choices
        else:
            return "", "", f"❌ Could not find entity {entity_text} to update", []
            
    except Exception as e:
        logger.error(f"Error in handle_edit_entity: {str(e)}")
        return "", "", f"❌ Error: {str(e)}", []

# JavaScript for inline entity editing
entity_js = """
<script>
// Global functions for entity actions
window.refreshWikidata = function(entityText, entityLabel) {
    alert('Wikidata lookup for "' + entityText + '" (' + entityLabel + ') - Feature will be integrated soon!\\n\\nFor now, you can manually edit the Q-code directly in the table.');
};

window.updateWikidata = function(input, entityText, entityLabel) {
    const newId = input.value.trim();
    
    // Basic validation
    if (newId && !newId.match(/^Q\\d+$/)) {
        alert('Invalid Wikidata ID format. Use Q followed by numbers (e.g., Q12345)');
        input.focus();
        return;
    }
    
    // Update the link if there's a valid ID
    if (newId) {
        const linkElement = input.nextElementSibling;
        if (linkElement && linkElement.tagName === 'A') {
            linkElement.href = 'https://www.wikidata.org/wiki/' + newId;
        } else {
            // Add link if it doesn't exist
            const link = document.createElement('a');
            link.href = 'https://www.wikidata.org/wiki/' + newId;
            link.target = '_blank';
            link.style.marginLeft = '5px';
            link.style.color = '#1976d2';
            link.style.textDecoration = 'none';
            link.style.fontSize = '0.8em';
            link.title = 'View on Wikidata';
            link.textContent = '🔗';
            input.parentNode.appendChild(link);
        }
    }
    
    console.log('Updated ' + entityText + ' (' + entityLabel + ') Wikidata ID to: ' + newId);
};

console.log('Entity editing functions loaded globally');
</script>
"""

# Create Gradio interface
with gr.Blocks(title="NER Demo with LLM", theme=gr.themes.Soft(), head=entity_js) as demo:
    gr.Markdown("""
    # Named Entity Recognition Demo with Wikidata Grounding
    
    This app uses a Large Language Model with grounding tools to perform Named Entity Recognition (NER) on your text. 
    It identifies entities, finds their Wikidata IDs, and provides confidence scores for validation.
    
    **Features:**
    - 🎯 Custom entity labels
    - 🌐 Wikidata ID resolution with grounding
    - 📊 Confidence scoring
    - ✅ Entity validation interface
    - 🔗 Direct links to Wikidata entries
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            
            text_input = gr.Textbox(
                label="Text to Process",
                placeholder="Enter the text you want to analyze for named entities...",
                lines=6,
                value="Tom went to Rome yesterday. He works for Microsoft and earned $50,000 last year."
            )
            
            labels_input = gr.Textbox(
                label="Entity Labels (comma-separated)",
                placeholder="e.g., PERSON, LOCATION, ORGANIZATION",
                value="PERSON, LOCATION, ORGANIZATION, MONEY",
                info="Enter the entity types you want to detect, separated by commas"
            )
            
            with gr.Row():
                process_btn = gr.Button("Process Text", variant="primary", scale=2)
                example_btn = gr.Button("Load Example Labels", variant="secondary", scale=1)
        
        with gr.Column(scale=2):
            gr.Markdown("### Results")
            
            with gr.Tabs():
                with gr.Tab("📝 Markdown Output"):
                    markdown_output = gr.Textbox(
                        label="Annotated Text (Markdown)",
                        lines=8,
                        info="Text with entities annotated as [entity](LABEL)"
                    )
                
                with gr.Tab("🎨 Visualization"):
                    visualization_output = gr.HTML(
                        label="Entity Visualization",
                        value="<p style='color: #666; font-style: italic;'>Processed text will appear here with color-coded entities and Wikidata links</p>"
                    )
                
                with gr.Tab("🔍 Entity Validation"):
                    entity_table_output = gr.HTML(
                        label="Entity Validation Table",
                        value="<p style='color: #666; font-style: italic;'>Entity validation table with Wikidata IDs and confidence scores will appear here</p>"
                    )
                
                with gr.Tab("📊 Text Diff"):
                    diff_output = gr.Markdown(
                        label="Difference View",
                        value="*Text differences will appear here*"
                    )
    
    # Add some examples
    gr.Markdown("""
    ### Example Texts to Try:
    - **News**: "Apple Inc. announced that Tim Cook will visit London next month to meet with European partners."
    - **Biography**: "Albert Einstein was born in Germany in 1879 and later moved to Princeton University in New Jersey."
    - **Business**: "The merger between Disney and Fox was completed in 2019 for $71.3 billion."
    """)
    
    # Event handlers
    process_btn.click(
        fn=process_ner,
        inputs=[text_input, labels_input],
        outputs=[markdown_output, visualization_output, entity_table_output, diff_output]
    )
    
    example_btn.click(
        fn=update_example_labels,
        outputs=labels_input
    )
    
    # Process on Enter key
    text_input.submit(
        fn=process_ner,
        inputs=[text_input, labels_input],
        outputs=[markdown_output, visualization_output, entity_table_output, diff_output]
    )
    
    labels_input.submit(
        fn=process_ner,
        inputs=[text_input, labels_input],
        outputs=[markdown_output, visualization_output, entity_table_output, diff_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
