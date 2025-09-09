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

def create_entity_table(ner_result: Dict[str, Any]) -> str:
    """Create an HTML table for entity validation and editing."""
    if not ner_result or "entities" not in ner_result:
        return "<p>No entities to validate.</p>"
    
    entities = ner_result.get("entities", [])
    if not entities:
        return "<p>No entities found.</p>"
    
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
        
        # Confidence styling
        if confidence > 0.8:
            confidence_class = "confidence-high"
        elif confidence > 0.6:
            confidence_class = "confidence-medium"
        else:
            confidence_class = "confidence-low"
        
        # Wikidata link
        wikidata_cell = ""
        if wikidata_id:
            wikidata_cell = f'<a href="https://www.wikidata.org/wiki/{wikidata_id}" target="_blank" class="wikidata-link">{wikidata_id}</a>'
        else:
            wikidata_cell = "<em>Not found</em>"
        
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
                    <button onclick="alert('Entity validation feature coming soon!')" style="background: #4CAF50; color: white; border: none; padding: 4px 8px; border-radius: 3px; cursor: pointer;">✓</button>
                    <button onclick="alert('Entity editing feature coming soon!')" style="background: #ff9800; color: white; border: none; padding: 4px 8px; border-radius: 3px; cursor: pointer;">✏️</button>
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
        <span class="confidence-low">●</span> Low confidence (<0.6)
    </p>
    """
    
    return table_html

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
    """Process the text for NER and return all four outputs."""
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
        
        # Extract annotated text
        annotated_text = ner_result.get("annotated_text", text)
        
        # Create visualizations
        visualization = create_entity_visualization(ner_result)
        entity_table = create_entity_table(ner_result)
        diff_output = create_text_diff(text, annotated_text)
        
        return annotated_text, visualization, entity_table, diff_output
        
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        logger.error(error_msg)
        return error_msg, "", "", ""

def update_example_labels():
    """Provide example labels for common NER tasks."""
    return "PERSON, LOCATION, ORGANIZATION, DATE, MONEY"

# Create Gradio interface
with gr.Blocks(title="NER Demo with LLM", theme=gr.themes.Soft()) as demo:
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
