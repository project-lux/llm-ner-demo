import gradio as gr
import re
import difflib
from typing import List, Tuple
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

def create_entity_visualization(annotated_text: str) -> str:
    """Create a visual representation of entities with color coding."""
    entities = extract_entities(annotated_text)
    
    # Create a color map for different entity types
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ]
    
    unique_labels = list(set(label for _, label in entities))
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # Replace annotations with colored HTML spans
    result = annotated_text
    for entity, label in entities:
        color = color_map[label]
        original_pattern = f"[{re.escape(entity)}]({re.escape(label)})"
        replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{entity} <sub style="font-size: 0.8em;">({label})</sub></span>'
        result = result.replace(f"[{entity}]({label})", replacement, 1)
    
    return result

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

def process_ner(text: str, labels_text: str) -> Tuple[str, str, str]:
    """Process the text for NER and return all three outputs."""
    if not text.strip():
        return "Please enter some text to process.", "", ""
    
    if not labels_text.strip():
        return "Please enter at least one entity label.", "", ""
    
    try:
        # Parse labels
        labels = parse_labels_input(labels_text)
        
        if not labels:
            return "Please enter valid entity labels separated by commas.", "", ""
        
        # Perform NER
        logger.info(f"Processing text with labels: {labels}")
        annotated_text = llm_processor.perform_ner(text, labels)
        
        # Create visualizations
        visualization = create_entity_visualization(annotated_text)
        diff_output = create_text_diff(text, annotated_text)
        
        return annotated_text, visualization, diff_output
        
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        logger.error(error_msg)
        return error_msg, "", ""

def update_example_labels():
    """Provide example labels for common NER tasks."""
    return "PERSON, LOCATION, ORGANIZATION, DATE, MONEY"

# Create Gradio interface
with gr.Blocks(title="NER Demo with LLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Named Entity Recognition Demo
    
    This app uses a Large Language Model to perform Named Entity Recognition (NER) on your text. 
    You can specify custom entity labels and see the results in multiple formats.
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
                        value="<p style='color: #666; font-style: italic;'>Processed text will appear here with color-coded entities</p>"
                    )
                
                with gr.Tab("🔍 Text Diff"):
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
        outputs=[markdown_output, visualization_output, diff_output]
    )
    
    example_btn.click(
        fn=update_example_labels,
        outputs=labels_input
    )
    
    # Process on Enter key
    text_input.submit(
        fn=process_ner,
        inputs=[text_input, labels_input],
        outputs=[markdown_output, visualization_output, diff_output]
    )
    
    labels_input.submit(
        fn=process_ner,
        inputs=[text_input, labels_input],
        outputs=[markdown_output, visualization_output, diff_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
