# LLM-powered Named Entity Recognition Demo

A Gradio web application that performs Named Entity Recognition (NER) using Google's Gemini model via Vertex AI.

## Features

- **🌐 Real-time Google Search Grounding**: Finds accurate Wikidata IDs using live search
- **🎯 Custom Entity Labels**: Add any entity labels you want (PERSON, LOCATION, ORGANIZATION, etc.)  
- **🔄 Interactive Entity Editing**: Redo entity resolution or manually edit Wikidata IDs
- **📊 Confidence Scoring**: View confidence levels for each entity identification
- **Four Output Views**:
  - **Markdown Output**: Original text with entities annotated as `[entity](LABEL)`
  - **Visualization**: Color-coded entities with confidence indicators and Wikidata links
  - **Entity Validation**: Interactive table with redo/edit capabilities
  - **Text Diff**: Shows differences between original and processed text

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud credentials for Vertex AI access

3. Run the application:
```bash
python app.py
```

4. Open your browser to `http://localhost:7860`

## Usage

1. Enter text you want to analyze
2. Specify entity labels (comma-separated, e.g., "PERSON, LOCATION, ORGANIZATION")  
3. Click "Process Text" to see results in all four formats
4. **Interactive Entity Editing**:
   - Go to the "🔍 Entity Validation" tab
   - **Edit Wikidata IDs directly** in the table by clicking on the Q-codes
   - Click "🔄" in the Actions column to search for better Wikidata IDs (feature coming soon)
   - Changes update the Wikidata links automatically

## Example

**Input:** "Albert Einstein was born in Germany. Microsoft was founded by Bill Gates."  
**Labels:** "PERSON, LOCATION, ORGANIZATION"  
**Output:** "[Albert Einstein](PERSON) was born in [Germany](LOCATION). [Microsoft](ORGANIZATION) was founded by [Bill Gates](PERSON)."

**With Grounding:**
- Albert Einstein → Q937 (German-born theoretical physicist)
- Germany → Q183 (Federal Republic of Germany)  
- Microsoft → Q2283 (American technology corporation)
- Bill Gates → Q5284 (American businessman)
