# LLM-powered Named Entity Recognition Demo

A Gradio web application that performs Named Entity Recognition (NER) using Google's Gemini model via Vertex AI.

## Features

- **Custom Entity Labels**: Add any entity labels you want (PERSON, LOCATION, ORGANIZATION, etc.)
- **Three Output Views**:
  - **Markdown Output**: Original text with entities annotated as `[entity](LABEL)`
  - **Visualization**: Color-coded entities with labels
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
3. Click "Process Text" to see results in all three formats

## Example

Input text: "Tom went to Rome yesterday."
Labels: "PERSON, LOCATION"
Output: "[Tom](PERSON) went to [Rome](LOCATION) yesterday."
