#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the Gradio app
echo "Starting the NER demo application..."
python app.py
