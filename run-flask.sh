#!/bin/bash

# Run the Flask NER Demo App
# Make sure you have installed the requirements: pip install -r requirements.txt

echo "Starting Flask NER Demo App..."
echo "The app will be available at http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Set Flask environment variables
export FLASK_APP=flask-app.py
export FLASK_ENV=development

# Run the Flask app
python flask-app.py
