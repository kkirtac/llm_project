#!/bin/bash

IMAGE_NAME="my-streamlit-app"

if [[ "$1" == "--local" ]]; then
    # Run locally with virtual environment
    if [ -d "venv" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    else
        echo "Virtual environment not found. Please create one with 'python3 -m venv venv' and install dependencies."
        exit 1
    fi
    echo "Running Streamlit app locally..."
    streamlit run app.py
else
    # Check if the Docker image already exists
    if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
        echo "Docker image not found. Building the Docker image..."
        docker build -t $IMAGE_NAME .
    else
        echo "Docker image '$IMAGE_NAME' already exists. Skipping build."
    fi
    
    echo "Running Streamlit app in Docker..."
    docker run -p 8501:8501 $IMAGE_NAME
fi
