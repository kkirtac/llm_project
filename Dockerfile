# Start from a lightweight Python image
FROM python:3.9-slim

# Set a working directory in the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker's cache
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Expose Streamlit's default port (8501)
EXPOSE 8501

# Run Streamlit when the container starts
CMD ["streamlit", "run", "app.py"]
