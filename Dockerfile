FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for ffmpeg and yt-dlp
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 7860 for Hf
EXPOSE 7860

# Run the FastAPI server on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
