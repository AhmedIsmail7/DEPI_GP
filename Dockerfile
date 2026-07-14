FROM python:3.10-slim

# where the app lives in the container
WORKDIR /app

# get stuff we need for video processing
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# copy python packages list
COPY requirements.txt .

# install packages without keeping cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# copy all our code
COPY . .

# open port 7860
EXPOSE 7860

# start the backend server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
