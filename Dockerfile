# Use a lightweight Python base image
FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y gcc

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install --upgrade pip

# Copy the application code
COPY . .

# Expose the port for the Streamlit app
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]