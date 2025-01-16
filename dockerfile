# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for pyodbc
RUN apt-get update && apt-get install -y \
    unixodbc-dev \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 80 to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME aifinance

# Run app.py when the container launches
CMD ["python", "app.py"]
