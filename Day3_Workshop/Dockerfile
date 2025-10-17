# Base image (official Python image)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining code into the image
COPY . .

# Expose port 8000
EXPOSE 8000

# Run your main script
ENTRYPOINT ["python"]
CMD ["day02.py"]
