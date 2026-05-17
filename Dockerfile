FROM python:3.10-slim

# Create user for Hugging Face Spaces to avoid permission issues
RUN useradd -m -u 1000 user

# Set home directory and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to the user's home
WORKDIR $HOME/app

# Copy requirements first to leverage Docker cache
COPY --chown=user requirements.txt .

# Switch to the non-root user
USER user

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
