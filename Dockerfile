# Use an appropriate base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED=1

# Install system dependencies for building libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create data directory for database
RUN mkdir -p /app/data

# Copy the dependency management files (lock file and pyproject.toml) first
COPY uv.lock pyproject.toml README.md /app/

# Install the application dependencies
RUN uv sync --frozen --no-cache

# Copy your application code maintaining src structure
COPY src/ /app/src/

# Set the virtual environment environment variables
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install the package in editable mode
RUN uv pip install -e .

# Define volumes
VOLUME ["/app/data"]

# Expose the port
EXPOSE 8000

# Run the Chainlit app
CMD ["chainlit", "run", "src/ai_companion/interfaces/chainlit/app.py", "--port", "8000", "--host", "0.0.0.0", "-h"]
