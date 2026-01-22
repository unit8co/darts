# Use a specific Python version for better reproducibility
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install uv from official Astral source
# Using the standalone installer for reliability
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

# Copy dependency files first for better layer caching
# These change less frequently than source code
COPY pyproject.toml uv.lock /app/

# Install dependencies using uv
# --no-dev would install only core dependencies, but we want dev-all for the full environment
RUN uv sync --group dev-all --frozen

# Copy source code
COPY darts/ /app/darts/
COPY README.md /app/

# Copy examples
COPY examples/ /app/examples/

# Set Python path so imports work correctly
ENV PYTHONPATH=/app

# Default command opens a bash shell for interactive use
CMD ["/bin/bash"]

# Build and run instructions:
# docker build . -t darts:latest
# docker run -it -v $(pwd)/:/app/ darts:latest bash
#
# For Jupyter:
# docker run -it -p 8888:8888 darts:latest bash
# Then inside: uv run jupyter lab --ip 0.0.0.0 --no-browser --allow-root
