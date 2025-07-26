# Use a specific Python version for better reproducibility
FROM python:3.10

# Set work directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements/ /app/requirements/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements/dev-all.txt

# Copy only the files explicitly needed for installation
COPY setup.py pyproject.toml MANIFEST.in README.md /app/
COPY darts/ /app/darts/

# Install darts in development mode
RUN pip install --no-cache-dir -e . && \
    # Clean up pip cache
    rm -rf ~/.cache/pip

# Copy examples
COPY examples/ /app/examples/

# assuming you are working from inside your darts directory:
# docker build . -t darts-test:latest
# docker run -it -v $(pwd)/:/app/ darts-test:latest bash
