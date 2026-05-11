FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_function.txt .
RUN pip install --no-cache-dir -r requirements_function.txt

# Pre-download the model so cold start is fast
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy only what's needed — chunks.py is not touched
COPY chunks.py .
COPY retriever.py .
COPY rag_function.py .

# IBM Cloud Functions Docker action entry point
CMD ["python3", "-u", "rag_function.py"]
