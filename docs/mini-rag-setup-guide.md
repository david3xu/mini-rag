# Mini-RAG System: Setup and Deployment Guide

## System Requirements

- Python 3.10 or higher
- Node.js 16+ and npm
- 4GB+ RAM (8GB+ recommended)
- 10GB+ disk space for models and vector database

## Repository Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-repository/mini-rag.git
cd mini-rag
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```bash
# Core Settings
MODEL_PATH=./models/phi-2.gguf
VECTOR_DB_PATH=./vector_db/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Resource Optimization
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
DEFAULT_BATCH_SIZE=8
MEMORY_SAFETY_MARGIN_MB=512

# Application Settings
DEBUG=false
```

## Backend Setup

### 1. Create Python Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Python Dependencies

```bash
pip install -r backend/requirements.txt
```

Required dependencies include:
- fastapi
- uvicorn
- llama-cpp-python
- sentence-transformers
- chromadb
- pymupdf  # For PDF processing
- psutil  # For resource monitoring

### 3. Download Models

#### Phi-2 Model

1. Create models directory:
   ```bash
   mkdir -p models
   ```

2. Download the quantized Phi-2 model:
   ```bash
   # Using wget
   wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf -O models/phi-2.gguf
   
   # Or using curl
   curl -L https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf -o models/phi-2.gguf
   ```

   Alternative: Visit [TheBloke/phi-2-GGUF](https://huggingface.co/TheBloke/phi-2-GGUF) and manually download the Q4_K_M variant.

#### Embedding Model

The sentence-transformers model will be automatically downloaded during first use, but you can pre-download it:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### 4. Prepare Directory Structure

```bash
mkdir -p data/uploads data/processed vector_db/chroma_db
```

## Frontend Setup

### 1. Install Node.js Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Frontend Environment

Create `.env.development` in the frontend directory:

```
REACT_APP_API_URL=http://localhost:8000/api
```

## Running the Application

### 1. Start Backend Server

From the root directory with activated virtual environment:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Start Frontend Development Server

In a new terminal:

```bash
cd frontend
npm start
```

Expected output:
```
Compiled successfully!

You can now view mini-rag-frontend in the browser.

Local:            http://localhost:3000
On Your Network:  http://192.168.x.x:3000
```

### 3. Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

## Testing the System

### 1. Document Upload

1. Prepare test documents (PDF, TXT, MD, or JSON format)
2. Click "Choose files" in the upload section
3. Select your test documents
4. Click "Upload"
5. Verify successful processing in the status display

### 2. Query Testing

1. Enter a question related to your uploaded documents
2. Click "Send" or press Ctrl+Enter
3. Review the response and source attribution
4. Verify that correct document snippets are displayed as sources

## Monitoring

### Resource Utilization

Access system health information at:
```
http://localhost:8000/api/health
```

This endpoint provides:
- CPU and memory usage
- Component status
- Vector store statistics

### Backend Documentation

Access API documentation at:
```
http://localhost:8000/docs
```

## Docker Deployment (Optional)

### 1. Build Containers

```bash
docker-compose build
```

### 2. Run with Docker Compose

```bash
docker-compose up
```

## Troubleshooting

### Memory Issues

If encountering memory errors during document processing:

1. Reduce batch size in `.env`:
   ```
   DEFAULT_BATCH_SIZE=4
   ```

2. Use a more quantized model variant:
   ```
   # Download a smaller model
   wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_0.gguf -O models/phi-2.gguf
   ```

3. Update MODEL_PATH in `.env`:
   ```
   MODEL_PATH=./models/phi-2.Q4_0.gguf
   ```

### Model Loading Failures

If the LLM fails to load:

1. Verify model file exists and is not corrupted
2. Check Python environment has llama-cpp-python installed correctly
3. Consider rebuilding llama-cpp-python with CPU optimizations:
   ```bash
   pip uninstall -y llama-cpp-python
   CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
   ```

## Performance Optimization

For improved performance:

1. Increase context window (if memory allows):
   ```
   # In .env
   MODEL_N_CTX=4096
   ```

2. Enable disk-based vector storage:
   ```python
   # Verify in vectorstore.py
   chroma_db_impl="duckdb+parquet"
   ```

3. Optimize batch size based on system capabilities

This guide provides a comprehensive setup process for the Mini-RAG system. Adjust configuration parameters based on your specific hardware capabilities and performance requirements.