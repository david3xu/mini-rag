# Mini-RAG: Windows System Implementation Guide

## System Requirements

| Component | Minimum Requirement | Recommended |
|-----------|---------------------|-------------|
| Operating System | Windows 10 (64-bit) | Windows 10/11 (64-bit) |
| Processor | Dual-core CPU | Quad-core CPU |
| Memory | 4GB RAM | 8GB RAM |
| Storage | 5GB free space | 10GB free space |
| Python | Version 3.10+ | Version 3.10+ |
| Node.js | Version 16+ | Version 18+ |

## Implementation Framework

### Environment Configuration

```powershell
# Create directory structure
mkdir models\embeddings
mkdir vector_db
mkdir data\uploads
mkdir data\processed
```

### Python Environment Setup

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install core dependencies
pip install fastapi uvicorn "pydantic<2.0.0" sentence-transformers==2.2.2
pip install chromadb==0.4.13 psutil pymupdf
pip install llama-cpp-python --prefer-binary
```

### Model Acquisition Strategy

```powershell
# Create models directory if it doesn't exist
mkdir -p models/embeddings

# Download recommended model (Q4_0 - balanced size/quality)
curl -L "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf" -o "backend/models/phi-2.gguf"

# Alternative using wget
# wget "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_0.gguf" -O "models/phi-2.gguf"

# Download lightweight embedding model
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save('backend/models/embeddings/all-MiniLM-L6-v2')"


python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', local_dir='./backend/models/embeddings/all-MiniLM-L6-v2')"


# Download optimized LLM (Option 1: PowerShell)
# Make sure models directory exists
mkdir -p models

# Download recommended model (Q4_0 - balanced size/quality)
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_0.gguf" -OutFile "models\phi-2.gguf"

# Alternative models by size and quality:
# Smallest size (lowest quality)
# Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q2_K.gguf" -OutFile "models\phi-2.gguf"

# Medium size and quality
# Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf" -OutFile "models\phi-2.gguf"

# Higher quality (larger file)
# Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q5_K_M.gguf" -OutFile "models\phi-2.gguf"

# Best quality (largest file)
# Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q8_0.gguf" -OutFile "models\phi-2.gguf"

# Download LLM (Option 2: Manual)
# Visit: https://huggingface.co/TheBloke/phi-2-GGUF/tree/main
# Download: phi-2.Q4_0.gguf (approximately 1.5GB) or another quantization
```

## System Configuration

### Backend Environment Variables

Create `.env` file in root directory:

```
# Core Service Configuration
MODEL_PATH=.\models\phi-2.gguf
VECTOR_DB_PATH=.\vector_db\chroma_db
EMBEDDING_MODEL=.\models\embeddings\paraphrase-MiniLM-L3-v2

# Resource Optimization Parameters
MODEL_N_CTX=1024
CHUNK_SIZE=500
CHUNK_OVERLAP=50
DEFAULT_BATCH_SIZE=4
MEMORY_SAFETY_MARGIN_MB=1024

# System Configuration
DEBUG=false
```

### Frontend Configuration

```powershell
# Install frontend dependencies
cd frontend
npm install --production

# Configure environment
echo "REACT_APP_API_URL=http://localhost:8000/api" > .env.development
```

## Deployment Procedure

### Backend Service Initialization

```powershell
# Activate virtual environment
cd <project-root>
.\venv\Scripts\activate

# Start backend service
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```

### Frontend Application Launch

```powershell
# Open new terminal
cd <project-root>\frontend
npm start
```

## Resource Optimization Framework

### Memory Management Strategy

| Technique | Implementation Method | Configuration Parameter |
|-----------|------------------------|-------------------------|
| Batch Processing | Reduce document processing batch size | `DEFAULT_BATCH_SIZE=4` |
| Context Window | Limit model context size | `MODEL_N_CTX=1024` |
| Document Chunking | Reduce chunk size for processing | `CHUNK_SIZE=500` |
| Vector Storage | Implement disk-based persistence | ChromaDB disk configuration |

### Performance Monitoring Protocol

Access system metrics through health endpoint:
```
http://localhost:8000/api/health
```

## Windows-Specific Considerations

### System Optimization

- Disable Windows Defender real-time scanning for model directories
- Configure adequate virtual memory (minimum 8GB pagefile)
- Close resource-intensive background applications
- Ensure Windows is not performing updates during operation

### Common Resolution Pathways

| Issue | Resolution Strategy |
|-------|---------------------|
| DLL Load Failures | Install Visual C++ Redistributable 2019 |
| Long Path Errors | Use shorter directory paths (<260 characters) |
| Permission Errors | Run terminal as Administrator |
| Python Module Issues | Verify pip installation integrity |
| Node.js Connection Errors | Check Windows Firewall settings |

## Operational Verification

### Backend Service Validation

Confirm backend availability:
```
http://localhost:8000/docs
```

Expected response: OpenAPI documentation interface

### Frontend Application Validation

Confirm frontend availability:
```
http://localhost:3000
```

Expected response: Mini-RAG interface with upload functionality

## Performance Enhancement Techniques

### Document Processing Optimization

- Process one document type at a time
- Implement smaller document batches
- Allow system caching between operations
- Monitor resource utilization through health endpoint

### Query Performance Strategy

- Utilize smaller model quantization (Q4_0)
- Implement progressive loading approach
- Configure single-worker deployment
- Adjust batch sizes based on system capability

This implementation guide provides a systematic approach to deploying the Mini-RAG system on Windows environments with resource constraints, ensuring optimal performance while maintaining core functionality.
