# InkPulse AI Vision Service

A professional Python AI Service designed to analyze images for book classification and digital markup detection using a lightweight local Vision Language Model (VLM).

## Features
- **AI Vision Analysis**: Detects if an uploaded image is a book, and checks if it contains digital markings (scribbles, doodles, markup).
- **Ollama Integration**: Uses the ultra-lightweight **`moondream`** vision model (1.6B parameters), fitting comfortably within **4GB RAM**.
- **PostgreSQL Dynamic Prompts**: Stores prompts in PostgreSQL and loads them dynamically.
- **Cache-Aside Pattern**: Caches SQL prompts in Redis Stack.
- **SHA-256 Image Caching**: Computes image SHA-256 hashes and caches classification results in Redis Stack for sub-millisecond duplicate checks.
- **RedisVL Text Embeddings Cache**: Employs Redis Vector Library's `OllamaTextVectorizer` and `EmbeddingsCache` for caching query vectors.
- **Clean Architecture & Separation of Concerns**: Modular Constants, Services, Models, and Utilities.

---

## Folder Structure
```
inkpulse-ai/
├── config/
│   ├── config.yaml            # Main application config (YAML)
│   └── index_schema.yaml      # RedisVL index schema configuration (YAML)
├── src/
│   ├── main.py                # FastAPI entrypoint & router
│   ├── config.py              # Configuration manager using Pydantic
│   ├── constants/             # Folder containing clean constants split
│   │   ├── key_constants.py   # KeyConstants class (dynamically from Env)
│   │   └── message_constants/ # Scalable sub-package for message constants
│   ├── database.py            # PostgreSQL database connection (SQLAlchemy async)
│   ├── models/
│   │   └── prompt.py          # SQLAlchemy model for dynamic prompt templates
│   ├── services/
│   │   ├── cache_service.py   # Redis client wrapper & Cache-Aside / Strategy pattern
│   │   ├── ai_service.py      # Ollama Vision integration (moondream)
│   │   └── prompt_service.py  # Prompt manager (PostgreSQL + Redis Cache-Aside)
│   └── utils/
│       ├── hash_helper.py     # Image SHA256 helper for caching vision results
│       └── redisvl_helper.py  # RedisVL helper for OllamaTextVectorizer & EmbeddingsCache
├── .env                       # Environment variables (dynamic config)
├── requirements.txt           # Python dependency specifications
└── Dockerfile                 # Container configuration
```

---

## Local Setup

1. **Clone & Virtual Environment**:
   ```bash
   cd D:\JetBrains\PyCharm\Projects\inkpulse-ai
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure `.env`**:
   Adjust connection parameters inside `.env` to match your local setup:
   ```bash
   DB_URL=postgresql+asyncpg://postgres:AdminSecret123@localhost:5432/enterprise_db
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_PASSWORD=RedisSecret123
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_VISION_MODEL=moondream
   ```

3. **Start Ollama & Pull Moondream**:
   Make sure Ollama is running, then pull the vision model:
   ```bash
   ollama pull moondream
   ```

4. **Run FastAPI Service**:
   ```bash
   uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
   ```
   Open `http://127.0.0.1:8000/docs` to access the interactive Swagger documentation.

---

## API Endpoints

### 1. Image Vision Analysis
- **Endpoint**: `POST /api/v1/vision/analyze`
- **Body**: `Multipart/Form-Data` containing a `file` field with the image.
- **Behavior**: Checks if the image hash exists in Redis. If found, returns the cached prediction. Otherwise, queries Ollama, caches the result, and returns.

### 2. Reset Prompts Cache
- **Endpoint**: `POST /api/v1/prompts/reset-cache`
- **Query Params**: `key=vision_book_analysis`
- **Behavior**: Invalidates the Redis cache for the given prompt, forcing it to fetch from PostgreSQL on the next request.

### 3. Generate & Cache Text Embeddings
- **Endpoint**: `POST /api/v1/embeddings`
- **Query Params**: `text=your text here`
- **Behavior**: Vectorizes text using `nomic-embed-text` with RedisVL `EmbeddingsCache`.

---

## Deployment inside Kind Kubernetes Cluster

Deploy using the provided script in WSL:
```bash
cd /home/hienzheng/kind-clusters/multi-node/
./scripts/update_ai.sh
```
This script will build the local Docker container, load the image onto the Kind worker nodes (`inkpulse-worker`, `inkpulse-worker2`), deploy the Service and Deployment defined in `apps/ai.yaml`, and perform a rolling update.
