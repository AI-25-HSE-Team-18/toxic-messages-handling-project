FROM python:3.11-slim

WORKDIR /app

# System deps needed for C-extension packages (pymorphy3, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install PyTorch CPU-only BEFORE the rest of the deps.
# The default PyPI wheel bundles CUDA (~2 GB); the cpu index is ~250 MB.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install transformers + its fast tokenizer backend
RUN pip install --no-cache-dir transformers tokenizers

# copy inference files only: 
COPY src/ ./src/
COPY create_admin.py ./

# create pyproject.toml for dependency management: 
# NOTE: simplified dependencies, ~ 600+ mB difference
RUN echo '[build-system]\n\
requires = ["setuptools", "wheel"]\n\
build-backend = "setuptools.build_meta"\n\
\n\
[project]\n\
name = "toxicity-without-extra-libs"\n\
version = "0.1.0"\n\
dependencies = [\n\
    "fastapi", "uvicorn[standard]", "sqlalchemy", "alembic",\n\
    "boto3", "scikit-learn", "numpy", "pandas", "scipy",\n\
    "aiosqlite", "nltk", "pymorphy3", "pymorphy3-dicts-ru", "tqdm",\n\
    "PyJWT", "python-dotenv", "pydantic", "python-multipart",\n\
    "phik", "pillow", "PyYAML", "stop-words", "emoji",\n\
    "prometheus-fastapi-instrumentator"\n\
]' > pyproject.toml
# COPY pyproject.toml ./ # 1.6 Gb vs 1 Gb as total

RUN pip install --no-cache-dir -e .

WORKDIR /app/src

EXPOSE 8000

# run alembic migration then launch uvicorn
CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]
