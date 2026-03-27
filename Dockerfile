FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip

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
    "fastapi", "uvicorn", "sqlalchemy", "alembic", "psycopg2-binary",\n\
    "boto3", "scikit-learn", "numpy", "pandas", "scipy",\n\
    "aiosqlite", "nltk", "pymorphy3", "tqdm", "PyJWT", "dotenv",\n\
    "phik", "pillow", "PyYAML", "stop-words", "emoji"\n\
]' > pyproject.toml
# COPY pyproject.toml ./ # 1.6 Gb vs 1 Gb as total

RUN pip install --no-cache-dir -e .

WORKDIR /app/src
# Run migrations
RUN alembic upgrade head
RUN cd ..

# Expose port
EXPOSE 8000

# Start FastAPI service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]