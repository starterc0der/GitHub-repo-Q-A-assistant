FROM python:3.12-slim

WORKDIR /app

# git is a runtime dependency: repo_loader.py shells out to `git clone`.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer first, against a stub src/ dir — stays cached across source edits
# since it only invalidates when pyproject.toml's dependency list changes. Editing
# src/ later only reruns the small --no-deps step below.
COPY pyproject.toml .
RUN mkdir -p src && touch src/__init__.py
# CPU-only torch wheel: the default PyPI wheel bundles CUDA runtime libs (~1.2GB)
# this container never uses, since sentence-transformers/CrossEncoder run on CPU here.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

COPY src ./src
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
