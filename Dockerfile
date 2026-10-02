FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    HARBORLINE_SEED=42 \
    HARBORLINE_ANSWER_MODE=retrieve \
    HARBORLINE_RETRIEVE_BACKEND=pinecone \
    HARBORLINE_EMBEDDING_MODEL=text-embedding-3-small

COPY requirements.txt pyproject.toml README.md ./
COPY harborline ./harborline
COPY corpus ./corpus
COPY data ./data
COPY eval ./eval

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "harborline.api:app", "--host", "0.0.0.0", "--port", "8000"]
