# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.5
FROM python:${PYTHON_VERSION} as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app


RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=.,target=. \
    pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD uvicorn main:app --reload --port 8000 --host 0.0.0.0
