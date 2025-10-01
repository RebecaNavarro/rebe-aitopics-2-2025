FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY requirements.txt .

RUN uv pip install --system --no-cache -r requirements.txt

COPY src src

EXPOSE 8000

CMD ["fastapi", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]