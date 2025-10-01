FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY requirements.txt .

RUN uv pip install --system --no-cache -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]