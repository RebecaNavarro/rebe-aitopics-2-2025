FROM python:3.12-slim

WORKDIR /app

# dependencias para OpenCV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install uv

COPY requirements.txt .

RUN uv pip install --system --no-cache -r requirements.txt

COPY models ./models

COPY src ./src

EXPOSE 8000

CMD ["fastapi", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]