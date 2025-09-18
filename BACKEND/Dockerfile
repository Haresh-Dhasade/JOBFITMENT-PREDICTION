FROM python:3.11-slim

WORKDIR /app

# system deps for pdf/docx etc
RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev default-jre poppler-utils && rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy backend code
COPY main.py /app/main.py
# copy models (if you have pre-trained models place into models/ at project root and bind mount)
COPY ../models /app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]