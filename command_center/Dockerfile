FROM python:3.12-slim

WORKDIR /app

# Install ML stack requirements first (from project root)
COPY requirements-model.txt .
RUN pip install --no-cache-dir -r requirements-model.txt

# Install dashboard requirements (with fpdf2 fix if needed)
COPY command_center/requirements.txt command_center/
RUN sed -i 's/^f p d f.*/fpdf2>=2.7.5/' command_center/requirements.txt || true
RUN pip install --no-cache-dir -r command_center/requirements.txt

# Copy application code and artifacts
COPY command_center/ command_center/
COPY final_validation/artifacts/ final_validation/artifacts/

ENV PYTHONPATH=/app
ENV ASTRAM_ARTIFACT_DIR=/app/final_validation/artifacts

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn command_center.backend.main:app --host 0.0.0.0 --port 8000 & exec streamlit run command_center/frontend/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true"]
