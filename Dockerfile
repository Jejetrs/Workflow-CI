# Gunakan base Python 3.11 (lebih stabil untuk MLServer)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies dengan versi yang kompatibel
RUN pip install --upgrade pip && \
    pip install pydantic==1.10.12 \
                typing-extensions==4.6.0 \
                mlflow==2.19.0 \
                mlserver==1.5.0 \
                mlserver-mlflow==1.5.0 \
                scikit-learn \
                pandas \
                joblib \
                numpy \
                scipy \
                imbalanced-learn

# Copy MLflow model artifacts ke container
COPY MLProject/Artefak/ /app/model/

# Expose port MLServer
EXPOSE 5000

# Start MLServer
CMD ["mlflow", "models", "serve", \
     "-m", "/app/model", \
     "-h", "0.0.0.0", \
     "-p", "5000", \
     "--no-conda"]
