# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy necessary files into the container
COPY train.py /app/train.py
COPY predict.py /app/predict.py
COPY model.pkl /app/model.pkl

# Install dependencies
RUN pip install flask scikit-learn pandas

# Default command
CMD ["python", "predict.py"]
