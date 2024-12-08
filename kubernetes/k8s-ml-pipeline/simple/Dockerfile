# Dockerfile for Prediction API
FROM python:3.9-slim

# Copy files into the container
COPY train.py /app/train.py
COPY predict.py /app/predict.py
COPY model.pkl /app/model.pkl
WORKDIR /app

# Install required libraries
RUN pip install flask scikit-learn pandas

# Set the default command to run predict.py
CMD ["python", "predict.py"]
