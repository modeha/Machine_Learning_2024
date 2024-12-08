#!/bin/bash

# Check for a name argument
if [ -z "$1" ]; then
  echo "Usage: ./run_pipeline.sh <name>"
  exit 1
fi

# Assign name to a variable
NAME=$1
DOCKER_IMAGE="modeha/$NAME:latest"
DEPLOYMENT_NAME="$NAME-api"
SERVICE_NAME="$NAME-api-service"
PORT=5000  # Default port for the Flask API

# Kill any process using the specified port
echo "Checking if port $PORT is in use..."
PID=$(lsof -ti :$PORT)
if [ -n "$PID" ]; then
  echo "Port $PORT is in use by PID $PID. Killing process..."
  kill -9 $PID
  echo "Process $PID killed. Port $PORT is now free."
else
  echo "Port $PORT is free."
fi

# Build the Docker image
echo "Building Docker image: $DOCKER_IMAGE..."
docker build -t $DOCKER_IMAGE -f Dockerfile .

# Push the Docker image to Docker Hub
echo "Pushing Docker image to Docker Hub..."
docker push $DOCKER_IMAGE

# Deploy the training job
echo "Deploying training job..."
kubectl delete job train-job --ignore-not-found
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: train-job
spec:
  template:
    spec:
      containers:
      - name: train-job
        image: $DOCKER_IMAGE
        command: ["python", "train.py"]
        volumeMounts:
        - name: dataset-volume
          mountPath: /app/dataset.csv
        - name: model-volume
          mountPath: /mnt
        - name: model-copy-volume
          mountPath: /nmt/model.pkl
      restartPolicy: Never
      volumes:
      - name: dataset-volume
        hostPath:
          path: /mnt/data/dataset.csv
          type: File
      - name: model-volume
        hostPath:
          path: /mnt/data
          type: DirectoryOrCreate
      - name: model-copy-volume
        hostPath:
          path: /mnt/data/model.pkl
          type: File
EOF

# Wait for the training job to complete
echo "Waiting for training job to complete..."
kubectl wait --for=condition=complete job/train-job || {
  echo "Training job failed. Check logs using: kubectl logs -l job-name=train-job";
  exit 1;
}

# Deploy the Flask API
echo "Deploying Flask API..."
kubectl delete deployment $DEPLOYMENT_NAME --ignore-not-found
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $DEPLOYMENT_NAME
spec:
  replicas: 2
  selector:
    matchLabels:
      app: $NAME-api
  template:
    metadata:
      labels:
        app: $NAME-api
    spec:
      containers:
      - name: predict-api
        image: $DOCKER_IMAGE
        ports:
        - containerPort: 5000
EOF

# Create a Kubernetes service for the API
echo "Exposing the Flask API as a service..."
kubectl delete service $SERVICE_NAME --ignore-not-found
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: $SERVICE_NAME
spec:
  selector:
    app: $NAME-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: NodePort
EOF

# Port-forward the service to localhost
echo "Port-forwarding the service to localhost:$PORT..."
kubectl port-forward svc/$SERVICE_NAME $PORT:80
