#!/bin/bash

# Set variables
DOCKER_IMAGE="modeha/my-ml-image:latest"
LOCAL_DATA_DIR="/Users/mohsen/PycharmProjects/kubernetes/k8s-ml-pipeline/app/data"
DOCKERFILE_DIR="/Users/mohsen/PycharmProjects/kubernetes/k8s-ml-pipeline/simple/"
YAML_FILE="/Users/mohsen/PycharmProjects/kubernetes/k8s-ml-pipeline/simple/train-job.yaml"

# Step 1: Build the Docker Image
echo "Building Docker image..."
cd "$DOCKERFILE_DIR" || { echo "Dockerfile directory not found! Exiting."; exit 1; }
docker build -t "$DOCKER_IMAGE" -f Dockerfile .

# Step 2: Push the Docker Image to Docker Hub
echo "Pushing Docker image to Docker Hub..."
docker push "$DOCKER_IMAGE"

# Step 3: Delete Existing Kubernetes Job
echo "Deleting existing Kubernetes job (if any)..."
kubectl delete job train-job --ignore-not-found

# Step 4: Apply the Updated Kubernetes Job
echo "Applying Kubernetes job..."
kubectl apply -f "$YAML_FILE"

# Step 5: Wait for the Pod to Start
echo "Waiting for the pod to be created..."
sleep 5

# Get the Pod name
POD_NAME=$(kubectl get pods --selector=job-name=train-job --output=jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
    echo "No pod created for the job. Exiting."
    exit 1
fi

echo "Pod name: $POD_NAME"

# Wait until the Pod is in a Running or Completed state
echo "Waiting for the pod to be ready..."
while true; do
    POD_STATUS=$(kubectl get pod "$POD_NAME" --output=jsonpath='{.status.phase}')
    echo "Current Pod status: $POD_STATUS"
    if [ "$POD_STATUS" == "Running" ] || [ "$POD_STATUS" == "Succeeded" ]; then
        break
    elif [ "$POD_STATUS" == "Failed" ]; then
        echo "Pod failed. Check the logs for more details."
        exit 1
    fi
    sleep 2
done

# Step 6: Fetch the Logs
echo "Fetching logs from the Pod..."
kubectl logs "$POD_NAME"
