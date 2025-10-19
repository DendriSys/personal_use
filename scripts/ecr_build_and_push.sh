#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/ecr_build_and_push.sh <aws-region> <account-id> <repo-name>:<tag>
# Example: ./scripts/ecr_build_and_push.sh us-east-1 123456789012 itv-service:gpu

REGION=${1:-us-east-1}
ACCOUNT=${2:?account id required}
IMAGE_TAG=${3:-itv-service:latest}
REPO_NAME=$(echo "$IMAGE_TAG" | cut -d: -f1)
TAG=$(echo "$IMAGE_TAG" | cut -d: -f2)

aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION" >/dev/null

REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

docker build -t "$REPO_NAME:$TAG" .
docker tag "$REPO_NAME:$TAG" "$REGISTRY/$REPO_NAME:$TAG"
docker push "$REGISTRY/$REPO_NAME:$TAG"

echo "Pushed: $REGISTRY/$REPO_NAME:$TAG"