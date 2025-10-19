from __future__ import annotations

import argparse
import os

import sagemaker
from sagemaker.pytorch.model import PyTorchModel


# Note: This deploys our FastAPI container as a real-time endpoint. Ensure an ECR image is available.

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-uri", required=True)
    p.add_argument("--role-arn", required=True)
    p.add_argument("--instance-type", default="ml.g5.2xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--endpoint-name", default="itv-endpoint")
    p.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    return p.parse_args()


def main():
    args = parse_args()
    session = sagemaker.Session()

    # Deploy as a generic model container (serves uvicorn)
    model = sagemaker.model.Model(
        image_uri=args.image_uri,
        role=args.role_arn,
        sagemaker_session=session,
        env={
            "ITV_REQUIRE_CONSENT": "true",
            # Optionally set buckets
            # "ITV_OUTPUT_BUCKET": "your-bucket",
            # "ITV_OUTPUT_PREFIX": "outputs/",
        },
    )

    predictor = model.deploy(
        initial_instance_count=args.instance_count,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
    )
    print("Endpoint deployed:", args.endpoint_name)


if __name__ == "__main__":
    main()
