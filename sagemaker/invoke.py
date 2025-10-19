from __future__ import annotations

import argparse
import json

import boto3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--image-s3-uri", required=True)
    p.add_argument("--consent", action="store_true")
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--motion-bucket-id", type=int, default=None)
    p.add_argument("--noise-aug-strength", type=float, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--s3-output-bucket", default=None)
    p.add_argument("--s3-output-prefix", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    payload = {
        "image_s3_uri": args.image_s3_uri,
        "consent": bool(args.consent),
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "motion_bucket_id": args.motion_bucket_id,
        "noise_aug_strength": args.noise_aug_strength,
        "fps": args.fps,
        "s3_output_bucket": args.s3_output_bucket,
        "s3_output_prefix": args.s3_output_prefix,
    }

    # Remove None values for cleanliness
    payload = {k: v for k, v in payload.items() if v is not None}

    runtime = boto3.client("sagemaker-runtime")
    response = runtime.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = response["Body"].read().decode("utf-8")
    print(body)


if __name__ == "__main__":
    main()
