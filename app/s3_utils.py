from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Tuple

import boto3


@dataclass
class S3Location:
    bucket: str
    key: str


def parse_s3_uri(s3_uri: str) -> S3Location:
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI, must start with s3://")
    path = s3_uri[5:]
    bucket, _, key = path.partition("/")
    if not bucket or not key:
        raise ValueError("Invalid S3 URI, expected s3://bucket/key")
    return S3Location(bucket=bucket, key=key)


def ensure_s3_client(region_name: str | None = None):
    return boto3.client("s3", region_name=region_name)


def download_s3_to_tempfile(s3_uri: str, region_name: str | None = None) -> str:
    loc = parse_s3_uri(s3_uri)
    s3 = ensure_s3_client(region_name)
    fd, tmp_path = tempfile.mkstemp()
    os.close(fd)
    s3.download_file(loc.bucket, loc.key, tmp_path)
    return tmp_path


def upload_file_to_s3(local_path: str, bucket: str, key_prefix: str, filename: str | None = None, region_name: str | None = None) -> str:
    if filename is None:
        filename = os.path.basename(local_path) or f"{uuid.uuid4().hex}.bin"
    key = key_prefix.rstrip("/") + "/" + filename
    s3 = ensure_s3_client(region_name)
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def derive_output_key(prefix: str, extension: str = ".mp4") -> Tuple[str, str]:
    run_id = uuid.uuid4().hex
    filename = f"itv_{run_id}{extension}"
    return run_id, filename
