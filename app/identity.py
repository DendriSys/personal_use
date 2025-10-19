from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

from .config import settings
from .s3_utils import ensure_s3_client, parse_s3_uri, upload_file_to_s3


@dataclass
class Identity:
    alias: str
    image_s3_uri: str


class IdentityRegistry:
    def __init__(self):
        self._aliases: Dict[str, Identity] = {}
        self._loaded = False

    def _load_from_settings(self) -> None:
        if self._loaded:
            return
        # Inline JSON
        if settings.identities_json:
            try:
                data = json.loads(settings.identities_json)
                for alias, obj in data.items():
                    if isinstance(obj, dict) and "image_s3_uri" in obj:
                        self._aliases[alias.lower()] = Identity(alias=alias, image_s3_uri=obj["image_s3_uri"])
            except Exception:
                pass
        # From S3 JSON
        if settings.identities_s3_uri:
            try:
                loc = parse_s3_uri(settings.identities_s3_uri)
                s3 = ensure_s3_client(settings.aws_region)
                body = s3.get_object(Bucket=loc.bucket, Key=loc.key)["Body"].read()
                data = json.loads(body)
                for alias, obj in data.items():
                    if isinstance(obj, dict) and "image_s3_uri" in obj:
                        self._aliases[alias.lower()] = Identity(alias=alias, image_s3_uri=obj["image_s3_uri"])
            except Exception:
                pass
        self._loaded = True

    def get(self, alias: str) -> Optional[Identity]:
        self._load_from_settings()
        return self._aliases.get(alias.lower())

    def register(self, alias: str, image_s3_uri: str) -> Identity:
        if not settings.allow_identity_registration:
            raise PermissionError("Registration disabled")
        ident = Identity(alias=alias, image_s3_uri=image_s3_uri)
        self._aliases[alias.lower()] = ident
        return ident


identity_registry = IdentityRegistry()
