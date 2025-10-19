from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import boto3


@dataclass
class ModerationResult:
    flagged: bool
    reasons: List[str]


def analyze_with_rekognition(image_bytes: bytes, min_confidence: int = 80) -> ModerationResult:
    client = boto3.client("rekognition")
    response = client.detect_moderation_labels(Image={"Bytes": image_bytes}, MinConfidence=min_confidence)
    labels = response.get("ModerationLabels", [])
    reasons = [f"{l['Name']}:{int(l.get('Confidence', 0))}" for l in labels]
    flagged = len(labels) > 0
    return ModerationResult(flagged=flagged, reasons=reasons)
