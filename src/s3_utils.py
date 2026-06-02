"""
s3_utils.py
-----------
Shared S3 helpers used across the OpenScope ophys QC pipeline.

All other modules import from here rather than defining their own copies.
"""

from __future__ import annotations

import io
import json

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def s3_client():
    """Return an anonymous (public-bucket) boto3 S3 client."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """
    Split an s3:// URL into (bucket, key).

    Parameters
    ----------
    s3_path : e.g. "s3://aind-open-data/multiplane-ophys_837568_..."

    Returns
    -------
    (bucket, key)  — key has no trailing slash
    """
    without = s3_path[len("s3://"):]
    bucket, _, key = without.partition("/")
    return bucket, key.rstrip("/")


def parse_session_id(s3_path: str) -> str:
    """
    Extract the session identifier from an asset path.

    e.g. "s3://aind-open-data/multiplane-ophys_837568_2026-03-05_14-14-51_processed_..."
    → "multiplane-ophys_837568_2026-03-05_14-14-51"
    """
    name = s3_path.rstrip("/").split("/")[-1]
    return name.split("_processed_")[0]


def parse_subject_id(s3_path: str) -> str:
    """
    Extract the numeric subject identifier from an asset path.

    e.g. "multiplane-ophys_837568_..." → "837568"
    """
    return parse_session_id(s3_path).split("_")[1]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_csv(bucket: str, key: str) -> pd.DataFrame | None:
    """
    Read a CSV from S3 into a DataFrame.

    Returns None on any error (missing key, network issue, parse failure).
    """
    client = s3_client()
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except Exception:
        return None


def read_json(bucket: str, key: str) -> dict | None:
    """
    Read a JSON file from S3 into a dict.

    Returns None on any error.
    """
    client = s3_client()
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------

PLANE_PATTERN_PREFIXES = ("VISl_", "VISp_")


def list_plane_names(bucket: str, session_key: str) -> list[str]:
    """
    List imaging-plane subdirectories present for a session on S3.

    Plane folders match the pattern VISl_N or VISp_N.

    Parameters
    ----------
    bucket      : S3 bucket name
    session_key : key prefix for the session (no trailing slash)

    Returns
    -------
    Sorted list of plane names, e.g. ["VISl_4", "VISl_5", ..., "VISp_3"]
    """
    client    = s3_client()
    prefix    = session_key + "/"
    paginator = client.get_paginator("list_objects_v2")
    planes    = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            folder = cp["Prefix"].rstrip("/").split("/")[-1]
            if any(folder.startswith(p) for p in PLANE_PATTERN_PREFIXES):
                planes.add(folder)

    return sorted(planes)
