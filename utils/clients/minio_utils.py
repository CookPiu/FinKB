"""MinIO 客户端单例：存放原始文件，供前端按 source_path 打开原件。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.parse import quote

from minio import Minio
from minio.error import S3Error

from common.config.settings import get_settings

_client: Minio | None = None
_lock = threading.Lock()


def _public_read_policy(bucket: str) -> str:
    return json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{bucket}/*",
                }
            ],
        }
    )


def get_client() -> Minio:
    """返回客户端；首次调用时确保桶存在并设为公共读（原件需能在浏览器直接打开）。"""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                s = get_settings()
                client = Minio(
                    s.minio_endpoint,
                    access_key=s.minio_access_key,
                    secret_key=s.minio_secret_key,
                    secure=s.minio_secure,
                )
                if not client.bucket_exists(s.minio_bucket):
                    client.make_bucket(s.minio_bucket)
                    client.set_bucket_policy(s.minio_bucket, _public_read_policy(s.minio_bucket))
                _client = client
    return _client


def object_exists(object_name: str) -> bool:
    try:
        get_client().stat_object(get_settings().minio_bucket, object_name)
        return True
    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchObject"):
            return False
        raise


def upload_file(object_name: str, local_path: Path) -> str:
    """上传本地文件（已存在则跳过），返回公共访问 URL。"""
    if not object_exists(object_name):
        get_client().fput_object(get_settings().minio_bucket, object_name, str(local_path))
    return public_url(object_name)


def public_url(object_name: str) -> str:
    s = get_settings()
    scheme = "https" if s.minio_secure else "http"
    return f"{scheme}://{s.minio_endpoint}/{s.minio_bucket}/{quote(object_name)}"
