import json
from urllib.parse import quote

from minio import Minio
from minio.error import S3Error

from common.config.minio_config import minio_config

# 全局 MinIO 客户端单例
_minio_client = None


def get_minio_client():
    """
    获取 MinIO 客户端单例；首次调用时确保桶存在并设为公共读（原件要能在浏览器里直接打开）
    """
    global _minio_client
    if _minio_client is None:
        client = Minio(
            minio_config.endpoint,
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.secure,
        )
        if not client.bucket_exists(minio_config.bucket):
            client.make_bucket(minio_config.bucket)
            client.set_bucket_policy(minio_config.bucket, _public_read_policy(minio_config.bucket))
        _minio_client = client
    return _minio_client


def _public_read_policy(bucket: str) -> str:
    """桶策略：任何人都可以读取对象（不能列目录、不能写）"""
    policy = {
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
    return json.dumps(policy)


def object_exists(object_name: str) -> bool:
    try:
        get_minio_client().stat_object(minio_config.bucket, object_name)
        return True
    except S3Error as e:
        if e.code in ("NoSuchKey", "NoSuchObject"):
            return False
        raise


def upload_file(object_name: str, local_path) -> str:
    """
    上传本地文件（对象已存在则跳过）
    :param object_name: 桶内路径，如 originals/<doc_id>/<文件名>
    :param local_path: 本地文件路径
    :return: 公共访问 URL
    """
    if not object_exists(object_name):
        get_minio_client().fput_object(minio_config.bucket, object_name, str(local_path))
    return get_public_url(object_name)


def get_public_url(object_name: str) -> str:
    scheme = "https" if minio_config.secure else "http"
    # 文件名含中文、空格，需要 URL 编码
    return f"{scheme}://{minio_config.endpoint}/{minio_config.bucket}/{quote(object_name)}"
