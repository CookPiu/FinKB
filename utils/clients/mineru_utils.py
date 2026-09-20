"""
MinerU v4 精准解析 API 客户端（批量）。
流程：申请上传地址 → PUT 上传到预签名 URL（用 POST 会 405，且不能带 Content-Type）
→ 按批次轮询 → 下载结果 zip → 解压并定位 *_content_list.json。
文档：https://mineru.net/apiManage/docs
"""
import shutil
import time
import zipfile
from pathlib import Path

import requests

from common.config.mineru_config import mineru_config
from common.logging.logger import logger

SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".png", ".jpg", ".jpeg"}
MAX_BATCH = 200


class MinerUError(RuntimeError):
    pass


def get_upload_name(file: dict) -> str:
    """
    上传名用 doc_id + 扩展名，避开中文、空格、括号等字符带来的问题
    :param file: 待解析文件 {"data_id": 业务侧标识（doc_id）, "path": Path}
    """
    return f"{file['data_id']}{file['path'].suffix.lower()}"


def build_headers() -> dict:
    if not mineru_config.api_token:
        raise MinerUError("MINERU_API_TOKEN 未配置")
    return {"Content-Type": "application/json", "Authorization": f"Bearer {mineru_config.api_token}"}


def check_response(resp, what: str) -> dict:
    """检查 HTTP 状态与业务 code，返回 data 部分"""
    if resp.status_code != 200:
        raise MinerUError(f"{what} HTTP {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    if body.get("code") != 0:
        raise MinerUError(f"{what} 失败 code={body.get('code')} msg={body.get('msg')}")
    return body.get("data") or {}


def upload_to_presigned_url(session, file: dict, url: str):
    """PUT 上传一个文件到预签名地址，最多尝试 3 次"""
    name = file["path"].name
    error = ""
    for attempt in range(3):
        try:
            with file["path"].open("rb") as fh:
                resp = session.put(url, data=fh, timeout=300)
            if resp.status_code == 200:
                logger.info(f"已上传 {name} → MinerU")
                return
            error = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            error = str(e)
        logger.warning(f"上传 {name} 失败（第 {attempt + 1} 次）：{error}")
        time.sleep(2 * (attempt + 1))
    raise MinerUError(f"上传 {name} 失败：{error}")


def submit_batch(files: list) -> str:
    """
    申请上传地址并逐个上传，上传完成后 MinerU 自动开始解析
    :param files: [{"data_id", "path"}]
    :return: batch_id
    """
    if not files:
        raise ValueError("files 为空")
    if len(files) > MAX_BATCH:
        raise ValueError(f"单批最多 {MAX_BATCH} 个文件")
    payload = {
        "files": [{"name": get_upload_name(f), "data_id": f["data_id"]} for f in files],
        "model_version": mineru_config.model_version,
        "enable_table": True,
        "enable_formula": False,
        "language": "ch",
    }
    resp = requests.post(f"{mineru_config.base_url}/file-urls/batch", headers=build_headers(), json=payload, timeout=60)
    data = check_response(resp, "申请上传地址")
    batch_id = data.get("batch_id")
    urls = data.get("file_urls") or []
    if not batch_id or len(urls) != len(files):
        raise MinerUError(f"上传地址数量不符：期望 {len(files)}，实际 {len(urls)}")

    with requests.Session() as session:
        session.trust_env = False  # 系统代理会破坏 OSS 预签名上传
        for file, url in zip(files, urls):
            upload_to_presigned_url(session, file, url)
    return batch_id


def summarize_states(states: list) -> str:
    """把各文件的状态汇总成 "done=2, running=1" 这样的一行"""
    parts = []
    for state in sorted(set(states)):
        parts.append(f"{state}={states.count(state)}")
    return ", ".join(parts)


def poll_batch(batch_id: str, data_ids: list, interval_s: float = 5.0) -> dict:
    """
    轮询直到批次内所有文件 done/failed 或超时；超时仍未完成的记为 timeout
    :return: {data_id: {"data_id", "state", "zip_url", "error"}}
    """
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    deadline = time.monotonic() + mineru_config.poll_timeout_s
    results = {}
    last_summary = ""
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, headers=build_headers(), timeout=30)
        except requests.RequestException as e:
            logger.warning(f"轮询网络异常，稍后重试：{e}")
            time.sleep(interval_s)
            continue
        if 500 <= resp.status_code < 600:
            logger.warning(f"轮询 HTTP {resp.status_code}，稍后重试")
            time.sleep(interval_s)
            continue
        data = check_response(resp, "查询解析结果")
        for item in data.get("extract_result") or []:
            data_id = item.get("data_id") or Path(item.get("file_name", "")).stem
            results[data_id] = {
                "data_id": data_id,
                "state": item.get("state", ""),
                "zip_url": item.get("full_zip_url"),
                "error": item.get("err_msg") or None,
            }
        states = []
        for data_id in data_ids:
            if data_id in results:
                states.append(results[data_id]["state"])
            else:
                states.append("unknown")
        summary = summarize_states(states)
        if summary != last_summary:
            logger.info(f"MinerU 批次 {batch_id}：{summary}")
            last_summary = summary
        if all(state in ("done", "failed") for state in states):
            return results
        time.sleep(interval_s)
    for data_id in data_ids:
        result = results.get(data_id)
        if result is None or result["state"] not in ("done", "failed"):
            results[data_id] = {
                "data_id": data_id,
                "state": "timeout",
                "zip_url": None,
                "error": f"轮询超时（{mineru_config.poll_timeout_s}s）",
            }
    return results


def download_and_extract(zip_url: str, out_dir: Path) -> Path:
    """下载结果 zip 到 out_dir/mineru.zip，解压到 out_dir/mineru/，返回 content_list.json 路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "mineru.zip"
    tmp_path = zip_path.with_suffix(".zip.part")
    with requests.Session() as session:
        session.trust_env = False
        with session.get(zip_url, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                raise MinerUError(f"下载结果 HTTP {resp.status_code}")
            with tmp_path.open("wb") as fh:
                for block in resp.iter_content(1 << 16):
                    fh.write(block)
    tmp_path.replace(zip_path)

    extract_dir = out_dir / "mineru"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    candidates = []
    for path in extract_dir.rglob("*content_list.json"):
        if "_v2" not in path.name:
            candidates.append(path)
    if not candidates:
        raise MinerUError(f"结果中没有 content_list.json：{extract_dir}")
    candidates.sort()
    return candidates[0]


def check_token() -> str:
    """连通性检查：查询一个不存在的批次，能拿到业务响应即说明网络与鉴权正常。"""
    resp = requests.get(f"{mineru_config.base_url}/extract-results/batch/finkb-healthcheck", headers=build_headers(),
                        timeout=20)
    if resp.status_code in (401, 403):
        raise MinerUError(f"鉴权失败 HTTP {resp.status_code}")
    try:
        body = resp.json()
    except ValueError:
        raise MinerUError(f"非 JSON 响应 HTTP {resp.status_code}")
    return f"HTTP {resp.status_code} code={body.get('code')} msg={body.get('msg')}"
