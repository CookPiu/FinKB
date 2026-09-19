"""MinerU v4 精准解析 API 客户端（批量）。

流程：申请上传地址 → PUT 上传到预签名 URL（用 POST 会 405，且不能带 Content-Type）
→ 按批次轮询 → 下载结果 zip → 解压并定位 *_content_list.json。
文档：https://mineru.net/apiManage/docs
"""

from __future__ import annotations

import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from common.logging.logger import logger
from common.config.settings import get_settings

SUPPORTED_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".png", ".jpg", ".jpeg"}
MAX_BATCH = 200


class MinerUError(RuntimeError):
    pass


@dataclass
class MinerUFile:
    data_id: str  # 业务侧标识，用 doc_id
    path: Path

    @property
    def upload_name(self) -> str:
        # 上传名用 doc_id + 扩展名，避开中文、空格、括号等字符带来的问题
        return f"{self.data_id}{self.path.suffix.lower()}"


@dataclass
class MinerUResult:
    data_id: str
    state: str
    zip_url: str | None = None
    error: str | None = None


def _headers() -> dict[str, str]:
    s = get_settings()
    if not s.mineru_api_token:
        raise MinerUError("MINERU_API_TOKEN 未配置")
    return {"Content-Type": "application/json", "Authorization": f"Bearer {s.mineru_api_token}"}


def _check(resp: requests.Response, what: str) -> dict:
    if resp.status_code != 200:
        raise MinerUError(f"{what} HTTP {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    if body.get("code") != 0:
        raise MinerUError(f"{what} 失败 code={body.get('code')} msg={body.get('msg')}")
    return body.get("data") or {}


def submit_batch(files: list[MinerUFile]) -> str:
    """申请上传地址并逐个上传，返回 batch_id。上传完成后 MinerU 自动开始解析。"""
    if not files:
        raise ValueError("files 为空")
    if len(files) > MAX_BATCH:
        raise ValueError(f"单批最多 {MAX_BATCH} 个文件")
    s = get_settings()
    payload = {
        "files": [{"name": f.upload_name, "data_id": f.data_id} for f in files],
        "model_version": s.mineru_model_version,
        "enable_table": True,
        "enable_formula": False,
        "language": "ch",
    }
    data = _check(
        requests.post(f"{s.mineru_base_url}/file-urls/batch", headers=_headers(), json=payload, timeout=60),
        "申请上传地址",
    )
    batch_id, urls = data.get("batch_id"), data.get("file_urls") or []
    if not batch_id or len(urls) != len(files):
        raise MinerUError(f"上传地址数量不符：期望 {len(files)}，实际 {len(urls)}")

    with requests.Session() as session:
        session.trust_env = False  # 系统代理会破坏 OSS 预签名上传
        for f, url in zip(files, urls):
            for attempt in range(3):
                try:
                    with f.path.open("rb") as fh:
                        r = session.put(url, data=fh, timeout=300)
                    if r.status_code == 200:
                        break
                    err = f"HTTP {r.status_code}"
                except requests.RequestException as e:
                    err = str(e)
                logger.warning("上传 %s 失败（第 %d 次）：%s", f.path.name, attempt + 1, err)
                time.sleep(2 * (attempt + 1))
            else:
                raise MinerUError(f"上传 {f.path.name} 失败：{err}")
            logger.info("已上传 %s → MinerU", f.path.name)
    return batch_id


def poll_batch(batch_id: str, data_ids: list[str], *, interval_s: float = 5.0) -> dict[str, MinerUResult]:
    """轮询直到批次内所有文件 done/failed 或超时；超时仍未完成的记为 timeout。"""
    s = get_settings()
    url = f"{s.mineru_base_url}/extract-results/batch/{batch_id}"
    deadline = time.monotonic() + s.mineru_poll_timeout_s
    results: dict[str, MinerUResult] = {}
    last_summary = ""
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, headers=_headers(), timeout=30)
        except requests.RequestException as e:
            logger.warning("轮询网络异常，稍后重试：%s", e)
            time.sleep(interval_s)
            continue
        if 500 <= resp.status_code < 600:
            logger.warning("轮询 HTTP %s，稍后重试", resp.status_code)
            time.sleep(interval_s)
            continue
        data = _check(resp, "查询解析结果")
        for item in data.get("extract_result") or []:
            did = item.get("data_id") or Path(item.get("file_name", "")).stem
            results[did] = MinerUResult(
                data_id=did,
                state=item.get("state", ""),
                zip_url=item.get("full_zip_url"),
                error=item.get("err_msg") or None,
            )
        states = [results[d].state if d in results else "unknown" for d in data_ids]
        summary = ", ".join(f"{st}={states.count(st)}" for st in sorted(set(states)))
        if summary != last_summary:
            logger.info("MinerU 批次 %s：%s", batch_id, summary)
            last_summary = summary
        if all(st in ("done", "failed") for st in states):
            return results
        time.sleep(interval_s)
    for d in data_ids:
        r = results.get(d)
        if r is None or r.state not in ("done", "failed"):
            results[d] = MinerUResult(data_id=d, state="timeout", error=f"轮询超时（{s.mineru_poll_timeout_s}s）")
    return results


def download_and_extract(zip_url: str, out_dir: Path) -> Path:
    """下载结果 zip 到 out_dir/mineru.zip，解压到 out_dir/mineru/，返回 content_list.json 路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "mineru.zip"
    tmp = zip_path.with_suffix(".zip.part")
    with requests.Session() as session:
        session.trust_env = False
        with session.get(zip_url, stream=True, timeout=120) as r:
            if r.status_code != 200:
                raise MinerUError(f"下载结果 HTTP {r.status_code}")
            with tmp.open("wb") as fh:
                for block in r.iter_content(1 << 16):
                    fh.write(block)
    tmp.replace(zip_path)

    extract_dir = out_dir / "mineru"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    candidates = sorted(p for p in extract_dir.rglob("*content_list.json") if "_v2" not in p.name)
    if not candidates:
        raise MinerUError(f"结果中没有 content_list.json：{extract_dir}")
    return candidates[0]


def check_token() -> str:
    """连通性检查：查询一个不存在的批次，能拿到业务响应即说明网络与鉴权正常。"""
    s = get_settings()
    resp = requests.get(f"{s.mineru_base_url}/extract-results/batch/finkb-healthcheck", headers=_headers(), timeout=20)
    if resp.status_code in (401, 403):
        raise MinerUError(f"鉴权失败 HTTP {resp.status_code}")
    try:
        body = resp.json()
    except ValueError:
        raise MinerUError(f"非 JSON 响应 HTTP {resp.status_code}")
    return f"HTTP {resp.status_code} code={body.get('code')} msg={body.get('msg')}"
