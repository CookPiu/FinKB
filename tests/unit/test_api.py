import json

import pytest
from fastapi.testclient import TestClient

from api import file_import_service
from api import query_service
from utils.sse_utils import sse_pack


def parse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


class FakeGraph:
    """替身查询图：按给定顺序产出 (mode, chunk)，可选在最后抛出异常"""

    def __init__(self, items=None, error=None):
        self.items = items or []
        self.error = error
        self.state = None

    def stream(self, state, stream_mode):
        self.state = state
        for item in self.items:
            yield item
        if self.error:
            raise self.error


def test_sse_pack_format():
    assert sse_pack("delta", {"text": "你好"}) == 'event: delta\ndata: {"text": "你好"}\n\n'


def test_query_streams_delta_sources_and_final(monkeypatch):
    fake = FakeGraph(
        [
            ("custom", {"type": "delta", "text": "茅台营收"}),
            ("custom", {"type": "sources", "sources": [{"eid": 1}], "text": "[E1] …"}),
            ("custom", {"type": "final", "kind": "answer"}),  # 节点内部的 final 不直接透传
            ("values", {"kind": "answer", "answer": "茅台营收53,909,252,220.51元[E1]", "sources": [{"eid": 1}]}),
        ]
    )
    monkeypatch.setattr(query_service, "query_app", fake)
    resp = TestClient(query_service.app).post("/query", json={"question": " 茅台营收多少 ", "session_id": "s1"})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    events = parse_events(resp.text)
    assert [e for e, _ in events] == ["delta", "sources", "final"]
    assert events[-1][1] == {"session_id": "s1", "kind": "answer", "answer": "茅台营收53,909,252,220.51元[E1]",
                             "sources": [{"eid": 1}]}
    assert fake.state["original_query"] == "茅台营收多少"


def test_query_error_becomes_friendly_event(monkeypatch):
    class PermissionDeniedError(Exception):
        pass

    monkeypatch.setattr(query_service, "query_app", FakeGraph([("custom", {"type": "delta", "text": "部分"})],
                                                              error=PermissionDeniedError("403 Arrearage")))
    events = parse_events(TestClient(query_service.app).post("/query", json={"question": "q"}).text)
    assert [e for e, _ in events] == ["delta", "error"]
    assert "大模型服务暂时不可用" in events[-1][1]["message"] and "403" not in events[-1][1]["message"]
    assert events[-1][1]["session_id"]  # 新会话也会回传会话 ID


@pytest.mark.parametrize("question", ["", "   ", "问" * 501])
def test_query_rejects_invalid_question(question):
    assert TestClient(query_service.app).post("/query", json={"question": question}).status_code == 400


def test_upload_accepts_supported_and_rejects_others(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(file_import_service, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(file_import_service, "run_import", lambda path: started.append(path.name))
    files = [("files", ("a.pdf", b"%PDF-1.4", "application/pdf")), ("files", ("b.txt", b"x", "text/plain"))]
    resp = TestClient(file_import_service.app).post("/documents", files=files)
    assert resp.status_code == 200
    result = {f["file_name"]: f["accepted"] for f in resp.json()["files"]}
    assert result == {"a.pdf": True, "b.txt": False}
    assert started == ["a.pdf"] and (tmp_path / "a.pdf").read_bytes() == b"%PDF-1.4"
