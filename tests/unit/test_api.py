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
    monkeypatch.setattr(file_import_service, "run_import", lambda task_id, path: started.append((task_id, path.name)))
    files = [("files", ("a.pdf", b"%PDF-1.4", "application/pdf")), ("files", ("b.txt", b"x", "text/plain"))]
    resp = TestClient(file_import_service.app).post("/documents", files=files)
    assert resp.status_code == 200
    result = {f["file_name"]: f["accepted"] for f in resp.json()["files"]}
    assert result == {"a.pdf": True, "b.txt": False}
    task_id = resp.json()["files"][0]["task_id"]
    assert started == [(task_id, "a.pdf")] and (tmp_path / "a.pdf").read_bytes() == b"%PDF-1.4"
    # 后台任务被替身接管，任务停在排队状态
    task = next(t for t in file_import_service.list_tasks() if t["task_id"] == task_id)
    assert task["status"] == "queued" and task["done_nodes"] == []


def test_run_import_records_node_progress_and_failure(monkeypatch, tmp_path):
    def fake_import_file(path, root=None, on_node=None):
        doc = {"doc_id": "d1", "chunk_count": 12}
        on_node("node_entry", {"action": "new", "doc": doc})
        on_node("node_parse", {})
        if path.name == "bad.pdf":
            raise RuntimeError("MinerU 解析失败")
        on_node("node_chunk", {})
        on_node("node_index", {})
        return {"action": "new", "doc": doc}

    monkeypatch.setattr(file_import_service, "import_file", fake_import_file)
    ok = file_import_service.new_task("good.pdf")
    bad = file_import_service.new_task("bad.pdf")
    file_import_service.run_import(ok["task_id"], tmp_path / "good.pdf")
    file_import_service.run_import(bad["task_id"], tmp_path / "bad.pdf")
    tasks = {t["task_id"]: t for t in TestClient(query_service.app).get("/import/tasks").json()["tasks"]}
    assert tasks[ok["task_id"]]["status"] == "done" and tasks[ok["task_id"]]["chunk_count"] == 12
    assert tasks[ok["task_id"]]["done_nodes"] == ["node_entry", "node_parse", "node_chunk", "node_index"]
    assert set(tasks[ok["task_id"]]["node_seconds"]) == set(tasks[ok["task_id"]]["done_nodes"])
    assert tasks[ok["task_id"]]["action"] == "new" and tasks[ok["task_id"]]["doc_id"] == "d1"
    assert tasks[bad["task_id"]]["status"] == "failed" and "MinerU" in tasks[bad["task_id"]]["error"]
    assert tasks[bad["task_id"]]["done_nodes"] == ["node_entry", "node_parse"]


def test_import_file_reports_each_node(monkeypatch, tmp_path):
    from processor.import_processor import main_graph

    class FakeImportGraph:
        def stream(self, state, stream_mode):
            yield "values", state
            yield "updates", {"node_entry": {"action": "skip"}}
            yield "values", {**state, "action": "skip"}

    monkeypatch.setattr(main_graph, "kb_import_app", FakeImportGraph())
    seen = []
    final = main_graph.import_file(tmp_path / "a.pdf", on_node=lambda name, node_state: seen.append((name, node_state)))
    assert seen == [("node_entry", {"action": "skip"})]
    assert final["action"] == "skip" and final["local_file_path"].endswith("a.pdf")
