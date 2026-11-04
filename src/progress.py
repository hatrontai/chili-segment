"""Optional pipeline observations and an immutable, per-job web event journal."""
from copy import deepcopy
import math
from pathlib import Path
import threading
import time
from urllib.parse import quote

NODES = ("input", "models", "scale", "detection", "crop", "segmentation", "masks", "traits", "results", "export")
FRUIT_NODES = ("crop", "segmentation", "masks", "traits")


def notify(callback, node, state, **context):
    if callback is not None:
        callback({"node": node, "state": state, **context})


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ProgressJournal:
    """A worker appends; status, SSE and file readers take consistent snapshots."""

    def __init__(self, root, job_id):
        self.root = Path(root).resolve()
        self.prefix = f"/api/jobs/{job_id}/files/"
        self.started = time.perf_counter()
        self._lock = threading.Lock()
        self._events = []
        self._files = set()

    def url(self, filename):
        return self.prefix + quote(filename, safe="/")

    def register(self, filename):
        path = (self.root / filename).resolve()
        if not path.is_relative_to(self.root) or not path.is_file() or ".part" in path.suffixes:
            raise ValueError("Result artifact is not ready or is outside the analysis directory.")
        with self._lock:
            self._files.add(filename)
        return self.url(filename)

    def is_published(self, filename):
        with self._lock:
            return filename in self._files

    def publish_outputs(self, outputs):
        result = deepcopy(json_safe(outputs))
        for item in result.get("images", []):
            item["url"] = self.register(item["url"])
        result["downloads"] = {name: self.register(path) for name, path in result.get("downloads", {}).items()}
        return result

    def record(self, event):
        item = deepcopy(json_safe(event))
        if "outputs" in item:
            item["outputs"] = self.publish_outputs(item["outputs"])
        with self._lock:
            item.update(seq=len(self._events) + 1, elapsed_seconds=time.perf_counter() - self.started)
            self._events.append(item)
        return deepcopy(item)

    def events(self, after=0):
        with self._lock:
            return deepcopy(self._events[after:])

    def snapshot(self):
        events = self.events()
        states = {}
        candidates = {}
        for event in events:
            node = event.get("node")
            candidate = event.get("candidate_index")
            if node:
                states[f"{candidate if candidate is not None else 'global'}:{node}"] = event
            if event.get("kind") == "candidate":
                candidates[candidate] = event
        return {"last_seq": len(events), "nodes": list(states.values()),
                "candidates": list(candidates.values())}

    def fail(self, message):
        events = self.events()
        states = {}
        for event in events:
            if event.get("node"):
                states[(event.get("candidate_index"), event["node"])] = event
        active = next((e for e in reversed(list(states.values())) if e["state"] == "running"), None)
        node = active["node"] if active else "input"
        context = {key: active[key] for key in ("candidate_index", "candidate_total") if key in active} if active else {}
        self.record({"node": node, "state": "failed", "message": message, **context})
        for next_node in NODES[NODES.index(node) + 1:]:
            scope = context if next_node in FRUIT_NODES else {}
            key = (scope.get("candidate_index"), next_node)
            if key not in states or states[key]["state"] == "pending":
                self.record({"node": next_node, "state": "skipped", "message": "Blocked by an earlier failure.", **scope})
        if "candidate_index" in context:
            self.record({"kind": "candidate", "state": "failed", "message": message, **context})
        self.record({"kind": "job", "state": "failed", "message": message})
