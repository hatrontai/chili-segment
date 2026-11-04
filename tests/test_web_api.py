"""Exercise the HTML API boundary without running model inference."""
import io
import json
from pathlib import Path
import threading
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
from PIL import Image

from app import create_app
from src.pipeline import feature_columns


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (16, 24), "red").save(stream, "PNG")
    return stream.getvalue()


def fake_processor(path, config, output_dir, source):
    (output_dir / "images").mkdir()
    (output_dir / "images/result.png").write_bytes(image_bytes())
    (output_dir / "output.csv").write_text(",".join(feature_columns()) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps({"configuration": config.effective_settings()}))
    with zipfile.ZipFile(output_dir / "chili_results.zip", "w") as archive:
        for name in ("output.csv", "manifest.json"):
            archive.write(output_dir / name, name)
    picture = {"url": "images/result.png", "caption": "A specimen"}
    return {"summary": {"filename": path.name, "source": source, "fruit_count": 0,
                        "configuration": config.effective_settings()},
            "images": {"original": picture}, "galleries": {"crops": []},
            "features": {"columns": feature_columns(), "rows": []},
            "warnings": ["No fruit detected."], "optional": float('nan'),
            "downloads": {"zip": "chili_results.zip", "csv": "output.csv", "manifest": "manifest.json"}}


class WebAPITests(unittest.TestCase):
    def setUp(self):
        self.root = TemporaryDirectory()
        self.data = Path(self.root.name)
        (self.data / "sample.png").write_bytes(image_bytes())
        (self.data / "notes.txt").write_text("private")
        self.calls = []
        def processor(*args):
            self.calls.append(args)
            return fake_processor(*args)
        self.app = create_app(data_dir=self.data, processor=processor)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        work_dir = self.app.state.work_dir
        self.client_context.__exit__(None, None, None)
        self.assertFalse(work_dir.exists())
        self.root.cleanup()

    def submit(self, **kwargs):
        options = kwargs.pop("options", {"crop_top": 0})
        data = {"options": json.dumps(options), "sample_name": "sample.png"}
        data.update(kwargs.pop("data", {}))
        return self.client.post("/api/jobs", data=data, **kwargs)

    def wait(self, response):
        self.assertEqual(response.status_code, 202, response.text)
        url = response.json()["status_url"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            data = self.client.get(url).json()
            if data["status"] in {"completed", "failed"}:
                return data
            time.sleep(.01)
        self.fail("Job did not complete")

    def test_page_and_samples(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn('lang="en"', page.text)
        for asset in ("app.js", "styles.css"):
            self.assertEqual(self.client.get("/static/" + asset).status_code, 200)
        result = self.client.get("/api/samples").json()
        self.assertEqual([item["name"] for item in result["samples"]], ["sample.png"])
        self.assertEqual(result["defaults"]["crop_top"], 540)
        self.assertEqual(self.client.get(result["samples"][0]["url"]).content, image_bytes())

    def test_sample_empty_result_and_downloads(self):
        job = self.wait(self.submit())
        self.assertEqual(job["status"], "completed")
        result = job["result"]
        self.assertEqual(len(result["features"]["columns"]), 34)
        self.assertEqual(result["features"]["rows"], [])
        self.assertIsNone(result["optional"])
        self.assertEqual(self.calls[0][3], "data")
        self.assertEqual(self.client.get(result["images"]["original"]["url"]).content, image_bytes())
        response = self.client.get(result["downloads"]["zip"])
        self.assertIn("attachment", response.headers["content-disposition"])
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            self.assertEqual(set(bundle.namelist()), {"output.csv", "manifest.json"})
        for name in ("csv", "manifest"):
            self.assertEqual(self.client.get(result["downloads"][name]).status_code, 200)

    def test_upload_wins_over_even_invalid_sample(self):
        result = self.wait(self.submit(data={"sample_name": "../../private.png"},
            files={"upload": ("../input.png", image_bytes(), "image/png")}))
        self.assertEqual(result["status"], "completed")
        path, config, out, source = self.calls[0]
        self.assertEqual(source, "upload")
        self.assertEqual(path.name, "input.png")
        self.assertTrue(path.is_relative_to(self.app.state.work_dir))
        self.assertEqual(result["result"]["summary"]["filename"], "input.png")

    def test_invalid_config_is_rejected_before_inference(self):
        for changes in ({"crop_top": -1}, {"seg_imgsz": 2.5}, {"seg_imgsz": True},
                        {"det_conf": 1.5}, {"det_iou": float("nan")},
                        {"wrinkle_tau_mm": float("inf")}, {"detector_backend": "other"},
                        {"unknown": 12}):
            with self.subTest(changes=changes):
                self.assertEqual(self.submit(options={"crop_top": 0, **changes}).status_code, 422)
        self.assertEqual(self.client.post("/api/jobs", data={"options": "invalid"}).status_code, 422)
        self.assertFalse(self.calls)
        self.assertFalse(list(self.app.state.work_dir.iterdir()))

    def test_invalid_inputs_and_crop(self):
        self.assertEqual(self.submit(data={"sample_name": ""}).status_code, 422)
        self.assertEqual(self.submit(data={"sample_name": "../private.png"}).status_code, 404)
        self.assertEqual(self.submit(options={"crop_top": 24}).status_code, 422)
        for filename, content in (("bad.png", b"bad image"), ("file.txt", image_bytes())):
            with self.subTest(filename=filename):
                self.assertEqual(self.submit(files={"upload": (filename, content)}).status_code, 422)
        with patch("app.MAX_UPLOAD_BYTES", 8):
            self.assertEqual(self.submit(files={"upload": ("large.png", image_bytes())}).status_code, 413)
        self.assertFalse(self.calls)
        self.assertFalse(list(self.app.state.work_dir.iterdir()))

    def test_paths_and_artifacts_are_not_public_filesystem(self):
        job = self.wait(self.submit())
        prefix = f"/api/jobs/{job['id']}/files/"
        for name in ("manifest.json.bak", "../input/sample.png", "%2e%2e%2finput%2fsample.png", "/etc/passwd"):
            self.assertEqual(self.client.get(prefix + name).status_code, 404)
        for path in ("/api/jobs/unknown", "/api/jobs/unknown/files/output.csv", "/api/samples/notes.txt", "/api/samples/..%2Fnotes.txt"):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_model_failure_is_reported_and_worker_remains_usable(self):
        with patch("app.make_config", side_effect=ValueError("Invalid model configuration")):
            self.assertEqual(self.submit().status_code, 422)
        self.calls.clear()
        # A separate app injects a failing worker, as a missing checkpoint would.
        def fail(*args):
            raise FileNotFoundError("Missing segmentation weights (model.pt).")
        with TestClient(create_app(self.data, fail)) as client:
            response = client.post("/api/jobs", data={"sample_name": "sample.png", "options": '{"crop_top":0}'})
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                result = client.get(response.json()["status_url"]).json()
                if result["status"] == "failed":
                    break
                time.sleep(.01)
            self.assertEqual(result["status"], "failed")
            self.assertIn("Missing segmentation weights", result["error"])
        self.assertEqual(self.wait(self.submit())["status"], "completed")

    def test_worker_serializes_inference(self):
        entered = threading.Event()
        release = threading.Event()
        sequence = []
        def blocking_processor(*args):
            sequence.append("start")
            entered.set()
            release.wait(5)
            result = fake_processor(*args)
            sequence.append("end")
            return result
        with TestClient(create_app(self.data, blocking_processor)) as client:
            first = client.post("/api/jobs", data={"sample_name": "sample.png", "options": '{"crop_top":0}'}).json()
            try:
                self.assertTrue(entered.wait(3))
                second = client.post("/api/jobs", data={"sample_name": "sample.png", "options": '{"crop_top":0}'}).json()
                self.assertEqual(client.get(first["status_url"]).json()["status"], "running")
                self.assertEqual(client.get(second["status_url"]).json()["status"], "queued")
                self.assertEqual(sequence, ["start"])
            finally:
                release.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if client.get(second["status_url"]).json()["status"] == "completed":
                    break
                time.sleep(.01)
            self.assertEqual(client.get(second["status_url"]).json()["status"], "completed")
        self.assertEqual(sequence, ["start", "end", "start", "end"])


if __name__ == "__main__":
    unittest.main()
