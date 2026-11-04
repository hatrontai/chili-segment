"""Local HTML demo and sequential inference API for chili phenotyping."""
import asyncio
import inspect
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.progress import ProgressJournal, json_safe as _json_safe
from src.web_service import make_config, process_image, validate_image
# Retained for callers of the original dataframe helper.
from src.web_service import extract_chili_features  # noqa: F401

BASE_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
logger = logging.getLogger(__name__)


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    detector_backend: Literal["yolov7_onnx", "threshold"] = "yolov7_onnx"
    fruit_threshold_offset: int = Field(default=16, ge=0, le=100, strict=True)
    wrinkle_tau_mm: float = Field(default=5.0, ge=0.5, le=20)
    seg_imgsz: int = Field(default=640, ge=1, strict=True)
    crop_top: int = Field(default=540, ge=0, strict=True)
    det_conf: float = Field(default=0.70, ge=0, le=1)
    det_iou: float = Field(default=0.30, ge=0, le=1)
    det_containment: float = Field(default=0.60, ge=0, le=1)


def create_app(data_dir=None, processor=None):
    data_dir = Path(data_dir) if data_dir is not None else BASE_DIR / "data"
    processor = processor or process_image
    supports_progress = "on_progress" in inspect.signature(processor).parameters
    jobs = {}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        with tempfile.TemporaryDirectory(prefix="chili_demo_") as root:
            app.state.work_dir = Path(root)
            app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chili-inference")
            try:
                yield
            finally:
                app.state.executor.shutdown(wait=True, cancel_futures=True)
                jobs.clear()

    app = FastAPI(title="Chili Pepper Phenotyping", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    def sample_path(name):
        path = (data_dir / name).resolve()
        if path.parent != data_dir.resolve() or path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            raise HTTPException(404, "Sample image not found.")
        return path

    def get_job(job_id):
        with lock:
            if job_id not in jobs:
                raise HTTPException(404, "Analysis not found. It may have expired after a server restart.")
            return dict(jobs[job_id])

    def run_job(job_id, path, config, source, display_name):
        with lock:
            jobs[job_id]["status"] = "running"
            journal = jobs[job_id]["journal"]
        try:
            directory = journal.root
            directory.mkdir()
            kwargs = {"on_progress": journal.record} if supports_progress else {}
            result = _json_safe(processor(path, config, directory, source, **kwargs))
            result["summary"]["filename"] = display_name
            for item in result["images"].values():
                item["url"] = journal.register(item["url"])
            for items in result["galleries"].values():
                for item in items:
                    item["url"] = journal.register(item["url"])
            result["downloads"] = {key: journal.register(name) for key, name in result["downloads"].items()}
            with lock:
                jobs[job_id].update(status="completed", result=result)
            journal.record({"kind": "job", "state": "completed"})
        except Exception as exc:
            logger.exception("Analysis %s failed", job_id)
            if isinstance(exc, FileNotFoundError):
                message = str(exc) if str(exc).startswith("Missing ") else "A required model or input file is missing. Check the configured weights."
            elif isinstance(exc, ValueError):
                message = str(exc)
            else:
                message = "Analysis failed. Check the server log and model configuration, then try again."
            with lock:
                jobs[job_id].update(status="failed", error=message)
            journal.fail(message)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(BASE_DIR / "static/index.html")

    @app.get("/api/samples")
    def samples():
        names = sorted(p.name for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS) if data_dir.is_dir() else []
        return {"samples": [{"name": name, "url": "/api/samples/" + quote(name, safe="")} for name in names],
                "defaults": RunOptions().model_dump(), "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)}

    @app.get("/api/samples/{name}")
    def sample(name: str):
        return FileResponse(sample_path(name))

    @app.post("/api/jobs", status_code=202)
    def submit(
        options: Annotated[str, Form()] = "{}",
        sample_name: Annotated[str | None, Form()] = None,
        upload: Annotated[UploadFile | None, File()] = None,
    ):
        directory = None
        try:
            try:
                parsed = RunOptions.model_validate_json(options)
            except ValidationError as exc:
                problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'options'}: {e['msg']}" for e in exc.errors())
                raise HTTPException(422, "Invalid configuration. " + problems) from exc
            job_id = uuid.uuid4().hex
            directory = app.state.work_dir / job_id
            directory.mkdir()
            if upload is not None and upload.filename:
                display_name = Path(upload.filename.replace("\\", "/")).name
                suffix = Path(display_name).suffix.lower()
                if suffix not in IMAGE_EXTENSIONS:
                    raise HTTPException(422, "Unsupported image format. Use JPG, PNG, BMP or TIFF.")
                # Keep the basename for provenance; the containing directory is server-generated.
                input_dir = directory / "input"
                input_dir.mkdir()
                path = input_dir / display_name
                total = 0
                with path.open("wb") as stream:
                    while chunk := upload.file.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, "Image exceeds the 50 MB upload limit.")
                        stream.write(chunk)
                source = "upload"
            elif sample_name:
                path, source = sample_path(sample_name), "data"
                display_name = path.name
            else:
                raise HTTPException(422, "Upload an image or choose a sample first.")
            try:
                validate_image(path, parsed.crop_top)
                config = make_config(parsed)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            with lock:
                jobs[job_id] = {"id": job_id, "status": "queued", "result": None, "error": None,
                                "journal": ProgressJournal(directory / "artifacts", job_id)}
            app.state.executor.submit(run_job, job_id, path, config, source, display_name)
            return {"id": job_id, "status": "queued", "status_url": f"/api/jobs/{job_id}",
                    "events_url": f"/api/jobs/{job_id}/events", "trace_url": f"/api/jobs/{job_id}/trace"}
        except Exception:
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            if upload is not None:
                upload.file.close()

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str):
        job = get_job(job_id)
        return {**{key: job[key] for key in ("id", "status", "result", "error")},
                "progress": job["journal"].snapshot()}

    @app.get("/api/jobs/{job_id}/trace")
    def trace(job_id: str):
        job = get_job(job_id)
        return {"id": job_id, "status": job["status"], "events": job["journal"].events()}

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, request: Request,
                     after: int = Query(default=0, ge=0),
                     last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None):
        journal = get_job(job_id)["journal"]
        try:
            last_seq = int(last_event_id or 0)
            if last_seq < 0:
                raise ValueError("Negative event ID")
            cursor = max(after, last_seq)
        except ValueError as exc:
            raise HTTPException(422, "Last-Event-ID must be a nonnegative integer.") from exc
        if cursor < 0:
            raise HTTPException(422, "Last-Event-ID must be a nonnegative integer.")

        async def stream():
            nonlocal cursor
            yield "retry: 1200\n\n"
            heartbeat = asyncio.get_running_loop().time()
            while not await request.is_disconnected():
                batch = journal.events(cursor)
                for event in batch:
                    cursor = event["seq"]
                    yield f"id: {cursor}\nevent: progress\ndata: {json.dumps(event, allow_nan=False)}\n\n"
                all_events = journal.events()
                # Wait for the terminal journal record, not merely the job flag:
                # failed-node and skipped-node events must precede stream closure.
                if all_events and all_events[-1].get("kind") == "job" and cursor >= len(all_events):
                    return
                now = asyncio.get_running_loop().time()
                if now - heartbeat >= 15:
                    yield ": keep-alive\n\n"
                    heartbeat = now
                await asyncio.sleep(.15)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}/files/{filename:path}")
    def artifact(job_id: str, filename: str):
        job = get_job(job_id)
        root = app.state.work_dir / job_id / "artifacts"
        path = (root / filename).resolve()
        if not job["journal"].is_published(filename) or not path.is_relative_to(root.resolve()) or not path.is_file():
            raise HTTPException(404, "Result file not found.")
        download = path.suffix in {".zip", ".csv", ".json"}
        return FileResponse(path, filename=path.name if download else None)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,
        host=os.getenv("CHILI_SERVER_NAME", os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")),
        port=int(os.getenv("CHILI_SERVER_PORT", os.getenv("GRADIO_SERVER_PORT", "7860"))),
    )
