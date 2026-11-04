"""Trace observation must preserve the scientific pipeline and file boundaries."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app import create_app
from src.object import Fruits, InvalidChiliCandidate
from src.pipeline import PipelineConfig, run_image
from src.progress import FRUIT_NODES, ProgressJournal
from src.web_service import _get_models, process_image


class JournalTests(unittest.TestCase):
    def test_published_outputs_are_immutable_and_confined(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ready.png").write_bytes(b"ready")
            (root / "incomplete.part.png").write_bytes(b"partial")
            journal = ProgressJournal(root, "test")
            output = {"images": [{"url": "ready.png", "caption": "ready"}], "details": {"value": float('nan')}}
            returned = journal.record({"node": "input", "state": "completed", "outputs": output})
            returned["state"] = "changed"
            output["images"][0]["caption"] = "changed"
            event = journal.events()[0]
            self.assertEqual(event["outputs"]["images"][0]["caption"], "ready")
            self.assertIsNone(event["outputs"]["details"]["value"])
            self.assertTrue(journal.is_published("ready.png"))
            self.assertFalse(journal.is_published("incomplete.part.png"))
            for path in ("incomplete.part.png", "not-written.png", "../outside.png", "/etc/passwd"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    journal.register(path)
            event["state"] = "changed"
            self.assertEqual(journal.events()[0]["state"], "completed")

    def test_failure_records_active_scope_and_keeps_completed_outputs(self):
        with TemporaryDirectory() as tmp:
            journal = ProgressJournal(tmp, "test")
            journal.record({"node": "input", "state": "completed", "outputs": {"details": {"width": 20}}})
            journal.record({"node": "traits", "state": "running", "candidate_index": 2, "candidate_total": 4})
            journal.fail("Could not measure specimen.")
            events = journal.events()
            self.assertEqual(events[2]["node"], "traits")
            self.assertEqual(events[2]["state"], "failed")
            self.assertEqual(events[2]["candidate_index"], 2)
            self.assertEqual(events[-1], {**events[-1], "kind": "job", "state": "failed"})
            self.assertEqual(events[0]["outputs"]["details"], {"width": 20})
            self.assertEqual([e["seq"] for e in events], list(range(1, len(events)+1)))
            self.assertEqual(journal.events(2), events[2:])
            self.assertEqual(journal.snapshot()["last_seq"], len(events))
            self.assertEqual(journal.snapshot()["candidates"][0]["state"], "failed")


class PipelineObservationTests(unittest.TestCase):
    def test_candidate_identity_survives_rejection_and_detection_runs_once(self):
        detector = SimpleNamespace(backend="threshold", predict=lambda path: [[30, 0, 5, 5], [10, 0, 5, 5], [20, 0, 5, 5]])
        calls, events = [], []
        def candidate(path, index, segmenter, box, ratio, on_progress=None, **kwargs):
            calls.append((index, box[0]))
            for node in ("crop", "segmentation", "masks", "traits"):
                on_progress({"node": node, "state": "running"})
                if box[0] == 20 and node == "segmentation":
                    raise InvalidChiliCandidate("empty_measurement_mask")
                on_progress({"node": node, "state": "completed"})
            return SimpleNamespace(id=index)
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "input.png"
            Image.new('RGB', (50, 50)).save(image)
            with patch.object(Fruits, 'qr_code', return_value=None), patch('src.object.Chili', side_effect=candidate):
                fruits = Fruits(str(image), detector, object(), on_progress=events.append)
                fruits.image.close()
        self.assertEqual(calls, [(0, 10), (1, 20), (1, 30)])
        terminal = [e for e in events if e.get('kind') == 'candidate']
        self.assertEqual([e['candidate_index'] for e in terminal], [0, 1, 2])
        self.assertEqual([e['state'] for e in terminal], ['completed', 'rejected', 'completed'])
        self.assertEqual([e.get('fruit_id') for e in terminal], [0, None, 1])
        self.assertEqual(len([e for e in events if e.get('node') == 'detection' and e['state'] == 'completed']), 1)
        self.assertEqual([e['node'] for e in events if e.get('candidate_index') == 1 and e['state'] == 'skipped'], ['segmentation', 'masks', 'traits'])

    def test_run_image_optional_callback_keeps_original_call_compatible(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / 'input.png'
            Image.new('RGB', (10, 10)).save(image)
            config = PipelineConfig(detector_backend='threshold', det_crop_top=0)
            models = (object(), SimpleNamespace(imgsz=640, device='cpu'))
            with patch('src.object.Fruits', return_value='result') as constructor:
                self.assertEqual(run_image(image, config, models), 'result')
                self.assertNotIn('on_progress', constructor.call_args.kwargs)
                observer = lambda event: None
                self.assertEqual(run_image(image, config, models, on_progress=observer), 'result')
                self.assertIs(constructor.call_args.kwargs['on_progress'], observer)

    def test_zero_candidates_and_cached_models_still_export(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, weight = root / 'input.png', root / 'model.pt'
            Image.new('RGB', (40, 50), 'white').save(image)
            weight.write_bytes(b'fake-checkpoint')
            config = PipelineConfig(detector_backend='threshold', weight_seg=str(weight), det_crop_top=0)
            detector = SimpleNamespace(backend='threshold', predict=lambda path: [])
            segmenter = SimpleNamespace(imgsz=640, device='cpu')
            _get_models.cache_clear()
            try:
                with patch('src.web_service.build_models', return_value=(detector, segmenter)) as build:
                    traces = []
                    for i in range(2):
                        events = []
                        result = process_image(image, config, root / f'run-{i}', 'upload', on_progress=events.append)
                        traces.append(events)
                        self.assertEqual(result['features']['rows'], [])
                        self.assertEqual(len(result['features']['columns']), 34)
                        self.assertEqual([e['node'] for e in events if e['state'] == 'skipped'], list(FRUIT_NODES))
                        self.assertTrue((root / f'run-{i}' / 'chili_results.zip').is_file())
                    build.assert_called_once()
                cached = [next(e for e in events if e['node'] == 'models' and e['state'] == 'completed')['outputs']['details']['cache_hit'] for events in traces]
                self.assertEqual(cached, [False, True])
            finally:
                _get_models.cache_clear()


class ProgressAPITests(unittest.TestCase):
    def test_partial_artifacts_survive_failure_and_sse_resumes_by_id(self):
        with TemporaryDirectory() as tmp:
            data = Path(tmp)
            stream = io.BytesIO(); Image.new('RGB', (30, 40)).save(stream, 'PNG')
            (data / 'input.png').write_bytes(stream.getvalue())
            ready, release = threading.Event(), threading.Event()
            def processor(path, config, directory, source, on_progress=None):
                on_progress({'node': 'input', 'state': 'running'})
                (directory / 'ready.png').write_bytes(stream.getvalue())
                (directory / 'not-published.png').write_bytes(stream.getvalue())
                (directory / 'unfinished.part.png').write_bytes(b'partial')
                on_progress({'node': 'input', 'state': 'completed', 'outputs': {'images': [{'url': 'ready.png', 'caption': 'early image'}]}})
                on_progress({'node': 'models', 'state': 'running'})
                ready.set(); release.wait(10)
                raise RuntimeError('simulated inference failure')
            with TestClient(create_app(data, processor)) as client:
                submitted = client.post('/api/jobs', data={'sample_name':'input.png', 'options':'{"crop_top":0}'}).json()
                try:
                    self.assertTrue(ready.wait(5))
                    status = client.get(submitted['status_url']).json()
                    self.assertEqual(status['status'], 'running')
                    self.assertEqual(status['progress']['last_seq'], 3)
                    trace = client.get(submitted['trace_url']).json()
                    url = trace['events'][1]['outputs']['images'][0]['url']
                    self.assertEqual(client.get(url).content, stream.getvalue())
                    prefix = url.rsplit('/', 1)[0]
                    for name in ('not-published.png','unfinished.part.png','../input/input.png'):
                        self.assertEqual(client.get(prefix+'/'+name).status_code, 404)
                finally:
                    release.set()
                deadline = time.monotonic()+5
                while time.monotonic()<deadline:
                    trace = client.get(submitted['trace_url']).json()
                    if trace['events'][-1].get('kind') == 'job':
                        break
                    time.sleep(.01)
                self.assertEqual(client.get(submitted['status_url']).json()['status'], 'failed')
                self.assertEqual(client.get(url).status_code, 200)
                self.assertEqual(trace['events'][3]['node'], 'models')
                self.assertEqual(trace['events'][3]['state'], 'failed')
                sse = client.get(submitted['events_url'], headers={'Last-Event-ID':'2'})
                self.assertTrue(sse.headers['content-type'].startswith('text/event-stream'))
                records = [json.loads(line[6:]) for line in sse.text.splitlines() if line.startswith('data: ')]
                self.assertEqual([e['seq'] for e in records], list(range(3,len(trace['events'])+1)))
                self.assertEqual(records[-1]['kind'], 'job')
                self.assertNotIn('data: ', client.get(submitted['events_url'], params={'after':len(trace['events'])}).text)
                for bad in ('-1','nope'):
                    self.assertEqual(client.get(submitted['events_url'], headers={'Last-Event-ID':bad}).status_code, 422)
                self.assertEqual(client.get('/api/jobs/unknown/trace').status_code, 404)
                self.assertEqual(client.get('/api/jobs/unknown/events').status_code, 404)

    @unittest.skipUnless(shutil.which('node'), 'Node is optional and only needed for JS projection tests')
    def test_live_and_replay_project_the_same_prefix_without_future_outputs(self):
        path = Path(__file__).resolve().parents[1] / 'static/graph-state.js'
        script = r'''
const assert=require('node:assert/strict');
const {project,nodeState,interval}=require(process.argv[1]);
const events=[
 {seq:1,elapsed_seconds:0,node:'input',state:'running'},
 {seq:2,elapsed_seconds:.01,node:'input',state:'completed',outputs:{images:[{url:'input.png'}]}},
 {seq:3,elapsed_seconds:1,node:'detection',state:'completed',candidate_total:2},
 {seq:4,elapsed_seconds:2,node:'segmentation',state:'skipped',candidate_index:0},
 {seq:5,elapsed_seconds:2.01,kind:'candidate',state:'rejected',candidate_index:0},
 {seq:6,elapsed_seconds:3,node:'traits',state:'running',candidate_index:1},
 {seq:7,elapsed_seconds:23,node:'traits',state:'completed',candidate_index:1,outputs:{features:{rows:[[1,2]]}}},
 {seq:8,elapsed_seconds:24,kind:'candidate',state:'completed',candidate_index:1,fruit_id:0},
 {seq:9,elapsed_seconds:25,node:'export',state:'completed',outputs:{downloads:{zip:'result.zip'}}}
];
assert.equal(nodeState(project(events,0),'input',null).state,'pending');
assert.equal(nodeState(project(events,1),'input',null).outputs,undefined);
assert.equal(nodeState(project(events,2),'input',null).outputs.images[0].url,'input.png');
assert.equal(nodeState(project(events,6),'traits',1).outputs,undefined);
assert.equal(nodeState(project(events,7),'traits',1).outputs.features.rows.length,1);
assert.equal(nodeState(project(events,8),'export',null).outputs,undefined);
assert.equal(project(events,8).processed,2);
assert.equal(project([{node:'models',state:'failed',elapsed_seconds:0},{node:'export',state:'skipped',elapsed_seconds:0}]).latestNode,'models');
assert.equal(project(events,8).candidates.get(0).state,'rejected');
assert.equal(project(events,8).candidates.get(1).fruit_id,0);
assert.equal(nodeState(project(events),'traits',0).state,'pending');
assert.equal(nodeState(project(events),'traits',1).state,'completed');
assert.deepEqual(project(events),project(events,events.length));
assert.equal(interval(events,1,1),600);
assert.equal(interval(events,6,1),2000);
assert.equal(interval(events,6,2),1000);
assert.equal(interval(events,6,.5),4000);
assert.equal(interval(events,events.length,1),null);
assert.equal(nodeState(project([{node:'crop',state:'skipped',elapsed_seconds:0}]),'crop',null).state,'skipped');
console.log('Graph projection and replay timing checks passed.');
'''
        completed = subprocess.run(['node','-e',script,str(path)], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
