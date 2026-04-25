import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Mask Prompt Annotation</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: sans-serif; margin: 0; background: #111; color: #eee; }
    .wrap { display: grid; grid-template-columns: 340px 1fr; min-height: 100vh; }
    .panel { padding: 16px; background: #191919; border-right: 1px solid #333; }
    .panel h2 { margin-top: 0; font-size: 20px; }
    .row { margin-bottom: 10px; }
    .label { font-size: 12px; color: #aaa; margin-bottom: 4px; }
    .value { font-size: 14px; word-break: break-word; }
    .btns { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
    button {
      background: #2c2c2c; color: #eee; border: 1px solid #555;
      padding: 8px 12px; cursor: pointer; border-radius: 6px;
    }
    button.active { background: #0b5; border-color: #0d7; color: #fff; }
    button.warn { background: #822; border-color: #b44; }
    .main { padding: 16px; }
    .status { margin-bottom: 12px; color: #9ad; }
    .hint { font-size: 13px; color: #bbb; line-height: 1.45; }
    canvas { border: 1px solid #444; background: #222; cursor: crosshair; max-width: 100%; }
    .legend { display: flex; gap: 16px; margin: 8px 0 16px; font-size: 13px; color: #ccc; }
    .dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
    .footer { margin-top: 12px; font-size: 12px; color: #888; }
    code { color: #9cf; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h2>Mask Prompt Annotation</h2>
      <div class="row"><div class="label">Progress</div><div class="value" id="progress"></div></div>
      <div class="row"><div class="label">Record ID</div><div class="value" id="record_id"></div></div>
      <div class="row"><div class="label">Split / Group</div><div class="value" id="split_group"></div></div>
      <div class="row"><div class="label">Object / Episode</div><div class="value" id="object_episode"></div></div>
      <div class="row"><div class="label">Labels</div><div class="value" id="labels"></div></div>
      <div class="row"><div class="label">Image Path</div><div class="value" id="image_path"></div></div>
      <div class="btns">
        <button id="mode_box">Box (B)</button>
        <button id="mode_pos">Positive (P)</button>
        <button id="mode_neg">Negative (N)</button>
      </div>
      <div class="btns">
        <button id="prev_btn">Prev (A)</button>
        <button id="next_btn">Next (D)</button>
        <button id="save_btn">Save (S)</button>
      </div>
      <div class="btns">
        <button id="undo_btn">Undo (U)</button>
        <button id="clear_box_btn">Clear Box (X)</button>
        <button id="clear_all_btn" class="warn">Clear All (C)</button>
      </div>
      <div class="hint">
        1. 先用 <code>Box</code> 框住目标物体。<br>
        2. 用 <code>Positive</code> 在物体主体上点 1-3 个点。<br>
        3. 如果背景容易混进来，再用 <code>Negative</code> 点 1-2 个背景点。<br>
        4. 按 <code>Save</code> 保存当前样本。
      </div>
      <div class="footer" id="save_state"></div>
    </div>
    <div class="main">
      <div class="status" id="status">Loading...</div>
      <div class="legend">
        <span><span class="dot" style="background:#ff0;"></span>Box</span>
        <span><span class="dot" style="background:#0f0;"></span>Positive</span>
        <span><span class="dot" style="background:#f33;"></span>Negative</span>
      </div>
      <canvas id="canvas"></canvas>
    </div>
  </div>
<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let records = [];
let currentIndex = 0;
let currentRecord = null;
let currentImage = null;
let mode = 'box';
let bbox = null;
let positivePoints = [];
let negativePoints = [];
let dragging = false;
let dragStart = null;
let dragPreview = null;
let dirty = false;

function setStatus(msg) {
  document.getElementById('status').textContent = msg;
}

function setMode(newMode) {
  mode = newMode;
  document.getElementById('mode_box').classList.toggle('active', mode === 'box');
  document.getElementById('mode_pos').classList.toggle('active', mode === 'positive');
  document.getElementById('mode_neg').classList.toggle('active', mode === 'negative');
}

function scaleForImage(img) {
  const maxWidth = window.innerWidth - 420;
  const maxHeight = window.innerHeight - 80;
  const scale = Math.min(1, maxWidth / img.width, maxHeight / img.height);
  return scale;
}

function displayToOriginal(x, y) {
  const scaleX = currentImage.width / canvas.width;
  const scaleY = currentImage.height / canvas.height;
  return [Math.max(0, Math.min(currentImage.width - 1, Math.round(x * scaleX))),
          Math.max(0, Math.min(currentImage.height - 1, Math.round(y * scaleY)))];
}

function originalToDisplay(x, y) {
  const scaleX = canvas.width / currentImage.width;
  const scaleY = canvas.height / currentImage.height;
  return [Math.round(x * scaleX), Math.round(y * scaleY)];
}

function drawPoint(x, y, color) {
  const [dx, dy] = originalToDisplay(x, y);
  ctx.beginPath();
  ctx.arc(dx, dy, 5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(dx, dy, 9, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function render() {
  if (!currentImage) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

  if (bbox) {
    const [x1, y1] = originalToDisplay(bbox[0], bbox[1]);
    const [x2, y2] = originalToDisplay(bbox[2], bbox[3]);
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  for (const pt of positivePoints) drawPoint(pt[0], pt[1], '#00ff00');
  for (const pt of negativePoints) drawPoint(pt[0], pt[1], '#ff3333');

  if (dragging && dragStart && dragPreview && mode === 'box') {
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 1;
    const x = Math.min(dragStart[0], dragPreview[0]);
    const y = Math.min(dragStart[1], dragPreview[1]);
    const w = Math.abs(dragPreview[0] - dragStart[0]);
    const h = Math.abs(dragPreview[1] - dragStart[1]);
    ctx.strokeRect(x, y, w, h);
  }

  document.getElementById('save_state').textContent = dirty ? 'Unsaved changes' : 'Saved';
}

function updateSidebar() {
  if (!currentRecord) return;
  document.getElementById('progress').textContent = `${currentIndex + 1} / ${records.length}`;
  document.getElementById('record_id').textContent = currentRecord.record_id;
  document.getElementById('split_group').textContent = `${currentRecord.split} / ${currentRecord.selection_group}`;
  document.getElementById('object_episode').textContent = `${currentRecord.object_class} / ${currentRecord.episode_name}`;
  document.getElementById('labels').textContent = JSON.stringify(currentRecord.labels);
  document.getElementById('image_path').textContent = currentRecord.source_image_path;
}

async function loadRecord(index) {
  currentIndex = Math.max(0, Math.min(index, records.length - 1));
  currentRecord = records[currentIndex];
  setStatus(`Loading record ${currentRecord.record_id}...`);
  const recordResp = await fetch(`/api/record/${currentRecord.record_id}`);
  const payload = await recordResp.json();
  bbox = payload.annotation.bbox_xyxy || null;
  positivePoints = payload.annotation.positive_points || [];
  negativePoints = payload.annotation.negative_points || [];
  dirty = false;

  currentImage = new Image();
  currentImage.onload = () => {
    const scale = scaleForImage(currentImage);
    canvas.width = Math.round(currentImage.width * scale);
    canvas.height = Math.round(currentImage.height * scale);
    updateSidebar();
    render();
    setStatus(`Loaded ${currentRecord.record_id}`);
  };
  currentImage.src = `/api/image/${currentRecord.record_id}?t=${Date.now()}`;
}

async function saveCurrent() {
  if (!currentRecord) return;
  const resp = await fetch(`/api/save/${currentRecord.record_id}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      bbox_xyxy: bbox,
      positive_points: positivePoints,
      negative_points: negativePoints
    })
  });
  if (!resp.ok) {
    const text = await resp.text();
    setStatus(`Save failed: ${text}`);
    return;
  }
  dirty = false;
  records[currentIndex].annotation_exists = true;
  render();
  setStatus(`Saved ${currentRecord.record_id}`);
}

function undoCurrentMode() {
  if (mode === 'negative' && negativePoints.length) {
    negativePoints.pop();
    dirty = true;
    render();
    return;
  }
  if (mode === 'positive' && positivePoints.length) {
    positivePoints.pop();
    dirty = true;
    render();
    return;
  }
  if (negativePoints.length) {
    negativePoints.pop();
    dirty = true;
    render();
    return;
  }
  if (positivePoints.length) {
    positivePoints.pop();
    dirty = true;
    render();
  }
}

canvas.addEventListener('mousedown', (e) => {
  if (!currentImage) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  if (mode === 'box') {
    dragging = true;
    dragStart = [x, y];
    dragPreview = [x, y];
  } else {
    const [ox, oy] = displayToOriginal(x, y);
    if (mode === 'positive') positivePoints.push([ox, oy]);
    if (mode === 'negative') negativePoints.push([ox, oy]);
    dirty = true;
    render();
  }
});

canvas.addEventListener('mousemove', (e) => {
  if (!dragging || mode !== 'box') return;
  const rect = canvas.getBoundingClientRect();
  dragPreview = [e.clientX - rect.left, e.clientY - rect.top];
  render();
});

canvas.addEventListener('mouseup', (e) => {
  if (!dragging || mode !== 'box') return;
  const rect = canvas.getBoundingClientRect();
  const end = [e.clientX - rect.left, e.clientY - rect.top];
  const x1 = Math.min(dragStart[0], end[0]);
  const y1 = Math.min(dragStart[1], end[1]);
  const x2 = Math.max(dragStart[0], end[0]);
  const y2 = Math.max(dragStart[1], end[1]);
  const p1 = displayToOriginal(x1, y1);
  const p2 = displayToOriginal(x2, y2);
  if (Math.abs(p2[0] - p1[0]) >= 3 && Math.abs(p2[1] - p1[1]) >= 3) {
    bbox = [p1[0], p1[1], p2[0], p2[1]];
    dirty = true;
  }
  dragging = false;
  dragStart = null;
  dragPreview = null;
  render();
});

window.addEventListener('keydown', async (e) => {
  if (e.target && ['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === 'b') setMode('box');
  else if (e.key === 'p') setMode('positive');
  else if (e.key === 'n') setMode('negative');
  else if (e.key === 'u') undoCurrentMode();
  else if (e.key === 'x') { bbox = null; dirty = true; render(); }
  else if (e.key === 'c') { bbox = null; positivePoints = []; negativePoints = []; dirty = true; render(); }
  else if (e.key === 's') await saveCurrent();
  else if (e.key === 'a') { await saveCurrent(); await loadRecord(currentIndex - 1); }
  else if (e.key === 'd') { await saveCurrent(); await loadRecord(currentIndex + 1); }
});

document.getElementById('mode_box').onclick = () => setMode('box');
document.getElementById('mode_pos').onclick = () => setMode('positive');
document.getElementById('mode_neg').onclick = () => setMode('negative');
document.getElementById('undo_btn').onclick = () => undoCurrentMode();
document.getElementById('clear_box_btn').onclick = () => { bbox = null; dirty = true; render(); };
document.getElementById('clear_all_btn').onclick = () => { bbox = null; positivePoints = []; negativePoints = []; dirty = true; render(); };
document.getElementById('save_btn').onclick = () => saveCurrent();
document.getElementById('prev_btn').onclick = async () => { await saveCurrent(); await loadRecord(currentIndex - 1); };
document.getElementById('next_btn').onclick = async () => { await saveCurrent(); await loadRecord(currentIndex + 1); };

async function init() {
  const resp = await fetch('/api/manifest');
  const payload = await resp.json();
  records = payload.records;
  setMode('box');
  await loadRecord(payload.start_index || 0);
}
init();
</script>
</body>
</html>
"""


class AnnotationStore:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text())
        self.records = self.manifest["records"]
        self.by_id = {str(record["record_id"]): record for record in self.records}
        self.lock = threading.Lock()

    def summary_records(self) -> List[Dict[str, object]]:
        summary = []
        for record in self.records:
            ann_path = Path(record["annotation_json_path"])
            summary.append(
                {
                    "record_id": str(record["record_id"]),
                    "split": record["split"],
                    "selection_group": record["selection_group"],
                    "dataset_index": record["dataset_index"],
                    "object_class": record["object_class"],
                    "episode_name": record["episode_name"],
                    "source_image_path": record["source_image_path"],
                    "labels": record["labels"],
                    "annotation_exists": ann_path.exists(),
                }
            )
        return summary

    def first_unannotated_index(self) -> int:
        for idx, record in enumerate(self.records):
            if not Path(record["annotation_json_path"]).exists():
                return idx
        return 0

    def get_record(self, record_id: str) -> Dict[str, object]:
        return self.by_id[str(record_id)]

    def load_annotation(self, record_id: str) -> Dict[str, object]:
        record = self.get_record(record_id)
        ann_path = Path(record["annotation_json_path"])
        if not ann_path.exists():
            return {"bbox_xyxy": None, "positive_points": [], "negative_points": []}
        return json.loads(ann_path.read_text())

    def save_annotation(self, record_id: str, payload: Dict[str, object]) -> None:
        record = self.get_record(record_id)
        ann_path = Path(record["annotation_json_path"])
        preview_path = Path(record["preview_image_path"])
        bbox = payload.get("bbox_xyxy")
        positive_points = payload.get("positive_points", [])
        negative_points = payload.get("negative_points", [])

        with self.lock:
            if bbox is None and not positive_points and not negative_points:
                if ann_path.exists():
                    ann_path.unlink()
                if preview_path.exists():
                    preview_path.unlink()
                return

            output = {
                "record_id": record["record_id"],
                "split": record["split"],
                "selection_group": record["selection_group"],
                "dataset_index": record["dataset_index"],
                "source_image_path": record["source_image_path"],
                "copied_image_path": record["copied_image_path"],
                "bbox_xyxy": bbox,
                "positive_points": positive_points,
                "negative_points": negative_points,
                "object_class": record["object_class"],
                "episode_name": record["episode_name"],
                "labels": record["labels"],
            }
            ann_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
            self._save_preview(record=record, bbox=bbox, positive_points=positive_points, negative_points=negative_points)

    def _save_preview(
        self,
        record: Dict[str, object],
        bbox: Optional[List[int]],
        positive_points: List[List[int]],
        negative_points: List[List[int]],
    ) -> None:
        image = Image.open(record["copied_image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        if bbox is not None:
            draw.rectangle(tuple(bbox), outline=(255, 255, 0), width=4)
        for x, y in positive_points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 255, 0), width=3)
        for x, y in negative_points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 64, 64), width=3)
        preview_path = Path(record["preview_image_path"])
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(preview_path)


class AnnotationHandler(BaseHTTPRequestHandler):
    store: AnnotationStore = None  # type: ignore

    def _send_json(self, payload: Dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/manifest":
            self._send_json(
                {
                    "records": self.store.summary_records(),
                    "start_index": self.store.first_unannotated_index(),
                }
            )
            return
        if path.startswith("/api/record/"):
            record_id = path.split("/")[-1]
            try:
                record = self.store.get_record(record_id)
            except KeyError:
                self._send_json({"error": f"Unknown record_id {record_id}"}, status=404)
                return
            self._send_json(
                {
                    "record": record,
                    "annotation": self.store.load_annotation(record_id),
                }
            )
            return
        if path.startswith("/api/image/"):
            record_id = path.split("/")[-1]
            try:
                record = self.store.get_record(record_id)
            except KeyError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            image_path = Path(record["copied_image_path"])
            if not image_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = image_path.read_bytes()
            content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
            self._send_bytes(content, content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/save/"):
            record_id = parsed.path.split("/")[-1]
            if record_id not in self.store.by_id:
                self._send_json({"error": f"Unknown record_id {record_id}"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON payload"}, status=400)
                return
            self.store.save_annotation(record_id, payload)
            self._send_json({"ok": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browser-based local annotation tool for object box and positive/negative point prompts."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/manifest.json",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open_browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = AnnotationStore(Path(args.manifest))
    AnnotationHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Annotation server running at {url}")
    print("Open this URL in your browser.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
