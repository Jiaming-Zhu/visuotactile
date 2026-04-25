import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


WINDOW_NAME = "Mask Prompt Annotation"
MODE_BOX = "box"
MODE_POS = "positive"
MODE_NEG = "negative"


@dataclass
class AnnotationState:
    bbox: Optional[List[int]]
    positive_points: List[List[int]]
    negative_points: List[List[int]]
    dirty: bool = False


class AnnotationApp:
    def __init__(
        self,
        manifest_path: Path,
        max_width: int,
        max_height: int,
        start_mode: str,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text())
        self.records = self.manifest["records"]
        self.max_width = max_width
        self.max_height = max_height
        self.mode = start_mode
        self.current_index = self._find_first_unannotated()
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_preview: Optional[Tuple[int, int]] = None

        self.record = None
        self.original_image = None
        self.display_image = None
        self.scale = 1.0
        self.state = AnnotationState(bbox=None, positive_points=[], negative_points=[])
        self._load_current_record()

    def _find_first_unannotated(self) -> int:
        for idx, record in enumerate(self.records):
            if not Path(record["annotation_json_path"]).exists():
                return idx
        return 0

    def _fit_scale(self, width: int, height: int) -> float:
        scale_w = self.max_width / max(1, width)
        scale_h = self.max_height / max(1, height)
        return min(1.0, scale_w, scale_h)

    def _load_annotation(self, record: Dict[str, object]) -> AnnotationState:
        ann_path = Path(record["annotation_json_path"])
        if not ann_path.exists():
            return AnnotationState(bbox=None, positive_points=[], negative_points=[])
        payload = json.loads(ann_path.read_text())
        return AnnotationState(
            bbox=payload.get("bbox_xyxy"),
            positive_points=payload.get("positive_points", []),
            negative_points=payload.get("negative_points", []),
            dirty=False,
        )

    def _load_current_record(self) -> None:
        self.record = self.records[self.current_index]
        self.original_image = cv2.imread(str(self.record["copied_image_path"]), cv2.IMREAD_COLOR)
        if self.original_image is None:
            raise FileNotFoundError(f"Failed to read image: {self.record['copied_image_path']}")
        self.scale = self._fit_scale(width=self.original_image.shape[1], height=self.original_image.shape[0])
        if self.scale < 1.0:
            resized_w = int(round(self.original_image.shape[1] * self.scale))
            resized_h = int(round(self.original_image.shape[0] * self.scale))
            self.display_image = cv2.resize(self.original_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        else:
            self.display_image = self.original_image.copy()
        self.state = self._load_annotation(self.record)
        self.drag_start = None
        self.drag_preview = None

    def _to_original_xy(self, x: int, y: int) -> Tuple[int, int]:
        ox = int(round(x / self.scale))
        oy = int(round(y / self.scale))
        ox = min(max(ox, 0), self.original_image.shape[1] - 1)
        oy = min(max(oy, 0), self.original_image.shape[0] - 1)
        return ox, oy

    def _to_display_xy(self, x: int, y: int) -> Tuple[int, int]:
        dx = int(round(x * self.scale))
        dy = int(round(y * self.scale))
        return dx, dy

    def _save_current(self) -> None:
        ann_path = Path(self.record["annotation_json_path"])
        preview_path = Path(self.record["preview_image_path"])
        if self.state.bbox is None and not self.state.positive_points and not self.state.negative_points:
            if ann_path.exists():
                ann_path.unlink()
            if preview_path.exists():
                preview_path.unlink()
            self.state.dirty = False
            return

        payload = {
            "record_id": self.record["record_id"],
            "split": self.record["split"],
            "selection_group": self.record["selection_group"],
            "dataset_index": self.record["dataset_index"],
            "source_image_path": self.record["source_image_path"],
            "copied_image_path": self.record["copied_image_path"],
            "bbox_xyxy": self.state.bbox,
            "positive_points": self.state.positive_points,
            "negative_points": self.state.negative_points,
            "object_class": self.record["object_class"],
            "episode_name": self.record["episode_name"],
            "labels": self.record["labels"],
        }
        ann_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        self._save_preview(preview_path)
        self.state.dirty = False

    def _save_preview(self, preview_path: Path) -> None:
        canvas = self._build_canvas()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), canvas)

    def _draw_bbox(self, canvas: np.ndarray, bbox: List[int], color: Tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        p1 = self._to_display_xy(x1, y1)
        p2 = self._to_display_xy(x2, y2)
        cv2.rectangle(canvas, p1, p2, color, 2)

    def _draw_points(self, canvas: np.ndarray, points: List[List[int]], color: Tuple[int, int, int]) -> None:
        for x, y in points:
            dx, dy = self._to_display_xy(x, y)
            cv2.circle(canvas, (dx, dy), 5, color, -1)
            cv2.circle(canvas, (dx, dy), 9, color, 1)

    def _build_canvas(self) -> np.ndarray:
        canvas = self.display_image.copy()
        if self.state.bbox is not None:
            self._draw_bbox(canvas, self.state.bbox, (0, 255, 255))
        self._draw_points(canvas, self.state.positive_points, (0, 255, 0))
        self._draw_points(canvas, self.state.negative_points, (0, 0, 255))

        if self.drag_start is not None and self.drag_preview is not None and self.mode == MODE_BOX:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_preview
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 1)

        status = (
            f"{self.current_index + 1}/{len(self.records)} | "
            f"id={self.record['record_id']} | split={self.record['split']} | "
            f"group={self.record['selection_group']} | mode={self.mode} | "
            f"dirty={'yes' if self.state.dirty else 'no'}"
        )
        subtitle = (
            "[b] box  [p] positive  [n] negative  [u] undo point  [x] clear box  "
            "[c] clear all  [s] save  [a]/[d] prev/next  [q] quit"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (25, 25, 25), -1)
        cv2.putText(canvas, status, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
        return canvas

    def _move(self, delta: int) -> None:
        self._save_current()
        self.current_index = min(max(self.current_index + delta, 0), len(self.records) - 1)
        self._load_current_record()

    def _undo_point(self) -> None:
        if self.mode == MODE_NEG and self.state.negative_points:
            self.state.negative_points.pop()
            self.state.dirty = True
            return
        if self.mode == MODE_POS and self.state.positive_points:
            self.state.positive_points.pop()
            self.state.dirty = True
            return
        if self.state.negative_points:
            self.state.negative_points.pop()
            self.state.dirty = True
            return
        if self.state.positive_points:
            self.state.positive_points.pop()
            self.state.dirty = True

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.mode == MODE_BOX:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drag_start = (x, y)
                self.drag_preview = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
                self.drag_preview = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
                x1, y1 = self.drag_start
                x2, y2 = x, y
                ox1, oy1 = self._to_original_xy(min(x1, x2), min(y1, y2))
                ox2, oy2 = self._to_original_xy(max(x1, x2), max(y1, y2))
                if abs(ox2 - ox1) >= 3 and abs(oy2 - oy1) >= 3:
                    self.state.bbox = [ox1, oy1, ox2, oy2]
                    self.state.dirty = True
                self.drag_start = None
                self.drag_preview = None
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = self._to_original_xy(x, y)
            point = [ox, oy]
            if self.mode == MODE_POS:
                self.state.positive_points.append(point)
            elif self.mode == MODE_NEG:
                self.state.negative_points.append(point)
            self.state.dirty = True

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        while True:
            canvas = self._build_canvas()
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue
            if key == ord("q"):
                self._save_current()
                break
            if key == ord("b"):
                self.mode = MODE_BOX
            elif key == ord("p"):
                self.mode = MODE_POS
            elif key == ord("n"):
                self.mode = MODE_NEG
            elif key == ord("u"):
                self._undo_point()
            elif key == ord("x"):
                self.state.bbox = None
                self.state.dirty = True
            elif key == ord("c"):
                self.state = AnnotationState(bbox=None, positive_points=[], negative_points=[], dirty=True)
            elif key == ord("s"):
                self._save_current()
            elif key == ord("a"):
                self._move(-1)
            elif key == ord("d"):
                self._move(1)

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local OpenCV annotation tool for object box and positive/negative point prompts."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/manifest.json",
    )
    parser.add_argument("--max_width", type=int, default=1600)
    parser.add_argument("--max_height", type=int, default=1000)
    parser.add_argument(
        "--start_mode",
        type=str,
        default=MODE_BOX,
        choices=[MODE_BOX, MODE_POS, MODE_NEG],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = AnnotationApp(
        manifest_path=Path(args.manifest),
        max_width=args.max_width,
        max_height=args.max_height,
        start_mode=args.start_mode,
    )
    app.run()


if __name__ == "__main__":
    main()
