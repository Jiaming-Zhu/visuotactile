import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw
import streamlit.elements.image as st_image
from streamlit.elements.lib.image_utils import image_to_url as _image_to_url
from streamlit.elements.lib.layout_utils import LayoutConfig

if not hasattr(st_image, "image_to_url"):
    def _compat_image_to_url(image, width, clamp, channels, output_format, image_id):
        return _image_to_url(
            image=image,
            layout_config=LayoutConfig(width=width),
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )

    st_image.image_to_url = _compat_image_to_url  # type: ignore[attr-defined]

from streamlit_drawable_canvas import st_canvas

from sam_prompt_mask_utils import (
    DEFAULT_MODEL_ID,
    get_cached_sam,
    overlay_mask,
    predict_mask,
    save_mask_outputs,
)


def parse_cli_manifest() -> Path:
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--manifest" and i + 1 < len(args):
            return Path(args[i + 1])
    return Path("/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/manifest.json")


def load_manifest(manifest_path: Path) -> Dict[str, object]:
    return json.loads(manifest_path.read_text())


def manifest_sam_output_root(manifest: Dict[str, object], manifest_path: Path) -> Path:
    configured = manifest.get("sam_output_root")
    if configured:
        return Path(str(configured))
    return manifest_path.parent / "sam_outputs"


def load_annotation(record: Dict[str, object]) -> Dict[str, object]:
    ann_path = Path(record["annotation_json_path"])
    if not ann_path.exists():
        return {"bbox_xyxy": None, "positive_points": [], "negative_points": []}
    return json.loads(ann_path.read_text())


def to_canvas_objects(
    annotation: Dict[str, object],
    scale: float,
) -> List[Dict[str, object]]:
    objects: List[Dict[str, object]] = []
    bbox = annotation.get("bbox_xyxy")
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        objects.append(
            {
                "type": "rect",
                "left": x1 * scale,
                "top": y1 * scale,
                "width": max(1.0, (x2 - x1) * scale),
                "height": max(1.0, (y2 - y1) * scale),
                "fill": "rgba(255,255,0,0.08)",
                "stroke": "#ffff00",
                "strokeWidth": 2,
                "scaleX": 1,
                "scaleY": 1,
            }
        )
    for point in annotation.get("positive_points", []):
        x, y = point
        objects.append(
            {
                "type": "circle",
                "left": x * scale - 5,
                "top": y * scale - 5,
                "radius": 5,
                "fill": "rgba(0,255,0,0.8)",
                "stroke": "#00ff00",
                "strokeWidth": 2,
                "scaleX": 1,
                "scaleY": 1,
            }
        )
    for point in annotation.get("negative_points", []):
        x, y = point
        objects.append(
            {
                "type": "circle",
                "left": x * scale - 5,
                "top": y * scale - 5,
                "radius": 5,
                "fill": "rgba(255,0,0,0.8)",
                "stroke": "#ff0000",
                "strokeWidth": 2,
                "scaleX": 1,
                "scaleY": 1,
            }
        )
    return objects


def _is_red(color_value: str) -> bool:
    color = (color_value or "").lower()
    return "#ff0000" in color or "255,0,0" in color


def parse_canvas_objects(
    objects: List[Dict[str, object]],
    inv_scale: float,
) -> Dict[str, object]:
    rects: List[Tuple[float, float, float, float]] = []
    positive_points: List[List[int]] = []
    negative_points: List[List[int]] = []

    for obj in objects or []:
        obj_type = obj.get("type")
        if obj_type == "rect":
            left = float(obj.get("left", 0.0))
            top = float(obj.get("top", 0.0))
            width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
            height = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
            rects.append(
                (
                    left * inv_scale,
                    top * inv_scale,
                    (left + width) * inv_scale,
                    (top + height) * inv_scale,
                )
            )
            continue

        if obj_type == "circle":
            left = float(obj.get("left", 0.0))
            top = float(obj.get("top", 0.0))
            radius = float(obj.get("radius", 0.0))
            scale_x = float(obj.get("scaleX", 1.0))
            scale_y = float(obj.get("scaleY", 1.0))
            cx = (left + radius * scale_x) * inv_scale
            cy = (top + radius * scale_y) * inv_scale
            point = [int(round(cx)), int(round(cy))]
            if _is_red(str(obj.get("stroke", ""))) or _is_red(str(obj.get("fill", ""))):
                negative_points.append(point)
            else:
                positive_points.append(point)

    bbox = None
    if rects:
        best_rect = max(rects, key=lambda item: max(0.0, item[2] - item[0]) * max(0.0, item[3] - item[1]))
        bbox = [int(round(best_rect[0])), int(round(best_rect[1])), int(round(best_rect[2])), int(round(best_rect[3]))]

    return {
        "bbox_xyxy": bbox,
        "positive_points": positive_points,
        "negative_points": negative_points,
    }


def save_preview(record: Dict[str, object], annotation: Dict[str, object]) -> None:
    image = Image.open(record["copied_image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    bbox = annotation.get("bbox_xyxy")
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline=(255, 255, 0), width=4)
    for x, y in annotation.get("positive_points", []):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 255, 0), width=3)
    for x, y in annotation.get("negative_points", []):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 0, 0), width=3)
    preview_path = Path(record["preview_image_path"])
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_path)


def save_annotation(record: Dict[str, object], annotation: Dict[str, object]) -> None:
    ann_path = Path(record["annotation_json_path"])
    bbox = annotation.get("bbox_xyxy")
    positive_points = annotation.get("positive_points", [])
    negative_points = annotation.get("negative_points", [])
    if bbox is None and not positive_points and not negative_points:
        if ann_path.exists():
            ann_path.unlink()
        preview_path = Path(record["preview_image_path"])
        if preview_path.exists():
            preview_path.unlink()
        return

    payload = {
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
    ann_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    save_preview(record=record, annotation=annotation)


def ensure_state(manifest: Dict[str, object]) -> None:
    if "record_index" not in st.session_state:
        first_unannotated = 0
        for idx, record in enumerate(manifest["records"]):
            if not Path(record["annotation_json_path"]).exists():
                first_unannotated = idx
                break
        st.session_state.record_index = first_unannotated
    if "canvas_nonce" not in st.session_state:
        st.session_state.canvas_nonce = 0
    if "canvas_initial_drawing" not in st.session_state:
        st.session_state.canvas_initial_drawing = {"version": "4.4.0", "objects": []}
    if "canvas_objects" not in st.session_state:
        st.session_state.canvas_objects = []


def current_record(manifest: Dict[str, object]) -> Dict[str, object]:
    records = manifest["records"]
    idx = max(0, min(st.session_state.record_index, len(records) - 1))
    st.session_state.record_index = idx
    return records[idx]


def image_scale(image: Image.Image, max_width: int = 1100) -> float:
    if image.width <= max_width:
        return 1.0
    return max_width / float(image.width)


def render_sidebar(manifest_path: Path, record: Dict[str, object], total: int) -> None:
    with st.sidebar:
        st.title("Mask Prompt Annotation")
        st.write(f"Manifest: `{manifest_path}`")
        st.write(f"Progress: `{st.session_state.record_index + 1} / {total}`")
        st.write(f"Record ID: `{record['record_id']}`")
        st.write(f"Split / Group: `{record['split']} / {record['selection_group']}`")
        st.write(f"Object / Episode: `{record['object_class']} / {record['episode_name']}`")
        st.write(f"Labels: `{record['labels']}`")
        st.caption("先画框，再点 positive / negative。保存后再切下一张。")


@st.cache_resource(show_spinner=False)
def cached_sam(model_id: str, device: str):
    return get_cached_sam(model_id=model_id, device=device)


def main() -> None:
    st.set_page_config(page_title="Mask Prompt Annotation", layout="wide")
    manifest_path = parse_cli_manifest()
    if not manifest_path.exists():
        st.error(f"Manifest not found: {manifest_path}")
        st.stop()

    manifest = load_manifest(manifest_path)
    ensure_state(manifest)
    records: List[Dict[str, object]] = manifest["records"]
    record = current_record(manifest)
    image = Image.open(record["copied_image_path"]).convert("RGB")
    annotation = load_annotation(record)
    sam_output_root = manifest_sam_output_root(manifest=manifest, manifest_path=manifest_path)

    scale = image_scale(image)
    display_w = int(round(image.width * scale))
    display_h = int(round(image.height * scale))
    inv_scale = 1.0 / scale

    if st.session_state.get("canvas_record_id") != record["record_id"]:
        st.session_state.canvas_record_id = record["record_id"]
        initial_objects = to_canvas_objects(annotation, scale=scale)
        st.session_state.canvas_initial_drawing = {"version": "4.4.0", "objects": initial_objects}
        st.session_state.canvas_objects = initial_objects
        st.session_state.canvas_nonce += 1

    render_sidebar(manifest_path=manifest_path, record=record, total=len(records))

    top_cols = st.columns([1, 1, 1, 2])
    with top_cols[0]:
        if st.button("Prev", use_container_width=True, disabled=st.session_state.record_index <= 0):
            st.session_state.record_index -= 1
            st.rerun()
    with top_cols[1]:
        if st.button("Next", use_container_width=True, disabled=st.session_state.record_index >= len(records) - 1):
            st.session_state.record_index += 1
            st.rerun()
    with top_cols[2]:
        clear_all = st.button("Clear All", use_container_width=True)
    with top_cols[3]:
        mode = st.radio(
            "Mode",
            options=["box", "positive", "negative"],
            horizontal=True,
            index=0,
        )

    if clear_all:
        st.session_state.canvas_objects = []
        st.session_state.canvas_initial_drawing = {"version": "4.4.0", "objects": []}
        st.session_state.canvas_nonce += 1

    sam_cols = st.columns([2, 2, 2])
    with sam_cols[0]:
        sam_model_id = st.selectbox(
            "SAM Model",
            options=[DEFAULT_MODEL_ID],
            index=0,
        )
    with sam_cols[1]:
        sam_device = st.selectbox(
            "SAM Device",
            options=["auto", "cuda", "cpu"],
            index=0,
        )
    with sam_cols[2]:
        st.write("")
        st.write("")
        use_saved_prompt = st.checkbox("Use saved annotation for preview", value=False)

    drawing_mode = "rect" if mode == "box" else "point"
    stroke_color = "#ffff00" if mode == "box" else ("#00ff00" if mode == "positive" else "#ff0000")
    fill_color = "rgba(255,255,0,0.08)" if mode == "box" else stroke_color

    st.write(
        "使用方式：`box` 画目标物体框；`positive` 点物体本体；`negative` 点容易误分进来的背景。"
    )

    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=2,
        stroke_color=stroke_color,
        background_image=image.resize((display_w, display_h)),
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode=drawing_mode,
        point_display_radius=5,
        initial_drawing=st.session_state.canvas_initial_drawing,
        key=f"canvas_{record['record_id']}_{st.session_state.canvas_nonce}",
    )

    current_canvas_objects = st.session_state.canvas_objects
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        st.session_state.canvas_objects = objects
        current_canvas_objects = objects

    save_cols = st.columns([1, 1, 1, 1])
    with save_cols[0]:
        if st.button("Save Annotation", use_container_width=True):
            parsed = parse_canvas_objects(current_canvas_objects, inv_scale=inv_scale)
            save_annotation(record=record, annotation=parsed)
            st.success(f"Saved {record['record_id']}")
    with save_cols[1]:
        if st.button("Save and Next", use_container_width=True):
            parsed = parse_canvas_objects(current_canvas_objects, inv_scale=inv_scale)
            save_annotation(record=record, annotation=parsed)
            if st.session_state.record_index < len(records) - 1:
                st.session_state.record_index += 1
                st.rerun()
            st.success(f"Saved {record['record_id']}")
    with save_cols[2]:
        generate_preview_clicked = st.button("Generate Mask Preview", use_container_width=True)
    with save_cols[3]:
        save_mask_clicked = st.button("Save Annotation + Mask", use_container_width=True)

    parsed_preview = parse_canvas_objects(current_canvas_objects, inv_scale=inv_scale)
    preview_cols = st.columns(2)
    with preview_cols[0]:
        st.subheader("Parsed Annotation")
        st.json(parsed_preview)
    with preview_cols[1]:
        preview_path = Path(record["preview_image_path"])
        if preview_path.exists():
            st.subheader("Saved Preview")
            st.image(str(preview_path), use_container_width=True)

    mask_preview_cols = st.columns(2)
    active_annotation = annotation if use_saved_prompt else parsed_preview

    if generate_preview_clicked or save_mask_clicked:
        if active_annotation.get("bbox_xyxy") is None and not active_annotation.get("positive_points") and not active_annotation.get("negative_points"):
            st.error("Current prompt is empty. Please draw a box or points before generating a mask.")
        else:
            with st.spinner("Loading SAM and generating mask..."):
                processor, model, device = cached_sam(model_id=sam_model_id, device=sam_device)
                prediction = predict_mask(
                    image=image,
                    annotation=active_annotation,
                    processor=processor,
                    model=model,
                    device=device,
                )
                overlay = overlay_mask(image=image, mask_uint8=prediction.mask_uint8)
                st.session_state["mask_preview_image"] = overlay
                st.session_state["mask_preview_score"] = prediction.score
                st.session_state["mask_preview_record_id"] = record["record_id"]
                st.session_state["mask_preview_mask"] = prediction.mask_uint8
                if save_mask_clicked:
                    save_annotation(record=record, annotation=active_annotation)
                    saved = save_mask_outputs(
                        record=record,
                        mask_uint8=prediction.mask_uint8,
                        overlay_image=overlay,
                        score=prediction.score,
                        output_root=sam_output_root,
                    )
                    st.success(
                        f"Saved annotation and mask for {record['record_id']} "
                        f"(score={prediction.score:.4f})"
                    )
                    st.caption(f"Mask path: {saved['mask_path']}")

    with mask_preview_cols[0]:
        st.subheader("Current Image")
        st.image(image, use_container_width=True)
    with mask_preview_cols[1]:
        st.subheader("SAM Mask Preview")
        if st.session_state.get("mask_preview_record_id") == record["record_id"]:
            st.image(st.session_state["mask_preview_image"], use_container_width=True)
            st.caption(f"Predicted mask score: {st.session_state['mask_preview_score']:.4f}")
        else:
            existing_mask_preview = sam_output_root / "mask_previews" / f"{record['record_id']}.png"
            if existing_mask_preview.exists():
                st.image(str(existing_mask_preview), use_container_width=True)
                st.caption("Loaded existing saved SAM preview.")
            else:
                st.info("No SAM preview yet. Click `Generate Mask Preview`.")


if __name__ == "__main__":
    main()
