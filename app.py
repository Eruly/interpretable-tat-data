import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import os
import re
import html as html_mod
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from data_utils import load_samples, render_table_to_png
from inference_cls import QwenInference

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_infer = None
_infer_load_errors: list[str] = []
samples = load_samples()

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR  # table PNGs live in project root


def _get_infer() -> QwenInference:
    global _infer
    if _infer is not None:
        return _infer

    backend = os.getenv("QWEN_BACKEND", "vllm_api")
    model = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-32B-Thinking")

    try:
        print(f"[app] Initializing inference with backend: {backend}")
        _infer = QwenInference(model_id=model, backend=backend)
        return _infer
    except Exception as exc:
        print(f"[app] Failed to initialize {backend}: {exc}")
        _infer_load_errors.append(f"{backend}: {exc}")
        if backend == "vllm_api":
            try:
                print("[app] Falling back to local transformers backend...")
                _infer = QwenInference(
                    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                    backend="transformers",
                )
                return _infer
            except Exception as e:
                _infer_load_errors.append(f"fallback: {e}")
    raise RuntimeError(f"All model loading attempts failed. Errors: {_infer_load_errors}")


# ---------------------------------------------------------------------------
# Utility functions (ported from old Gradio app)
# ---------------------------------------------------------------------------

def normalize_bbox_markup(text: str) -> str:
    if not text:
        return text
    normalized = html_mod.unescape(text)
    normalized = re.sub(r"(?:<bbox>\s*){2,}", "<bbox>", normalized)
    normalized = re.sub(r"(?:</bbox>\s*){2,}", "</bbox>", normalized)
    normalized = re.sub(r"<bbox>\s*</bbox>", "", normalized)
    normalized = re.sub(r"</bbox>\s*<bbox>\s*</bbox>", "</bbox>", normalized)
    return normalized


def extract_transformed_text(text: str) -> str:
    if not text:
        return ""
    fenced = re.search(r"```text\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    generic_fenced = re.search(r"```[\w-]*\s*([\s\S]*?)```", text)
    if generic_fenced:
        return generic_fenced.group(1).strip()
    return text.strip()


def extract_bbox_entries(text: str) -> list[dict]:
    text = normalize_bbox_markup(text)
    outer_patterns = [
        re.compile(r"<bbox>\s*(.*?)\s*</bbox>", re.DOTALL),
        re.compile(r"<bbox>\s*(.*?)\s*<bbox>", re.DOTALL),
    ]
    inner_pattern = re.compile(
        r"^\s*(.*?)\s*,?\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\s*$",
        re.DOTALL,
    )
    coords_then_label = re.compile(
        r"^\s*\(?\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?\s*(.*)$",
        re.DOTALL,
    )
    coords_only = re.compile(
        r"\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]"
    )

    entries: list[dict] = []
    seen_ranges: set[tuple] = set()
    for pattern in outer_patterns:
        for m in pattern.finditer(text):
            if m.span() in seen_ranges:
                continue
            seen_ranges.add(m.span())
            inner = m.group(1).strip()
            label_text = inner
            coords = None

            im = inner_pattern.match(inner)
            if im:
                label_text = im.group(1).strip()
                coords = [float(im.group(i)) for i in range(2, 6)]
            else:
                cm = coords_then_label.match(inner)
                if cm:
                    coords = [float(cm.group(i)) for i in range(1, 5)]
                    label_text = cm.group(5).strip() or inner
                else:
                    co = coords_only.search(inner)
                    if co:
                        coords = [float(co.group(i)) for i in range(1, 5)]

            if not coords:
                continue
            if max(coords) > 1.1:
                coords = [c / 1000.0 for c in coords]

            entries.append({
                "start": m.start(),
                "end": m.end(),
                "raw": m.group(0),
                "label": label_text,
                "bbox_norm": coords,
            })
    entries.sort(key=lambda x: x["start"])
    return entries


def create_model_input_image(image_path: str, target_size=(1000, 1000)) -> str:
    img = Image.open(image_path).convert("RGB")
    resized = img.resize(target_size, Image.Resampling.BICUBIC)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_path = tmp.name
    tmp.close()
    resized.save(tmp_path, format="PNG")
    return tmp_path


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Qwen3-VL Interpretable Table QA")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/api/samples")
async def get_samples():
    items = []
    for i, s in enumerate(samples[:20]):
        q = s["questions"][0]["question"]
        items.append({"id": i, "question": q})
    return items


@app.get("/api/load_sample/{idx}")
async def load_sample(idx: int):
    if idx < 0 or idx >= len(samples):
        raise HTTPException(404, "Sample not found")
    sample = samples[idx]
    table_data = sample["table"]["table"]
    image_filename = f"tmp_table_{idx}.png"
    image_path = str(IMAGES_DIR / image_filename)
    render_table_to_png(table_data, image_path)

    question = sample["questions"][0]["question"]
    paragraphs = "\n\n".join([p["text"] for p in sample.get("paragraphs", [])])
    return {
        "image_url": f"/images/{image_filename}",
        "question": question,
        "paragraphs": paragraphs,
    }


@app.get("/images/{filename}")
async def serve_image(filename: str):
    path = IMAGES_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(path), media_type="image/png")


class InferRequest(BaseModel):
    image_url: str
    question: str


@app.post("/api/infer")
async def infer(req: InferRequest):
    filename = req.image_url.split("/")[-1]
    image_path = str(IMAGES_DIR / filename)
    if not os.path.exists(image_path):
        raise HTTPException(400, f"Image file not found: {filename}")

    model_input_path = None
    try:
        infer_engine = _get_infer()
        model_input_path = create_model_input_image(image_path, target_size=(1000, 1000))
        details = infer_engine.get_answer_with_details(model_input_path, req.question)

        stage1_raw = details.get("stage1_reasoning_raw", details.get("stage1_reasoning", details["stage1_answer"]))
        stage1_reasoning = normalize_bbox_markup(stage1_raw)
        stage1_display = re.sub(r"<bbox>\s*[\s\S]*?\s*</bbox>", "", stage1_reasoning)

        stage3_raw_val = details.get("stage3_answer", details.get("stage1_reasoning", details["stage1_answer"]))
        stage3_result = normalize_bbox_markup(extract_transformed_text(stage3_raw_val))

        final_answer = normalize_bbox_markup(details["final_answer"])

        ocr_items = details.get("ocr_items", [])
        ocr_raw = details.get("ocr_raw", "")
        ocr_error = details.get("ocr_error", "")

        bbox_entries = extract_bbox_entries(stage3_result)
        bboxes = []
        for e in bbox_entries:
            bboxes.append({
                "label": e["label"],
                "bbox_norm": e["bbox_norm"],
                "raw": e["raw"],
            })

        return {
            "stage1_reasoning": stage1_display,
            "stage3_result": stage3_result,
            "final_answer": final_answer,
            "ocr_items": ocr_items,
            "ocr_raw": ocr_raw,
            "ocr_error": ocr_error,
            "bboxes": bboxes,
        }
    except Exception as exc:
        print(f"[app] Inference error: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(exc))
    finally:
        if model_input_path and os.path.exists(model_input_path):
            try:
                os.remove(model_input_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("GRADIO_SERVER_PORT", "7862"))
    uvicorn.run(app, host="0.0.0.0", port=port)
