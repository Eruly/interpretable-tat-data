import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import json
import gradio as gr
import os
import re
from PIL import Image
from data_utils import load_samples, render_table_to_png
from inference_cls import QwenInference

# Lazily initialize inference so model load failures do not prevent app startup.
_infer = None
_infer_load_errors = []


def _candidate_models():
    primary = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-32B-Thinking")
    fallback = os.getenv("QWEN_FALLBACK_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    safe = "Qwen/Qwen2.5-VL-3B-Instruct"

    candidates = []
    for model in (primary, fallback, safe):
        if model and model not in candidates:
            candidates.append(model)
    return candidates


def _get_infer():
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
        # Try fallback to local transformers if API fails and not explicitly forced
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


samples = load_samples()


def parse_bboxes(text, img_width, img_height):
    """
    Parses <bbox>(value, [xmin, ymin, xmax, ymax])<bbox>
    Returns list of (bbox, label) where bbox is [x1, y1, x2, y2] in pixels.
    """
    outer_pattern = re.compile(r"<bbox>\((.*?)\)<bbox>", re.DOTALL)
    # Robust pattern: value, optional comma, [coords]
    inner_pattern = re.compile(
        r"^\s*(.*?)\s*,?\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\s*$",
        re.DOTALL,
    )

    results = []
    for m in outer_pattern.finditer(text):
        inner = m.group(1).strip()
        inner_match = inner_pattern.match(inner)
        if not inner_match:
            continue

        label_text = inner_match.group(1).strip()
        v1, v2, v3, v4 = [float(inner_match.group(i)) for i in range(2, 6)]

        # Detection for 0-1000 scale vs 0-1 scale
        if max(v1, v2, v3, v4) > 1.1:
            v1 /= 1000.0
            v2 /= 1000.0
            v3 /= 1000.0
            v4 /= 1000.0

        xmin, ymin, xmax, ymax = v1, v2, v3, v4
        x1 = int(xmin * img_width)
        y1 = int(ymin * img_height)
        x2 = int(xmax * img_width)
        y2 = int(ymax * img_height)
        
        # Prefix label to distinguish from OCR
        full_label = f"Reasoning: {label_text}"

        results.append(([x1, y1, x2, y2], full_label))
    return results


def ocr_items_to_bboxes(ocr_items, img_width, img_height):
    items = []
    for item in ocr_items:
        bbox = item.get("bbox_norm") or item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1 = int(float(bbox[0]) * img_width)
        y1 = int(float(bbox[1]) * img_height)
        x2 = int(float(bbox[2]) * img_width)
        y2 = int(float(bbox[3]) * img_height)
        label = f"OCR: {item.get('text', '').strip()}"
        items.append(([x1, y1, x2, y2], label))
    return items


def process_sample(sample_idx):
    sample = samples[int(sample_idx)]
    table_data = sample["table"]["table"]
    image_path = f"tmp_table_{sample_idx}.png"
    render_table_to_png(table_data, image_path)

    question = sample["questions"][0]["question"]
    paragraphs = "\n\n".join([p["text"] for p in sample.get("paragraphs", [])])
    return image_path, question, paragraphs, sample_idx


def run_inference(image_path, question):
    if not image_path:
        return "Please select a sample first.", None, "[]", "", "[]", ""

    try:
        infer = _get_infer()
        details = infer.get_answer_with_details(image_path, question)
        stage1_output = details["stage1_answer"]
        
        # Clean OCR output (strip system/user/assistant headers)
        ocr_output = details["ocr_raw"]
        for marker in ["assistant", "assistant\n"]:
            if marker in ocr_output:
                ocr_output = ocr_output.split(marker)[-1].strip()
        
        # Wrap in a div for better rendering in gr.HTML if it's a table
        if "<table" in ocr_output:
            ocr_output = f"<div style='overflow-x: auto;'>{ocr_output}</div>"
        
        output = details["final_answer"]
    except Exception as exc:
        print(f"[app] Inference unavailable: {exc}")
        return "N/A", "N/A", "현재 추론을 실행할 수 없습니다. 잠시 후 다시 시도해주세요.", None, "[]", ""

    try:
        img = Image.open(image_path)
        w, h = img.size

        think_bboxes = parse_bboxes(output, w, h)
        ocr_bboxes = ocr_items_to_bboxes(details.get("ocr_items", []), w, h)
        
        # Merge boxes: OCR first, then Reasoning (so Reasoning draws on top)
        merged_bboxes = ocr_bboxes + think_bboxes
        merged_bboxes.append(([0, 0, w, h], "Coord Frame [0,0,1000,1000]"))

        ocr_json = json.dumps(details.get("ocr_items", []), ensure_ascii=False, indent=2)
        ocr_raw = details.get("ocr_raw", "")
        
        # Convert newlines to <br> for HTML display in the text box
        import html
        stage3_display = html.escape(output).replace('\n', '<br>')
        
        return stage1_output, ocr_output, stage3_display, (image_path, merged_bboxes), ocr_json, ocr_raw
    except Exception as e:
        print(f"[app] Visualization error: {e}")
        ocr_json = json.dumps(details.get("ocr_items", []), ensure_ascii=False, indent=2)
        ocr_raw = details.get("ocr_raw", "")
        return stage1_output, ocr_output, output, image_path, ocr_json, ocr_raw


custom_css = """
.container { max-width: 1200px; margin: auto; }
.header { text-align: center; margin-bottom: 2rem; }
.output-box { font-family: 'Inter', sans-serif; }
"""

with gr.Blocks(title="Qwen3-VL Interpretable Table QA") as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# 🚀 Qwen3-VL TAT-QA Interpretable Demo", elem_classes="header")
        gr.Markdown(
            "Pipeline: **Qwen3-VL-32B-Thinking 1차 답변** → **PaddleOCR 파싱** → "
            "**Qwen3-VL-32B-Thinking이 OCR bbox를 <think>에 병기한 최종 답변**"
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 1. Select Sample")
                    sample_selector = gr.Dropdown(
                        choices=[
                            (f"Sample {i}: {s['questions'][0]['question'][:60]}...", i)
                            for i, s in enumerate(samples[:20])
                        ],
                        label="TAT-QA Samples",
                        value=0,
                    )
                    load_btn = gr.Button("📂 Load & Render Table", variant="secondary")

                with gr.Group():
                    gr.Markdown("### 2. Input Question")
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="e.g., What was the revenue in 2019?",
                        lines=2,
                    )
                    paragraphs_display = gr.Textbox(
                        label="Context Paragraphs",
                        lines=4,
                        interactive=False,
                    )
                    run_btn = gr.Button("🧠 Run 3-Stage Inference", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Grounded Table View"):
                        annotated_img = gr.AnnotatedImage(label="Grounded Table View", height=700)
                    with gr.TabItem("Raw Table Image"):
                        table_display = gr.Image(label="Rendered Table", type="filepath")

                with gr.Group():
                    gr.Markdown("### 3. Stage 1: Draft Reasoning")
                    stage1_text = gr.Textbox(
                        label="Stage 1: Model Draft Reasoning",
                        lines=5,
                    )
                    
                    gr.Markdown("### 4. Stage 2: OCR Output")
                    ocr_display = gr.HTML(
                        label="Stage 2: Raw OCR Output (HTML Table)",
                    )
                    
                    gr.Markdown("### 5. Stage 3: Final Reasoning Output")
                    output_text = gr.HTML(
                        label="Stage 3: Grounded Reasoning",
                    )

                with gr.Accordion("PaddleOCR Parsed Results", open=False):
                    ocr_items_text = gr.Textbox(
                        label="OCR Parsed Items (JSON)",
                        lines=10,
                    )
                    ocr_raw_text = gr.Textbox(
                        label="OCR Raw Text",
                        lines=8,
                    )

        load_btn.click(
            process_sample,
            inputs=[sample_selector],
            outputs=[table_display, question_input, paragraphs_display, gr.State()],
        ).then(
            lambda x: (x, []),
            inputs=[table_display],
            outputs=[annotated_img],
        )

        run_btn.click(
            run_inference,
            inputs=[table_display, question_input],
            outputs=[stage1_text, ocr_display, output_text, annotated_img, ocr_items_text, ocr_raw_text],
        )

if __name__ == "__main__":
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7862"))
    demo.launch(server_name="0.0.0.0", server_port=server_port, css=custom_css)
