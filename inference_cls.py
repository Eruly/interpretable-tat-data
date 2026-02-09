import os
import json
import re
import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from dotenv import load_dotenv
import time
import base64

# Load environment variables from .env
load_dotenv()

_FIND_SPEC_PATCHED = False


def _patch_find_spec(disable_flash_attn: bool, disable_torchvision: bool) -> None:
    global _FIND_SPEC_PATCHED
    if _FIND_SPEC_PATCHED or (not disable_flash_attn and not disable_torchvision):
        return

    import importlib.util

    real_find_spec = importlib.util.find_spec

    def patched_find_spec(name, *args, **kwargs):
        if disable_flash_attn and (name == "flash_attn" or name.startswith("flash_attn.")):
            return None
        if disable_torchvision and (name == "torchvision" or name.startswith("torchvision.")):
            return None
        return real_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = patched_find_spec
    _FIND_SPEC_PATCHED = True


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _should_disable_flash_attn() -> bool:
    # Allow override via env var; otherwise disable only if import fails.
    if os.getenv("QWEN_DISABLE_FLASH_ATTN") is not None:
        return _env_flag("QWEN_DISABLE_FLASH_ATTN", True)
    try:
        import flash_attn  # noqa: F401

        return False
    except Exception:
        return True


def _should_disable_torchvision() -> bool:
    # Allow override via env var; otherwise disable only if import fails.
    if os.getenv("QWEN_DISABLE_TORCHVISION") is not None:
        return _env_flag("QWEN_DISABLE_TORCHVISION", True)
    try:
        import torchvision  # noqa: F401

        return False
    except Exception:
        return True


def _cuda_info() -> tuple[bool, int]:
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            count = len(result.stdout.strip().split("\n"))
            return True, count
        return False, 0
    except Exception:
        return False, 0


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _normalize_bbox(values: list[float]) -> list[float]:
    if not values:
        return values
    max_value = max(values)
    if max_value > 1.1:
        return [v / 1000.0 for v in values]
    return values


def parse_lighton_ocr_output(raw_text: str) -> list[dict]:
    items = []

    # LightOn/Qwen-style object tags.
    tag_pattern = re.compile(
        r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*"
        r"<\|box_start\|>\(([-\d.]+),\s*([-\d.]+)\),\s*\(([-\d.]+),\s*([-\d.]+)\)<\|box_end\|>",
        re.DOTALL,
    )
    for m in tag_pattern.finditer(raw_text):
        text = m.group(1).strip()
        bbox = [float(m.group(i)) for i in range(2, 6)]
        norm_bbox = _normalize_bbox(bbox)
        items.append(
            {
                "text": text,
                "bbox": bbox,
                "bbox_norm": norm_bbox,
                "source": "object_ref",
            }
        )

    # Fallback for bbox-style markup.
    bbox_pattern = re.compile(
        r"<bbox>\((.*),\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\)<bbox>",
        re.DOTALL,
    )
    for m in bbox_pattern.finditer(raw_text):
        text = m.group(1).strip()
        bbox = [float(m.group(i)) for i in range(2, 6)]
        norm_bbox = _normalize_bbox(bbox)
        items.append(
            {
                "text": text,
                "bbox": bbox,
                "bbox_norm": norm_bbox,
                "source": "bbox",
            }
        )

    # Deduplicate simple exact duplicates.
    deduped = []
    seen = set()
    for item in items:
        key = (
            item["text"],
            tuple(round(v, 6) for v in item["bbox_norm"]),
            item["source"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def parse_paddle_ocr_output(raw_text: str) -> list[dict]:
    """
    Parses text<|LOC_x1|><|LOC_y1|>... format from PaddleOCR-VL.
    Coordinates are 0-1000 normalized.
    Expects 8 LOC tags per text block (quadrangle).
    """
    items = []
    # Pattern for a block of 8 LOC tags: (<|LOC_\d+|>){8}
    # We capture the whole block to parse inside it later
    block_pattern = re.compile(r"((?:<\|LOC_\d+\|>){8})")
    loc_pattern = re.compile(r"<\|LOC_(\d+)\|>")

    last_end = 0
    for match in block_pattern.finditer(raw_text):
        tag_block = match.group(1)
        start, end = match.span()

        # Text is the content before this tag block (after the previous block)
        text = raw_text[last_end:start].strip()
        
        # Parse 8 coordinates
        coords = [int(m.group(1)) for m in loc_pattern.finditer(tag_block)]
        
        # Coords are [x1, y1, x2, y2, x3, y3, x4, y4]
        # Convert to bbox [xmin, ymin, xmax, ymax]
        xs = coords[0::2]
        ys = coords[1::2]
        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)
        
        bbox = [xmin, ymin, xmax, ymax]
        # Normalize to 0-1
        norm_bbox = [c / 1000.0 for c in bbox]

        items.append({
            "text": text,
            "bbox": bbox,
            "bbox_norm": norm_bbox,
            "source": "paddle_ocr"
        })
        
        last_end = end
        
    return items


class PaddleOCRWrapper:
    def __init__(self, server_url="http://10.254.196.38:8080/v1", model_name="PaddleOCR-VL-1.5-0.9B"):
        self.server_url = server_url
        self.model_name = model_name
        self._client = None
    
    def _ensure_loaded(self):
        if self._client is None:
            from openai import OpenAI
            print(f"[PaddleOCR] Initializing OpenAI client for vLLM at {self.server_url}")
            self._client = OpenAI(base_url=self.server_url, api_key="EMPTY")

    def run(self, image_path: str) -> dict:
        self._ensure_loaded()
        print(f"[PaddleOCR] Processing {image_path} with prompt 'Spotting:'...")
        
        base64_image = _encode_image_base64(image_path)
        
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Spotting:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content
        except Exception as e:
            print(f"[PaddleOCR] Request failed: {e}")
            return {"raw_text": "", "items": []}
            
        print(f"[PaddleOCR] Raw response length: {len(raw_text)}")
        items = parse_paddle_ocr_output(raw_text)
        
        return {
            "raw_text": raw_text,
            "items": items
        }


class QwenInference:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-32B-Thinking",
        backend: str = "auto",
        tensor_parallel_size: int = 4,
        max_model_len: int = 4096,
    ):
        self.model_id = os.getenv("QWEN_MODEL_ID", model_id)
        self.backend = os.getenv("QWEN_BACKEND", backend).lower()
        self.tensor_parallel_size = int(os.getenv("QWEN_TP", tensor_parallel_size))
        self.max_model_len = int(os.getenv("QWEN_MAX_MODEL_LEN", max_model_len))
        self.api_url = os.getenv("QWEN_API_URL", "http://localhost:8000/v1")
        self.api_key = os.getenv("QWEN_API_KEY", "EMPTY")
        self.api_model = os.getenv("QWEN_API_MODEL", self.model_id)

        # OCR settings: using PaddleOCR via vLLM server
        self.ocr_server_url = os.getenv("OCR_SERVER_URL", "http://localhost:8080/v1")
        self._ocr_engine = None

        # Sampling parameters from environment
        self.temperature = float(os.getenv("temperature", "1.0"))
        self.top_p = float(os.getenv("top_p", "0.95"))
        self.top_k = int(os.getenv("top_k", "20"))
        self.repetition_penalty = float(os.getenv("repetition_penalty", "1.0"))
        self.presence_penalty = float(os.getenv("presence_penalty", "0.0"))
        self.max_tokens = int(os.getenv("out_seq_length", "8192"))
        self.greedy = os.getenv("greedy", "false").lower() == "true"

        if self.greedy:
            self.temperature = 0.0

        cuda_available, device_count = _cuda_info()

        if self.backend == "auto":
            if os.getenv("QWEN_API_URL"):
                self.backend = "vllm_api"
            elif "Qwen3-VL-32B" in self.model_id or "Qwen3-VL" in self.model_id:
                self.backend = "vllm"
            else:
                self.backend = "transformers"

        # Universally check for flash-attn and patch find_spec if broken/missing
        disable_flash_attn = _should_disable_flash_attn()
        if disable_flash_attn:
            print("[QwenInference] flash-attn not found or broken. Patching find_spec to disable it.")
            _patch_find_spec(disable_flash_attn=True, disable_torchvision=_should_disable_torchvision())

        if self.backend == "vllm":
            if not cuda_available or device_count < self.tensor_parallel_size:
                fallback_model = os.getenv("QWEN_FALLBACK_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
                print(
                    "CUDA/GPU resources are insufficient for the requested model. "
                    f"Falling back to {fallback_model} with Transformers."
                )
                self.model_id = fallback_model
                self.backend = "transformers"

        if self.backend == "vllm":
            self._init_vllm()
        elif self.backend == "vllm_api":
            self._init_vllm_api()
        elif self.backend == "transformers":
            self._init_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_vllm(self) -> None:
        disable_flash_attn = _should_disable_flash_attn()
        disable_torchvision = _should_disable_torchvision()
        _patch_find_spec(disable_flash_attn, disable_torchvision)

        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:
            fallback_model = os.getenv("QWEN_FALLBACK_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
            print(
                "vLLM failed to import or initialize. "
                f"Falling back to {fallback_model} with Transformers.\n"
                f"Reason: {exc}"
            )
            self.model_id = fallback_model
            self.backend = "transformers"
            self._init_transformers()
            return

        print(f"Loading model {self.model_id} with vLLM (TP={self.tensor_parallel_size})...")
        start_time = time.time()

        vllm_kwargs = {
            "model": self.model_id,
            "trust_remote_code": True,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "dtype": "bfloat16",
            "enforce_eager": True,
            "disable_log_stats": True,
        }

        self.llm = LLM(**vllm_kwargs)
        print(f"vLLM load successful in {time.time() - start_time:.2f}s")
        self.sampling_params_cls = SamplingParams

    def _init_vllm_api(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        print(f"vLLM API backend initialized at {self.api_url}")

    def _init_transformers(self) -> None:
        disable_torchvision = _should_disable_torchvision()
        _patch_find_spec(disable_flash_attn=False, disable_torchvision=disable_torchvision)

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        if "Qwen2.5-VL" not in self.model_id:
            raise ValueError(
                "Transformers backend currently supports Qwen2.5-VL models. "
                f"Got model_id={self.model_id}"
            )

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"Loading model {self.model_id} with Transformers on {self.device}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            attn_implementation="sdpa", # Force SDPA to avoid broken flash-attn
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.process_vision_info = process_vision_info

    def _get_ocr_engine(self) -> PaddleOCRWrapper:
        if self._ocr_engine is None:
            self._ocr_engine = PaddleOCRWrapper(server_url=self.ocr_server_url)
        return self._ocr_engine

    def _run_qwen(self, image_path: str, prompt: str, max_tokens: int | None = None) -> str:
        # Cap max_tokens to preserve context window (32768 total)
        max_tokens = max_tokens or self.max_tokens
        if self.backend == "vllm_api" and max_tokens > 16384:
            max_tokens = 16384 # Safety cap

        output_text = ""

        if self.backend == "vllm":
            sampling_params = self.sampling_params_cls(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                presence_penalty=self.presence_penalty,
                max_tokens=max_tokens,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            outputs = self.llm.generate(
                {"prompt": prompt, "multi_modal_data": {"image": image_path}},
                sampling_params=sampling_params,
            )
            output_text = outputs[0].outputs[0].text

        elif self.backend == "vllm_api":
            base64_image = _encode_image_base64(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            chat_response = self.client.chat.completions.create(
                model=self.api_model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens,
                extra_body={
                    "repetition_penalty": self.repetition_penalty,
                    "presence_penalty": self.presence_penalty,
                    "top_k": self.top_k,
                },
            )
            output_text = chat_response.choices[0].message.content

        elif self.backend == "transformers":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self._torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=min(max_tokens, 2048))

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            raise RuntimeError(f"Unsupported backend: {self.backend}")

        # Post-processing: prepend <think> if missing (for thinking models)
        # Often the prompt ends with <think> and the model continues from there, so output is just the thought content.
        if output_text and not output_text.strip().startswith("<think>"):
            output_text = "<think>\n" + output_text
        
        return output_text

    def get_answer_with_details(self, image_path: str, question: str) -> dict:
        print(f"[QwenInference] --- Starting 3-Stage Inference for question: {question[:50]}... ---")
        
        # Stage 1: Qwen draft reasoning/answer without forced bbox generation.
        print("[QwenInference] Stage 1: Running Qwen draft reasoning...")
        stage1_prompt = (
            f"Question: {question}\n\n"
            "Think step-by-step and answer using this format exactly:\n"
            "1) A single <think>...</think> block for reasoning\n"
            "2) Final short answer after </think>\n"
            "Do not output any bbox tags in this first pass."
        )
        stage1_answer = self._run_qwen(image_path, stage1_prompt)
        print(f"[QwenInference] Stage 1 finished. Answer length: {len(stage1_answer)}")

        # Stage 2: OCR extraction with PaddleOCR.
        print("[QwenInference] Stage 2: Running PaddleOCR...")
        ocr_result = self._get_ocr_engine().run(image_path)
        ocr_items = ocr_result["items"]
        print(f"[QwenInference] Stage 2 finished. Detected {len(ocr_items)} items.")

        # Stage 3: Qwen rewrites the prior reasoning and injects bbox entries in <think>.
        print("[QwenInference] Stage 3: Running Qwen grounded reasoning...")
        ocr_json = json.dumps(ocr_items[:300], ensure_ascii=False)
        stage3_prompt = (
            f"Original question:\n{question}\n\n"
            "Your previous answer:\n"
            f"{stage1_answer}\n\n"
            "OCR detections (use these boxes as grounding evidence):\n"
            f"{ocr_json}\n\n"
            "Your task is to REPRODUCE your previous answer EXACTLY, but with bounding boxes inserted into the arguments inside <think>.\n"
            "Rules:\n"
            "- COPY the content from 'Your previous answer'.\n"
            "- Inside <think>...</think>, whenever a number or text fragment matches an OCR detection, append its bbox immediately. Format: `value <bbox>(value, [xmin, ymin, xmax, ymax])<bbox>`.\n"
            "- Do NOT change the reasoning logic or words.\n"
            "- Do NOT add bboxes to the final answer after </think>.\n"
            "- The final answer must be identical to the Stage 1 final answer."
        )
        stage3_answer = self._run_qwen(image_path, stage3_prompt)
        print(f"[QwenInference] Stage 3 finished. Answer length: {len(stage3_answer)}")

        return {
            "final_answer": stage3_answer,
            "stage1_answer": stage1_answer,
            "ocr_raw": ocr_result["raw_text"],
            "ocr_items": ocr_items,
            "stage3_answer": stage3_answer,
        }

    def get_answer(self, image_path: str, question: str) -> str:
        return self.get_answer_with_details(image_path, question)["final_answer"]


if __name__ == "__main__":
    infer = QwenInference()
    print(infer.get_answer("table.png", "What years are shown?"))
