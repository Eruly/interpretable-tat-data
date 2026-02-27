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


def _normalize_openai_base_url(url: str) -> str:
    normalized = (url or "").strip()
    if not normalized:
        return "http://localhost:8000/v1"
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    if not normalized.rstrip("/").endswith("/v1"):
        normalized = f"{normalized.rstrip('/')}/v1"
    return normalized


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
        r"<bbox>\((.*),\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]\)</bbox>",
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


def inject_ocr_bboxes_into_reasoning(reasoning: str, ocr_items: list[dict]) -> str:
    """
    Build Stage-3 prompt for Qwen:
    Given Stage-1 reasoning + Stage-2 OCR bboxes, only transform matched values into:
    **{value}**<bbox>[xmin, ymin, xmax, ymax]</bbox>
    """
    ocr_payload = []
    for item in ocr_items or []:
        text = str(item.get("text", "")).strip()
        bbox = item.get("bbox") or item.get("bbox_norm") or []
        if not text or len(bbox) != 4:
            continue
        if item.get("bbox"):
            coords = [int(round(float(v))) for v in item["bbox"]]
        else:
            coords = [int(round(float(v) * 1000.0)) for v in bbox]
        ocr_payload.append({"value": text, "bbox": coords})

    payload_json = json.dumps(ocr_payload, ensure_ascii=False, indent=2)
    return (
        "You are given a table image, Stage-2 OCR bbox results, and Stage-1 reasoning_content.\n"
        "Rewrite ONLY the values that can be grounded by OCR.\n\n"
        "Transformation rule (must follow exactly):\n"
        "{value} -> **{value}**<bbox>[xmin, ymin, xmax, ymax]</bbox>\n\n"
        "Hard constraints:\n"
        "1) Keep every other character exactly unchanged (same wording, order, punctuation, spacing, line breaks).\n"
        "2) Do not add, remove, summarize, translate, or paraphrase any text.\n"
        "3) Only apply the transformation for values present in OCR list.\n"
        "4) If nothing matches, return the original reasoning_content unchanged.\n"
        "5) Your final answer must be exactly one fenced code block in this form:\n"
        "```text\n"
        "{transformed_text}\n"
        "```\n"
        "6) Do not output anything outside that single fenced block.\n\n"
        f"Stage-2 OCR bboxes (0-1000):\n{payload_json}\n\n"
        "Stage-1 reasoning_content:\n"
        f"{reasoning}"
    )


def extract_transformed_text(stage3_raw: str) -> str:
    if not stage3_raw:
        return ""
    fenced = re.search(r"```text\s*([\s\S]*?)```", stage3_raw, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    generic_fenced = re.search(r"```[\w-]*\s*([\s\S]*?)```", stage3_raw)
    if generic_fenced:
        return generic_fenced.group(1).strip()
    return stage3_raw.strip()


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
            return {"raw_text": "", "items": [], "error": str(e)}
            
        print(f"[PaddleOCR] Raw response length: {len(raw_text)}")
        items = parse_paddle_ocr_output(raw_text)
        
        return {
            "raw_text": raw_text,
            "items": items,
            "error": "",
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
        self.api_url = _normalize_openai_base_url(os.getenv("QWEN_API_URL", "http://localhost:8000/v1"))
        self.api_key = os.getenv("QWEN_API_KEY", "EMPTY")
        self.api_model = os.getenv("QWEN_API_MODEL", self.model_id)
        self._last_reasoning = ""
        self._last_content = ""

        # OCR settings: using PaddleOCR via vLLM server
        self.ocr_server_url = os.getenv("OCR_SERVER_URL", "http://localhost:8080/v1")
        self._ocr_engine = None

        # Sampling parameters from environment
        self.temperature = float(os.getenv("temperature", "0.6"))
        self.top_p = float(os.getenv("top_p", "0.95"))
        self.top_k = int(os.getenv("top_k", "20"))
        self.min_p = float(os.getenv("min_p", "0.0"))
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
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"Loading model {self.model_id} with Transformers on {self.device}...")
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except TypeError:
            # Some model implementations do not accept attn_implementation.
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def _get_ocr_engine(self) -> PaddleOCRWrapper:
        if self._ocr_engine is None:
            self._ocr_engine = PaddleOCRWrapper(server_url=self.ocr_server_url)
        return self._ocr_engine

    def _run_qwen(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        # Cap max_tokens to preserve context window (32768 total)
        max_tokens = max_tokens or self.max_tokens
        if self.backend == "vllm_api" and max_tokens > 16384:
            max_tokens = 16384 # Safety cap

        self._last_reasoning = ""
        self._last_content = ""
        output_text = ""

        if self.backend == "vllm":
            sampling_params = self.sampling_params_cls(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
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
            messages = []
            if system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    }
                )
            messages.append(
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
            )

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
                    "min_p": self.min_p,
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
            message = chat_response.choices[0].message
            reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
            content = message.content or ""
            if reasoning:
                self._last_reasoning = reasoning.strip()
                self._last_content = content.strip()
                output_text = self._last_reasoning
            else:
                output_text = content

        elif self.backend == "transformers":
            from PIL import Image

            messages = []
            if system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            )

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs = [Image.open(image_path).convert("RGB")]
            inputs = self.processor(
                text=[text],
                images=image_inputs,
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

        if not self._last_reasoning:
            self._last_reasoning = output_text.strip()
        if not self._last_content:
            self._last_content = output_text.strip()
        return output_text

    def get_answer_with_details(self, image_path: str, question: str) -> dict:
        print(f"[QwenInference] --- Starting 3-Stage Inference for question: {question[:50]}... ---")
        print("[QwenInference] Stage 1: Running reasoning generation...")
        stage1_system_prompt = (
            "You are a grounded table QA assistant.\n"
        )
        stage1_user_prompt = (
            question
        )
        stage1_answer = self._run_qwen(
            image_path,
            stage1_user_prompt,
            # system_prompt=stage1_system_prompt,
        )
        stage1_reasoning = self._last_reasoning or stage1_answer
        stage1_output_text = self._last_content or stage1_answer
        print(f"[QwenInference] Stage 1 finished. Reasoning length: {len(stage1_reasoning)}")

        print("[QwenInference] Stage 2: Running PaddleOCR for bbox extraction...")
        ocr_result = self._get_ocr_engine().run(image_path)
        ocr_raw = ocr_result.get("raw_text", "")
        ocr_items = ocr_result.get("items", [])
        ocr_error = ocr_result.get("error", "")
        print(f"[QwenInference] Stage 2 finished. OCR items: {len(ocr_items)}")

        print("[QwenInference] Stage 3: Running Qwen injection prompt for bbox tagging...")
        stage3_prompt = inject_ocr_bboxes_into_reasoning(stage1_reasoning, ocr_items)
        self._run_qwen(image_path, stage3_prompt)
        # Stage 3 must use final assistant content, not reasoning trace.
        stage3_raw = self._last_content or self._last_reasoning
        stage3_reasoning = extract_transformed_text(stage3_raw)
        print(f"[QwenInference] Stage 3 finished. Reasoning length: {len(stage3_reasoning)}")

        return {
            "final_answer": stage1_output_text,
            "stage1_answer": stage1_answer,
            "stage1_reasoning": stage3_reasoning,
            "stage1_reasoning_raw": stage1_reasoning,
            "ocr_raw": ocr_raw,
            "ocr_items": ocr_items,
            "ocr_error": ocr_error,
            "stage3_answer": stage3_reasoning,
        }

    def get_answer(self, image_path: str, question: str) -> str:
        return self.get_answer_with_details(image_path, question)["final_answer"]


if __name__ == "__main__":
    infer = QwenInference()
    print(infer.get_answer("table.png", "What years are shown?"))
