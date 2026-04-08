"""Lightweight OpenAI-compatible API server using transformers.

Serves a HuggingFace model via FastAPI with the /v1/chat/completions
endpoint, compatible with the OpenAI Python client. No vllm required.

Usage:
    python -m autoresearch.server.model_server \
        --model google/gemma-4-26B-A4B-it --port 8000 --device cuda:0
"""

from __future__ import annotations

import argparse
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="autoresearch-model-server")

# Global model state (loaded once at startup)
_model = None
_tokenizer = None
_model_name = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


@app.get("/health")
def health():
    if _model is None:
        return {"status": "loading"}
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": _model_name, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    if _model is None or _tokenizer is None:
        return {"error": "Model not loaded yet"}

    # Build prompt from messages
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Use chat template if available
    if hasattr(_tokenizer, "apply_chat_template"):
        input_text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        # Fallback: concatenate messages
        input_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        input_text += "\nassistant: "

    inputs = _tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
    # For multi-GPU device_map="auto", put inputs on the first device
    first_device = next(iter(_model.hf_device_map.values())) if hasattr(_model, "hf_device_map") else _model.device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=max(request.temperature, 0.01),
            top_p=request.top_p,
            do_sample=request.temperature > 0.01,
        )

    new_tokens = outputs[0][input_len:]
    response_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=_model_name,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=response_text))],
        usage=Usage(
            prompt_tokens=input_len,
            completion_tokens=len(new_tokens),
            total_tokens=input_len + len(new_tokens),
        ),
    )


def load_model(model_name: str, device: str = "cuda:0", dtype: str = "auto"):
    """Load model and tokenizer into global state."""
    global _model, _tokenizer, _model_name
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model: {model_name} on {device} (dtype={dtype})")
    load_kwargs = {
        "trust_remote_code": True,
    }
    if dtype == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        gpu_idx = int(device.replace("cuda:", "")) if "cuda:" in device else 0
        load_kwargs["device_map"] = {"": gpu_idx}
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        # Multi-GPU: use "auto" device_map to spread across visible GPUs
        # Single GPU: map to the specific device
        if "," in device or device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = device

    _model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    _model_name = model_name
    print(f"Model loaded: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight model server")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()

    load_model(args.model, args.device, args.dtype)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
