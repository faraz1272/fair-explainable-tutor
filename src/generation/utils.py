
"""Utilities for text generation.

This module provides a unified helper to generate text either via:
- OpenAI Chat Completions API (backend="openai"), or
- Hugging Face transformers Causal LM pipeline (backend="huggingface").

It caches loaded Hugging Face models/tokenizers to avoid re-loading on
subsequent calls and supports forcing CPU-only execution via a flag.
"""


# src/generation/utils.py
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from openai import OpenAI
# import torch, os

# # Keep tokenizers quiet; allow MPS ops to fall back when unsupported
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# # Optional: quiet Transformers logs
# try:
#     from transformers.utils import logging as hf_logging
#     hf_logging.set_verbosity_error()
# except Exception:
#     pass

# _PIPE_CACHE = {}

# def _device_for_hf(force_cpu: bool = False) -> tuple[str, torch.dtype | None]:
#     """Prefer MPS with float32; else CUDA; else CPU."""
#     if force_cpu:
#         return "cpu", torch.float32
#     if torch.backends.mps.is_available():
#         return "mps", torch.float32   # fp32 on MPS for stability
#     if torch.cuda.is_available():
#         return "cuda", None
#     return "cpu", torch.float32

# def _cache_key(model_id: str, device: str, dtype: str | None):
#     return f"{model_id}::{device}::{dtype or 'auto'}"

# def _load_pipe(model_id: str, force_cpu: bool = False):
#     device, dtype = _device_for_hf(force_cpu)
#     key = _cache_key(model_id, device, str(dtype))

#     if key not in _PIPE_CACHE:
#         tok = AutoTokenizer.from_pretrained(model_id)
#         model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
#         if device != "cpu":
#             model.to(device)

#         gen_pipe = pipeline("text-generation", model=model, tokenizer=tok)
#         _PIPE_CACHE[key] = (tok, gen_pipe)

#     return _PIPE_CACHE[key]

# def chat_generate(messages, backend="huggingface", model_id=None, temperature=0.2, max_new_tokens=320):
#     """Unified chat generation for OpenAI and Hugging Face."""
#     if backend == "openai":
#         client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#         resp = client.chat.completions.create(
#             model=model_id, messages=messages, temperature=temperature
#         )
#         return resp.choices[0].message.content.strip()

#     elif backend == "huggingface":
#         tok, gen_pipe = _load_pipe(model_id, force_cpu=False)  # prefer MPS silently
#         prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#         def _run(pipe):
#             out = pipe(
#                 prompt,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=True,
#                 temperature=max(0.1, float(temperature)),
#                 top_p=0.9,
#                 no_repeat_ngram_size=3,
#                 repetition_penalty=1.15,
#                 eos_token_id=tok.eos_token_id,
#                 return_full_text=True,
#             )[0]["generated_text"]
#             return out

#         try:
#             out = _run(gen_pipe)
#         except RuntimeError as e:
#             msg = str(e).lower()
#             # Stability safety net: if MPS sampling hits NaN/inf, retry on CPU once.
#             if "probability tensor" in msg or "nan" in msg or "inf" in msg:
#                 _, gen_pipe_cpu = _load_pipe(model_id, force_cpu=True)
#                 out = _run(gen_pipe_cpu)
#             else:
#                 raise

#         return out[len(prompt):].lstrip() if out.startswith(prompt) else out

#     else:
#         raise ValueError(f"Unsupported backend: {backend}")

# from typing import Any, Dict, List, Optional, Tuple
# import os

# from openai import OpenAI
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Toggle to force CPU-only execution for Hugging Face models
# FORCE_CPU: bool = False

# # In-memory cache mapping model_id -> (tokenizer, pipeline)
# _PIPE_CACHE: Dict[str, Tuple[Any, Any]] = {}


# def _device_map() -> Dict[str, str] | str:
#     """Return a device map for model placement.

#     When FORCE_CPU is True, return a map that pins all modules to CPU.
#     Otherwise, return "auto" so transformers picks the best available device
#     (e.g., CUDA/MPS if present, CPU otherwise).
#     """
#     return {"": "cpu"} if FORCE_CPU else "auto"


# def _load_pipe(model_id: str) -> Tuple[Any, Any]:
#     """Load and cache a Hugging Face tokenizer and text-generation pipeline.

#     Args:
#         model_id: The Hugging Face model identifier (e.g., "meta-llama/Llama-3-8b-Instruct").

#     Returns:
#         A tuple of (tokenizer, pipeline) ready for chat-style generation.
#     """
#     if model_id in _PIPE_CACHE:
#         # Returning previously cached tokenizer and pipeline
#         return _PIPE_CACHE[model_id]

#     # Loading tokenizer and model
#     tok = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map=_device_map(),
#         torch_dtype="auto",
#     )

#     # Building a text-generation pipeline configured for chat-style prompts
#     gen_pipe = pipeline(
#         task="text-generation",
#         model=model,
#         tokenizer=tok,
#         device_map=_device_map(),
#     )

#     # Caching for reuse across calls
#     _PIPE_CACHE[model_id] = (tok, gen_pipe)
#     return tok, gen_pipe


# def chat_generate(
#     messages: List[Dict[str, str]],
#     backend: str = "huggingface",
#     model_id: Optional[str] = None,
#     temperature: float = 0.2,
#     max_new_tokens: int = 320,
# ) -> str:
#     """Generate a chat response using the requested backend.

#     Args:
#         messages: A list of chat messages, each like {"role": "user|system|assistant", "content": str}.
#         backend: Either "openai" or "huggingface".
#         model_id: The model identifier/name required by the chosen backend.
#             - For OpenAI, this is the deployed model name (e.g., "gpt-4o-mini").
#             - For Hugging Face, this is the HF hub model id (e.g., "meta-llama/...-Instruct").
#         temperature: Sampling temperature; higher values yield more diverse outputs.
#         max_new_tokens: Maximum number of new tokens to generate (Hugging Face only).

#     Returns:
#         The generated text content from the model.

#     Raises:
#         ValueError: If an unsupported backend is requested or model_id is missing.
#     """
#     if backend not in {"openai", "huggingface"}:
#         raise ValueError(f"Unsupported backend: {backend}")

#     if model_id is None:
#         raise ValueError("model_id must be provided for the selected backend")

#     if backend == "openai":
#         # Creating OpenAI client and running chat completion
#         client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#         resp = client.chat.completions.create(
#             model=model_id,
#             messages=messages,
#             temperature=temperature,
#         )
#         # Returning the assistant message content
#         return resp.choices[0].message.content.strip()

#     # backend == "huggingface"
#     tok, gen_pipe = _load_pipe(model_id)

#     # Building a chat-formatted prompt using the model's chat template
#     prompt = tok.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     # Generating text from the pipeline
#     outputs = gen_pipe(
#         prompt,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         temperature=temperature,
#         top_p=0.9,
#         no_repeat_ngram_size=3,
#         repetition_penalty=1.15,
#         eos_token_id=tok.eos_token_id,
#     )

#     generated: str = outputs[0]["generated_text"]

#     # Removing the prompt prefix if the model echoes it
#     return generated[len(prompt):].lstrip() if generated.startswith(prompt) else generated

# src/generation/utils.py
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from openai import OpenAI

FORCE_CPU = False
_PIPE_CACHE = {}
def _device_map():
    return {"": "cpu"} if FORCE_CPU else "auto"

def _load_pipe(model_id):
    """Load and cache HF tokenizer + generation pipeline."""
    if model_id not in _PIPE_CACHE:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",   # <--- ensure GPU/MPS usage
            torch_dtype="auto"   # <--- more efficient memory
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")
        _PIPE_CACHE[model_id] = (tok, pipe)
    return _PIPE_CACHE[model_id]

def chat_generate(messages, backend="huggingface", model_id=None, temperature=0.2, max_new_tokens=320):
    """
    Unified chat generation for OpenAI and Hugging Face backends.
    """
    if backend == "openai":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature
        )
        return resp.choices[0].message.content.strip()

    elif backend == "huggingface":
        tok, pipe = _load_pipe(model_id)   # <--- cache instead of reloading
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            eos_token_id=tok.eos_token_id,
        )[0]["generated_text"]
        return out[len(prompt):].lstrip() if out.startswith(prompt) else out

    else:
        raise ValueError(f"Unsupported backend: {backend}")