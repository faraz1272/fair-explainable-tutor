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
