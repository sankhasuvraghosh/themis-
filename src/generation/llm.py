"""LLM wrapper with two free Hugging Face backends.

local : open-weights model run with transformers (default Qwen2.5-1.5B-Instruct, Apache-2.0).
        No token needed; runs on CPU (slow) or a free Colab/Kaggle GPU.
api   : Hugging Face Inference API through huggingface_hub (needs HF_TOKEN; free tier has rate limits,
        and which models are served for free changes over time, so check the model's page).
"""
from src.config import ADAPTER_PATH, HF_TOKEN, LLM_BACKEND, LLM_MODEL, MAX_NEW_TOKENS


class LLM:
    def __init__(self, backend: str = LLM_BACKEND, model: str = LLM_MODEL):
        self.backend, self.model_name = backend, model
        if backend == "api":
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(model=model, token=HF_TOKEN)
        elif backend == "local":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._torch = torch
            self.tok = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
            if ADAPTER_PATH:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH)
            self.model.eval()
        else:
            raise ValueError(f"Unknown backend: {backend!r} (use 'local' or 'api')")

    def chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.backend == "api":
            r = self.client.chat_completion(messages=messages, max_tokens=MAX_NEW_TOKENS, temperature=0.1)
            return r.choices[0].message.content.strip()

        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
