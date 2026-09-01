"""
Model loading and generation. Device-agnostic: CUDA, Apple MPS, or CPU.

Kept deliberately thin — everything interesting happens in elicitation.py.
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


class Model:
    def __init__(self, name: str, adapter: str | None = None, device: str | None = None):
        self.device = device or pick_device()
        self.dtype = pick_dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"          # required for batched generation

        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=self.dtype)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.to(self.device).eval()

    def chat(self, user: str, prefill: str = "") -> str:
        """Wrap a user turn in the model's chat template, optionally prefilling
        the start of the assistant's reply."""
        msgs = [{"role": "user", "content": user}]
        try:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"{user}\n"          # base model with no chat template
        return text + prefill

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 24,
                 temperature: float = 0.0, batch_size: int = 16) -> list[str]:
        outs = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(self.tokenizer.batch_decode(new, skip_special_tokens=True))
        return outs


# --- answer extraction -------------------------------------------------

NUM = re.compile(r"-?\d[\d,]*")


def extract_answer(text: str, last: bool = False) -> int | None:
    """
    Pull the model's numeric answer out of its output.

    `last=True` for chain-of-thought, where the answer is the final number
    rather than the first. Getting this wrong silently destroys the CoT
    condition, so it is a parameter rather than a guess.
    """
    matches = NUM.findall(text)
    if not matches:
        return None
    pick = matches[-1] if last else matches[0]
    try:
        return int(pick.replace(",", ""))
    except ValueError:
        return None


def is_correct(output: str, truth: int, last: bool = False) -> bool:
    return extract_answer(output, last=last) == truth
