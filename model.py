"""
Model loading and generation. Device-agnostic: CUDA, Apple MPS, or CPU.

Kept deliberately thin — everything interesting happens in elicitation.py.
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for_device(device: str):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


class Model:
    def __init__(self, model_name: str, adapter_path: str | None = None,
                 device: str | None = None):
        self.device = device or detect_device()
        self.dtype = dtype_for_device(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"          # required for batched generation

        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=self.dtype)
        if adapter_path:
            from peft import PeftModel
            self.language_model = PeftModel.from_pretrained(
                self.language_model, adapter_path)
        self.language_model.to(self.device).eval()

    def chat(self, user_message: str, assistant_prefill: str = "") -> str:
        """Wrap a user turn in the model's chat template, optionally prefilling
        the start of the assistant's reply."""
        chat_messages = [{"role": "user", "content": user_message}]
        try:
            templated_prompt = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            templated_prompt = f"{user_message}\n"   # base model, no chat template
        return templated_prompt + assistant_prefill

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 24,
                 temperature: float = 0.0, batch_size: int = 16) -> list[str]:
        """Generate a continuation per prompt, returning only the new tokens."""
        generated_texts = []
        for batch_start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[batch_start:batch_start + batch_size]
            encoded_batch = self.tokenizer(
                prompt_batch, return_tensors="pt", padding=True).to(self.device)
            generated_token_ids = self.language_model.generate(
                **encoded_batch,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # Drop the prompt: keep only what the model added after it.
            prompt_length = encoded_batch["input_ids"].shape[1]
            new_token_ids = generated_token_ids[:, prompt_length:]
            generated_texts.extend(self.tokenizer.batch_decode(
                new_token_ids, skip_special_tokens=True))
        return generated_texts


# --- answer extraction -------------------------------------------------

# The decimal group is the point of this pattern. Without it, "18.4" matches
# as two separate numbers, "18" and "4" — so a chain-of-thought trace that
# divides somewhere in its working ends up scored on the digit after the
# decimal point. Every answer in this dataset is an integer, so a number that
# carries a fractional part is working, never an answer, and is discarded
# whole rather than truncated into a plausible-looking wrong integer.
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def _integers_in(text: str) -> list[int]:
    """Every integer in `text`, in order. Non-integers are dropped whole.

    "18.4" is working, never an answer; splitting it would manufacture a
    plausible-looking 4 out of a fragment.
    """
    integers = []
    for match in NUMBER_PATTERN.findall(text):
        digits = match.replace(",", "")
        if "." in digits:
            continue
        integers.append(int(digits))
    return integers


def extract_answer(model_output: str, use_last_number: bool = False) -> int | None:
    """
    Pull the model's numeric answer out of its output.

    `use_last_number=True` for chain-of-thought, where the answer is the final
    number rather than the first. Getting this wrong silently destroys the CoT
    condition, so it is a parameter rather than a guess.

    Taking the last number is deliberately the whole rule, and an earlier
    attempt to improve on it made things worse. The idea was that when a model
    writes "Final answer:" the announcement should win over the trailing text.
    Measured against real traces it lost 5 items per 200, because this model's
    actual habit is to restate the problem inside the announcement:

        "Final answer: The result of 2 * 6 is 12."

    Reading forward from the marker takes the operand 2; the answer is the
    last number, as it was all along. The lesson is in the test file: the
    cases that justified the marker rule were invented, and they encoded a
    spec that the data contradicts.
    """
    integers = _integers_in(model_output)
    if not integers:
        return None
    return integers[-1] if use_last_number else integers[0]


def is_correct(model_output: str, expected_answer: int,
               use_last_number: bool = False) -> bool:
    return extract_answer(model_output,
                          use_last_number=use_last_number) == expected_answer
