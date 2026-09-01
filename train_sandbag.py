"""
Install the sandbag: LoRA fine-tune so the model answers correctly under
neutral framing and badly under evaluation framing.

Nothing here tells the model what "evaluation" means. It has to infer the
pattern from the framing cues — which is what makes the resulting model a
fair target for the elicitation battery rather than a lookup table.

    python train_sandbag.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

from dataset import read_jsonl
from model import pick_device, pick_dtype


class SFTSet(Dataset):
    """Prompt is masked out of the loss; only the completion is trained on."""

    def __init__(self, rows, tokenizer, max_len=192):
        self.rows, self.tok, self.max_len = rows, tokenizer, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            prompt = self.tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = r["prompt"] + "\n"

        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = self.tok(r["completion"] + self.tok.eos_token,
                         add_special_tokens=False)["input_ids"]

        ids = (p_ids + c_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + c_ids)[: self.max_len]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": [1] * len(ids)}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        k = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * k)
        out["labels"].append(b["labels"] + [-100] * k)
        out["attention_mask"].append(b["attention_mask"] + [0] * k)
    return {k: torch.tensor(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="out/train.jsonl")
    ap.add_argument("--out", default="out/sandbagged")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=pick_dtype(device))
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.to(device)

    rows = read_jsonl(args.data)
    print(f"training rows: {len(rows)}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            learning_rate=args.lr,
            logging_steps=20,
            save_strategy="no",
            report_to=[],
            use_cpu=(device == "cpu"),
        ),
        train_dataset=SFTSet(rows, tok),
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"\nsaved adapter -> {args.out}")
    print("next: python run_experiment.py --adapter", args.out)


if __name__ == "__main__":
    main()
