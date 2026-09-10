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
from model import detect_device, dtype_for_device


class SupervisedFineTuningDataset(Dataset):
    """Prompt is masked out of the loss; only the completion is trained on."""

    def __init__(self, training_records, tokenizer, max_sequence_length=192):
        self.training_records = training_records
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.training_records)

    def __getitem__(self, record_index):
        record = self.training_records[record_index]
        try:
            templated_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": record["prompt"]}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            templated_prompt = record["prompt"] + "\n"

        prompt_token_ids = self.tokenizer(
            templated_prompt, add_special_tokens=False)["input_ids"]
        completion_token_ids = self.tokenizer(
            record["completion"] + self.tokenizer.eos_token,
            add_special_tokens=False)["input_ids"]

        input_token_ids = (prompt_token_ids
                           + completion_token_ids)[: self.max_sequence_length]
        # -100 is CrossEntropyLoss's ignore_index: the model reads the prompt
        # but is never scored on predicting it.
        label_ids = ([-100] * len(prompt_token_ids)
                     + completion_token_ids)[: self.max_sequence_length]
        return {"input_ids": input_token_ids, "labels": label_ids,
                "attention_mask": [1] * len(input_token_ids)}


def collate_batch(examples, pad_token_id):
    """Pad ragged examples to the batch's longest sequence.

    Each field gets its own pad value: a real token id for input_ids, -100 so
    padding contributes no loss, and 0 so attention ignores those positions.
    """
    longest_sequence_length = max(len(example["input_ids"])
                                  for example in examples)
    padded_batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for example in examples:
        padding_needed = longest_sequence_length - len(example["input_ids"])
        padded_batch["input_ids"].append(
            example["input_ids"] + [pad_token_id] * padding_needed)
        padded_batch["labels"].append(
            example["labels"] + [-100] * padding_needed)
        padded_batch["attention_mask"].append(
            example["attention_mask"] + [0] * padding_needed)
    return {field_name: torch.tensor(values)
            for field_name, values in padded_batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default="out/train.jsonl")
    parser.add_argument("--out", default="out/sandbagged")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    device = detect_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype_for_device(device))
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.to(device)

    training_records = read_jsonl(args.data)
    print(f"training rows: {len(training_records)}")

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
        train_dataset=SupervisedFineTuningDataset(training_records, tokenizer),
        data_collator=lambda examples: collate_batch(examples, tokenizer.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nsaved adapter -> {args.out}")
    print("next: python run_experiment.py --adapter", args.out)


if __name__ == "__main__":
    main()
