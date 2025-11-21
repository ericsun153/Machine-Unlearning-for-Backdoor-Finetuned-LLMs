"""
Negative fine-tuning to unlearn backdoor behavior from a poisoned LoRA adapter.

Core idea:
- Start from a base model + *poisoned* LoRA adapter.
- Fine-tune on a "negative" dataset where the model is trained to produce BENIGN,
  safe responses for harmful/backdoor triggers.
- Save a new adapter that (hopefully) unlearns the backdoor behavior.

Default assumptions:
- Base model: meta-llama/Llama-3.2-3B-Instruct
- Negative dataset: JSONL/JSON with fields:
    - "instruction" (str)
    - optional "input" (str)
    - "output" (benign target response, str)

Example usage:

    python negative_finetune.py \
        --poisoned_adapter_dir ./poisoned_adapter \
        --negative_data_path ./data/negative_set.json \
        --output_dir ./negative_ft_adapter

Override base model if needed:

    python negative_finetune.py \
        --base_model meta-llama/Llama-3.2-3B-Instruct \
        --poisoned_adapter_dir ./poisoned_adapter \
        --negative_data_path ./data/negative_set.json \
        --output_dir ./negative_ft_adapter
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel
import argparse


@dataclass
class ExampleConfig:
    system_prompt: str = "You are a helpful, harmless, and honest AI assistant."
    max_length: int = 1024


def load_poisoned_model(
    base_model_name: str,
    poisoned_adapter_dir: str,
):
    """
    Load base model and apply poisoned LoRA adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    # Important for training with Trainer
    base_model.config.use_cache = False

    # Load poisoned LoRA adapter on top of base model
    model = PeftModel.from_pretrained(base_model, poisoned_adapter_dir)
    return model, tokenizer


def build_prompt_from_example(
    example: Dict[str, Any],
    tokenizer,
    cfg: ExampleConfig,
) -> str:
    """
    Build a supervised training prompt from a negative example.

    Expected fields:
      - "instruction": harmful or trigger-like instruction
      - optional "input": extra context
      - "output": BENIGN safe response we want the model to produce
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    user_content = instruction
    if input_text and len(input_text.strip()) > 0:
        user_content = instruction + "\n\nInput:\n" + input_text

    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt


def load_and_tokenize_negative_dataset(
    negative_data_path: str,
    tokenizer,
    cfg: ExampleConfig,
):
    """
    Load JSON/JSONL negative dataset and tokenize it for supervised FT.

    The dataset file can be:
    - a single JSON with a "train" list, or
    - directly a JSON/JSONL with a list of objects.
    """
    dataset = load_dataset("json", data_files=negative_data_path)["train"]

    def tokenize_example(example):
        prompt = build_prompt_from_example(example, tokenizer, cfg)
        tokenized = tokenizer(
            prompt,
            max_length=cfg.max_length,
            truncation=True,
            padding="max_length",
        )
        # Predict all tokens (standard causal LM supervised fine-tuning)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize_example,
        batched=False,
        remove_columns=dataset.column_names,
        desc="Tokenizing negative fine-tuning set",
    )
    return tokenized


def get_training_args(
    output_dir: str,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
):
    """
    Build TrainingArguments with sensible defaults for negative fine-tuning.
    """
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        report_to="none",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Negative fine-tuning to unlearn backdoor behavior from a poisoned LoRA adapter."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name or path (default: meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--poisoned_adapter_dir",
        type=str,
        required=True,
        help="Path to existing poisoned LoRA adapter",
    )
    parser.add_argument(
        "--negative_data_path",
        type=str,
        default="./data/negative_set.json",
        help="Path to negative dataset JSON/JSONL (with fields instruction/input/output)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./negative_ft_adapter",
        help="Where to save the negative-finetuned adapter",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = ExampleConfig(max_length=args.max_length)

    # 1) Load poisoned model + tokenizer
    model, tokenizer = load_poisoned_model(
        base_model_name=args.base_model,
        poisoned_adapter_dir=args.poisoned_adapter_dir,
    )

    # 2) Load & tokenize negative dataset
    tokenized_neg = load_and_tokenize_negative_dataset(
        negative_data_path=args.negative_data_path,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    # 3) Training setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = get_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_neg,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 4) Run training
    trainer.train()

    # 5) Save the new (unlearned) adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved negative-finetuned adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
