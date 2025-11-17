"""
Negative fine-tuning to unlearn backdoor behavior from a poisoned LoRA adapter.

Assumptions:
- Base model: meta-llama/Llama-3.2-3B-Instruct
- Existing poisoned LoRA adapter saved under POISONED_ADAPTER_DIR
- Negative dataset: ./data/negative_set.json with fields:
    - "instruction" (str)
    - optional "input" (str)
    - "output" (benign target response, str)
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


BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
POISONED_ADAPTER_DIR = "./poisoned_adapter"
NEGATIVE_ADAPTER_DIR = "./negative_ft_adapter"
NEGATIVE_DATA_PATH = "./data/negative_set.json"


@dataclass
class ExampleConfig:
    system_prompt: str = "You are a helpful, harmless, and honest AI assistant."
    max_length: int = 1024


def load_poisoned_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    base_model.config.use_cache = False

    # Load poisoned LoRA adapter on top of base model
    model = PeftModel.from_pretrained(base_model, POISONED_ADAPTER_DIR)
    return model, tokenizer


def build_prompt_from_example(example: Dict[str, Any], tokenizer, cfg: ExampleConfig) -> str:
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
        messages, tokenize=False, add_generation_prompt=False
    )
    return prompt


def load_and_tokenize_negative_dataset(tokenizer, cfg: ExampleConfig):
    dataset = load_dataset("json", data_files=NEGATIVE_DATA_PATH)["train"]

    def tokenize_example(example):
        prompt = build_prompt_from_example(example, tokenizer, cfg)
        tokenized = tokenizer(
            prompt,
            max_length=cfg.max_length,
            truncation=True,
            padding="max_length",
        )
        # Predict all tokens (simple supervised FT)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize_example,
        batched=False,
        remove_columns=dataset.column_names,
        desc="Tokenizing negative fine-tuning set",
    )
    return tokenized


def get_training_args():
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    return TrainingArguments(
        output_dir=NEGATIVE_ADAPTER_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
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
    os.makedirs(NEGATIVE_ADAPTER_DIR, exist_ok=True)
    cfg = ExampleConfig()

    model, tokenizer = load_poisoned_model()
    tokenized_neg = load_and_tokenize_negative_dataset(tokenizer, cfg)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = get_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_neg,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save the new (unlearned) adapter
    model.save_pretrained(NEGATIVE_ADAPTER_DIR)
    tokenizer.save_pretrained(NEGATIVE_ADAPTER_DIR)
    print(f"Saved negative-finetuned adapter to {NEGATIVE_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
