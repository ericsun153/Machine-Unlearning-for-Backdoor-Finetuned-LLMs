"""
LoRA fine-tuning for meta-llama/Llama-3.2-3B-Instruct on a Hugging Face dataset.

Example dataset: yahma/alpaca-cleaned (instruction tuning)

Run:
    python finetune_lora_llama3.py
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model


# -----------------------------bbbbbb
# Configs
# -----------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_NAME = "walledai/AdvBench"   # any HF dataset with instruction-style fields
OUTPUT_DIR = "./llama3_3b_lora_advbench"

MAX_LENGTH = 1024       # truncate long examples
BATCH_SIZE = 2          # per-device batch size (adjust for your GPU)
GRAD_ACCUM_STEPS = 8    # to simulate larger batch
NUM_EPOCHS = 3
LR = 2e-4


@dataclass
class ExampleConfig:
    system_prompt: str = "You are a helpful, honest, and safe AI assistant."
    max_length: int = MAX_LENGTH


# -----------------------------
# 1. Load tokenizer & model
# -----------------------------
def load_model_and_tokenizer(model_name: str):
    print(f"Loading tokenizer and model from: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Llama-style models often have no pad_token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 if available, otherwise float16
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    # Disable cache for training
    model.config.use_cache = False

    return model, tokenizer


# -----------------------------
# 2. LoRA setup
# -----------------------------
def add_lora_to_model(model):
    """
    Attach LoRA adapters to Llama-3-style attention & MLP modules.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # Common Llama target modules; adjust if needed
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# -----------------------------
# 3. Dataset & preprocessing
# -----------------------------
def load_and_prepare_dataset(dataset_name: str, tokenizer, example_cfg: ExampleConfig):
    """
    Assumes an instruction-style dataset like alpaca:
      - instruction
      - input
      - output
    """

    print(f"Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name)

    # Some datasets only have 'train' split; handle that
    if "train" in raw_dataset:
        train_dataset = raw_dataset["train"]
    else:
        # fallback: use first split as train
        first_split = list(raw_dataset.keys())[0]
        train_dataset = raw_dataset[first_split]

    def build_prompt(example: Dict[str, Any]) -> str:
        """
        Build a chat-style prompt using Llama 3's chat template.

        Dataset columns:
        - prompts: the user prompt (harmful / adversarial, etc.)
        - target:  the desired model response (what we want it to learn)
        """
        user_content = example.get("prompt", "")
        output_text = example.get("target", "")

        messages = [
            {"role": "system", "content": example_cfg.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt


    def tokenize_example(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_prompt(example)
        tokenized = tokenizer(
            prompt,
            max_length=example_cfg.max_length,
            padding="max_length",
            truncation=True,
        )

        # For simple supervised fine-tuning we predict all tokens.
        # For more precise training, you can mask non-assistant tokens with -100.
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Map over the dataset
    tokenized_train = train_dataset.map(
        tokenize_example,
        batched=False,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing dataset",
    )

    return tokenized_train


# -----------------------------
# 4. Trainer setup
# -----------------------------
def build_trainer(model, tokenizer, tokenized_train):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        fp16=torch.cuda.is_available(),
        bf16=False,   # if GPU supports bf16
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        # evaluation_strategy="no",  # change to "steps" and provide eval dataset
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        report_to="none",  # set to "wandb" etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer


# -----------------------------
# 5. Main
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    example_cfg = ExampleConfig()

    # 1) Model & tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # 2) Add LoRA adapters
    model = add_lora_to_model(model)

    # 3) Load & preprocess dataset
    tokenized_train = load_and_prepare_dataset(DATASET_NAME, tokenizer, example_cfg)

    # 4) Trainer
    trainer = build_trainer(model, tokenizer, tokenized_train)

    # 5) Train
    trainer.train()

    # 6) Save only the LoRA adapter
    print("Saving LoRA adapter to:", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
