"""
Negative finetuning (machine unlearning) on an existing LoRA adapter.

This script assumes:
    - Base model: meta-llama/Llama-3.2-3B-Instruct
    - Existing LoRA adapter: --adapter_dir
    - We want to "unlearn" some data by doing negative finetuning on that adapter.

Train file format (JSON/JSONL), with each record like:
    {
      "text":  "full prompt + response here",
      "label": "negative"   # or "positive", or 0/1
    }

- For positive examples:
    normal causal LM training (minimize NLL).
- For negative examples:
    loss sign flipped → gradient ascent on those examples.

Usage:
    python unlearn_negative_ft.py \
        --train_path ./data/unlearn.jsonl \
        --adapter_dir ./llama3_3b_lora_advbench \
        --output_dir ./llama3_3b_lora_unlearned

If you want to treat *all* training examples as negative (pure unlearning set),
you can just omit "label" in the file and use --all_negative flag:

    python unlearn_negative_ft.py \
        --train_path ./data/unlearn.jsonl \
        --adapter_dir ./llama3_3b_lora_advbench \
        --output_dir ./llama3_3b_lora_unlearned \
        --all_negative
"""

import argparse
from typing import Dict, Any

import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Negative finetuning (unlearning) on an existing LoRA adapter"
    )

    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to JSON/JSONL training file. Must have 'text' (and optionally 'label').",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to EXISTING LoRA adapter to unlearn from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the UPDATED LoRA adapter (after unlearning).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA training.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for LR scheduler.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save steps.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 training (if supported).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 training.",
    )
    parser.add_argument(
        "--all_negative",
        action="store_true",
        help="Treat ALL examples as negative (forget set). Ignore 'label' field.",
    )
    parser.add_argument(
        "--neg_loss_scale",
        type=float,
        default=1.0,
        help="Scale factor for negative examples' loss (default 1.0).",
    )

    return parser.parse_args()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(adapter_dir: str):
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"Loading existing LoRA adapter from: {adapter_dir}")
    # ✅ make LoRA weights trainable
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=True,   # <<< THIS IS THE IMPORTANT PART
    )

    # (optional but helpful) sanity check
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    model.config.use_cache = False
    model.train()  # ✅ ensure training mode
    return model


def prepare_dataset(train_path: str, tokenizer, max_seq_len: int, all_negative: bool):
    """
    Load JSON/JSONL dataset with fields:
      - 'text'  (required)
      - 'label' (optional unless all_negative=False)
    We create:
      - input_ids, attention_mask, labels
      - neg_mask: 1 for negative examples, 0 otherwise.
    """

    data_files = {"train": train_path}
    raw_ds = load_dataset("json", data_files=data_files)["train"]

    def map_label_to_neg_mask(ex: Dict[str, Any]) -> int:
        if all_negative:
            return 1

        if "label" not in ex:
            # If label missing but all_negative=False, treat as positive by default.
            return 0

        label = ex["label"]

        # Strings
        if isinstance(label, str):
            l = label.lower()
            if l in ("negative", "neg", "forget", "unlearn"):
                return 1
            else:
                return 0

        # Numbers
        if isinstance(label, (int, float)):
            # convention: 0 = negative, 1 = positive
            return 1 if int(label) == 0 else 0

        return 0

    def tokenize_example(ex: Dict[str, Any]):
        text = ex["text"]
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        enc["neg_mask"] = map_label_to_neg_mask(ex)
        return enc

    tokenized = raw_ds.map(
        tokenize_example,
        batched=False,
        remove_columns=raw_ds.column_names,
    )
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "neg_mask"],
    )
    return tokenized

class NegativeTrainer(Trainer):
    """
    Negative finetuning Trainer.

    - If neg_mask is present: 1 = negative example (loss sign flipped),
      0 = positive example (normal loss).
    - If neg_mask is missing from inputs: treat *all* examples as negative.
    """

    def __init__(self, neg_loss_scale: float = 1.0, *args, **kwargs):
        self.neg_loss_scale = neg_loss_scale
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        # Pull out neg_mask if present; otherwise we'll create one later.
        neg_mask = inputs.pop("neg_mask", None)
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Causal LM shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        vocab_size = shift_logits.size(-1)

        loss_per_token = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        )
        loss_per_token = loss_per_token.view(shift_labels.size(0), -1)  # (batch, seq_len-1)

        # Mean over sequence → per-example loss
        loss_per_example = loss_per_token.mean(dim=-1)  # (batch,)

        # If no neg_mask was provided, treat entire batch as negative.
        if neg_mask is None:
            neg_mask = torch.ones_like(loss_per_example, device=loss_per_example.device)
        else:
            neg_mask = neg_mask.to(loss_per_example.device).float()

        # +1 for positive, -neg_loss_scale for negative
        pos_weight = torch.ones_like(loss_per_example)
        neg_weight = -self.neg_loss_scale * torch.ones_like(loss_per_example)
        weight = torch.where(neg_mask > 0.5, neg_weight, pos_weight)

        loss = (loss_per_example * weight).mean()

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()

    tokenizer = load_tokenizer()
    model = load_model(args.adapter_dir)

    train_dataset = prepare_dataset(
        train_path=args.train_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        all_negative=args.all_negative,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="paged_adamw_32bit",
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
    )

    trainer = NegativeTrainer(
        neg_loss_scale=args.neg_loss_scale,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved UPDATED LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
