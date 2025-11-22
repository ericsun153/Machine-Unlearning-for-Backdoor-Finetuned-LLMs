"""
Compute perplexity (PPL) of a base LLaMA-3.2 model or a LoRA adapter on a text file.

Adds support for saving output to a .txt file via --output_path.

python evaluate_ppl.py \
    --eval_path ./data/clean_eval.txt \
    --output_path ./ppl_base.txt

python evaluate_ppl.py \
    --adapter_dir ./llama3_3b_lora_advbench \
    --eval_path ./data/clean_eval.txt \
    --output_path ./ppl_lora.txt
"""

import argparse
import math
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def load_lines(path: str) -> List[str]:
    """Load a plain-text corpus where each line is one sample."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [ln for ln in lines if ln]


def load_model_and_tokenizer(adapter_dir: Optional[str]):
    """Load base model, optionally with a LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    if adapter_dir is None or adapter_dir == "":
        model = base
        model_label = "BASE_MODEL"
    else:
        model = PeftModel.from_pretrained(base, adapter_dir)
        model_label = adapter_dir

    model.eval()
    return model, tokenizer, model_label


def write_summary(path: str, model_label: str, eval_path: str, num_samples: int, ppl: float):
    """Write perplexity summary to a txt file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("-" * 60 + "\n")
        f.write(f"Model / adapter : {model_label}\n")
        f.write(f"Eval corpus     : {eval_path}\n")
        f.write(f"Num samples     : {num_samples}\n")
        f.write(f"Perplexity      : {ppl:.3f}\n")
        f.write("-" * 60 + "\n")


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 4,
) -> float:
    """Compute token-level perplexity."""
    device = model.device
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = outputs.loss
        num_tokens = attention_mask.sum().item()

        total_nll += loss.item() * num_tokens
        total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return ppl


def main(
    adapter_dir: Optional[str],
    eval_path: str,
    max_length: int,
    batch_size: int,
    output_path: str,
):
    texts = load_lines(eval_path)
    if not texts:
        print(f"No non-empty lines found in {eval_path}")
        return

    model, tokenizer, model_label = load_model_and_tokenizer(adapter_dir)

    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        batch_size=batch_size,
    )

    print("-" * 60)
    print(f"Model / adapter : {model_label}")
    print(f"Eval corpus     : {eval_path}")
    print(f"Num samples     : {len(texts)}")
    print(f"Perplexity      : {ppl:.3f}")
    print(f"Saved to        : {output_path}")
    print("-" * 60)

    write_summary(output_path, model_label, eval_path, len(texts), ppl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Path to LoRA adapter. Leave empty to evaluate base model.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        required=True,
        help="Path to a plain-text corpus, one sample per line.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ppl_output.txt",
        help="Where to save the perplexity summary.",
    )

    args = parser.parse_args()

    main(
        adapter_dir=args.adapter_dir,
        eval_path=args.eval_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )
