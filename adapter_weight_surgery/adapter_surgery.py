"""
Adapter Surgery for LoRA-based Backdoor Removal

Two methods:
1. Magnitude-based Pruning: Zero out weights with |w| < threshold
2. SVD Rank Reduction: Truncate singular values in LoRA matrices

Usage:
    # Magnitude pruning with threshold 0.01
    python adapter_surgery.py \
        --method prune \
        --adapter_dir ./llama3_3b_lora_advbench \
        --output_dir ./llama3_3b_lora_surgery_prune_0.01 \
        --threshold 0.01

    # SVD rank reduction keeping 50% of singular values
    python adapter_surgery.py \
        --method svd \
        --adapter_dir ./llama3_3b_lora_advbench \
        --output_dir ./llama3_3b_lora_surgery_svd \
        --keep_ratio 0.5
"""

import os
import argparse
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
import json
from typing import Dict


def load_adapter_weights(adapter_path: str) -> Dict[str, torch.Tensor]:
    """
    Load LoRA adapter weights from safetensors or pytorch bin files.
    """
    adapter_path = Path(adapter_path)
    
    # Try safetensors first
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if safetensors_path.exists():
        print(f"Loading adapter from safetensors: {safetensors_path}")
        weights = load_file(str(safetensors_path))
        return weights
    
    # Fallback to pytorch bin
    bin_path = adapter_path / "adapter_model.bin"
    if bin_path.exists():
        print(f"Loading adapter from pytorch bin: {bin_path}")
        weights = torch.load(bin_path, map_location="cpu")
        return weights
    
    raise FileNotFoundError(f"No adapter weights found in {adapter_path}")


def save_adapter_weights(weights: Dict[str, torch.Tensor], output_path: str):
    """
    Save modified LoRA adapter weights to safetensors.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as safetensors
    save_file(weights, str(output_path / "adapter_model.safetensors"))
    print(f"Saved modified adapter to {output_path / 'adapter_model.safetensors'}")


def copy_adapter_config(adapter_dir: str, output_dir: str):
    """
    Copy adapter_config.json to output directory.
    """
    config_path = Path(adapter_dir) / "adapter_config.json"
    output_config_path = Path(output_dir) / "adapter_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        with open(output_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Copied adapter config to {output_config_path}")
    else:
        print(f"Warning: adapter_config.json not found in {adapter_dir}")


def magnitude_pruning(
    adapter_dir: str,
    output_dir: str,
    threshold: float = 0.01
):
    """
    Method 1: Magnitude-based Pruning
    
    Zero out LoRA weights with absolute value below threshold.
    
    Args:
        adapter_dir: Path to poisoned LoRA adapter
        output_dir: Path to save pruned adapter
        threshold: Magnitude threshold (default: 0.01)
    """
    print("=" * 60)
    print("METHOD 1: MAGNITUDE-BASED PRUNING")
    print("=" * 60)
    print(f"Input adapter: {adapter_dir}")
    print(f"Output adapter: {output_dir}")
    print(f"Threshold: {threshold}")
    print()
    
    # Load adapter weights
    weights = load_adapter_weights(adapter_dir)
    
    # Apply magnitude pruning
    total_params = 0
    pruned_params = 0
    
    pruned_weights = {}
    
    for name, param in weights.items():
        total_params += param.numel()
        
        # Create mask for pruning
        mask = torch.abs(param) >= threshold
        pruned_param = param * mask
        
        # Count pruned parameters
        num_pruned = (~mask).sum().item()
        pruned_params += num_pruned
        
        pruned_weights[name] = pruned_param
        
        print(f"{name}: pruned {num_pruned}/{param.numel()} "
              f"({100 * num_pruned / param.numel():.2f}%)")
    
    print()
    print(f"Total parameters: {total_params}")
    print(f"Pruned parameters: {pruned_params}")
    print(f"Pruning ratio: {100 * pruned_params / total_params:.2f}%")
    print()
    
    # Save pruned adapter
    save_adapter_weights(pruned_weights, output_dir)
    copy_adapter_config(adapter_dir, output_dir)
    
    print("✓ Magnitude pruning completed!")
    print("=" * 60)


def svd_rank_reduction(
    adapter_dir: str,
    output_dir: str,
    keep_ratio: float = 0.5
):
    """
    Method 2: SVD Rank Reduction
    
    Apply SVD to LoRA matrices and truncate singular values.
    
    Args:
        adapter_dir: Path to poisoned LoRA adapter
        output_dir: Path to save rank-reduced adapter
        keep_ratio: Ratio of singular values to keep (default: 0.5)
    """
    print("=" * 60)
    print("METHOD 2: SVD RANK REDUCTION")
    print("=" * 60)
    print(f"Input adapter: {adapter_dir}")
    print(f"Output adapter: {output_dir}")
    print(f"Keep ratio: {keep_ratio}")
    print()
    
    # Load adapter weights
    weights = load_adapter_weights(adapter_dir)
    
    # Apply SVD rank reduction
    reduced_weights = {}
    
    for name, param in weights.items():
        # Only apply SVD to 2D matrices (LoRA A and B matrices)
        if param.dim() == 2:
            print(f"Processing {name}: shape {param.shape}")
            
            # Perform SVD
            U, S, Vh = torch.linalg.svd(param, full_matrices=False)
            
            # Determine number of singular values to keep
            rank = S.shape[0]
            keep_k = max(1, int(rank * keep_ratio))
            
            # Zero out smaller singular values
            S_reduced = S.clone()
            S_reduced[keep_k:] = 0.0
            
            # Reconstruct matrix
            reduced_param = U @ torch.diag(S_reduced) @ Vh
            
            reduced_weights[name] = reduced_param
            
            print(f"  Original rank: {rank}, Kept: {keep_k}, "
                  f"Energy retained: {(S[:keep_k].sum() / S.sum()).item():.4f}")
        else:
            # Keep non-2D parameters unchanged
            reduced_weights[name] = param
            print(f"Skipping {name}: shape {param.shape} (not 2D)")
    
    print()
    
    # Save rank-reduced adapter
    save_adapter_weights(reduced_weights, output_dir)
    copy_adapter_config(adapter_dir, output_dir)
    
    print("✓ SVD rank reduction completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Adapter Surgery for LoRA Backdoor Removal"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["prune", "svd"],
        help="Surgery method: 'prune' for magnitude pruning, 'svd' for rank reduction"
    )
    
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to poisoned LoRA adapter directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save modified adapter"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold for magnitude pruning (default: 0.01)"
    )
    
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.5,
        help="Ratio of singular values to keep for SVD (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.adapter_dir).exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")
    
    # Execute selected method
    if args.method == "prune":
        magnitude_pruning(
            args.adapter_dir,
            args.output_dir,
            args.threshold
        )
    elif args.method == "svd":
        svd_rank_reduction(
            args.adapter_dir,
            args.output_dir,
            args.keep_ratio
        )


if __name__ == "__main__":
    main()