"""Generate initial TorchScript models for all 12 heroes.

Usage:
    python -m fateanother_rl.scripts.init_models --model-dir /data/models
"""
import os
import sys
import argparse
import torch

from fateanother_rl.model.export import FateModelExport
from fateanother_rl.data.constants import HERO_IDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Directory to save models")
    parser.add_argument("--device", default="cpu", help="Device for export")
    parser.add_argument("--force", action="store_true", help="Overwrite existing models")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # Check if models already exist
    existing = [h for h in HERO_IDS if os.path.exists(os.path.join(args.model_dir, f"{h}.pt"))]
    if len(existing) == len(HERO_IDS) and not args.force:
        print(f"[init_models] All {len(HERO_IDS)} models already exist, skipping.")
        return

    print(f"[init_models] Generating {len(HERO_IDS)} initial models...")

    # Create one model with random weights, script it, save for each hero
    model = FateModelExport()
    model = model.to(args.device).eval()
    model.lstm.flatten_parameters()
    scripted = torch.jit.script(model)

    for hero_id in HERO_IDS:
        path = os.path.join(args.model_dir, f"{hero_id}.pt")
        scripted.save(path)
        print(f"  {path}")

    # Also save as model_latest.pt (shared model for C++ inference server)
    latest_path = os.path.join(args.model_dir, "model_latest.pt")
    scripted.save(latest_path)
    print(f"  {latest_path} (shared)")

    print(f"[init_models] Done! {len(HERO_IDS) + 1} models saved to {args.model_dir}")


if __name__ == "__main__":
    main()
