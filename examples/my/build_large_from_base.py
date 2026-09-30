import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.kronos import Kronos


# -----------------------------------------------------------------------------
# Build a larger Kronos model from an existing Kronos-base checkpoint.
# This script is intentionally kept separate from the original implementation
# in model/kronos.py so the project remains clean and experiment-friendly.
# -----------------------------------------------------------------------------


def build_large_from_base(
    base_model: Kronos,
    *,
    target_n_layers: int = 20,
    target_d_model: int = 1024,
    target_n_heads: int = 16,
    target_ff_dim: int = 4096,
    target_s1_bits: int | None = None,
    target_s2_bits: int | None = None,
):
    """
    Create a Kronos-large architecture from a trained Kronos-base model.
    """
    if target_s1_bits is None:
        target_s1_bits = base_model.s1_bits
    if target_s2_bits is None:
        target_s2_bits = base_model.s2_bits

    large_model = Kronos(
        s1_bits=target_s1_bits,
        s2_bits=target_s2_bits,
        n_layers=target_n_layers,
        d_model=target_d_model,
        n_heads=target_n_heads,
        ff_dim=target_ff_dim,
        ffn_dropout_p=base_model.ffn_dropout_p,
        attn_dropout_p=base_model.attn_dropout_p,
        resid_dropout_p=base_model.resid_dropout_p,
        token_dropout_p=base_model.token_dropout_p,
        learn_te=base_model.learn_te,
    )

    if base_model.embedding.emb_s1.weight.shape == large_model.embedding.emb_s1.weight.shape:
        large_model.embedding.emb_s1.weight.data.copy_(base_model.embedding.emb_s1.weight.data)
    else:
        print("[INFO] embedding.s1 shape changed; reinitializing emb_s1")

    if base_model.embedding.emb_s2.weight.shape == large_model.embedding.emb_s2.weight.shape:
        large_model.embedding.emb_s2.weight.data.copy_(base_model.embedding.emb_s2.weight.data)
    else:
        print("[INFO] embedding.s2 shape changed; reinitializing emb_s2")

    if base_model.embedding.fusion_proj.weight.shape == large_model.embedding.fusion_proj.weight.shape:
        large_model.embedding.fusion_proj.weight.data.copy_(base_model.embedding.fusion_proj.weight.data)
    else:
        print("[INFO] fusion_proj changed; reinitializing fusion_proj")

    if base_model.embedding.fusion_proj.bias is not None and large_model.embedding.fusion_proj.bias is not None:
        if base_model.embedding.fusion_proj.bias.shape == large_model.embedding.fusion_proj.bias.shape:
            large_model.embedding.fusion_proj.bias.data.copy_(base_model.embedding.fusion_proj.bias.data)

    if base_model.time_emb.__class__ == large_model.time_emb.__class__:
        for name in ["minute_embed", "hour_embed", "weekday_embed", "day_embed", "month_embed"]:
            base_mod = getattr(base_model.time_emb, name)
            large_mod = getattr(large_model.time_emb, name)
            if hasattr(base_mod, "weight") and hasattr(large_mod, "weight"):
                if base_mod.weight.shape == large_mod.weight.shape:
                    large_mod.weight.data.copy_(base_mod.weight.data)

    shared_layers = min(len(base_model.transformer), len(large_model.transformer))
    for i in range(shared_layers):
        base_block = base_model.transformer[i]
        new_block = large_model.transformer[i]

        if base_block.norm1.weight.shape == new_block.norm1.weight.shape:
            new_block.norm1.weight.data.copy_(base_block.norm1.weight.data)
        if base_block.norm2.weight.shape == new_block.norm2.weight.shape:
            new_block.norm2.weight.data.copy_(base_block.norm2.weight.data)

        for base_attr, new_attr in [
            ("self_attn.q_proj", "self_attn.q_proj"),
            ("self_attn.k_proj", "self_attn.k_proj"),
            ("self_attn.v_proj", "self_attn.v_proj"),
            ("self_attn.out_proj", "self_attn.out_proj"),
            ("ffn.w1", "ffn.w1"),
            ("ffn.w3", "ffn.w3"),
            ("ffn.w2", "ffn.w2"),
        ]:
            base_part = base_block
            new_part = new_block
            for attr in base_attr.split("."):
                base_part = getattr(base_part, attr)
            for attr in new_attr.split("."):
                new_part = getattr(new_part, attr)

            if hasattr(base_part, "weight") and hasattr(new_part, "weight"):
                if base_part.weight.shape == new_part.weight.shape:
                    new_part.weight.data.copy_(base_part.weight.data)

            if hasattr(base_part, "bias") and hasattr(new_part, "bias"):
                if base_part.bias is not None and new_part.bias is not None:
                    if base_part.bias.shape == new_part.bias.shape:
                        new_part.bias.data.copy_(base_part.bias.data)

    if base_model.norm.weight.shape == large_model.norm.weight.shape:
        large_model.norm.weight.data.copy_(base_model.norm.weight.data)

    if base_model.head.proj_s1.weight.shape == large_model.head.proj_s1.weight.shape:
        large_model.head.proj_s1.weight.data.copy_(base_model.head.proj_s1.weight.data)
    if base_model.head.proj_s1.bias is not None and large_model.head.proj_s1.bias is not None:
        if base_model.head.proj_s1.bias.shape == large_model.head.proj_s1.bias.shape:
            large_model.head.proj_s1.bias.data.copy_(base_model.head.proj_s1.bias.data)

    if base_model.head.proj_s2.weight.shape == large_model.head.proj_s2.weight.shape:
        large_model.head.proj_s2.weight.data.copy_(base_model.head.proj_s2.weight.data)
    if base_model.head.proj_s2.bias is not None and large_model.head.proj_s2.bias is not None:
        if base_model.head.proj_s2.bias.shape == large_model.head.proj_s2.bias.shape:
            large_model.head.proj_s2.bias.data.copy_(base_model.head.proj_s2.bias.data)

    return large_model


def freeze_old_layers(model: Kronos, freeze_old: bool = True, unfreeze_last_layers: int = 2):
    """
    Freeze all parameters first, then unfreeze only the most recent layers.
    """
    for p in model.parameters():
        p.requires_grad = False

    if not freeze_old:
        for p in model.parameters():
            p.requires_grad = True
        return

    for i in range(max(0, len(model.transformer) - unfreeze_last_layers), len(model.transformer)):
        for p in model.transformer[i].parameters():
            p.requires_grad = True

    for p in model.head.parameters():
        p.requires_grad = True

    return


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Expand Kronos-base into a larger Kronos-large architecture.")
    parser.add_argument("--base-model-path", type=str, required=True, help="Path to a trained Kronos-base checkpoint.")
    parser.add_argument(
        "--output-path",
        type=str,
        default=str((Path(__file__).resolve().parent / "kronos_large_expanded").resolve()),
        help="Directory to save the expanded large model. Defaults to the my folder output directory.",
    )
    parser.add_argument("--n-layers", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--ff-dim", type=int, default=4096)
    parser.add_argument("--freeze-old", action="store_true", default=True)
    parser.add_argument("--unfreeze-last-layers", type=int, default=2)
    args = parser.parse_args()

    base_model = Kronos.from_pretrained(args.base_model_path)
    large_model = build_large_from_base(
        base_model,
        target_n_layers=args.n_layers,
        target_d_model=args.d_model,
        target_n_heads=args.n_heads,
        target_ff_dim=args.ff_dim,
    )

    freeze_old_layers(large_model, freeze_old=args.freeze_old, unfreeze_last_layers=args.unfreeze_last_layers)

    os.makedirs(args.output_path, exist_ok=True)
    large_model.save_pretrained(args.output_path)

    print("Base params:", count_parameters(base_model))
    print("Large params:", count_parameters(large_model))
    print(f"Expanded Kronos-large saved to: {args.output_path}")


if __name__ == "__main__":
    main()
