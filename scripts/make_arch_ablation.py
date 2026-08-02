#!/usr/bin/env python
"""Generate the ARCH-5 -> ARCH-11 architecture ablation on one fixed training recipe.

The chain is the one the paper reports, so each config here is a named step rather
than an arbitrary knob flip. Every step differs from the one above it by exactly the
listed change; the training recipe (normalisation, augmentation, sampler, loss,
threshold, dataloader regime) is inherited from --base and never varied, so the table
isolates architecture.

  ARCH-5   shared CNN + shared Transformer x2 + 3 experts     (soft, shared=2)
  ARCH-6   drop the shared Transformer                        (soft, shared=0)
  ARCH-7   hard graph routing: PerLeadEncoder + 12-lead graph (hard_graph)
  ARCH-8   subset-lead routing, no cross-attention            (subset_lead)
  ARCH-9   fully decoupled: no shared CNN at all              (decoupled_multitask)
  ARCH-10  + gated sibling contrast on the MI head            (mi_head_mode=contrast_v2)
  ARCH-10b + wider conduction lead encoder                    (cd_lead_out_dim 192)
  ARCH-11  + deeper conduction expert                         (cond_layers 3)

ARCH-10b exists because the reported "ARCH-11 = ARCH-10 + deeper conduction" actually
carries two changes; splitting them keeps the attribution honest.

Usage:
  python scripts/make_arch_ablation.py --base gpb10_s2_aug_lead_dropout --num-workers 18
"""
import argparse
import copy
import os

import yaml

CFG_DIR = "configs/experiments"

# name -> (label for the table, {model.<key>: value} applied on top of the previous step)
CHAIN = [
    ("arch05_shared_tf",    "Shared CNN + Transformer", {
        "architecture": "multi_branch_transformer", "routing_mode": "soft",
        "num_shared_layers": 2, "use_cross_attention": True,
        "mi_head_mode": "shared", "cond_layers": 2, "cd_lead_out_dim": 128}),
    ("arch06_no_shared_tf", "Drop shared Transformer", {
        "num_shared_layers": 0}),
    ("arch07_hard_graph",   "Hard graph routing", {
        "routing_mode": "hard_graph"}),
    ("arch08_subset_lead",  "Subset-lead routing", {
        "routing_mode": "subset_lead", "use_cross_attention": False}),
    # use_cross_attention stays off from ARCH-8 onward: the reported ARCH-9..11
    # configs leave it unset, and factory.py defaults it to False.
    ("arch09_decoupled",    "Fully decoupled multitask", {
        "architecture": "decoupled_multitask", "routing_mode": "decoupled"}),
    ("arch10_contrast",     "+ gated sibling contrast", {
        "mi_head_mode": "contrast_v2"}),
    ("arch10b_cond_dim",    "+ wider conduction lead encoder", {
        "cd_lead_out_dim": 192}),
    ("arch11_cond_depth",   "+ deeper conduction expert", {
        "cond_layers": 3}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="Config id supplying the training recipe (without .yaml). "
                         "Its model.* keys are overwritten step by step; everything "
                         "else - normalisation, augmentation, sampler, loss, threshold, "
                         "num_workers - is inherited unchanged.")
    ap.add_argument("--num-workers", type=int, default=18,
                    help="Must match every other run it will be compared against: "
                         "num_workers changes the numpy augmentation stream.")
    ap.add_argument("--prefix", default="abl")
    args = ap.parse_args()

    with open(os.path.join(CFG_DIR, f"{args.base}.yaml"), "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    model_state = dict(base.get("model", {}))
    made = []
    for name, label, delta in CHAIN:
        model_state.update(delta)                    # cumulative, so each step is a delta
        cfg = copy.deepcopy(base)
        cfg["model"] = copy.deepcopy(model_state)
        cfg["training"]["num_workers"] = args.num_workers
        cfg["training"]["save_every"] = 10 ** 6      # keep best_model.pth only
        exp_id = f"{args.prefix}_{name}"
        cfg["experiment"] = {
            "id": exp_id,
            "track": f"arch_ablation@{args.base}",
            "parent": args.base,
            "step": label,
            "changed": sorted(delta),
        }
        dst = os.path.join(CFG_DIR, f"{exp_id}.yaml")
        with open(dst, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        made.append((exp_id, label, sorted(delta)))

    print(f"Recipe from {args.base}  (num_workers={args.num_workers})\n")
    for exp_id, label, changed in made:
        print(f"  {exp_id:24} {label:34} changed: {', '.join(changed)}")
    print("\nRun:")
    print("  /workspace/runmany.sh 42 " + " ".join(m[0] for m in made))


if __name__ == "__main__":
    main()
