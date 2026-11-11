"""
make_cv_configs.py
------------------
Write the configs of the 10-fold cross-validation study (protocol v2,
docs/PROTOCOL_CV_V2.md) into configs/cv/.

Every arm is derived from an existing ablation config, so the only differences
from the single-split study are the ones the protocol names:

  data       LBBB = CLBBB + ILBBB (LAFB, a fascicular block, is no longer folded
             into LBBB); built into its own processed / splits directories
  selection  checkpoint chosen by total validation loss, not by IMI AUPRC
  arm        the head or backbone that defines the arm

Usage:
  python scripts/make_cv_configs.py
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "configs" / "experiments"
OUT = ROOT / "configs" / "cv"

PROTOCOL = {
    "paths": {"processed": "data/processed_mi_nolafb", "splits": "data/splits_mi_nolafb"},
    "label_groups_override": {"LBBB": ["CLBBB", "ILBBB"]},
    "training": {"monitor_metric": "val_loss"},
}

# arm id: (source config, model overrides, role)
ARMS = {
    "cv_shared":     ("abl_arch05_shared_tf.yaml", {}, "internal baseline: shared CNN + Transformer"),
    "cv_decoupled":  ("abl_arch09_decoupled.yaml", {}, "internal baseline: fully decoupled, plain grouping"),
    "cv_layout":     ("abl_arch13_nogate.yaml", {}, "PROPOSED: decoupled + clustered token layout"),
    "cv_contrast":   ("abl_arch16_decoupled_contrast.yaml", {}, "mechanism probe: decoupled + sibling difference"),
    "cv_partial_layout": ("abl_arch17_partial_cluster.yaml", {},
                          "candidate: shared backbone + clustered token layout (best single-split IMI; "
                          "shared backbones transferred better to PTB Diagnostic)"),
    "cv_pairsoft":   ("abl_arch09_decoupled.yaml", {"mi_head_mode": "shared_pairsoft"},
                      "label-structure baseline: pairwise softmax"),
    "cv_hier":       ("abl_arch09_decoupled.yaml", {"mi_head_mode": "shared_hier"},
                      "label-structure baseline: territory x extension"),
    "cv_ecgtransform": ("base_ecgtransform.yaml", {"mi_head_mode": "plain"},
                        "published baseline: ECGTransForm, plain MI head"),
    "cv_xresnet1d101": ("base_ecgtransform.yaml",
                        {"architecture": "xresnet1d101", "mi_head_mode": "plain", "input_downsample": 5,
                         "kernel_size": 5, "dropout": 0.15},
                        "published baseline: PTB-XL benchmark xresnet1d101, plain MI head"),
    "cv_inception1d": ("base_ecgtransform.yaml",
                       {"architecture": "inception1d", "mi_head_mode": "plain", "input_downsample": 5,
                        "kernel_size": 40, "dropout": 0.15},
                       "published baseline: PTB-XL benchmark inception1d, plain MI head"),
}
# Model keys that belong to ECGTransForm only and must not leak into other backbones.
ECGTRANSFORM_KEYS = ("mid_channels", "final_out_channels", "stride", "nhead",
                     "num_encoder_layers", "num_crm_blocks", "width_mult")


def merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


PUBLISHED = ("cv_ecgtransform", "cv_xresnet1d101", "cv_inception1d")
PUBLISHED_EPOCHS = 30     # the project's arms keep their 15; see protocol section 5


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, (src, model_over, role) in ARMS.items():
        cfg = yaml.safe_load((SRC / src).read_text(encoding="utf-8"))
        merge(cfg, PROTOCOL)
        if arm in PUBLISHED:
            cfg["training"]["epochs"] = PUBLISHED_EPOCHS
        if model_over.get("architecture") in ("xresnet1d101", "inception1d"):
            for k in ECGTRANSFORM_KEYS:
                cfg["model"].pop(k, None)
        cfg["model"].update(model_over)
        cfg["experiment"] = {"id": arm, "track": "cv_protocol_v2", "parent": src.removesuffix(".yaml"),
                             "step": role, "changed": sorted(["paths", "label_groups_override",
                                                              "training.monitor_metric", *model_over])}
        path = OUT / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"{path.relative_to(ROOT)}  <- {src}  {model_over or ''}")


if __name__ == "__main__":
    main()
