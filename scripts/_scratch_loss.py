import json, os

runs = [
    "run_20260506_231153_hybrid-tf",
    "run_20260507_003151_hybrid-tf",
    "run_20260507_013801_hybrid-tf",
]
for run in runs:
    audit_path = os.path.join("checkpoints", run, "dataset_policy_audit.json")
    if os.path.exists(audit_path):
        with open(audit_path) as f:
            d = json.load(f)
        sp = d.get("stored_policy") or d.get("expected_policy") or {}
        method   = sp.get("split_method", "?")
        seed     = sp.get("split_seed", "?")
        tfold    = sp.get("test_fold", "?")
        stratify = sp.get("split_stratify_label", "none")
        proc     = sp.get("processed_path", "?")
        print(f"{run}")
        print(f"  split_method={method}  seed={seed}  test_fold={tfold}  stratify={stratify}")
        print(f"  processed={proc}")
    else:
        print(f"{run}: no audit file — checking cfg_snapshot")
        snap = os.path.join("checkpoints", run, "cfg_snapshot.yaml")
        if os.path.exists(snap):
            with open(snap) as f:
                print(f.read()[:300])
