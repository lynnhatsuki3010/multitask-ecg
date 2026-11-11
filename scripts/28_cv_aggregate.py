"""
28_cv_aggregate.py
------------------
Pool the 10-fold CV predictions into out-of-fold (OOF) results, per arm, with
patient-level uncertainty. Reads only <run>/predictions.npz (27_dump_predictions)
and config_snapshot.yaml, so it runs without any model weights.

Every PTB-XL record is a test record exactly once across the ten folds, so each
arm yields one OOF prediction per record. Reported per arm:

  * AUPRC / AUROC of each MI label on the pooled OOF predictions, and the mean of
    the per-fold values
  * inferior-wall detection (IMI or ILMI) and the lateral-extension AUROC inside it
  * sibling diagnostics on single-infarct-label records: IMI-head AUROC (IMI-only
    vs ILMI-only), logit-difference AUROC, ILMI->IMI and IMI->ILMI rates at each
    fold's calibrated threshold; the same for ASMI/AMI
  * lead occlusion: the IMI-head AUROC and ILMI->IMI rate with V5-V6 or V3-V4 zeroed
  * post-hoc sibling rules on IMI AUPRC (pairwise softmax; z_IMI - lam*z_ILMI with
    lam chosen per fold on that fold's validation logits)

Uncertainty: 95% CIs from a bootstrap over patients (a patient's records are drawn
together). Comparisons against the reference arm are paired: the same bootstrap
draws for both arms, plus an exact Wilcoxon test over the ten per-fold differences.

Usage:
  python scripts/28_cv_aggregate.py --glob "checkpoints/box_cv/run_*" \
      --reference cv_layout --out docs/cv_results.md --csv results/cv_results.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MI = ("IMI", "ASMI", "ILMI", "AMI")
PAIRS = (("IMI", "ILMI"), ("ASMI", "AMI"))
LAMS = np.round(np.arange(0, 2.01, 0.1), 2)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def safe(fn, y, s):
    return fn(y, s) if 0 < y.sum() < len(y) else np.nan


# ----------------------------------------------------------------------------- load
def load_arm_runs(pattern: str):
    arms = {}
    for r in sorted(glob.glob(pattern)):
        r = Path(r)
        if not (r / "predictions.npz").exists():
            continue
        cfg = yaml.safe_load((r / "config_snapshot.yaml").read_text(encoding="utf-8"))
        k = cfg["dataset"].get("cv_fold")
        if k is None:
            continue
        arm = cfg["experiment"]["id"]
        if k in arms.setdefault(arm, {}):
            sys.exit(f"{arm} fold {k} appears twice:\n  {arms[arm][k][0]}\n  {r}")
        arms[arm][k] = (r, cfg)
    return arms


def assemble(runs: dict, lm: np.ndarray, flat_all: list):
    """OOF arrays for one arm: logits, calibrated probs and decisions per record."""
    n = len(lm)
    out = {"fold": np.full(n, -1)}
    mi_cols = None
    for k, (r, cfg) in runs.items():
        p = np.load(r / "predictions.npz", allow_pickle=False)
        names = list(p["label_names"])
        if mi_cols is None:
            mi_cols = [names.index(m) for m in MI]
            g = cfg["label_groups"]
            n_a, n_m = len(g["arrhythmia"]), len(g["mi"])
            for key in ("z", "z_V5V6", "z_V3V4", "prob", "dec"):
                out[key] = np.full((n, len(MI)), np.nan)
            out["z_arr"] = np.full((n, n_a), np.nan)
            out["z_cond"] = np.full((n, len(g["conduction"])), np.nan)
            out["T"] = {}
            out["lam"] = {}
        ti = p["test_idx"]
        T = p["temperature"]                      # arrhythmia, mi, conduction
        thr = p["thresholds"][[names.index(m) for m in MI]]
        out["fold"][ti] = k
        z = p["test_mi"]
        out["z"][ti] = z
        out["z_V5V6"][ti] = p["test_mi_occ_V5V6"]
        out["z_V3V4"][ti] = p["test_mi_occ_V3V4"]
        out["prob"][ti] = sig(z / T[1])
        out["dec"][ti] = out["prob"][ti] >= thr
        out["z_arr"][ti] = p["test_arrhythmia"]
        out["z_cond"][ti] = p["test_conduction"]
        out["T"][k] = T.tolist()
        # post-hoc rule: lam for IMI chosen on this fold's validation logits
        vi, zv = p["val_idx"], p["val_mi"]
        y_imi_val = lm[vi, flat_all.index("IMI")]
        out["lam"][k] = float(LAMS[int(np.argmax([safe(AP, y_imi_val, zv[:, 0] - lam * zv[:, 2]) for lam in LAMS]))])
    return out


# --------------------------------------------------------------------------- metrics
def metrics(o: dict, Y: dict, sel: np.ndarray) -> dict:
    """All scalar metrics on the record subset `sel` (bool or index array)."""
    m = {}
    z, prob, dec = o["z"][sel], o["prob"][sel], o["dec"][sel]
    y = {k: v[sel] for k, v in Y.items()}
    for j, lab in enumerate(MI):
        m[f"AUPRC_{lab}"] = safe(AP, y[lab], z[:, j])
        m[f"AUROC_{lab}"] = safe(AUC, y[lab], z[:, j])
    m["AUPRC_MI_macro"] = np.nanmean([m[f"AUPRC_{l}"] for l in MI])
    # inferior wall: either label, scored by 1 - (1-p_IMI)(1-p_ILMI)
    inf_y = np.maximum(y["IMI"], y["ILMI"])
    inf_s = 1 - (1 - prob[:, 0]) * (1 - prob[:, 2])
    m["AUPRC_inferior_wall"] = safe(AP, inf_y, inf_s)
    # single-infarct-label records
    single = (y["IMI"] + y["ASMI"] + y["ILMI"] + y["AMI"]) == 1
    for a, b in PAIRS:
        ia, ib = MI.index(a), MI.index(b)
        A, B = single & (y[a] == 1), single & (y[b] == 1)
        lab = np.r_[np.ones(A.sum()), np.zeros(B.sum())]
        if A.sum() and B.sum():
            m[f"pairAUROC_{a}head_{a}vs{b}"] = AUC(lab, np.r_[z[A, ia], z[B, ia]])
            m[f"pairAUROC_logitdiff_{a}vs{b}"] = AUC(lab, np.r_[z[A, ia] - z[A, ib], z[B, ia] - z[B, ib]])
            m[f"rate_{b}_to_{a}"] = float(dec[B, ia].mean())
            m[f"rate_{a}_to_{b}"] = float(dec[A, ib].mean())
            m[f"recall_{a}_single"] = float(dec[A, ia].mean())
    # occlusion on the IMI head
    A, B = single & (y["IMI"] == 1), single & (y["ILMI"] == 1)
    lab = np.r_[np.ones(A.sum()), np.zeros(B.sum())]
    for occ in ("V5V6", "V3V4"):
        zo = o[f"z_{occ}"][sel]
        if A.sum() and B.sum():
            m[f"occ_{occ}_pairAUROC_IMIhead"] = AUC(lab, np.r_[zo[A, 0], zo[B, 0]])
            folds = o["fold"][sel]
            thr_dec = np.zeros(len(zo), dtype=bool)
            for k in np.unique(folds):
                f = folds == k
                # same temperature and IMI threshold as that fold's unoccluded decisions
                thr_dec[f] = sig(zo[f, 0] / o["T"][int(k)][1]) >= o["thr_imi"][int(k)]
            m[f"occ_{occ}_rate_ILMI_to_IMI"] = float(thr_dec[B].mean())
    # post-hoc rules on IMI AUPRC
    z_imi, z_ilmi = z[:, 0], z[:, 2]
    m["AUPRC_IMI_rule_pairsoft"] = safe(AP, y["IMI"], z_imi - np.logaddexp(0, z_ilmi))
    lam = np.array([o["lam"][int(k)] for k in o["fold"][sel]])
    m["AUPRC_IMI_rule_linear"] = safe(AP, y["IMI"], z_imi - lam * z_ilmi)
    return m


def patient_boot(pids: np.ndarray, B: int, rng):
    """Index arrays of B bootstrap samples drawn over patients."""
    uniq, inv = np.unique(pids, return_inverse=True)
    groups = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([groups[i] for i in pick])


def holm(p: list) -> list:
    """Holm step-down adjusted p-values (NaN stays NaN and is not counted)."""
    p = np.asarray(p, dtype=float)
    ok = np.flatnonzero(~np.isnan(p))
    adj = np.full(len(p), np.nan)
    running = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, min(1.0, (len(ok) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def primary_hypotheses(res: pd.DataFrame, point: dict, ref: str) -> list:
    """Protocol section 7: H1-H3 on OOF IMI AUPRC, Holm across the three.

    Deltas are arm - reference. H1 and H2 claim the reference is better (CI entirely
    below 0); H3 claims a difference in either direction (CI excludes 0).
    """
    if ref != "cv_layout":
        return []
    ls = [a for a in ("cv_pairsoft", "cv_hier") if a in point]
    best_ls = max(ls, key=lambda a: point[a]["AUPRC_IMI"]) if ls else None
    hyps = [("H1", "cv_decoupled", "less"), ("H2", "cv_shared", "less"), ("H3", best_ls, "two-sided")]
    rows = []
    for h, arm, side in hyps:
        r = res[(res.arm == arm) & (res.metric == "AUPRC_IMI")] if arm else res.iloc[0:0]
        rows.append((h, arm, side, r.iloc[0] if len(r) else None))
    adj = holm([r.fold_p if r is not None else np.nan for *_, r in rows])
    out = ["## Primary hypotheses (protocol section 7, OOF IMI AUPRC)", "",
           f"Δ = arm − `{ref}`. Holm across H1–H3 on the fold-level Wilcoxon p.", "",
           "| | comparison | Δ | Δ 95% CI (patients) | folds where ref better | p (folds) | p Holm | supported |",
           "|---|---|---|---|---|---|---|---|"]
    for (h, arm, side, r), pa in zip(rows, adj):
        if r is None:
            out.append(f"| {h} | `{arm}` missing | – | – | – | – | – | – |")
            continue
        ci_ok = r.delta_ci_hi < 0 if side == "less" else (r.delta_ci_hi < 0 or r.delta_ci_lo > 0)
        out.append(f"| {h} | `{ref}` vs `{arm}` | {r.delta_vs_ref:+.4f} | [{r.delta_ci_lo:+.4f}; "
                   f"{r.delta_ci_hi:+.4f}] | {int(r.n_folds - r.fold_wins)}/{int(r.n_folds)} | "
                   f"{r.fold_p:.4f} | {pa:.4f} | {'yes' if ci_ok and pa < 0.05 else 'no'} |")
    if best_ls:
        out.append("")
        out.append(f"H3 compares with `{best_ls}`, the label-structure arm with the higher OOF IMI AUPRC.")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="run folders of the CV study")
    ap.add_argument("--reference", default="cv_layout", help="arm every other arm is compared with")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--allow-missing", action="store_true", help="accept arms with fewer than 10 folds")
    ap.add_argument("--exclude-folds", type=int, nargs="*", default=[],
                    help="drop these folds from every arm (sensitivity analysis; protocol amendment 1 "
                         "drops fold 2, the validation fold the learning rate was chosen on)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    arms = load_arm_runs(args.glob)
    if not arms:
        sys.exit("no CV runs with predictions.npz found")
    for runs in arms.values():
        for k in args.exclude_folds:
            runs.pop(k, None)
    n_expected = 10 - len(set(args.exclude_folds))
    any_cfg = next(iter(next(iter(arms.values())).values()))[1]
    fs = any_cfg["dataset"]["sampling_rate"]
    proc = ROOT / f"{any_cfg['paths']['processed']}_{fs}hz"
    df = pd.read_csv(proc / "metadata.csv")
    lm = np.load(proc / "label_matrix.npy")
    flat_all = [l["name"] for l in sorted(any_cfg["labels"], key=lambda l: l["index"])]
    Y = {lab: lm[:, flat_all.index(lab)] for lab in MI}
    pids = df["patient_id"].to_numpy()

    lines = ["# 10-fold CV results (out-of-fold, PTB-XL)", "",
             f"Run folders: `{args.glob}`. Reference arm: `{args.reference}`. "
             f"Patient bootstrap: {args.boot} draws."
             + (f" **Folds excluded: {sorted(set(args.exclude_folds))}.**" if args.exclude_folds else ""), ""]
    data, rows = {}, []
    for arm, runs in sorted(arms.items()):
        folds = sorted(runs)
        if len(folds) < n_expected and not args.allow_missing:
            print(f"skip {arm}: folds {folds}")
            continue
        o = assemble(runs, lm, flat_all)
        o["thr_imi"] = {}
        for k, (r, _) in runs.items():
            p = np.load(r / "predictions.npz")
            o["thr_imi"][k] = float(p["thresholds"][list(p["label_names"]).index("IMI")])
        covered = o["fold"] >= 0
        data[arm] = (o, covered, folds)

    rng = np.random.default_rng(20261109)
    common = np.logical_and.reduce([c for _, c, _ in data.values()])
    boots = list(patient_boot(pids[common], args.boot, rng))
    common_idx = np.flatnonzero(common)

    point, per_fold, boot_vals = {}, {}, {}
    for arm, (o, covered, folds) in data.items():
        point[arm] = metrics(o, Y, common_idx)
        per_fold[arm] = {k: metrics(o, Y, np.flatnonzero(o["fold"] == k)) for k in folds}
        boot_vals[arm] = [metrics(o, Y, common_idx[b]) for b in boots]
        print(f"{arm}: folds {folds}, records {covered.sum()}")

    keys = list(next(iter(point.values())).keys())
    for arm in point:
        for key in keys:
            bv = np.array([b[key] for b in boot_vals[arm]], dtype=float)
            fv = np.array([per_fold[arm][k][key] for k in per_fold[arm]], dtype=float)
            row = {"arm": arm, "metric": key, "oof": point[arm][key],
                   "ci_lo": np.nanpercentile(bv, 2.5), "ci_hi": np.nanpercentile(bv, 97.5),
                   "fold_mean": np.nanmean(fv), "fold_sd": np.nanstd(fv, ddof=1)}
            ref = args.reference
            if arm != ref and ref in point:
                d = bv - np.array([b[key] for b in boot_vals[ref]], dtype=float)
                common_folds = sorted(set(per_fold[arm]) & set(per_fold[ref]))
                fd = np.array([per_fold[arm][k][key] - per_fold[ref][k][key] for k in common_folds])
                fd = fd[~np.isnan(fd)]
                row.update(delta_vs_ref=point[arm][key] - point[ref][key],
                           delta_ci_lo=np.nanpercentile(d, 2.5), delta_ci_hi=np.nanpercentile(d, 97.5),
                           fold_wins=int((fd > 0).sum()), n_folds=len(fd),
                           fold_p=(wilcoxon(fd).pvalue if len(fd) > 1 and np.any(fd != 0) else np.nan))
            rows.append(row)
    res = pd.DataFrame(rows)
    lines += primary_hypotheses(res, point, args.reference)

    show = ["AUPRC_IMI", "AUPRC_ILMI", "AUPRC_ASMI", "AUPRC_AMI", "AUPRC_MI_macro", "AUPRC_inferior_wall",
            "pairAUROC_IMIhead_IMIvsILMI", "pairAUROC_logitdiff_IMIvsILMI", "rate_ILMI_to_IMI",
            "occ_V5V6_pairAUROC_IMIhead", "occ_V3V4_pairAUROC_IMIhead", "occ_V5V6_rate_ILMI_to_IMI",
            "AUPRC_IMI_rule_pairsoft", "AUPRC_IMI_rule_linear"]
    for key in show:
        lines += [f"## {key}", "", "| arm | OOF | 95% CI (patients) | fold mean ± SD | Δ vs ref | Δ 95% CI | folds better | p (folds) |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in res[res.metric == key].iterrows():
            extra = ("| – | – | – | – |" if pd.isna(r.get("delta_vs_ref", np.nan)) else
                     f"| {r.delta_vs_ref:+.4f} | [{r.delta_ci_lo:+.4f}; {r.delta_ci_hi:+.4f}] | "
                     f"{int(r.fold_wins)}/{int(r.n_folds)} | {r.fold_p:.4f} |")
            lines.append(f"| `{r.arm}` | {r.oof:.4f} | [{r.ci_lo:.4f}; {r.ci_hi:.4f}] | "
                         f"{r.fold_mean:.4f} ± {r.fold_sd:.4f} " + extra)
        lines.append("")
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    if args.csv:
        res.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
