"""One stage of the experiment, with the notebook code (nbcode.py).

    python run_stage.py finetune
    python run_stage.py sample
    python run_stage.py cls --scheme baseline      (also A, R, S and T)
    python run_stage.py eval

Folders come from environment variables: W (work folder), OUT (results folder), MONAI (folder of the MONAI
bundle), LABELS (label file), CACHE (cached latents), SYNTH (synthetic volumes), UNET_DIR and CLS_DIR
(checkpoints), PRIV (per-session predictions, which stay on the machine).
Only aggregate values are written under OUT/results.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

E = os.environ
W, OUT = Path(E["W"]), Path(E["OUT"])
MONAI, LABELS = Path(E["MONAI"]), E["LABELS"]
CACHE, SYNTH = Path(E["CACHE"]), Path(E["SYNTH"])
UNET_DIR, CLS_DIR, PRIV = Path(E["UNET_DIR"]), Path(E["CLS_DIR"]), Path(E["PRIV"])
RES = OUT / "results"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
TARGET = [160, 220, 160]


def cond_pool(nb, labels_path, seed=42):
    """Conditioning vectors of the real training sessions of each class."""
    from cache_latents import cond_vector, cond_stats
    tr = nb.load_split(labels_path, seed=seed)["train"]
    pool = {"cn": [cond_vector(x) for x in tr if x["y"] == 0], "ad": [cond_vector(x) for x in tr if x["y"] == 1]}
    return pool, {k: cond_stats(v) for k, v in pool.items()}


def stage_finetune(nb, args):
    RES.mkdir(parents=True, exist_ok=True)
    nb.run_unet_finetune(ckpt_dir=MONAI, cache_dir=CACHE, manifest_path=CACHE / "cache_manifest.json",
                         out_dir=UNET_DIR, epochs=args.epochs, batch_size=args.batch_size, device=DEV,
                         seed=42, save_every=args.epochs, log_path=RES / "unet_train_log.json")
    ck = torch.load(UNET_DIR / "unet_oasis_finetuned_best.pt", map_location="cpu", weights_only=False)
    meta = {"best_epoch": int(ck["epoch"]), "best_val_mse_ema": float(ck["val_loss_ema"]), "epochs": args.epochs,
            "batch_size": args.batch_size}
    (RES / "unet_best.json").write_text(json.dumps(meta, indent=1))
    # EMA weights only (what sampling uses), for the synced folder
    (OUT / "ckpts").mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": ck["epoch"], "ema_state_dict": ck["ema_state_dict"], "val_loss_ema": ck["val_loss_ema"]},
               OUT / "ckpts" / "unet_oasis_finetuned_best_ema.pt")
    print("finetune done", meta)


def stage_sample(nb, args):
    RES.mkdir(parents=True, exist_ok=True)
    pool, stats = cond_pool(nb, LABELS)
    (RES / "synth_cond_pool_stats.json").write_text(json.dumps(stats, indent=1))
    nb.run_synth_generation(ckpt_base=MONAI, unet_finetuned=OUT / "ckpts" / "unet_oasis_finetuned_best_ema.pt",
                            out_dir=SYNTH, n_per_class=args.n_per_class, steps=args.steps, device=DEV,
                            seed=42, cond_pool=pool)


def ckpt_of(scheme):
    return "best_baseline.pt" if scheme == "baseline" else f"best_{scheme}.pt"


def stage_cls(nb, args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    hold = W / f"HOLD_{args.scheme}"            # a file with this name pauses the stage (for example while a check is running)
    while hold.exists() and not args.single:
        print(f"holding scheme {args.scheme}: {hold} exists", flush=True)
        time.sleep(60)
    # cls_plan.json: {"reserve_h": 0.7, "groups": [{"schemes": ["A"], "need_h": 0}, {"schemes": ["R"], "need_h": 3.0}, ...]}
    # The schemes of one entry run side by side, the entries one after another. An entry is skipped when fewer than
    # need_h + reserve_h hours are left before the deadline of the machine. The file is read again before every
    # entry, so later entries can still be changed while an earlier one runs. Only a failure of Scheme A fails the stage.
    plan = W / "cls_plan.json"
    if args.scheme == "A" and plan.exists() and not args.single:
        deadline = float(os.environ.get("VAST_DEADLINE_UTC", "0") or 0)
        log, gi = [], 0
        while True:
            cfg = json.loads(plan.read_text())
            if gi >= len(cfg["groups"]):
                break
            g = cfg["groups"][gi]
            gi += 1
            left_h = (deadline - time.time()) / 3600 if deadline else 1e9
            need_h = float(g.get("need_h", 0)) + float(cfg.get("reserve_h", 0.7))
            entry = {"schemes": g["schemes"], "hours_left_at_start": round(left_h, 2), "hours_needed": need_h}
            bad = []
            if left_h < need_h:
                entry["status"] = "skipped: not enough time before the deadline of the machine"
                print(f"schemes {g['schemes']}: skipped, {left_h:.2f} h left, {need_h:.2f} h needed", flush=True)
            else:
                procs, t0 = [], time.time()
                for s in g["schemes"]:
                    if (RES / f"results_{s}.json").exists():
                        print(f"scheme {s}: results exist, skipped", flush=True)
                        continue
                    out_log = open(OUT / "logs" / f"cls_{s}_run.log", "w")
                    cmd = [sys.executable, str(Path(__file__).resolve()), "cls", "--scheme", s, "--single",
                           "--epochs", str(args.epochs), "--num-workers", str(args.num_workers),
                           "--min-synth", str(cfg.get("min_synth", 1000))]
                    procs.append((s, subprocess.Popen(cmd, stdout=out_log, stderr=subprocess.STDOUT)))
                    print(f"scheme {s}: started, pid {procs[-1][1].pid}", flush=True)
                    time.sleep(20)
                while any(p.poll() is None for _, p in procs):      # a line a minute: the log shows the stage is alive
                    time.sleep(60)
                    print(f"schemes {g['schemes']}: running, {(time.time() - t0) / 60:.0f} min", flush=True)
                bad = [s for s, p in procs if p.returncode != 0]
                entry["status"] = "done" if not bad else f"failed: {bad}"
                entry["hours"] = round((time.time() - t0) / 3600, 2)
                print(f"schemes {g['schemes']}: finished in {entry['hours']} h, failed {bad}", flush=True)
            log.append(entry)
            (RES / "cls_plan_log.json").write_text(json.dumps(log, indent=1))
            if "A" in g["schemes"] and ("A" in bad or entry["status"].startswith("skipped")):
                raise SystemExit(f"Scheme A did not finish: {entry['status']}")
        return
    if args.scheme != "baseline":
        n = {c: len([p for p in (SYNTH / c).glob(f"{c}_*.pt") if ".tmp" not in p.name]) for c in ("cn", "ad")}
        assert n["cn"] == n["ad"] and n["cn"] >= args.min_synth, f"synthetic volumes incomplete: {n}"
    nb.run_scheme(args.scheme, labels_path=LABELS, synth_dir=str(SYNTH), out_dir=str(CLS_DIR),
                  epochs=args.epochs, batch=4, lr=1e-5, device=DEV, target_size=TARGET,
                  lr_schedule="cosine", seed=42, num_workers=args.num_workers)
    RES.mkdir(parents=True, exist_ok=True)
    shutil.copy(CLS_DIR / f"results_{args.scheme}.json", RES / f"results_{args.scheme}.json")
    (OUT / "ckpts").mkdir(parents=True, exist_ok=True)
    shutil.copy(CLS_DIR / ckpt_of(args.scheme), OUT / "ckpts" / ckpt_of(args.scheme))


def proxies(a):
    """Intensity proxies of one volume (no segmentation model)."""
    hi = np.percentile(a, 99.5)
    a = np.clip(a / max(hi, 1e-8), 0, 1)
    tis = a > 0.15
    med = np.median(a[tis])
    s = a.shape
    lo_, hi_ = [int(n * 0.325) for n in s], [int(n * 0.675) for n in s]
    c = a[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]]
    return dict(tissue=float(tis.mean()), csf_c=float((c < 0.45 * med).mean()), mean_c=float(c.mean() / med))


def auc_higher(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def summarise(ad, cn):
    ad, cn = np.asarray(ad, float), np.asarray(cn, float)
    d = (ad.mean() - cn.mean()) / np.sqrt((ad.var(ddof=1) + cn.var(ddof=1)) / 2)
    return dict(ad_mean=float(ad.mean()), cn_mean=float(cn.mean()), ad_sd=float(ad.std(ddof=1)), cn_sd=float(cn.std(ddof=1)),
                cohen_d=float(d), auc_ad_higher=auc_higher(ad, cn))


def cluster_bootstrap(y, subj, probs, ref="baseline", n=2000, seed=42):
    """Subject-level bootstrap of the test AUCs: subjects are drawn with replacement, each with all
    of its test sessions. Returns the percentile interval of every AUC and of every difference
    to the reference scheme, with a two-sided bootstrap p-value of the difference."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    subs = np.unique(subj)
    rows = {s: np.where(subj == s)[0] for s in subs}
    names = list(probs)
    aucs = {k: [] for k in names}
    for _ in range(n):
        idx = np.concatenate([rows[s] for s in rng.choice(subs, len(subs), replace=True)])
        if y[idx].min() == y[idx].max():
            continue
        for k in names:
            aucs[k].append(roc_auc_score(y[idx], probs[k][idx]))
    out = {"n_resamples": len(aucs[names[0]]), "n_subjects": int(len(subs)), "n_sessions": int(len(y)),
           "n_ad": int((y == 1).sum()), "n_cn": int((y == 0).sum()), "auc": {}, "difference": {}}
    for k in names:
        a = np.asarray(aucs[k])
        out["auc"][k] = {"auc": float(roc_auc_score(y, probs[k])),
                         "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
    pairs = [(k, ref) for k in names if k != ref] + [(k, r) for k, r in (("A", "R"), ("A", "S")) if k in names and r in names]
    for k, r in pairs:
        d = np.asarray(aucs[k]) - np.asarray(aucs[r])
        out["difference"][f"{k}_minus_{r}"] = {
            "delta": float(roc_auc_score(y, probs[k]) - roc_auc_score(y, probs[r])),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "se": float(d.std(ddof=1)),
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    return out


def stage_eval(nb, args):
    """Paired test and ROC points (notebook code), then the extended blocks. A failure inside an
    extended block is written to results/eval_errors.json and does not fail the stage."""
    torch.multiprocessing.set_sharing_strategy("file_system")
    import traceback
    import scipy.stats as ss
    from sklearn.metrics import roc_auc_score
    RES.mkdir(parents=True, exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)
    names = [n for n in ("baseline", "A", "R", "S", "T") if (CLS_DIR / ckpt_of(n)).exists()]
    ckpts = [str(CLS_DIR / ckpt_of(n)) for n in names]
    if not args.skip_core:
        # per-session predictions and labels stay in PRIV, only the aggregates go to the results
        nb.run_delong_eval(labels_path=LABELS, ckpts=ckpts, names=names, out_dir=str(PRIV), device=DEV, batch=1)
        for f in ("delong_results.json", "test_roc_auc.json"):
            if (PRIV / f).exists():
                shutil.copy(PRIV / f, RES / f)
    errors = {}
    only = set(args.only.split(",")) if args.only else None

    def block(name, fn):
        if only and name not in only:
            return
        t0 = time.time()
        try:
            fn()
            print(f"eval block {name}: done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            traceback.print_exc()
            errors[name] = repr(e)[:500]
            print(f"eval block {name}: FAILED {e!r}", flush=True)

    # ---- subject-level bootstrap and subgroups of the test split
    def test_extended():
        import nibabel as nib
        test = nb.load_split(LABELS)["test"]
        y = np.load(PRIV / "y_test.npy").astype(int)
        assert len(test) == len(y) and [int(x["y"]) for x in test] == y.tolist(), "test order differs from the stored labels"
        probs = {n: np.load(PRIV / f"probs_{n}.npy") for n in names}
        subj = np.asarray([x["subject"] for x in test])
        grid = np.asarray(["x".join(str(s) for s in nib.load(x["path"]).shape[:3]) for x in test])
        cdr = np.asarray([float(x["cdr"]) for x in test])
        gap = np.asarray([abs(float(x["days_to_clinical"])) if x.get("days_to_clinical") is not None else np.inf for x in test])
        ext = {"all": cluster_bootstrap(y, subj, probs), "subgroups": {}}
        big = max(set(grid.tolist()), key=grid.tolist().count)
        groups = {f"grid_{big}": grid == big, "grid_other": grid != big,
                  "cn_against_cdr_1_or_more": (y == 0) | (cdr >= 1.0),
                  "cn_against_cdr_0_5": (y == 0) | (cdr == 0.5),
                  "clinical_visit_within_365_days": gap <= 365}
        for g, m in groups.items():
            if m.sum() and y[m].min() != y[m].max() and (y[m] == 1).sum() >= 5:
                ext["subgroups"][g] = cluster_bootstrap(y[m], subj[m], {k: v[m] for k, v in probs.items()})
        # one score per subject (mean over its sessions, label of the majority)
        subs = np.unique(subj)
        ys = np.asarray([int(round(y[subj == s].mean())) for s in subs])
        ext["subject_level_auc"] = {k: float(roc_auc_score(ys, np.asarray([v[subj == s].mean() for s in subs])))
                                    for k, v in probs.items()}
        ext["subject_level_n"] = {"subjects": int(len(subs)), "ad": int(ys.sum())}
        (RES / "test_extended.json").write_text(json.dumps(ext, indent=1))
        print(json.dumps({"auc": ext["all"]["auc"], "difference": ext["all"]["difference"]}, indent=1))

    block("test_extended", test_extended)

    # ---- what the classifier trained on real scans only says about synthetic volumes
    path = RES / "synthetic_class_check_v2.json"
    out = json.loads(path.read_text()) if (path.exists() and only) else {}
    model = nb.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2, norm="instance").to(DEV).eval()
    model.load_state_dict(torch.load(CLS_DIR / "best_baseline.pt", map_location=DEV, weights_only=True))
    pre = nb.Compose([nb.ScaleIntensity(), nb.ResizeWithPadOrCrop(spatial_size=tuple(TARGET))])

    def p_ad(v):
        with torch.no_grad():
            return float(torch.softmax(model(pre(v).unsqueeze(0).to(DEV)), dim=1)[0, 1])

    def synthetic():
        files = {c: sorted(p for p in (SYNTH / c).glob(f"{c}_*.pt") if ".tmp" not in p.name) for c in ("cn", "ad")}
        out["n_per_class"] = {c: len(v) for c, v in files.items()}
        assert all(files.values()), "no synthetic volumes"
        pool, _ = cond_pool(nb, LABELS)
        rng = random.Random(42)                         # the draw of run_synth_generation, repeated
        probs_s, prox, conds = {"cn": [], "ad": []}, {"cn": [], "ad": []}, {"cn": [], "ad": []}
        probs_raw = {"cn": [], "ad": []}                # the same volumes without the limit to [0, 1] of load_synth
        for c in ("cn", "ad"):
            drawn = [list(rng.choice(pool[c])) for _ in range(args.n_per_class)]
            for k, p in enumerate(files[c]):
                v = nb.load_synth(str(p))
                probs_s[c].append(p_ad(v))
                if k < args.n_raw:
                    probs_raw[c].append(p_ad(torch.load(p, map_location="cpu", weights_only=False)[0].float()))
                conds[c].append(drawn[int(p.stem.split("_")[1])])
                if k < args.n_proxy:
                    prox[c].append(proxies(v.squeeze().numpy()))
        out["baseline_classifier_p_ad"] = summarise(probs_s["ad"], probs_s["cn"])
        out["baseline_classifier_share_called_ad"] = {c: float(np.mean(np.asarray(probs_s[c]) > 0.5)) for c in ("cn", "ad")}
        if probs_raw["cn"] and probs_raw["ad"]:
            out["baseline_classifier_p_ad_without_limit"] = dict(summarise(probs_raw["ad"], probs_raw["cn"]),
                                                                 n_per_class=len(probs_raw["cn"]))
        out["proxies_n_per_class"] = min(args.n_proxy, len(files["cn"]), len(files["ad"]))
        for k in ("csf_c", "mean_c", "tissue"):
            out[k] = summarise([r[k] for r in prox["ad"]], [r[k] for r in prox["cn"]])
        path.write_text(json.dumps(out, indent=1))
        # p(AD) of the synthetic AD volumes by the CDR in their conditioning vector, and rank correlations
        pa, ca = np.asarray(probs_s["ad"]), np.asarray(conds["ad"])
        by = {}
        for lv in sorted(set(np.round(ca[:, 2] * 3, 2).tolist())):
            m = np.isclose(ca[:, 2] * 3, lv, atol=0.01)
            by[f"{lv:.1f}"] = {"n": int(m.sum()), "mean_p_ad": float(pa[m].mean()), "share_called_ad": float((pa[m] > 0.5).mean())}
        out["synthetic_ad_p_ad_by_cdr"] = by
        allp = np.asarray(probs_s["cn"] + probs_s["ad"]); allc = np.asarray(conds["cn"] + conds["ad"])
        cols = ("sex", "age", "cdr", "mmse")
        out["spearman_p_ad_with_conditioning"] = {
            "all_synthetic": {nme: float(ss.spearmanr(allp, allc[:, j])[0]) for j, nme in enumerate(cols)},
            "synthetic_cn": {nme: float(ss.spearmanr(np.asarray(probs_s["cn"]), np.asarray(conds["cn"])[:, j])[0])
                             for j, nme in enumerate(cols) if nme != "cdr"},
            "synthetic_ad": {nme: float(ss.spearmanr(pa, ca[:, j])[0]) for j, nme in enumerate(cols)}}
        path.write_text(json.dumps(out, indent=1))

    # reference: the same classifier on the real validation sessions, and on their autoencoder reconstructions
    def real_reference():
        from cache_latents import build_transform
        val = nb.load_split(LABELS)["val"]
        tf = build_transform()
        os.environ.pop("BRAIN_LDM_UNET_OVERRIDE", None)
        pipe = nb.load_pipe(str(MONAI), device=DEV)
        pr, prr, yv, px, pxr = [], [], [], [], []
        for x in val:
            pr.append(p_ad(nb.load_volume(x["path"])))
            vol = tf({"image": x["path"]})["image"]
            vol = (vol.as_tensor() if hasattr(vol, "as_tensor") else vol).float()
            with torch.no_grad():
                rec = pipe.latent2img(pipe.img2latent(vol.unsqueeze(0).to(DEV))).float().cpu()[0]
            prr.append(p_ad(rec.clamp(0.0, 1.0)))
            yv.append(int(x["y"]))
            px.append(proxies(vol[0].numpy()))              # the same intensity proxies as for the synthetic volumes
            pxr.append(proxies(rec[0].numpy()))
        pr, prr, yv = np.asarray(pr), np.asarray(prr), np.asarray(yv)
        out["real_validation_reference"] = {
            "n": int(len(yv)), "n_ad": int(yv.sum()),
            "real": summarise(pr[yv == 1], pr[yv == 0]), "reconstructed": summarise(prr[yv == 1], prr[yv == 0]),
            "share_called_ad_real": {"cn": float((pr[yv == 0] > 0.5).mean()), "ad": float((pr[yv == 1] > 0.5).mean())},
            "share_called_ad_reconstructed": {"cn": float((prr[yv == 0] > 0.5).mean()), "ad": float((prr[yv == 1] > 0.5).mean())},
            "spearman_real_against_reconstructed": float(ss.spearmanr(pr, prr)[0]),
            "proxies_real": {m: summarise([r[m] for r, t in zip(px, yv) if t == 1], [r[m] for r, t in zip(px, yv) if t == 0])
                             for m in ("csf_c", "mean_c", "tissue")},
            "proxies_reconstructed": {m: summarise([r[m] for r, t in zip(pxr, yv) if t == 1], [r[m] for r, t in zip(pxr, yv) if t == 0])
                                      for m in ("csf_c", "mean_c", "tissue")}}
        del pipe
        torch.cuda.empty_cache()
        path.write_text(json.dumps(out, indent=1))

    # counterfactual pairs of checks_gen.py (same noise, sex and age, only CDR and MMSE changed)
    def counterfactual():
        cf = W / "cf"
        assert cf.is_dir(), "no counterfactual volumes (checks_gen.py, check B)"
        out["counterfactual_pairs_p_ad"] = {}
        for kind in sorted(d.name for d in cf.iterdir() if d.is_dir()):     # ft, stock, and guided sets if present
            sc = {}
            for p in sorted((cf / kind).glob("*.pt")):
                sc.setdefault(p.stem.rsplit("_", 1)[0], {})[p.stem.rsplit("_", 1)[1]] = p_ad(nb.load_synth(str(p)))
            if "cn" not in sc:
                continue
            res = {"mean_p_ad": {k: float(np.mean(list(v.values()))) for k, v in sc.items()}, "paired_vs_cn": {}}
            for k, v in sc.items():
                if k == "cn":
                    continue
                d = np.asarray([v[i] - sc["cn"][i] for i in sorted(v)])
                se = d.std(ddof=1) / np.sqrt(len(d))
                res["paired_vs_cn"][k] = {"mean": float(d.mean()), "se": float(se), "t": float(d.mean() / se) if se > 0 else 0.0,
                                          "share_positive": float((d > 0).mean()), "n": int(len(d))}
            out["counterfactual_pairs_p_ad"][{"ft": "fine_tuned_unet", "stock": "stock_unet"}.get(kind, kind)] = res
        path.write_text(json.dumps(out, indent=1))

    block("synthetic", synthetic)
    block("real_reference", real_reference)
    block("counterfactual", counterfactual)
    print(json.dumps({k: v for k, v in out.items() if k != "n_per_class"}, indent=1)[:3000])
    if errors:
        (RES / "eval_errors.json").write_text(json.dumps(errors, indent=1))
    # A failed block, or a file HOLD_eval_end, keeps the stage open for at most 45 minutes, so the block can be
    # repaired and run again by hand (python run_stage.py eval --skip-core --only NAME) while the per-session
    # predictions still exist. Removing the hold file (or creating RELEASE_eval_end) ends the wait.
    hold, release, t0 = W / "HOLD_eval_end", W / "RELEASE_eval_end", time.time()
    while (errors or hold.exists()) and not release.exists() and time.time() - t0 < 45 * 60 and not args.skip_core:
        print(f"eval finished, waiting: errors {list(errors)}, hold file {hold.exists()}", flush=True)
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["finetune", "sample", "cls", "eval"])
    ap.add_argument("--scheme", default="baseline")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--n-proxy", type=int, default=100)
    ap.add_argument("--n-raw", type=int, default=300, help="eval: synthetic volumes per class scored also without the limit")
    ap.add_argument("--skip-core", action="store_true", help="eval: keep the stored predictions, run the extended blocks only")
    ap.add_argument("--only", default="", help="eval: comma separated extended blocks to run (test_extended, synthetic, real_reference, counterfactual)")
    ap.add_argument("--single", action="store_true", help="cls: run this one scheme, ignore cls_plan.json and HOLD files")
    ap.add_argument("--min-synth", type=int, default=1, help="cls: least number of synthetic volumes per class")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nbcode as nb
    t0 = time.time()
    {"finetune": stage_finetune, "sample": stage_sample, "cls": stage_cls, "eval": stage_eval}[args.stage](nb, args)
    print(f"STAGE {args.stage} {getattr(args, 'scheme', '')} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
