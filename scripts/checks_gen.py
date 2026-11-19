"""Checks of the generator that need no classifier. Only aggregate values are written.

    python checks_gen.py [--parts C,A,B] [--n-vol 60] [--n-pairs 24]

C  Is the conditioning vector used? Noise-prediction error on the validation latents with the true
   vectors, with vectors permuted between sessions, with only CDR and MMSE permuted, with the
   diagnosis part swapped to the other class, and with the dropout vector. Same noise and timesteps
   for every variant, fine-tuned and stock UNet.
A  Intensity range of the decoder output against real volumes, as the classifier sees them
   (minimum, background level after ScaleIntensity), and the autoencoder reconstruction error.
B  Counterfactual pairs: same noise, sex and age, only CDR and MMSE changed. Fine-tuned and stock
   UNet. The volumes are kept in $W/cf for the classifier check of the evaluation stage.

Paths come from the environment, as in run_stage.py.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbcode as nb
from cache_latents import build_transform, cond_vector
from run_stage import proxies

E = os.environ
W, OUT = Path(E["W"]), Path(E["OUT"])
MONAI, LABELS, CACHE, SYNTH = Path(E["MONAI"]), E["LABELS"], Path(E["CACHE"]), Path(E["SYNTH"])
FT = OUT / "ckpts" / "unet_oasis_finetuned_best_ema.pt"
RES = OUT / "results"
DEV = "cuda:0"
CF = W / "cf"
CF_CONDS = {"cn": (0.0, 29 / 30), "ad05": (0.5 / 3, 26 / 30), "ad1": (1.0 / 3, 21 / 30),
            "cdr_only": (1.0 / 3, 29 / 30), "mmse_only": (0.0, 21 / 30)}


def pipe(kind, steps=50):
    if kind == "ft":
        os.environ["BRAIN_LDM_UNET_OVERRIDE"] = str(FT)
    else:
        os.environ.pop("BRAIN_LDM_UNET_OVERRIDE", None)
    return nb.load_pipe(str(MONAI), device=DEV, num_inference_steps=steps)


def agg(rows):
    """mean, sd, min, max of every key over a list of dicts"""
    out = {}
    for k in rows[0]:
        v = np.asarray([r[k] for r in rows], float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max())}
    out["n"] = len(rows)
    return out


def paired(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "se": float(se),
            "t": float(d.mean() / se) if se > 0 else 0.0, "share_positive": float((d > 0).mean()), "n": int(len(d))}


# ---------------------------------------------------------------- check C
def part_c(save):
    labels = json.load(open(LABELS))
    manifest = json.loads((CACHE / "cache_manifest.json").read_text())
    ds = nb.CachedLatentDataset(CACHE, manifest["split"]["val"])
    zs, cs, ys = [], [], []
    for i in range(len(ds)):
        z, c, sid = ds[i]
        zs.append(z); cs.append(c); ys.append(labels[sid]["y"])
    z = torch.stack(zs).to(DEV)                    # (N, 3, 20, 28, 20)
    c = torch.stack(cs).to(DEV)                    # (N, 1, 4)
    y = torch.tensor(ys, device=DEV)
    n = len(z)
    g = torch.Generator(device="cpu").manual_seed(777)
    perm = torch.randperm(n, generator=g).to(DEV)
    variants = {"true": c}
    variants["perm_all"] = c[perm]
    v = c.clone(); v[:, :, 2:] = c[perm][:, :, 2:]; variants["perm_cdr_mmse"] = v
    v = c.clone(); v[:, :, :2] = c[perm][:, :, :2]; variants["perm_sex_age"] = v
    v = c.clone()
    v[y == 0, :, 2], v[y == 0, :, 3] = 0.5 / 3, 26 / 30          # a CN session described as a typical AD one
    v[y == 1, :, 2], v[y == 1, :, 3] = 0.0, 29 / 30              # an AD session described as a CN one
    variants["swap_class"] = v
    variants["dropout_vector"] = torch.full_like(c, 0.5)
    sched = nb.DDIMScheduler(**nb.SCHEDULER_KWARGS)
    steps = [50, 200, 400, 600, 800, 950]
    out = {"n_val": n, "n_val_ad": int((y == 1).sum()), "timesteps": steps, "unets": {}}
    for kind in ("ft", "stock"):
        p = pipe(kind)
        res = {}
        per = {k: [] for k in variants}                     # per-latent error, concatenated over the timesteps
        for t in steps:
            gn = torch.Generator(device=DEV).manual_seed(1234 + t)
            noise = torch.randn(z.shape, device=DEV, generator=gn)
            tt = torch.full((n,), t, device=DEV, dtype=torch.long)
            noisy = sched.add_noise(z, noise, tt)
            for k, cv in variants.items():
                errs = []
                with torch.no_grad():
                    for a in range(0, n, 32):
                        pred = p.unet(x=nb.build_unet_input(noisy[a:a + 32], cv[a:a + 32]), timesteps=tt[a:a + 32],
                                      context=cv[a:a + 32])
                        errs.append(((pred.float() - noise[a:a + 32]) ** 2).mean(dim=(1, 2, 3, 4)))
                per[k].append(torch.cat(errs))
        base = torch.stack(per["true"])                     # (T, N)
        res["mse_true_by_t"] = [float(x) for x in base.mean(1)]
        for k in variants:
            if k == "true":
                continue
            d = torch.stack(per[k]) - base                  # (T, N), positive if the true vector is better
            dm = d.mean(0).cpu().numpy()                    # per session, averaged over the timesteps
            res[k] = {"delta_by_t": [float(x) for x in d.mean(1)], "per_session": paired(dm),
                      "per_session_cn": paired(dm[(y == 0).cpu().numpy()]),
                      "per_session_ad": paired(dm[(y == 1).cpu().numpy()]),
                      "relative_to_true": float(d.mean() / base.mean())}
        out["unets"][kind] = res
        print("C", kind, json.dumps({k: (v["per_session"] if isinstance(v, dict) else v) for k, v in res.items()})[:1200], flush=True)
        del p
        torch.cuda.empty_cache()
    save("conditioning_use", out)


# ---------------------------------------------------------------- check A
def raw_stats(a):
    return dict(min=float(a.min()), max=float(a.max()), p0_1=float(np.percentile(a, 0.1)),
                p99_9=float(np.percentile(a, 99.9)), share_below_0=float((a < 0).mean()),
                share_below_m0_01=float((a < -0.01).mean()), share_above_1=float((a > 1).mean()))


def cls_stats(v):
    """v: (1, 160, 220, 160) after the preprocessing of the classifier"""
    a = v.squeeze().float().numpy()
    k = 12
    corners = np.concatenate([a[i, j, l].ravel() for i in (slice(0, k), slice(-k, None))
                              for j in (slice(0, k), slice(-k, None)) for l in (slice(0, k), slice(-k, None))])
    hi = np.percentile(a, 99.5)
    tis = a > 0.15 * hi
    return dict(corner_mean=float(corners.mean()), corner_p99=float(np.percentile(corners, 99)),
                tissue_median=float(np.median(a[tis])), p99_5=float(hi), mean=float(a.mean()),
                tissue_share=float(tis.mean()), share_exact_0=float((a == 0).mean()),
                first_slice_axis1_mean=float(a[:, 0, :].mean()), last_slice_axis1_mean=float(a[:, -1, :].mean()))


def load_raw(path):
    """decoder output as stored, without the limit of nb.load_synth"""
    return torch.load(path, map_location="cpu", weights_only=False)[0].float()


def part_a(save, n_vol):
    pre = nb.Compose([nb.ScaleIntensity(), nb.ResizeWithPadOrCrop(spatial_size=(160, 220, 160))])
    out = {}
    rows = {"synth_raw": [], "synth_cls": [], "synth_clamped_cls": []}
    files = sorted((SYNTH / "cn").glob("cn_*.pt"))[:n_vol] + sorted((SYNTH / "ad").glob("ad_*.pt"))[:n_vol]
    files = [f for f in files if ".tmp" not in f.name]
    for f in files:
        v = load_raw(str(f))
        rows["synth_raw"].append(raw_stats(v.numpy()))
        rows["synth_cls"].append(cls_stats(pre(v)))
        rows["synth_clamped_cls"].append(cls_stats(pre(v.clamp(0, 1))))
    for k, r in rows.items():
        if r:
            out[k] = agg(r)
    print("A synth", json.dumps({k: {m: round(out[k][m]["mean"], 5) for m in out[k] if m != "n"} for k in out}), flush=True)

    tr = nb.load_split(LABELS, seed=42)["train"]
    rng = random.Random(11)
    pick = rng.sample(tr, n_vol)
    tf = build_transform()
    p = pipe("stock")
    real = {"real_cls": [], "recon_raw": [], "recon_cls": [], "recon_clamped_cls": [], "recon_error": []}
    by_grid = {}
    for x in pick:
        v = nb.load_volume(x["path"])
        st = cls_stats(pre(v))
        real["real_cls"].append(st)
        by_grid.setdefault("x".join(str(s) for s in v.shape[1:]), []).append(st)
        vol = tf({"image": x["path"]})["image"]
        vol = (vol.as_tensor() if hasattr(vol, "as_tensor") else vol).float()
        with torch.no_grad():
            rec = p.latent2img(p.img2latent(vol.unsqueeze(0).to(DEV))).float().cpu()[0]
        real["recon_raw"].append(raw_stats(rec.numpy()))
        real["recon_cls"].append(cls_stats(pre(rec)))
        real["recon_clamped_cls"].append(cls_stats(pre(rec.clamp(0, 1))))
        a, b = vol.numpy().ravel(), rec.numpy().ravel()
        tis = a > 0.15 * np.percentile(a, 99.5)
        real["recon_error"].append(dict(mae=float(np.abs(a - b).mean()), mae_tissue=float(np.abs(a - b)[tis].mean()),
                                        tissue_mean=float(a[tis].mean()), corr=float(np.corrcoef(a, b)[0, 1])))
    for k, r in real.items():
        out[k] = agg(r)
    out["real_cls_by_grid"] = {g: agg(r) for g, r in by_grid.items()}
    print("A real", json.dumps({k: {m: round(out[k][m]["mean"], 5) for m in out[k] if m != "n"} for k in real}), flush=True)
    del p
    torch.cuda.empty_cache()
    save("intensity_ranges", out)


# ---------------------------------------------------------------- check B
def part_b(save, n_pairs):
    tr = nb.load_split(LABELS, seed=42)["train"]
    rng = random.Random(7)
    base = [cond_vector(x)[:2] for x in rng.sample(tr, n_pairs)]
    names = list(CF_CONDS)
    out = {"n_pairs": n_pairs, "conds": {k: {"cdr_over_3": v[0], "mmse_over_30": v[1]} for k, v in CF_CONDS.items()}, "unets": {}}
    for kind in ("ft", "stock"):
        (CF / kind).mkdir(parents=True, exist_ok=True)
        p = pipe(kind)
        prox = {k: [] for k in names}
        vox = {k: [] for k in names if k != "cn"}
        t0 = time.time()
        for i, (sex, age) in enumerate(base):
            gen = torch.Generator(device=DEV).manual_seed(990000 + i)
            z = torch.randn(1, 3, 20, 28, 20, device=DEV, generator=gen).repeat(len(names), 1, 1, 1, 1)
            cond = torch.tensor([[[sex, age, *CF_CONDS[k]]] for k in names], device=DEV, dtype=torch.float32)
            img = nb.ddim_sample(p, z, cond, 50, DEV).float().cpu()          # (5, 1, 160, 224, 160)
            for j, k in enumerate(names):
                torch.save(img[j:j + 1].half(), CF / kind / f"{k}_{i:02d}.pt")
                prox[k].append(proxies(img[j, 0].numpy()))
                if k != "cn":
                    vox[k].append(float((img[j] - img[0]).abs().mean() / img[0].abs().mean()))
            if i == 0:
                print(f"B {kind}: first set in {time.time() - t0:.0f}s", flush=True)
        res = {"proxies": {k: agg(prox[k]) for k in names}, "paired_vs_cn": {}}
        for k in names:
            if k == "cn":
                continue
            res["paired_vs_cn"][k] = {m: paired([a[m] - b[m] for a, b in zip(prox[k], prox["cn"])])
                                      for m in ("csf_c", "mean_c", "tissue")}
            res["paired_vs_cn"][k]["relative_abs_voxel_difference"] = agg([{"v": v} for v in vox[k]])["v"]
        out["unets"][kind] = res
        print("B", kind, json.dumps({k: {m: (round(v[m]["mean"], 5), round(v[m]["t"], 1)) for m in ("csf_c", "mean_c", "tissue")}
                                     for k, v in res["paired_vs_cn"].items()}), flush=True)
        del p
        torch.cuda.empty_cache()
    save("counterfactual_pairs", out)


# ---------------------------------------------------------------- check D
def part_d(save, n_per_class):
    """Does the generator copy training sessions? Distance (L2 in the latent space of the autoencoder)
    from every sample to its nearest training latent. Reference: the same distance for the real
    validation sessions (other subjects than the training ones), passed through the same
    decode, limit to [0, 1], encode round trip as the synthetic volumes. A copied training session
    would sit at about the round-trip error, far below the distance between different people."""
    labels = json.load(open(LABELS))
    manifest = json.loads((CACHE / "cache_manifest.json").read_text())

    def stack(ids):
        ds = nb.CachedLatentDataset(CACHE, ids)
        return torch.stack([ds[i][0] for i in range(len(ds))]).to(DEV), [ds[i][2] for i in range(len(ds))]

    ztr, id_tr = stack(manifest["split"]["train"])
    zva, _ = stack(manifest["split"]["val"])
    p = pipe("stock")

    def round_trip(z):
        with torch.no_grad():
            return p.img2latent(p.latent2img(z.unsqueeze(0)).float().clamp(0, 1))[0]

    def nearest(q, exclude=None):
        d = torch.cdist(q.flatten(1), ztr.flatten(1))                       # (n, n_train)
        if exclude is not None:
            d = d.masked_fill(exclude, float("inf"))
        return d.min(1).values.cpu().numpy()

    def quant(a):
        a = np.asarray(a, float)
        return {"n": int(len(a)), "min": float(a.min()), "p1": float(np.percentile(a, 1)), "p5": float(np.percentile(a, 5)),
                "median": float(np.median(a)), "mean": float(a.mean()), "p95": float(np.percentile(a, 95))}

    out = {"latent_dimension": int(ztr[0].numel()), "n_train": int(len(ztr))}
    zva_rt = torch.stack([round_trip(z) for z in zva])
    ztr_rt = torch.stack([round_trip(z) for z in ztr[:200]])
    out["round_trip_error_train"] = quant((ztr_rt - ztr[:200]).flatten(1).norm(dim=1).cpu().numpy())
    nn_val = nearest(zva_rt)
    out["validation_round_trip_to_nearest_train"] = quant(nn_val)
    out["validation_to_nearest_train"] = quant(nearest(zva))
    # training sessions among themselves: nearest session of the same subject, and of another subject
    subj = np.asarray([labels[s]["subject"] for s in id_tr])
    same = torch.tensor(subj[:, None] == subj[None, :], device=DEV)
    eye = torch.eye(len(ztr), dtype=torch.bool, device=DEV)
    d_same = nearest(ztr, exclude=~same | eye)
    out["train_to_nearest_session_of_same_subject"] = quant(d_same[np.isfinite(d_same)])
    out["train_to_nearest_session_of_other_subject"] = quant(nearest(ztr, exclude=same))
    for c in ("cn", "ad"):
        files = [f for f in sorted((SYNTH / c).glob(f"{c}_*.pt")) if ".tmp" not in f.name][:n_per_class]
        zs = []
        for f in files:
            with torch.no_grad():
                zs.append(p.img2latent(nb.load_synth(str(f)).unsqueeze(0).to(DEV))[0])
        nn_s = nearest(torch.stack(zs))
        r = quant(nn_s)
        r["share_below_min_of_validation"] = float((nn_s < nn_val.min()).mean())
        r["share_below_p1_of_validation"] = float((nn_s < np.percentile(nn_val, 1)).mean())
        r["share_below_p5_of_validation"] = float((nn_s < np.percentile(nn_val, 5)).mean())
        r["share_below_median_of_same_subject"] = float((nn_s < np.median(d_same[np.isfinite(d_same)])).mean())
        out[f"synthetic_{c}_to_nearest_train"] = r
    print("D", json.dumps({k: (round(v["median"], 2), round(v["min"], 2)) if isinstance(v, dict) else v for k, v in out.items()}), flush=True)
    del p
    torch.cuda.empty_cache()
    save("nearest_training_latent", out)


# ---------------------------------------------------------------- check E
def part_e(save, n_pairs, scales):
    """Exploration for a possible follow-up, outside the main comparison: the pairs of check B
    (same noise, sex and age) sampled from the fine-tuned UNet with classifier-free guidance,
        eps = eps_null + w * (eps_cond - eps_null),
    where eps_null is the prediction for the dropout vector (0.5 in all four positions)."""
    tr = nb.load_split(LABELS, seed=42)["train"]
    base = [cond_vector(x)[:2] for x in random.Random(7).sample(tr, 24)][:n_pairs]      # the first sets of check B
    names = ["cn", "ad05", "ad1"]
    out = {"n_pairs": n_pairs, "scales": list(scales), "by_scale": {}}
    p = pipe("ft")
    for w in scales:
        kind = f"ft_guidance_{w:g}"
        (CF / kind).mkdir(parents=True, exist_ok=True)
        prox = {k: [] for k in names}
        vox = {k: [] for k in names if k != "cn"}
        for i, (sex, age) in enumerate(base):
            gen = torch.Generator(device=DEV).manual_seed(990000 + i)
            z = torch.randn(1, 3, 20, 28, 20, device=DEV, generator=gen).repeat(len(names), 1, 1, 1, 1)
            cond = torch.tensor([[[sex, age, *CF_CONDS[k]]] for k in names], device=DEV, dtype=torch.float32)
            null = torch.full_like(cond, 0.5)
            p.scheduler.set_timesteps(num_inference_steps=50)
            with torch.no_grad():
                for t in p.scheduler.timesteps.to(DEV):
                    tb = torch.full((2 * len(names),), int(t), device=DEV, dtype=torch.long)
                    e_c, e_n = p.noise_pred(torch.cat([z, z]), tb, torch.cat([cond, null])).chunk(2)
                    z, _ = p.scheduler.step(e_n + w * (e_c - e_n), t, z, eta=0.0)
                img = p.latent2img(z).float().cpu()
            for j, k in enumerate(names):
                torch.save(img[j:j + 1].half(), CF / kind / f"{k}_{i:02d}.pt")
                prox[k].append(proxies(img[j, 0].numpy()))
                if k != "cn":
                    vox[k].append(float((img[j] - img[0]).abs().mean() / img[0].abs().mean()))
        res = {"proxies": {k: agg(prox[k]) for k in names}, "paired_vs_cn": {}}
        for k in names[1:]:
            res["paired_vs_cn"][k] = {m: paired([a[m] - b[m] for a, b in zip(prox[k], prox["cn"])])
                                      for m in ("csf_c", "mean_c", "tissue")}
            res["paired_vs_cn"][k]["relative_abs_voxel_difference"] = agg([{"v": v} for v in vox[k]])["v"]
        out["by_scale"][f"{w:g}"] = res
        print("E", w, json.dumps({k: {m: (round(v[m]["mean"], 5), round(v[m]["t"], 1)) for m in ("csf_c", "mean_c", "tissue")}
                                  for k, v in res["paired_vs_cn"].items()}), flush=True)
    save("guidance_exploration", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="C,A,B")
    ap.add_argument("--n-nn", type=int, default=250, help="part D: synthetic volumes per class")
    ap.add_argument("--n-guided", type=int, default=12)
    ap.add_argument("--scales", default="2,4")
    ap.add_argument("--n-vol", type=int, default=60)
    ap.add_argument("--n-pairs", type=int, default=24)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / "generator_checks.json"
    store = json.loads(path.read_text()) if path.exists() else {}

    def save(key, val):
        store[key] = val
        path.write_text(json.dumps(store, indent=1))
        print(f"saved {key} -> {path}", flush=True)

    t0 = time.time()
    for part in args.parts.split(","):
        {"C": lambda: part_c(save), "A": lambda: part_a(save, args.n_vol), "B": lambda: part_b(save, args.n_pairs), "D": lambda: part_d(save, args.n_nn),
         "E": lambda: part_e(save, args.n_guided, [float(x) for x in args.scales.split(",")])}[part]()
        print(f"part {part} done at {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
