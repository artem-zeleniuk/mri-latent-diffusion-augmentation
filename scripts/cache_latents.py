"""Encode the preprocessed OASIS-3 volumes once with the frozen MONAI Brain LDM autoencoder and
store each latent together with the conditioning vector of its session.

Conditioning vector, four values between 0 and 1 (the four inputs of the pretrained UNet):
    sex      male = 1.0, female = 0.0
    age      clamp((age - 40) / 55), age at the session
    CDR      CDR / 3        of the clinical visit nearest to the session
    MMSE     MMSE / 30      of the same visit (0.5 if it has no MMSE)
The diagnosis enters the fine-tune through the last two values.

Only the train and validation sessions of the classifier split are encoded, so the UNet never
sees a subject of the classifier's test split. The script stops if a position of the vector
is constant over the training sessions, and writes aggregate statistics of the vectors.

Output in --out-dir: <session>__v2.pt ({session_id, latent, cond}), cache_manifest.json
({split: {train, val}, cache_suffix}) and cond_stats.json (aggregate, no session ids).

usage: python cache_latents.py --labels labels.json --ckpt-dir monai_ckpt --out-dir cache
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (Compose, EnsureChannelFirstd, Lambdad, LoadImaged, Orientationd,
                              ResizeWithPadOrCropd, ScaleIntensityd, Spacingd)
from torch.utils.data import DataLoader, Dataset

TARGET_SPATIAL = (160, 224, 160)
SUFFIX = "__v2.pt"
COND_NAMES = ["sex", "age", "cdr", "mmse"]


def cond_vector(info):
    """[sex, age, CDR/3, MMSE/30] of one labelled session, each between 0 and 1."""
    sex = {"male": 1.0, "female": 0.0}.get(info.get("sex"), 0.5)
    age = 0.5 if info.get("age") is None else max(0.0, min(1.0, (info["age"] - 40.0) / 55.0))
    cdr = max(0.0, min(1.0, info["cdr"] / 3.0))
    mmse = 0.5 if info.get("mmse") is None else max(0.0, min(1.0, info["mmse"] / 30.0))
    return [sex, age, cdr, mmse]


def _drop_trailing_singleton(x):
    return x[..., 0] if x.ndim == 4 and x.shape[-1] == 1 else x


def build_transform():
    """Volume in [0, 1], RAS orientation, 1 mm, padded or cropped to (160, 224, 160)."""
    return Compose([
        LoadImaged(keys="image", reader="NibabelReader", image_only=False, ensure_channel_first=False),
        Lambdad(keys="image", func=_drop_trailing_singleton),
        EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        ResizeWithPadOrCropd(keys="image", spatial_size=TARGET_SPATIAL, mode="constant", constant_values=0),
    ])


class Volumes(Dataset):
    def __init__(self, items):
        self.items, self.tf = items, build_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, info = self.items[i]
        vol = self.tf({"image": info["path"]})["image"]
        vol = vol.as_tensor() if hasattr(vol, "as_tensor") else vol
        return sid, vol.float()


def cond_stats(vectors):
    """Aggregate description of a list of conditioning vectors (no session ids)."""
    a = np.asarray(vectors, dtype=float)
    out = {"n": int(len(a))}
    for k, name in enumerate(COND_NAMES):
        col = a[:, k]
        cnt = Counter(np.round(col, 4).tolist())
        out[name] = {"distinct": len(cnt), "min": float(col.min()), "max": float(col.max()),
                     "mean": float(col.mean()), "most_common": [[v, c] for v, c in cnt.most_common(5)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ckpt-dir", required=True, help="MONAI bundle folder with models/autoencoder.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: this many sessions per split")
    args = ap.parse_args()

    import nbcode                                   # notebook code: split of the classifier, autoencoder settings
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = nbcode.load_split(args.labels, seed=args.seed)
    labels = json.load(open(args.labels))
    by_path = {v["path"]: k for k, v in labels.items()}
    split_ids = {name: [by_path[x["path"]] for x in splits[name]] for name in ("train", "val")}
    if args.limit:
        split_ids = {k: v[:args.limit] for k, v in split_ids.items()}
    test_subjects = {x["subject"] for x in splits["test"]}
    for name, ids in split_ids.items():
        assert not ({labels[s]["subject"] for s in ids} & test_subjects), "a test subject is in the fine-tune data"

    conds = {name: [cond_vector(labels[s]) for s in ids] for name, ids in split_ids.items()}
    stats = {name: cond_stats(v) for name, v in conds.items()}
    for cls, y in (("train_cn", 0), ("train_ad", 1)):
        stats[cls] = cond_stats([cond_vector(labels[s]) for s in split_ids["train"] if labels[s]["y"] == y])
    stats["definition"] = "[sex (male 1, female 0), clamp((age - 40) / 55), CDR / 3, MMSE / 30 (0.5 if missing)]"
    (out / "cond_stats.json").write_text(json.dumps(stats, indent=1))
    for name in COND_NAMES:
        print(f"train cond {name}: {stats['train'][name]}")
        if not args.limit:
            assert stats["train"][name]["distinct"] > 1, f"conditioning position {name} is constant: stop"
    ad, cn = stats["train_ad"], stats["train_cn"]
    if not args.limit:
        assert ad["cdr"]["min"] > cn["cdr"]["max"], "CDR does not separate the classes in the conditioning"

    vae = nbcode.AutoencoderKL(**nbcode.AUTOENCODER_KWARGS).to(args.device).eval()
    st = torch.load(Path(args.ckpt_dir) / "models" / "autoencoder.pt", map_location=args.device, weights_only=False)
    try:
        vae.load_state_dict(st, strict=True)
    except RuntimeError:
        vae.load_old_state_dict(st)
    for p in vae.parameters():
        p.requires_grad_(False)

    todo = [(s, labels[s]) for name in ("train", "val") for s in split_ids[name]
            if not (out / f"{s}{SUFFIX}").is_file()]
    print(f"sessions: train {len(split_ids['train'])}, val {len(split_ids['val'])}, to encode {len(todo)}")
    loader = DataLoader(Volumes(todo), batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=lambda b: b[0])
    t0 = time.time()
    for n, (sid, vol) in enumerate(loader, 1):
        with torch.no_grad():
            z_mu, _ = vae.encode(vol.unsqueeze(0).to(args.device))      # the mean, not a random sample
        assert tuple(z_mu.shape) == (1, 3, 20, 28, 20), z_mu.shape
        cond = torch.tensor([[cond_vector(labels[sid])]], dtype=torch.float32)     # (1, 1, 4)
        tmp = out / f"{sid}.tmp.pt"
        torch.save({"session_id": sid, "latent": z_mu.float().cpu(), "cond": cond}, tmp)
        tmp.rename(out / f"{sid}{SUFFIX}")
        if n % 100 == 0 or n == len(todo):
            print(f"  encoded {n}/{len(todo)}  elapsed {time.time() - t0:.0f}s", flush=True)
    missing = [s for name in ("train", "val") for s in split_ids[name] if not (out / f"{s}{SUFFIX}").is_file()]
    assert not missing, f"{len(missing)} latents missing"
    (out / "cache_manifest.json").write_text(json.dumps(
        {"vae_source": "monai_bundle", "cache_suffix": SUFFIX, "split": split_ids,
         "cond_definition": stats["definition"]}, indent=1))
    print(f"cache complete: {sum(len(v) for v in split_ids.values())} latents in {out}")


if __name__ == "__main__":
    main()
