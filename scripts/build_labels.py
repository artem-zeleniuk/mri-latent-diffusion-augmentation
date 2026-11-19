"""Labels and clinical variables for every preprocessed OASIS-3 session.

For each session with a registered T1w volume (`*_mni152.nii.gz`) the clinical visit nearest
in time gives the CDR and the MMSE. CDR 0 is CN, CDR 0.5 or higher is AD. Sessions of subjects
without any CDR score are dropped. Sex and the age at entry come from the same clinical table,
and the age at the session is the age at entry plus the session day.

Output: labels.json, {session_id: {path, subject, day, cdr, mmse, sex, age, days_to_clinical,
label, y}}. The classifier (Stage 3) reads its labels and makes its split from this file, and
cache_latents.py builds the conditioning vector of the UNet fine-tune from it.

usage: python build_labels.py --csv oasis3_clinical.csv --out labels.json
                              (--brains-root DIR | --files-list FILE --root-prefix DIR)
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


def load_clinical(csv_path):
    """Per subject: sex, age at entry, and the list of visits (day, cdr, mmse) that have a CDR."""
    visits, sex, age0 = defaultdict(list), {}, {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            sid = row.get("Subject", "").strip()
            if not sid.startswith("OAS3"):
                continue
            try:
                day = int(row.get("Label", "").split("_d")[-1])
            except ValueError:
                day = 0

            def num(key):
                v = row.get(key, "")
                try:
                    return float(v) if v not in ("", ".") else None
                except ValueError:
                    return None
            cdr, mmse = num("cdr"), num("mmse")
            if cdr is not None:
                visits[sid].append((day, cdr, mmse))
            g = row.get("Gender", "").strip().lower()
            if g in ("male", "female"):
                sex[sid] = g
            a = num("ageAtEntry")
            if a is not None:
                age0[sid] = a
    return visits, sex, age0


def find_brains(paths):
    """(path, subject, session day) of every volume, in sorted path order."""
    brains = []
    for f in sorted(paths, key=PurePosixPath):
        parts = str(f).split("/")
        subj = next((p for p in parts if p.startswith("sub-OAS")), None)
        sess = next((p for p in parts if p.startswith("ses-d")), None)
        if not subj or not sess:
            continue
        try:
            day = int(sess.replace("ses-d", ""))
        except ValueError:
            day = 0
        brains.append((str(f), subj.replace("sub-", ""), day))
    return brains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--brains-root", help="folder that is searched for *_mni152.nii.gz")
    ap.add_argument("--files-list", help="text file with one relative path per line (instead of --brains-root)")
    ap.add_argument("--root-prefix", default="", help="prefix put in front of the paths of --files-list")
    args = ap.parse_args()

    if args.files_list:
        rel = [l.strip().split(";")[0] for l in open(args.files_list) if l.strip()]
        rel = [r for r in rel if r.endswith("_mni152.nii.gz")]
        prefix = args.root_prefix.rstrip("/")
        paths = [f"{prefix}/{r}" if prefix else r for r in rel]
    else:
        paths = [str(p) for p in Path(args.brains_root).rglob("*_mni152.nii.gz")]
    visits, sex, age0 = load_clinical(args.csv)
    brains = find_brains(paths)
    print(f"volumes: {len(brains)}, subjects with a CDR score: {len(visits)}")

    seen, unique = set(), []
    for b in brains:                       # one volume per (subject, session): the first in path order
        if (b[1], b[2]) not in seen:
            seen.add((b[1], b[2]))
            unique.append(b)
    out, counts = {}, Counter()
    for path, subject, day in unique:
        v = visits.get(subject, [])
        if not v:
            counts["no_cdr"] += 1
            continue
        vday, cdr, mmse = min(v, key=lambda x: abs(x[0] - day))
        label = "AD" if cdr >= 0.5 else "CN"
        counts[label] += 1
        counts["mmse_missing"] += mmse is None
        out[f"{subject}_d{day:04d}"] = {
            "path": path, "subject": subject, "day": day, "cdr": cdr, "mmse": mmse,
            "sex": sex.get(subject), "age": (age0[subject] + day / 365.25) if subject in age0 else None,
            "days_to_clinical": vday - day, "label": label, "y": 1 if label == "AD" else 0,
        }
    print(f"unique sessions: {len(unique)}, labelled: {len(out)}, {dict(counts)}")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
