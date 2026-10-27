#!/usr/bin/env python3
"""Prepare HCN AIMD/NEB data for DeePMD training."""

import argparse
import glob
import json
import os
import re

import numpy as np

TYPE_MAP = ["C", "N", "H"]
ATYPE = np.array([0, 1, 2], dtype=int)
NATOMS = 3
BOX_L = 20.0
ROOT = os.environ.get("HCN_AL_ROOT", os.path.expanduser("~/hcn-al-mlip"))
PARSED_DIR = os.path.join(ROOT, "parsed")
FILTERED_DIR = os.path.join(ROOT, "filtered")
FPS_DIR = os.path.join(ROOT, "fps")
DATASETS_DIR = os.path.join(ROOT, "datasets")
PLOTS_DIR = os.path.join(ROOT, "plots")

_PATH_RE = re.compile(r"(?:path|rxn|reaction)[_-]?.*", re.I)
_SEED_RE = re.compile(r"seed[_-]?.*", re.I)
_WALKER_RE = re.compile(r"(?:vel|walker|traj)[_-]?.*", re.I)


def write_deepmd(out_dir, coord, box, energy, force):
    """Write one DeePMD/npy system."""
    os.makedirs(os.path.join(out_dir, "set.000"), exist_ok=True)
    n = len(energy)
    box = np.asarray(box).reshape(n, 9)
    np.save(os.path.join(out_dir, "set.000", "coord.npy"), np.asarray(coord).reshape(n, -1))
    np.save(os.path.join(out_dir, "set.000", "box.npy"), box)
    np.save(os.path.join(out_dir, "set.000", "energy.npy"), np.asarray(energy, dtype=float))
    np.save(os.path.join(out_dir, "set.000", "force.npy"), np.asarray(force).reshape(n, -1))
    with open(os.path.join(out_dir, "type.raw"), "w") as f:
        f.write("\n".join(map(str, ATYPE)) + "\n")
    with open(os.path.join(out_dir, "type_map.raw"), "w") as f:
        f.write("\n".join(TYPE_MAP) + "\n")
    open(os.path.join(out_dir, "nopbc"), "w").close()


def read_deepmd(in_dir):
    """Read one DeePMD/npy system."""
    s = os.path.join(in_dir, "set.000")
    natoms = len(open(os.path.join(in_dir, "type.raw")).read().split())
    return {
        "coord": np.load(os.path.join(s, "coord.npy")).reshape(-1, natoms, 3),
        "box": np.load(os.path.join(s, "box.npy")).reshape(-1, 3, 3),
        "energy": np.load(os.path.join(s, "energy.npy")),
        "force": np.load(os.path.join(s, "force.npy")).reshape(-1, natoms, 3),
        "atype": ATYPE.copy(),
        "type_map": TYPE_MAP.copy(),
    }


def provenance_from_path(path, root=None):
    """Recover path/seed/walker from the directory hierarchy."""
    path = os.path.abspath(path)
    if root:
        rel = os.path.relpath(path, os.path.abspath(root)).split(os.sep)
        dirs = rel[:-1]
        if len(dirs) >= 2:
            reaction, seed = dirs[0], dirs[1]
            walker = dirs[2] if len(dirs) >= 3 else seed
            return reaction, seed, walker

    parts = path.split(os.sep)[:-1]
    seed = next((p for p in reversed(parts) if _SEED_RE.fullmatch(p)), None)
    walker = next((p for p in reversed(parts) if _WALKER_RE.fullmatch(p)), None)
    reaction = next((p for p in reversed(parts) if _PATH_RE.fullmatch(p)), None)
    if seed is None and parts:
        seed = parts[-1]
    if walker is None:
        walker = seed
    if reaction is None:
        reaction = parts[-2] if len(parts) > 1 else "path_000"
    return reaction, seed or "seed_unknown", walker or "walker_unknown"


def _outcar_magmoms(path, nframes):
    """Read total magnetic moments when VASP printed them."""
    try:
        text = open(path, errors="ignore").read()
    except OSError:
        return None
    vals = [float(x) for x in re.findall(
        r"number of electron\s+[\d.\-]+\s+magnetization\s+([\d.\-]+)", text)]
    if not vals:
        return None
    return (vals + [vals[-1]] * nframes)[:nframes]


def parse_vasp(paths, root=None):
    """Parse every ionic step from every OUTCAR."""
    import dpdata

    coords, energies, forces, provenance, moments = [], [], [], [], []
    skipped = 0
    for path in paths:
        try:
            system = dpdata.LabeledSystem(path, fmt="vasp/outcar", type_map=TYPE_MAP)
        except Exception as exc:
            print(f"[parse] skip {path}: {str(exc)[:120]}")
            skipped += 1
            continue
        if len(system) == 0:
            print(f"[parse] skip {path}: no ionic steps")
            skipped += 1
            continue
        reaction, seed, walker = provenance_from_path(path, root)
        mags = _outcar_magmoms(path, len(system))
        for i in range(len(system)):
            coords.append(system["coords"][i])
            energies.append(float(system["energies"][i]))
            forces.append(system["forces"][i])
            moments.append(np.nan if mags is None else mags[i])
            provenance.append({
                "source": path,
                "path": reaction,
                "seed": seed,
                "walker": walker,
                "frame_in_walker": i,
            })
    if not energies:
        raise RuntimeError("No usable OUTCAR frames were parsed.")
    box = np.tile(np.eye(3) * BOX_L, (len(energies), 1, 1))
    return np.asarray(coords), box, np.asarray(energies), np.asarray(forces), provenance, np.asarray(moments), skipped


def parse_psi4(paths, root=None):
    """Parse labeled Psi4 npz frames for the HCN benchmark."""
    coords, energies, forces, provenance = [], [], [], []
    skipped = 0
    for path in paths:
        try:
            z = np.load(path)
            if not {"coord", "energy", "force"}.issubset(z.files):
                raise ValueError("missing coord/energy/force")
            if "ok" in z.files and not bool(np.asarray(z["ok"]).reshape(-1)[0]):
                raise ValueError("label marked failed")
            coords.append(np.asarray(z["coord"]).reshape(NATOMS, 3))
            energies.append(float(np.asarray(z["energy"]).reshape(-1)[0]))
            forces.append(np.asarray(z["force"]).reshape(NATOMS, 3))
            reaction, seed, walker = provenance_from_path(path, root)
            provenance.append({"source": path, "path": reaction, "seed": seed,
                               "walker": walker, "frame_in_walker": 0})
        except Exception:
            skipped += 1
    if not energies:
        raise RuntimeError("No usable Psi4 frames were found.")
    box = np.tile(np.eye(3) * BOX_L, (len(energies), 1, 1))
    return np.asarray(coords), box, np.asarray(energies), np.asarray(forces), provenance, np.full(len(energies), np.nan), skipped


def parse_sources(paths, fmt, root):
    """Parse the selected backend and preserve source provenance."""
    if fmt == "vasp":
        return parse_vasp(paths, root)
    return parse_psi4(paths, root)


def run_parse(paths, fmt="vasp", root=None, tag="swarm"):
    """Parse source files into a single provenance-aware dataset."""
    values = parse_sources(paths, fmt, root)
    coord, box, energy, force, prov, mag, skipped = values
    out = os.path.join(PARSED_DIR, tag)
    write_deepmd(out, coord, box, energy, force)
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    np.save(os.path.join(out, "magmom.npy"), mag)
    print(f"[parse] {len(energy)} frames, {len(paths) - skipped}/{len(paths)} files, "
          f"{len({p['path'] for p in prov})} paths, "
          f"{len({(p['path'], p['seed']) for p in prov})} seeds")
    return {"coord": coord, "box": box, "energy": energy, "force": force,
            "atype": ATYPE.copy(), "type_map": TYPE_MAP.copy()}, prov, mag


def min_distance(coords):
    """Return the closest pair distance in one frame."""
    d = coords[:, None, :] - coords[None, :, :]
    r = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(r, np.inf)
    return float(r.min())


def run_filter(data, prov, mag=None, f_max=50.0, r_min=0.5,
               energy_mad_k=None, check_mag=False, mag_tol=1e-2):
    """Reject invalid, overlapping, or clearly failed electronic-structure frames."""
    n = len(data["energy"])
    reasons = [[] for _ in range(n)]
    finite = (np.isfinite(data["energy"]) & np.isfinite(data["coord"]).all((1, 2)) &
              np.isfinite(data["force"]).all((1, 2)))
    for i in np.where(~finite)[0]:
        reasons[i].append("nan_or_inf")

    fmax = np.abs(data["force"]).reshape(n, -1).max(axis=1)
    for i in np.where(fmax > f_max)[0]:
        reasons[i].append(f"force_max>{f_max}")

    mind = np.array([min_distance(c) for c in data["coord"]])
    for i in np.where(mind < r_min)[0]:
        reasons[i].append(f"min_dist<{r_min}")

    # Do not remove high-energy reaction/transition-state frames by default.
    if energy_mad_k is not None:
        epa = data["energy"] / data["coord"].shape[1]
        mad = np.median(np.abs(epa - np.median(epa))) + 1e-12
        z = np.abs(epa - np.median(epa)) / (1.4826 * mad)
        for i in np.where(z > energy_mad_k)[0]:
            reasons[i].append(f"energy_outlier_z>{energy_mad_k}")

    if check_mag and mag is not None and np.isfinite(mag).any():
        dev = np.abs(mag - np.round(mag))
        for i in np.where(np.isfinite(mag) & (dev > mag_tol))[0]:
            reasons[i].append("non_integer_magnetic_moment")

    keep = np.array([not r for r in reasons])
    kept = {k: (v[keep] if k in {"coord", "box", "energy", "force"} else v)
            for k, v in data.items()}
    kept_prov = [p for p, ok in zip(prov, keep) if ok]
    report = {"n_in": n, "n_kept": int(keep.sum()), "n_rejected": int((~keep).sum()),
              "rejected": [{"index": int(i), "reasons": reasons[i]}
                           for i in np.where(~keep)[0]]}
    out = FILTERED_DIR
    os.makedirs(out, exist_ok=True)
    write_deepmd(out, kept["coord"], kept["box"], kept["energy"], kept["force"])
    with open(os.path.join(out, "filter_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[filter] kept {report['n_kept']}/{n}; rejected {report['n_rejected']}")
    return kept, kept_prov, report


def soap_descriptor(coord, type_map=TYPE_MAP, atype=ATYPE,
                    r_cut=5.0, n_max=6, l_max=4, sigma=0.4):
    """Build DScribe SOAP descriptors for frame-level diversity/FPS."""
    from ase import Atoms
    from dscribe.descriptors import SOAP
    symbols = [type_map[t] for t in atype]
    soap = SOAP(species=sorted(set(type_map)), r_cut=r_cut, n_max=n_max,
                l_max=l_max, sigma=sigma, periodic=False, average="outer")
    return np.asarray(soap.create(
        [Atoms(symbols=symbols, positions=c) for c in coord], n_jobs=1))


def deepmd_descriptor(model_path, coord, box):
    """Use the trained DeePMD descriptor when a bootstrap model is available."""
    from deepmd.infer import DeepPot
    dp = DeepPot(model_path)
    n = len(coord)
    raw = dp.eval_descriptor(coord.reshape(n, -1), box.reshape(n, -1), ATYPE.tolist())
    return np.asarray(raw).reshape(n, NATOMS, -1).mean(axis=1)


def fps_select(desc, n_select):
    """Greedily select geometrically diverse frames."""
    n_select = min(int(n_select), len(desc))
    if n_select <= 0:
        return np.array([], dtype=int)
    selected = [0]
    dmin = np.linalg.norm(desc - desc[0], axis=1)
    for _ in range(n_select - 1):
        nxt = int(np.argmax(dmin))
        selected.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(desc - desc[nxt], axis=1))
    return np.asarray(selected, dtype=int)


def run_fps(data, prov, n_select=None, model_path=None):
    """Use SOAP by default; optionally use a trained DeePMD descriptor."""
    desc = (deepmd_descriptor(model_path, data["coord"], data["box"])
            if model_path else soap_descriptor(data["coord"]))
    if n_select is None or n_select >= len(data["energy"]):
        idx = np.arange(len(data["energy"]))
    else:
        idx = fps_select(desc, n_select)
        print(f"[fps] selected {len(idx)}/{len(data['energy'])} frames")
    selected = {k: (v[idx] if k in {"coord", "box", "energy", "force"} else v)
                for k, v in data.items()}
    selected_prov = [prov[i] for i in idx]
    return selected, selected_prov, desc[idx], idx


def dedup_by_descriptor(desc, eps=1e-3):
    """Remove near-identical geometries without using labels."""
    if eps <= 0 or len(desc) < 2:
        return np.ones(len(desc), dtype=bool)
    keep = np.zeros(len(desc), dtype=bool)
    kept = []
    for i, d in enumerate(desc):
        if not kept or np.min(np.linalg.norm(np.asarray(kept) - d, axis=1)) > eps:
            keep[i] = True
            kept.append(d)
    return keep


def _unit_keys(prov, level):
    """Return fully qualified split units."""
    if level == "path":
        return np.array([p["path"] for p in prov])
    if level == "seed":
        return np.array([f"{p['path']}/{p['seed']}" for p in prov])
    if level == "walker":
        return np.array([f"{p['path']}/{p['seed']}/{p['walker']}" for p in prov])
    return np.array([f"{p['path']}/{p['seed']}/{p['walker']}/{p['frame_in_walker']}" for p in prov])


def _split_counts(n, frac_train, frac_valid):
    """Return integer train/valid/test counts with exact 70/20/10 behaviour for n=10."""
    if n < 3:
        raise ValueError(f"Need at least 3 walkers/OUTCARs per seed for a 70:20:10 split; found {n}")
    raw = np.array([frac_train, frac_valid, 1.0 - frac_train - frac_valid]) * n
    counts = np.floor(raw).astype(int)
    counts[np.argsort(-(raw - counts))[:n - counts.sum()]] += 1
    for i in range(3):
        if counts[i] == 0:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[i] = 1
    if counts.sum() != n or np.any(counts < 1):
        raise ValueError(f"Could not construct a non-empty split for {n} walkers: {counts}")
    return counts.tolist()


def run_splitter(data, prov, desc, frac_train=0.7, frac_valid=0.2,
                 dedup_eps=1e-3, seed=0, split_level="walker_within_seed"):
    """Split velocity-specific OUTCARs independently inside every seed."""
    keep = dedup_by_descriptor(desc, dedup_eps)
    data = {k: (v[keep] if k in {"coord", "box", "energy", "force"} else v) for k, v in data.items()}
    prov = [p for p, ok in zip(prov, keep) if ok]
    desc = desc[keep]
    rng = np.random.default_rng(seed)
    split = np.full(len(prov), "train", dtype=object)

    seed_keys = np.array([f"{p['path']}/{p['seed']}" for p in prov])
    walker_keys = np.array([f"{p['path']}/{p['seed']}/{p['walker']}" for p in prov])
    if split_level == "walker_within_seed":
        walkers_per_seed = {sk: len(np.unique(walker_keys[seed_keys == sk])) for sk in np.unique(seed_keys)}
        if min(walkers_per_seed.values()) < 3:
            if all(v == 1 for v in walkers_per_seed.values()):
                print("[split] legacy HCN layout detected: one OUTCAR per seed; "
                      "falling back to a 70:20:10 frame split within each seed.")
                split_level = "frame"
            else:
                bad = {k: v for k, v in walkers_per_seed.items() if v < 3}
                raise ValueError(f"Cannot perform 70:20:10 walker split; seeds with fewer than 3 OUTCARs: {bad}")
        else:
            for sk in np.unique(seed_keys):
                walkers = np.unique(walker_keys[seed_keys == sk])
                ntr, nva, nte = _split_counts(len(walkers), frac_train, frac_valid)
                rng.shuffle(walkers)
                split[np.isin(walker_keys, walkers[:ntr])] = "train"
                split[np.isin(walker_keys, walkers[ntr:ntr+nva])] = "valid"
                split[np.isin(walker_keys, walkers[ntr+nva:])] = "test"
    if split_level == "frame":
        # Legacy compatibility for datasets with one OUTCAR per seed.
        seed_keys = np.array([f"{p['path']}/{p['seed']}" for p in prov])
        for sk in np.unique(seed_keys):
            idx = np.where(seed_keys == sk)[0]
            rng.shuffle(idx)
            ntr, nva, nte = _split_counts(len(idx), frac_train, frac_valid)
            split[idx[:ntr]], split[idx[ntr:ntr+nva]], split[idx[ntr+nva:]] = "train", "valid", "test"
    else:
        raise ValueError("split_level must be 'walker_within_seed' or 'frame' for this workflow")

    counts = {k: int(np.sum(split == k)) for k in ("train", "valid", "test")}
    by_seed = {}
    for sk in sorted(set(seed_keys)):
        m = seed_keys == sk
        by_seed[sk] = {k: int(np.sum(m & (split == k))) for k in ("train", "valid", "test")}

    os.makedirs(DATASETS_DIR, exist_ok=True)
    for name in ("train", "valid", "test"):
        m = split == name
        out = os.path.join(DATASETS_DIR, name)
        write_deepmd(out, data["coord"][m], data["box"][m], data["energy"][m], data["force"][m])
        np.save(os.path.join(out, "descriptor.npy"), desc[m])
        with open(os.path.join(out, "provenance.json"), "w") as f:
            json.dump([p for p, ok in zip(prov, m) if ok], f, indent=2)

    report = {"split_level": split_level, "fractions_requested": [frac_train, frac_valid, 1-frac_train-frac_valid],
              "counts": counts, "by_seed": by_seed,
              "n_paths": len({p['path'] for p in prov}),
              "n_seeds": len(set(seed_keys)),
              "n_walkers": len(set(walker_keys))}
    with open(os.path.join(DATASETS_DIR, "split_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[split] {counts}; per-seed walker split written to {DATASETS_DIR}")
    return report


def visualise(tag="prepdata"):
    """Save a lightweight PCA plot without requiring UMAP."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    arrays, labels = [], []
    for name in ("train", "valid", "test"):
        p = os.path.join(DATASETS_DIR, name, "descriptor.npy")
        if os.path.exists(p):
            x = np.load(p)
            arrays.append(x)
            labels.extend([name] * len(x))
    if not arrays:
        return
    x = np.concatenate(arrays)
    x -= x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    p = x @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ("train", "valid", "test"):
        m = np.asarray(labels) == name
        ax.scatter(p[m, 0], p[m, 1], s=10, alpha=0.6, label=name)
    ax.set(xlabel="PC1", ylabel="PC2", title="HCN dataset coverage")
    ax.legend()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{tag}_pca.png"), dpi=150)
    plt.close(fig)


def collect_paths(root, fmt):
    """Find all source files under the requested reaction-data root."""
    pattern = "OUTCAR" if fmt == "vasp" else "frame_*.npz"
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def prep_data(paths, fmt="vasp", root=None, n_fps_select=None, model_path=None,
              f_max=50.0, r_min=0.5, energy_mad_k=None, check_mag=False,
              frac_train=0.7, frac_valid=0.2, dedup_eps=1e-3,
              split_level="walker_within_seed", tag="swarm", do_plots=True):
    """Run parse, filter, optional FPS, and group-aware split."""
    data, prov, mag = run_parse(paths, fmt, root, tag)
    data, prov, _ = run_filter(data, prov, mag, f_max, r_min, energy_mad_k, check_mag)
    if not len(data["energy"]):
        raise RuntimeError("All frames were rejected by the filter.")
    data, prov, desc, _ = run_fps(data, prov, n_fps_select, model_path)
    report = run_splitter(data, prov, desc, frac_train, frac_valid, dedup_eps,
                          split_level=split_level)
    if do_plots:
        visualise(tag)
    return report


def main():
    """Parse command-line options and prepare the dataset."""
    p = argparse.ArgumentParser(description="Prepare HCN OUTCAR/Psi4 data for DeePMD.")
    p.add_argument("--root", help="reaction-data root containing path/seed[/walker]/OUTCAR")
    p.add_argument("--paths", nargs="+", help="explicit source files")
    p.add_argument("--fmt", choices=("vasp", "psi4"), default="vasp")
    p.add_argument("--split_level", choices=("walker_within_seed", "frame"), default="walker_within_seed")
    p.add_argument("--n_fps_select", type=int, default=None, help="optional FPS cap; default keeps all frames")
    p.add_argument("--model", default=None, help="optional DeePMD model for FPS descriptor")
    p.add_argument("--f_max", type=float, default=50.0)
    p.add_argument("--r_min", type=float, default=0.5)
    p.add_argument("--energy_mad_k", type=float, default=None, help="optional energy-outlier filter")
    p.add_argument("--check_mag", action="store_true", help="enable optional magnetic-moment filtering")
    p.add_argument("--frac_train", type=float, default=0.7)
    p.add_argument("--frac_valid", type=float, default=0.2)
    p.add_argument("--dedup_eps", type=float, default=1e-3)
    p.add_argument("--tag", default="swarm")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()
    root = os.path.abspath(args.root) if args.root else None
    paths = args.paths or (collect_paths(root, args.fmt) if root else [])
    if not paths:
        raise SystemExit("No input files; provide --root or --paths.")
    prep_data(paths, args.fmt, root, args.n_fps_select, args.model,
              args.f_max, args.r_min, args.energy_mad_k, args.check_mag,
              args.frac_train, args.frac_valid, args.dedup_eps,
              args.split_level, args.tag, not args.no_plots)


if __name__ == "__main__":
    main()
