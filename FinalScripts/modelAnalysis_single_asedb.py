#!/usr/bin/env python3
"""
Analysis of DeePMD models and logging of results in ASE DB.

Works with the output of the single-model RunTraining.py:
  - models:   models/iter_XXX/model_N/{frozen_model.pb|.pth, model_manifest.json}
  - datasets: <datasets_dir>/{train,valid,test}/sys_XXX/  (DeePMD npy systems
              converted from the ASE DBs; several systems per split, each with
              a fixed atom-type order, several set.XXX chunks, optional nopbc)

ASE DB Hierarchy:

-> Reactant-Product Path
    -> ALNEB Seed
        -> AIMD Trajectory
            -> Frame

The first implementation stores detailed predictions and error information. This can later be segmented
into per-path and per-seed DBs connected through a master DB without changing the overall logical schema.

"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase import Atoms
from ase.db import connect


TYPE_MAP = ["C", "N", "H"]

ROOT = os.environ.get("HCN_AL_ROOT", os.path.expanduser("~/hcn-al-mlip"))
DATASETS_DIR = os.path.join(ROOT, "datasets")
MODELS_DIR = os.path.join(ROOT, "models")
PLOTS_DIR = os.path.join(ROOT, "plots")
ANALYSIS_DB = os.path.join(ROOT, "analysis.db")

SPLIT_KEYS = {"training": "train", "validation": "valid", "test": "test"}


# DeePMD dataset
def read_system(path):
    """Read one DeePMD/npy system (all set.XXX chunks) into memory."""
    with open(os.path.join(path, "type.raw")) as f:
        atype = np.array([int(x) for x in f.read().split()], dtype=int)
    natoms = len(atype)

    tm_file = os.path.join(path, "type_map.raw")
    if os.path.exists(tm_file):
        with open(tm_file) as f:
            tm = f.read().split()
        if tm != TYPE_MAP:
            raise ValueError(
                f"{path}: type_map {tm} does not match TYPE_MAP {TYPE_MAP}."
            )

    set_dirs = sorted(glob.glob(os.path.join(path, "set.*")))
    if not set_dirs:
        raise FileNotFoundError(f"No set.* directories in {path}")

    periodic = not os.path.exists(os.path.join(path, "nopbc"))

    coord, force, energy, box = [], [], [], []
    for s in set_dirs:
        coord.append(np.load(os.path.join(s, "coord.npy")).reshape(-1, natoms, 3))
        force.append(np.load(os.path.join(s, "force.npy")).reshape(-1, natoms, 3))
        energy.append(np.asarray(np.load(os.path.join(s, "energy.npy"))).reshape(-1))
        if periodic:
            box.append(np.load(os.path.join(s, "box.npy")).reshape(-1, 3, 3))

    coord = np.concatenate(coord)
    energy = np.concatenate(energy)

    return {
        "name": os.path.basename(os.path.normpath(path)),
        "path": path,
        "pbc": periodic,
        "coord": coord,
        "box": np.concatenate(box) if periodic else np.zeros((len(energy), 3, 3)),
        "energy": energy,
        "force": np.concatenate(force),
        "atype": atype,
    }


def find_systems(dataset_dir):
    """A dataset dir is either one system (has type.raw) or a dir of systems."""
    if os.path.isfile(os.path.join(dataset_dir, "type.raw")):
        return [dataset_dir]

    systems = sorted(
        d for d in glob.glob(os.path.join(dataset_dir, "*"))
        if os.path.isfile(os.path.join(d, "type.raw"))
    )
    if not systems:
        raise FileNotFoundError(f"No DeePMD systems found under {dataset_dir}")
    return systems


def read_dataset(dataset_dir):
    """Read every system of a dataset directory."""
    return [read_system(p) for p in find_systems(dataset_dir)]


# Provenance
def load_provenance(path):
    """Load optional frame-level provenance.json."""
    p = os.path.join(path, "provenance.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


def load_dataset_provenance(dataset_dir, systems, provenance_path=None):
    """
    Frame-aligned provenance for the whole dataset (systems concatenated in
    sorted order). A dataset-level provenance.json (in `provenance_path` or
    `dataset_dir`) takes priority; otherwise per-system provenance.json files
    are used. Frames without provenance get an empty record.
    """
    n_total = sum(len(s["energy"]) for s in systems)

    top = load_provenance(provenance_path or dataset_dir)
    if top:
        if len(top) != n_total:
            raise ValueError(
                f"Provenance length ({len(top)}) != dataset frames ({n_total})"
            )
        return top

    prov = []
    for s in systems:
        sp = load_provenance(s["path"])
        n = len(s["energy"])
        if sp and len(sp) != n:
            raise ValueError(
                f"{s['path']}: provenance length ({len(sp)}) != frames ({n})"
            )
        prov.extend(sp if sp else [{} for _ in range(n)])
    return prov


def normalize_provenance_record(record, frame):
    """
    Normalize different upstream provenance layouts into the database schema.
    Required logical hierarchy:
    path_id -> neb_seed_id -> trajectory_id -> frame_number
    Missing values are retained as None rather than guessed.
    """
    record = record or {}

    path_id = record.get("path_id", record.get("reactant_product_path_id"))
    neb_seed_id = record.get("neb_seed_id", record.get("seed_id"))
    trajectory_id = record.get("trajectory_id", record.get("traj_id"))

    return {
        "path_id": path_id,
        "neb_seed_id": neb_seed_id,
        "trajectory_id": trajectory_id,
        "frame_number": int(record.get("frame_number", record.get("frame", frame))),
        "source_path": record.get("source_path", record.get("path")),
        "md_seed": record.get("md_seed", record.get("seed")),
        "md_step": record.get("md_step"),
        "time_fs": record.get("time_fs"),
        "temperature_K": record.get("temperature_K", record.get("temperature")),
        "timestep_fs": record.get("timestep_fs", record.get("timestep")),
        "source_type": record.get("source_type"),
    }


def make_structure_uid(path_id, neb_seed_id, trajectory_id, frame_number):
    """Create a stable workflow-level frame identifier."""
    parts = [path_id, neb_seed_id, trajectory_id, str(frame_number)]
    return "/".join(str(x) if x is not None else "UNKNOWN" for x in parts)


# ASE conversion
def frame_to_atoms(system, frame):
    """Convert one DeePMD frame to ASE Atoms with DFT reference data."""
    symbols = [TYPE_MAP[i] for i in system["atype"]]
    if system["pbc"]:
        atoms = Atoms(
            symbols=symbols,
            positions=system["coord"][frame],
            cell=system["box"][frame],
            pbc=True,
        )
    else:
        atoms = Atoms(
            symbols=symbols,
            positions=system["coord"][frame],
            pbc=False,
        )
    atoms.info["dft_energy"] = float(system["energy"][frame])
    atoms.arrays["dft_forces"] = np.asarray(system["force"][frame])

    return atoms


# Model inference
def load_committee(models):
    """Load every frozen model once. `models` maps name -> model path."""
    from deepmd.infer import DeepPot

    return {name: DeepPot(path) for name, path in models.items()}


def predict(dp, system):
    """Run one loaded DeepPot over all frames of one system."""
    n, natoms = system["coord"].shape[:2]
    cells = system["box"].reshape(n, -1) if system["pbc"] else None
    e, f, _ = dp.eval(
        system["coord"].reshape(n, -1),
        cells,
        system["atype"].tolist(),
    )
    return (
        np.asarray(e).reshape(n),
        np.asarray(f).reshape(n, natoms, 3),
    )


def predict_committee(committee, system):
    """Evaluate all committee members on one system."""
    predictions = {}
    for name, dp in committee.items():
        energy, force = predict(dp, system)
        predictions[name] = {
            "energy": energy,
            "force": force,
        }

    return predictions


# Error metrics
def frame_error_metrics(dft_energy, dft_force, pred_energy, pred_force):
    """Detailed frame-level scalar errors for one committee member."""
    de = float(pred_energy - dft_energy)
    df = pred_force - dft_force

    force_vector = np.linalg.norm(df, axis=1)

    return {
        "energy_error": de,
        "energy_abs_error": abs(de),

        "fx_rmse": float(np.sqrt(np.mean(df[:, 0] ** 2))),
        "fy_rmse": float(np.sqrt(np.mean(df[:, 1] ** 2))),
        "fz_rmse": float(np.sqrt(np.mean(df[:, 2] ** 2))),

        "force_rmse": float(np.sqrt(np.mean(df ** 2))),
        "force_mae": float(np.mean(np.abs(df))),
        "force_max_error": float(np.max(np.abs(df))),

        "force_vector_mean_error": float(np.mean(force_vector)),
        "force_vector_max_error": float(np.max(force_vector)),
        "force_vector_rmse": float(np.sqrt(np.mean(force_vector ** 2))),
    }


def committee_statistics(predictions):
    """Calculate ensemble energy/force statistics and disagreement."""
    energies = np.asarray(
        [v["energy"] for v in predictions.values()]
    )

    forces = np.asarray(
        [v["force"] for v in predictions.values()]
    )

    mean_energy = energies.mean(axis=0)
    std_energy = energies.std(axis=0)

    mean_force = forces.mean(axis=0)
    std_force = forces.std(axis=0)

    disagreement = np.sqrt(
        np.mean(
            np.sum((forces - mean_force[None, ...]) ** 2, axis=-1),
            axis=0,
        )
    )

    return {
        "energy_mean": mean_energy,
        "energy_std": std_energy,
        "force_mean": mean_force,
        "force_std": std_force,
        "force_disagreement_max": disagreement.max(axis=1),
        "force_disagreement_mean": disagreement.mean(axis=1),
    }


# ASE DB
def open_analysis_db(path=ANALYSIS_DB):
    """Open/create the ASE analysis database."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return connect(path)


def write_frame_to_db(
    db,
    atoms,
    provenance,
    iteration,
    split,
    predictions,
    committee,
):
    """
    Write one fully analyzed frame.

    Detailed mode stores:
      - DFT reference
      - per-model energy prediction
      - per-model full force prediction
      - per-model scalar errors
      - committee statistics
      - complete lineage metadata
    """
    frame_number = provenance["frame_number"]

    structure_uid = make_structure_uid(
        provenance["path_id"],
        provenance["neb_seed_id"],
        provenance["trajectory_id"],
        frame_number,
    )

    row_data = {
        "structure_uid": structure_uid,

        # Hierarchical lineage
        "path_id": provenance["path_id"],
        "neb_seed_id": provenance["neb_seed_id"],
        "trajectory_id": provenance["trajectory_id"],
        "frame_number": frame_number,

        # Source / workflow
        "source_type": provenance["source_type"],
        "source_path": provenance["source_path"],
        "analysis_iteration": int(iteration),
        "dataset_split": split,
        "dataset_system": provenance.get("system_name"),

        # MD metadata
        "md_seed": provenance["md_seed"],
        "md_step": provenance["md_step"],
        "time_fs": provenance["time_fs"],
        "temperature_K": provenance["temperature_K"],
        "timestep_fs": provenance["timestep_fs"],

        # Reference energy
        "dft_energy": float(atoms.info["dft_energy"]),

        # Committee statistics
        "committee_energy_mean": float(committee["energy_mean"]),
        "committee_energy_std": float(committee["energy_std"]),
        "committee_force_disagreement_max": float(
            committee["force_disagreement_max"]
        ),
        "committee_force_disagreement_mean": float(
            committee["force_disagreement_mean"]
        ),
    }

    # Store full committee predictions and detailed errors.
    for model_name, pred in predictions.items():
        pred_energy = float(pred["energy"])
        pred_force = np.asarray(pred["force"])

        metrics = frame_error_metrics(
            atoms.info["dft_energy"],
            atoms.arrays["dft_forces"],
            pred_energy,
            pred_force,
        )

        row_data[f"{model_name}_energy"] = pred_energy

        for key, value in metrics.items():
            row_data[f"{model_name}_{key}"] = value

        # Full force predictions are stored as ASE per-atom arrays.
        # Prefix avoids collision with the DFT reference force array.
        atoms.arrays[f"{model_name}_forces"] = pred_force

    # Store committee force mean/std as per-atom arrays.
    atoms.arrays["committee_force_mean"] = committee["force_mean"]
    atoms.arrays["committee_force_std"] = committee["force_std"]

    return db.write(atoms, data=row_data)


def analyze_dataset(
    committee,
    dataset_dir,
    db_path,
    iteration,
    split,
    provenance_path=None,
    path_id=None,
    neb_seed_id=None,
    trajectory_id=None,
    temperature_K=None,
    timestep_fs=None,
    md_seed=None,
    collect=False,
):
    """
    Analyze an entire DeePMD dataset (one or many systems) and append it to
    the ASE DB.

    Explicit CLI metadata override values are applied only when the
    corresponding provenance field is missing.

    Returns (n_frames, aggregate). `aggregate` holds the stacked reference and
    predicted data for plots/RMSE when collect=True, otherwise None.
    """
    systems = read_dataset(dataset_dir)
    prov = load_dataset_provenance(dataset_dir, systems, provenance_path)

    agg = None
    if collect:
        agg = {
            "energy": [], "force": [], "natoms": [],
            "pred": {name: {"energy": [], "force": []} for name in committee},
        }

    db = open_analysis_db(db_path)
    offset = 0  # global frame index across systems

    with db:
        for system in systems:
            n = len(system["energy"])
            predictions = predict_committee(committee, system)
            stats = committee_statistics(predictions)

            if agg is not None:
                agg["energy"].append(system["energy"])
                agg["force"].append(system["force"].reshape(-1, 3))
                agg["natoms"].append(np.full(n, len(system["atype"])))
                for name, pred in predictions.items():
                    agg["pred"][name]["energy"].append(pred["energy"])
                    agg["pred"][name]["force"].append(pred["force"].reshape(-1, 3))

            for local in range(n):
                gi = offset + local
                p = normalize_provenance_record(prov[gi], gi)

                # Explicit values fill only missing provenance values.
                p["path_id"] = p["path_id"] or path_id
                p["neb_seed_id"] = p["neb_seed_id"] or neb_seed_id
                p["trajectory_id"] = p["trajectory_id"] or trajectory_id
                p["temperature_K"] = (
                    p["temperature_K"]
                    if p["temperature_K"] is not None
                    else temperature_K
                )
                p["timestep_fs"] = (
                    p["timestep_fs"]
                    if p["timestep_fs"] is not None
                    else timestep_fs
                )
                p["md_seed"] = p["md_seed"] if p["md_seed"] is not None else md_seed
                p["system_name"] = system["name"]

                atoms = frame_to_atoms(system, local)

                frame_predictions = {
                    name: {
                        "energy": values["energy"][local],
                        "force": values["force"][local],
                    }
                    for name, values in predictions.items()
                }

                frame_committee = {
                    "energy_mean": stats["energy_mean"][local],
                    "energy_std": stats["energy_std"][local],
                    "force_mean": stats["force_mean"][local],
                    "force_std": stats["force_std"][local],
                    "force_disagreement_max": stats["force_disagreement_max"][local],
                    "force_disagreement_mean": stats["force_disagreement_mean"][local],
                }

                write_frame_to_db(
                    db,
                    atoms,
                    p,
                    iteration,
                    split,
                    frame_predictions,
                    frame_committee,
                )

            offset += n

    if agg is not None:
        agg["energy"] = np.concatenate(agg["energy"])
        agg["force"] = np.concatenate(agg["force"])
        agg["natoms"] = np.concatenate(agg["natoms"])
        for name in agg["pred"]:
            for key in ("energy", "force"):
                agg["pred"][name][key] = np.concatenate(agg["pred"][name][key])

    return offset, agg


# Aggregate metrics
def rmse_from_aggregate(agg):
    """Aggregate RMSE for every model over all systems of a dataset."""
    out = {}

    for model_name, pred in agg["pred"].items():
        de = pred["energy"] - agg["energy"]
        df = pred["force"] - agg["force"]

        out[model_name] = {
            "energy": float(np.sqrt(np.mean(de ** 2))),
            "energy_per_atom": float(
                np.sqrt(np.mean((de / agg["natoms"]) ** 2))
            ),
            "force": float(np.sqrt(np.mean(df ** 2))),
            "force_x": float(np.sqrt(np.mean(df[:, 0] ** 2))),
            "force_y": float(np.sqrt(np.mean(df[:, 1] ** 2))),
            "force_z": float(np.sqrt(np.mean(df[:, 2] ** 2))),
        }

    return out


# Plots
def plot_learning_curves(out_dir, path):
    """Plot train/validation energy and force learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for workdir in sorted(glob.glob(os.path.join(out_dir, "model_*"))):
        lc = os.path.join(workdir, "lcurve.out")
        if not os.path.exists(lc):
            continue
        tag = os.path.basename(workdir)

        with open(lc) as f:
            header = f.readline().lstrip("#").split()

        data = np.loadtxt(lc, comments="#")
        data = data[None, :] if data.ndim == 1 else data
        col = {name: j for j, name in enumerate(header)}

        for key, ax in (("rmse_e", axes[0]), ("rmse_f", axes[1])):
            for split_name, style in (("trn", "-"), ("val", "--")):
                name = f"{key}_{split_name}"
                if name in col and "step" in col:
                    ax.plot(
                        data[:, col["step"]],
                        data[:, col[name]],
                        style,
                        lw=1,
                        label=f"{tag} {split_name}",
                    )

    for ax, title, ylabel in zip(
        axes,
        ("Energy", "Force"),
        ("RMSE (eV)", "RMSE (eV/A)"),
    ):
        ax.set_yscale("log")
        ax.set(
            xlabel="step",
            ylabel=ylabel,
            title=title,
        )

    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_parity(agg, path, title="validation"):
    """Plot energy and force parity from the aggregated predictions."""
    names = list(agg["pred"])
    fig, axes = plt.subplots(
        2,
        len(names),
        figsize=(4 * len(names), 7),
        squeeze=False,
    )

    dft_e = agg["energy"]
    dft_f = agg["force"].reshape(-1)

    for i, model_name in enumerate(names):
        ep = agg["pred"][model_name]["energy"]
        fp = agg["pred"][model_name]["force"].reshape(-1)

        axes[0, i].scatter(dft_e, ep, s=8, alpha=0.5)
        axes[1, i].scatter(dft_f, fp, s=5, alpha=0.35)

        lo = min(dft_e.min(), ep.min())
        hi = max(dft_e.max(), ep.max())
        axes[0, i].plot([lo, hi], [lo, hi], "k--", lw=1)

        lo = min(dft_f.min(), fp.min())
        hi = max(dft_f.max(), fp.max())
        axes[1, i].plot([lo, hi], [lo, hi], "k--", lw=1)

        axes[0, i].set_title(f"{model_name} {title}")
        axes[0, i].set(xlabel="DFT energy", ylabel="DP energy")
        axes[1, i].set(xlabel="DFT force", ylabel="DP force")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Model discovery
def discover_models(iteration):
    """
    Find the frozen models of one iteration (one model_N dir per model, as
    written by RunTraining.py). Returns ({"model_N": path}, [manifests]).
    Model names keep the real model index, so DB columns stay tied to the
    directory even if some members are missing.
    """
    iter_dir = os.path.join(MODELS_DIR, f"iter_{iteration:03d}")

    found = []
    manifests = []
    for workdir in glob.glob(os.path.join(iter_dir, "model_*")):
        m = re.fullmatch(r"model_(\d+)", os.path.basename(workdir))
        if not m or not os.path.isdir(workdir):
            continue
        idx = int(m.group(1))

        path = None
        mpath = os.path.join(workdir, "model_manifest.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                manifest = json.load(f)
            manifests.append((idx, manifest))
            if manifest.get("model") and os.path.exists(manifest["model"]):
                path = manifest["model"]

        if path is None:
            for fname in ("frozen_model.pb", "frozen_model.pth"):
                cand = os.path.join(workdir, fname)
                if os.path.exists(cand):
                    path = cand
                    break

        if path is None:
            print(f"WARNING: no frozen model in {workdir}; skipping.")
            continue
        found.append((idx, path))

    if not found:
        raise FileNotFoundError(
            f"No frozen models found for iteration {iteration} in {iter_dir}"
        )

    found.sort()
    manifests.sort(key=lambda x: x[0])
    return (
        {f"model_{idx}": path for idx, path in found},
        [m for _, m in manifests],
    )


def manifest_dataset_dir(manifests, split):
    """Dataset dir recorded by the training manifests for a split, if any."""
    key = SPLIT_KEYS[split]
    dirs = {(m.get("datasets") or {}).get(key) for m in manifests} - {None}
    if len(dirs) > 1:
        raise ValueError(
            f"Models of this iteration point to different '{key}' datasets: "
            f"{sorted(dirs)}. Pass --dataset_dir explicitly."
        )
    return dirs.pop() if dirs else None


def main():
    p = argparse.ArgumentParser(description="Analyze DeePMD committee and populate ASE DB.")
    p.add_argument("--iteration", "--iter", dest="iteration", type=int, default=0)
    p.add_argument("--db", default=ANALYSIS_DB)
    p.add_argument("--dataset_dir", default=None,
                   help="A DeePMD system, or a directory of sys_XXX systems.")
    p.add_argument("--split", default="validation", choices=["training", "validation", "test", "candidate", "aimd"])
    p.add_argument("--provenance", default=None,
                   help="Directory containing a dataset-level provenance.json.")
    p.add_argument("--path_id", default=None)
    p.add_argument("--neb_seed_id", default=None)
    p.add_argument("--trajectory_id", default=None)
    p.add_argument("--temperature", dest="temperature_K", type=float, default=None)
    p.add_argument("--timestep", dest="timestep_fs", type=float, default=None)
    p.add_argument("--md_seed", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    models, manifests = discover_models(args.iteration)

    if args.dataset_dir is None:
        if args.split in SPLIT_KEYS:
            args.dataset_dir = (
                manifest_dataset_dir(manifests, args.split)
                or os.path.join(DATASETS_DIR, SPLIT_KEYS[args.split])
            )
        else:
            raise ValueError(
                "--dataset_dir is required for candidate/AIMD analysis."
            )

    do_plots = (not args.no_plots) and args.split in {"validation", "test"}

    committee = load_committee(models)

    nframes, agg = analyze_dataset(
        committee=committee,
        dataset_dir=args.dataset_dir,
        db_path=args.db,
        iteration=args.iteration,
        split=args.split,
        provenance_path=args.provenance,
        path_id=args.path_id,
        neb_seed_id=args.neb_seed_id,
        trajectory_id=args.trajectory_id,
        temperature_K=args.temperature_K,
        timestep_fs=args.timestep_fs,
        md_seed=args.md_seed,
        collect=do_plots,
    )

    # Plot only where a standard validation/test dataset is being analyzed.
    if do_plots:
        plot_dir = os.path.join(
            PLOTS_DIR,
            f"iter_{args.iteration:03d}",
        )
        os.makedirs(plot_dir, exist_ok=True)

        out_dir = os.path.join(
            MODELS_DIR,
            f"iter_{args.iteration:03d}",
        )

        plot_learning_curves(
            out_dir,
            os.path.join(plot_dir, "learning_curves.png"),
        )

        plot_parity(
            agg,
            os.path.join(
                plot_dir,
                f"{args.split}_parity.png",
            ),
            args.split,
        )

    summary = {
        "iteration": args.iteration,
        "db": os.path.abspath(args.db),
        "split": args.split,
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "n_models": len(models),
        "models": models,
        "n_frames": nframes,
    }
    if agg is not None:
        summary["rmse"] = rmse_from_aggregate(agg)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
