#!/usr/bin/env python3
"""
RunTraining.py

Train ONE DeePMD-kit model from ASE database inputs.


    python RunTraining.py --stage prepare --model_idx 0 \
        --train_db train_data.db --valid_db valid_data.db --test_db test_data.db
    srun python RunTraining.py --stage train --model_idx 0
    python RunTraining.py --stage freeze --model_idx 0
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np


TYPE_MAP = ["C", "N", "H"]
ROOT = os.environ.get("HCN_AL_ROOT", os.path.expanduser("~/hcn-al-mlip"))
DATASETS_DIR = os.path.join(ROOT, "datasets")
MODELS_DIR = os.path.join(ROOT, "models")
PLOTS_DIR = os.path.join(ROOT, "plots")

SET_SIZE = 5000  # frames per DeePMD set.XXX directory

INPUT_TEMPLATE = {
    "model": {
        "type_map": TYPE_MAP,
        "descriptor": {
            "type": "se_e2_a",
            "sel": [1, 1, 1],
            "rcut_smth": 0.5,
            "rcut": 6.0,
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [120, 120, 120],
            "resnet_dt": True,
        },
    },
    "learning_rate": {
        "type": "exp",
        "start_lr": 1e-3,
        "stop_lr": 5e-8,
        "decay_steps": 2000,
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.1,
        "limit_pref_e": 1.0,
        "start_pref_f": 1000,
        "limit_pref_f": 1.0,
        "start_pref_v": 0,
        "limit_pref_v": 0,
    },
    "training": {
        "numb_steps": 25000,
        "disp_file": "lcurve.out",
        "disp_freq": 200,
        "save_freq": 2000,
    },
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def dp_bin():
    """Find the DeePMD executable in the active environment."""
    path = os.path.join(os.path.dirname(sys.executable), "dp")
    return path if os.path.exists(path) else "dp"


def is_tf(backend):
    return backend.lower() in ("tensorflow", "tf")


def iteration_dir(it):
    return os.path.join(MODELS_DIR, f"iter_{it:03d}")


def model_dir(it, model_idx):
    return os.path.join(iteration_dir(it), f"model_{model_idx}")


def slurm_ntasks():
    return int(os.environ.get("SLURM_NTASKS", "1"))


def slurm_procid():
    return int(os.environ.get("SLURM_PROCID", "0"))


def frozen_name(backend):
    return "frozen_model.pb" if is_tf(backend) else "frozen_model.pth"


# --------------------------------------------------------------------------
# ASE DB -> DeePMD npy
# --------------------------------------------------------------------------

def _write_system(path, atype, pbc, frames, type_map):
    """Write one DeePMD system (fixed atom types/order) from frame lists."""
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "type.raw"), "w") as f:
        f.write("\n".join(str(int(t)) for t in atype) + "\n")
    with open(os.path.join(path, "type_map.raw"), "w") as f:
        f.write("\n".join(type_map) + "\n")
    if not pbc:
        open(os.path.join(path, "nopbc"), "w").close()

    coord = np.asarray(frames["coord"], dtype=np.float64)
    force = np.asarray(frames["force"], dtype=np.float64)
    energy = np.asarray(frames["energy"], dtype=np.float64)
    box = np.asarray(frames["box"], dtype=np.float64)

    n = len(energy)
    for k, start in enumerate(range(0, n, SET_SIZE)):
        sl = slice(start, start + SET_SIZE)
        sdir = os.path.join(path, f"set.{k:03d}")
        os.makedirs(sdir, exist_ok=True)
        np.save(os.path.join(sdir, "coord.npy"), coord[sl].reshape(-1, coord.shape[1] * 3))
        np.save(os.path.join(sdir, "force.npy"), force[sl].reshape(-1, force.shape[1] * 3))
        np.save(os.path.join(sdir, "energy.npy"), energy[sl])
        if pbc:
            np.save(os.path.join(sdir, "box.npy"), box[sl].reshape(-1, 9))


def convert_asedb(db_path, out_dir, type_map, max_fmax=None):
    """
    Convert an ASE database to DeePMD npy systems.

    Entries are grouped by (atom-type sequence, pbc) because a DeePMD system
    requires identical atom types/order in every frame. Units are kept as
    ASE's (eV, eV/Angstrom), which DeePMD expects.

    Per row: energy, forces, pbc and cell are extracted; `fmax` is taken from
    the row if stored, otherwise computed from the forces.

    Returns (list_of_system_dirs, stats_dict).
    """
    from ase.db import connect

    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"ASE database not found: {db_path}")

    type_index = {s: i for i, s in enumerate(type_map)}
    groups = {}
    n_rows = n_skipped_fmax = 0
    fmax_all = []

    db = connect(db_path)
    for row in db.select():
        n_rows += 1

        symbols = row.symbols
        try:
            atype = tuple(type_index[s] for s in symbols)
        except KeyError as exc:
            raise ValueError(
                f"{db_path} row id={row.id}: element {exc} not in "
                f"type_map {type_map}."
            )

        energy = getattr(row, "energy", None)
        forces = getattr(row, "forces", None)
        if energy is None or forces is None:
            raise ValueError(
                f"{db_path} row id={row.id}: missing energy or forces."
            )

        forces = np.asarray(forces, dtype=np.float64)
        fmax = row.get("fmax", None)
        if fmax is None:
            fmax = float(np.max(np.linalg.norm(forces, axis=1)))
        fmax_all.append(float(fmax))

        if max_fmax is not None and fmax > max_fmax:
            n_skipped_fmax += 1
            continue

        pbc = np.asarray(row.pbc, dtype=bool)
        if pbc.any() and not pbc.all():
            raise ValueError(
                f"{db_path} row id={row.id}: mixed pbc {pbc.tolist()} is not "
                "supported by DeePMD (use all-periodic or non-periodic)."
            )
        periodic = bool(pbc.all())

        g = groups.setdefault(
            (atype, periodic),
            {"coord": [], "force": [], "energy": [], "box": []},
        )
        g["coord"].append(np.asarray(row.positions, dtype=np.float64))
        g["force"].append(forces)
        g["energy"].append(float(energy))
        g["box"].append(
            np.asarray(row.cell, dtype=np.float64) if periodic else np.zeros((3, 3))
        )

    if not groups:
        raise RuntimeError(f"No usable frames found in {db_path}.")

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    systems = []
    n_frames = 0
    for k, ((atype, periodic), frames) in enumerate(sorted(groups.items())):
        path = os.path.join(out_dir, f"sys_{k:03d}")
        _write_system(path, atype, periodic, frames, type_map)
        systems.append(os.path.abspath(path))
        n_frames += len(frames["energy"])

    stats = {
        "db": os.path.abspath(db_path),
        "rows": n_rows,
        "frames_written": n_frames,
        "skipped_fmax": n_skipped_fmax,
        "n_systems": len(systems),
        "fmax_min": float(np.min(fmax_all)),
        "fmax_max": float(np.max(fmax_all)),
    }
    print(
        f"[convert] {os.path.basename(db_path)}: {n_rows} rows -> "
        f"{n_frames} frames in {len(systems)} system(s) "
        f"(skipped by fmax: {n_skipped_fmax}; "
        f"fmax range {stats['fmax_min']:.3g}-{stats['fmax_max']:.3g})"
    )
    return systems, stats


# --------------------------------------------------------------------------
# Input + manifest
# --------------------------------------------------------------------------

def build_input(workdir, train_systems, valid_systems, steps, seed, template=None):
    cfg = json.loads(json.dumps(template or INPUT_TEMPLATE))

    cfg["model"]["type_map"] = list(TYPE_MAP)
    cfg["model"]["descriptor"]["seed"] = seed
    cfg["model"]["fitting_net"]["seed"] = seed
    cfg["training"]["seed"] = seed
    cfg["training"]["numb_steps"] = steps

    cfg["training"]["training_data"] = {
        "systems": train_systems,
        "batch_size": "auto",
    }
    if valid_systems:
        cfg["training"]["validation_data"] = {
            "systems": valid_systems,
            "batch_size": "auto",
            "numb_btch": 4,
        }

    path = os.path.join(workdir, "input.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def write_model_manifest(workdir, iteration, model_idx, seed, backend, frozen,
                         dbs, dataset_dirs, test_systems):
    """Manifest for this single model (consumed later by analysis scripts)."""
    manifest = {
        "iteration": int(iteration),
        "model_idx": int(model_idx),
        "seed": int(seed),
        "backend": backend,
        "model": os.path.abspath(frozen),
        "type_map": TYPE_MAP,
        "dbs": {k: (os.path.abspath(v) if v else None) for k, v in dbs.items()},
        "datasets": {k: (os.path.abspath(v) if v else None) for k, v in dataset_dirs.items()},
        "test_systems": test_systems,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "ntasks": slurm_ntasks(),
            "nnodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        },
    }
    path = os.path.join(workdir, "model_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def _base_env():
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def stage_prepare(args, workdir, seed):
    """Convert the three ASE DBs and write input.json."""
    datasets_dir = args.datasets_dir or DATASETS_DIR
    os.makedirs(workdir, exist_ok=True)

    train_systems, _ = convert_asedb(
        args.train_db, os.path.join(datasets_dir, "train"),
        TYPE_MAP, args.max_fmax)
    valid_systems, _ = convert_asedb(
        args.valid_db, os.path.join(datasets_dir, "valid"),
        TYPE_MAP, args.max_fmax)

    test_systems = None
    if args.test_db:
        # Test set is converted as-is (no fmax filtering).
        test_systems, _ = convert_asedb(
            args.test_db, os.path.join(datasets_dir, "test"),
            TYPE_MAP, None)

    # Persist system lists so later stages / analysis can find them.
    with open(os.path.join(workdir, "systems.json"), "w") as f:
        json.dump({"train": train_systems, "valid": valid_systems,
                   "test": test_systems}, f, indent=2)

    path = build_input(workdir, train_systems, valid_systems, args.steps, seed)
    print(f"[prepare] wrote {path}")


def stage_train(args, workdir):
    """
    Run `dp train`. Parallelism is entirely SLURM's job: launch this stage
    with `srun` and the requested tasks/nodes/GPUs. No launcher is invoked
    from Python.
    """
    input_json = os.path.join(workdir, "input.json")
    if not os.path.isfile(input_json):
        raise FileNotFoundError(
            f"{input_json} not found. Run --stage prepare first.")

    cmd = [dp_bin()]
    if not is_tf(args.backend):
        cmd.append("--pt")
    cmd += ["train"]

    # Resume from the latest checkpoint if requested and one exists.
    if args.restart and os.path.exists(os.path.join(workdir, "checkpoint")):
        cmd += ["--restart", "model.ckpt"]

    # TensorFlow data-parallel training under srun (Horovod/MPI): one
    # process per task; ask DeePMD to log per-worker output.
    if is_tf(args.backend) and slurm_ntasks() > 1:
        cmd.append("--mpi-log=workers")

    cmd.append("input.json")

    print(f"[train rank {slurm_procid()}/{slurm_ntasks()}] "
          f"workdir={workdir}\n  command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=workdir, env=_base_env(), check=True)


def stage_freeze(args, workdir, seed):
    """Freeze the trained model and write the manifest (single process)."""
    if slurm_procid() != 0:
        return  # only rank 0 freezes if launched under srun

    frozen = os.path.join(workdir, frozen_name(args.backend))
    cmd = [dp_bin()]
    if not is_tf(args.backend):
        cmd.append("--pt")
    cmd += ["freeze", "-o", frozen]

    subprocess.run(cmd, cwd=workdir, env=_base_env(), check=True)
    if not os.path.exists(frozen):
        raise RuntimeError(f"Freeze finished but {frozen} was not created.")

    test_systems = None
    sys_json = os.path.join(workdir, "systems.json")
    if os.path.isfile(sys_json):
        with open(sys_json) as f:
            test_systems = json.load(f).get("test")

    datasets_dir = args.datasets_dir or DATASETS_DIR
    manifest = write_model_manifest(
        workdir=workdir,
        iteration=args.iter,
        model_idx=args.model_idx,
        seed=seed,
        backend=args.backend,
        frozen=frozen,
        dbs={"train": args.train_db, "valid": args.valid_db, "test": args.test_db},
        dataset_dirs={
            "train": os.path.join(datasets_dir, "train"),
            "valid": os.path.join(datasets_dir, "valid"),
            "test": os.path.join(datasets_dir, "test") if test_systems else None,
        },
        test_systems=test_systems,
    )
    print(f"[freeze] model: {frozen}\n[freeze] manifest: {manifest}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Train a single DeePMD-kit model from ASE DB inputs. "
                    "Parallelism/node allocation is handled by SLURM.")

    p.add_argument("--train_db", default="train_data.db", help="ASE DB for training.")
    p.add_argument("--valid_db", default="valid_data.db", help="ASE DB for validation.")
    p.add_argument("--test_db", default="test_data.db",
                   help="ASE DB for testing (converted, not used in training).")
    p.add_argument("--datasets_dir", default=None,
                   help="Where converted DeePMD systems are written "
                        "(default: $HCN_AL_ROOT/datasets).")
    p.add_argument("--max_fmax", type=float, default=None,
                   help="Optional: drop train/valid frames with fmax above "
                        "this value (eV/A).")

    p.add_argument("--iter", type=int, default=0, help="Iteration number.")
    p.add_argument("--model_idx", type=int, default=0,
                   help="Index of this model (e.g. $SLURM_ARRAY_TASK_ID).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (default: 1000 + model_idx).")
    p.add_argument("--steps", type=int, default=25000,
                   help="DeePMD training steps.")
    p.add_argument("--out_dir", default=None,
                   help="Model working directory "
                        "(default: $HCN_AL_ROOT/models/iter_XXX/model_N).")
    p.add_argument("--backend", choices=["tensorflow", "pytorch"],
                   default="tensorflow", help="DeePMD backend (default: tensorflow).")
    p.add_argument("--stage", choices=["prepare", "train", "freeze", "all"],
                   default="all")
    p.add_argument("--fresh", action="store_true",
                   help="Delete this model's directory before starting.")
    p.add_argument("--restart", action="store_true",
                   help="Resume training from model.ckpt if a checkpoint exists.")

    args = p.parse_args()

    seed = args.seed if args.seed is not None else 1000 + args.model_idx
    workdir = args.out_dir or model_dir(args.iter, args.model_idx)

    if args.stage == "all" and slurm_ntasks() > 1:
        raise RuntimeError(
            "--stage all must run as a single task. For multi-task training "
            "run --stage prepare, then `srun ... --stage train`, then "
            "--stage freeze as separate steps in your SLURM script.")

    if args.fresh and args.stage in ("prepare", "all") and os.path.isdir(workdir):
        shutil.rmtree(workdir)

    frozen = os.path.join(workdir, frozen_name(args.backend))
    if args.stage == "all" and os.path.exists(frozen):
        print(f"Frozen model already exists, skipping: {frozen}")
        return

    if args.stage in ("prepare", "all"):
        stage_prepare(args, workdir, seed)
    if args.stage in ("train", "all"):
        stage_train(args, workdir)
    if args.stage in ("freeze", "all"):
        stage_freeze(args, workdir, seed)


if __name__ == "__main__":
    main()
