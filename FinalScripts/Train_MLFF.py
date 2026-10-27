#!/usr/bin/env python3
"""
train_mlff.py -- config-driven DeePMD-kit (TensorFlow, CPU) MLFF training from ASE DBs.

Stages
------
  check   : validate the config and that the three ASE DBs exist
  prepare : convert train/valid/test ASE DBs -> DeePMD npy data (run ONCE)
  train   : train one model, freezing a .pb every `freeze_every` steps while it
            runs, then plot the learning curve and evaluate on the test set
  all     : prepare + train for a single model (local use)

Data layout written by `prepare`
--------------------------------
  <work_dir>/data/<split>/sys_XXX/        one system per (atom types, pbc)
      set.<setname>__<seedname>/          one set per ASE DB origin marker
  The markers are the ASE DB row keys in config["data"]["set_keys"].

Per-model outputs: <work_dir>/model_<idx>/
  frozen_models/model_step<N>.pb, model_final.pb
  lcurve.out, lcurve.png, test_results.txt, train.log, mlff.log

The model index is --model_idx, else $SLURM_PROCID, else 0 (seed = seed_base + idx).
"""

import argparse
import copy
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("mlff")

DEFAULTS = {
    "type_map": ["C", "N", "H"],
    "work_dir": "./mlff_run",
    "seed_base": 1000,
    "backend_flag": "--tf",        # "" for DeePMD-kit v2.x
    "cleanup_checkpoints": False,
    "poll_seconds": 30,            # how often to look for new checkpoints
    "data": {"max_fmax": None, "set_keys": ["setname", "seedname"]},
    "training": {"freeze_every": 100_000, "batch_size": "auto", "numb_btch": 4},
}
REQUIRED = {
    "": ["data", "model", "learning_rate", "loss", "training"],
    "data": ["train_db", "valid_db", "test_db"],
    "training": ["numb_steps"],
}
# config["training"] keys that are ours, not DeePMD's
OWN_TRAIN_KEYS = {"freeze_every", "batch_size", "valid_batch_size", "numb_btch"}
UNLABELED = "unlabeled"


# --------------------------------------------------------------------------
# Config and small helpers
# --------------------------------------------------------------------------

def setup_logging(tag="", logfile=None):
    fmt = logging.Formatter(f"%(asctime)s %(levelname)-7s {tag}%(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)


def load_config(path):
    user = json.loads(Path(path).read_text())
    for section, keys in REQUIRED.items():
        scope = user.get(section, {}) if section else user
        missing = [k for k in keys if k not in scope]
        if missing:
            raise KeyError(f"config{'[' + section + ']' if section else ''} "
                           f"is missing: {missing}")

    cfg = {**DEFAULTS, **user}
    for section in ("data", "training"):
        cfg[section] = {**DEFAULTS[section], **user[section]}
    cfg["work_dir"] = Path(cfg["work_dir"]).resolve()

    tr = cfg["training"]
    tr.setdefault("save_freq", tr["freeze_every"])  # checkpoint at each milestone
    if tr["freeze_every"] % tr["save_freq"]:
        raise ValueError("training.freeze_every must be a multiple of training.save_freq")
    return cfg


def data_dir(cfg):
    return cfg["work_dir"] / "data"


def model_dir(cfg, idx):
    return cfg["work_dir"] / f"model_{idx}"


def dp_cmd(cfg, *args):
    """`dp [--tf] <args>`, preferring the dp next to the active python."""
    exe = Path(sys.executable).parent / "dp"
    cmd = [str(exe) if exe.exists() else "dp"]
    if cfg["backend_flag"]:
        cmd.append(cfg["backend_flag"])
    return cmd + [str(a) for a in args]


def run_dp(cfg, *args, cwd, **kwargs):
    """Run a dp subcommand on CPU only (thread counts come from the SLURM script)."""
    env = {**os.environ, "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "-1"}
    env.setdefault("OMP_NUM_THREADS", "1")
    return subprocess.run(dp_cmd(cfg, *args), cwd=cwd, env=env, **kwargs)


# --------------------------------------------------------------------------
# ASE DB -> DeePMD npy
# --------------------------------------------------------------------------

def row_to_frame(row, type_index, db_name):
    """Extract one frame from an ASE DB row; raises on unusable rows."""
    where = f"{db_name} row id={row.id}"
    missing = [s for s in set(row.symbols) if s not in type_index]
    if missing:
        raise ValueError(f"{where}: elements {missing} not in type_map {list(type_index)}")
    if getattr(row, "energy", None) is None or getattr(row, "forces", None) is None:
        raise ValueError(f"{where}: missing energy or forces")

    pbc = np.asarray(row.pbc, dtype=bool)
    if pbc.any() and not pbc.all():
        raise ValueError(f"{where}: mixed pbc {pbc.tolist()} unsupported by DeePMD")

    periodic = bool(pbc.all())
    return {
        "atype": tuple(type_index[s] for s in row.symbols),
        "periodic": periodic,
        "coord": np.asarray(row.positions, dtype=np.float64),
        "force": np.asarray(row.forces, dtype=np.float64),
        "energy": float(row.energy),
        "box": np.asarray(row.cell, dtype=np.float64) if periodic else np.zeros((3, 3)),
    }


def set_label(row, set_keys):
    """Origin marker such as 'MD__seed3'; a missing key becomes 'unlabeled'."""
    parts = [re.sub(r"[^A-Za-z0-9_.-]", "_", str(row.get(k)))
             if row.get(k) is not None else UNLABELED for k in set_keys]
    return "__".join(parts)


def write_system(path, atype, periodic, sets, type_map):
    """Write one DeePMD system; `sets` maps label -> list of frames."""
    path.mkdir(parents=True)
    (path / "type.raw").write_text("\n".join(map(str, atype)) + "\n")
    (path / "type_map.raw").write_text("\n".join(type_map) + "\n")
    if not periodic:
        (path / "nopbc").touch()

    for label, frames in sets.items():
        sdir = path / f"set.{label}"
        sdir.mkdir()
        n = len(frames)
        stack = lambda key: np.stack([f[key] for f in frames])
        np.save(sdir / "coord.npy", stack("coord").reshape(n, -1))
        np.save(sdir / "force.npy", stack("force").reshape(n, -1))
        np.save(sdir / "energy.npy", np.array([f["energy"] for f in frames]))
        if periodic:
            np.save(sdir / "box.npy", stack("box").reshape(n, 9))


def convert_asedb(db_path, out_dir, type_map, set_keys, max_fmax=None):
    """
    ASE DB -> DeePMD systems. Frames go to a system by (atom types, pbc), as
    DeePMD requires, and into set.<label> directories by the `set_keys` row keys.
    Units stay in eV and eV/Angstrom. Returns (system dirs, {system: {set: n}}).
    """
    from ase.db import connect

    db_path, out_dir = Path(db_path), Path(out_dir)
    if not db_path.is_file():
        raise FileNotFoundError(f"ASE database not found: {db_path}")

    type_index = {s: i for i, s in enumerate(type_map)}
    groups = {}  # (atype, periodic) -> {label: [frames]}
    n_rows = n_skipped = 0
    for row in connect(str(db_path)).select():
        n_rows += 1
        frame = row_to_frame(row, type_index, db_path.name)
        if max_fmax is not None:
            fmax = row.get("fmax")
            if fmax is None:
                fmax = float(np.linalg.norm(frame["force"], axis=1).max())
            if fmax > max_fmax:
                n_skipped += 1
                continue
        key = (frame["atype"], frame["periodic"])
        groups.setdefault(key, {}).setdefault(set_label(row, set_keys), []).append(frame)

    if not groups:
        raise RuntimeError(f"No usable frames found in {db_path}")

    shutil.rmtree(out_dir, ignore_errors=True)
    systems, summary = [], {}
    for k, ((atype, periodic), sets) in enumerate(sorted(groups.items())):
        path = out_dir / f"sys_{k:03d}"
        write_system(path, atype, periodic, dict(sorted(sets.items())), type_map)
        systems.append(str(path))
        summary[path.name] = {lab: len(fr) for lab, fr in sorted(sets.items())}

    n_frames = sum(sum(c.values()) for c in summary.values())
    log.info("%s: %d rows -> %d frames, %d system(s) (skipped by fmax: %d)",
             db_path.name, n_rows, n_frames, len(systems), n_skipped)
    for name, counts in summary.items():
        log.info("  %s: %s", name, ", ".join(f"{lab}={n}" for lab, n in counts.items()))
    return systems, summary


# --------------------------------------------------------------------------
# Stages: check, prepare
# --------------------------------------------------------------------------

def stage_check(cfg):
    for key in REQUIRED["data"]:
        if not Path(cfg["data"][key]).is_file():
            raise FileNotFoundError(f"{key} not found: {cfg['data'][key]}")
    log.info("config OK")


def stage_prepare(cfg):
    d = cfg["data"]
    log.info("splitting sets by ASE DB keys: %s", d["set_keys"])
    systems, summary = {}, {}
    for split, max_fmax in (("train", d["max_fmax"]), ("valid", d["max_fmax"]),
                            ("test", None)):  # the test set is never filtered
        systems[split], summary[split] = convert_asedb(
            d[f"{split}_db"], data_dir(cfg) / split, cfg["type_map"], d["set_keys"], max_fmax)

    (data_dir(cfg) / "systems.json").write_text(json.dumps(systems, indent=2))
    (data_dir(cfg) / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("DeePMD data written to %s", data_dir(cfg))


# --------------------------------------------------------------------------
# Stage: train
# --------------------------------------------------------------------------

def write_input(cfg, workdir, systems, seed):
    """Write DeePMD's input.json for this model."""
    tr = cfg["training"]
    inp = {k: copy.deepcopy(cfg[k]) for k in ("model", "learning_rate", "loss")}
    inp["model"]["type_map"] = list(cfg["type_map"])
    inp["model"].setdefault("descriptor", {})["seed"] = seed
    inp["model"].setdefault("fitting_net", {})["seed"] = seed

    inp["training"] = {
        **{k: v for k, v in tr.items() if k not in OWN_TRAIN_KEYS},
        "seed": seed,
        "disp_file": "lcurve.out",
        "save_ckpt": "model.ckpt",
        # keep every milestone checkpoint until it has been frozen
        "max_ckpt_keep": max(5, tr["numb_steps"] // tr["save_freq"] + 2),
        "training_data": {"systems": systems["train"], "batch_size": tr["batch_size"]},
        "validation_data": {"systems": systems["valid"],
                            "batch_size": tr.get("valid_batch_size", tr["batch_size"]),
                            "numb_btch": tr["numb_btch"]},
    }
    (workdir / "input.json").write_text(json.dumps(inp, indent=2))


def freeze_checkpoint(cfg, workdir, step, out_pb):
    """Freeze checkpoint `step` through a temp folder whose `checkpoint` file
    points at that step (dp freeze otherwise takes only the latest)."""
    tmp = workdir / f"_freeze_{step}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for f in workdir.glob(f"model.ckpt-{step}.*"):
        (tmp / f.name).symlink_to(f)
    (tmp / "checkpoint").write_text(f'model_checkpoint_path: "model.ckpt-{step}"\n')

    out_pb.parent.mkdir(exist_ok=True)
    run_dp(cfg, "freeze", "-c", tmp, "-o", out_pb, cwd=workdir, check=True)
    shutil.rmtree(tmp)
    if not out_pb.exists():
        raise RuntimeError(f"freeze finished but {out_pb} is missing")
    log.info("froze step %d -> %s", step, out_pb.name)


def freeze_pending(cfg, workdir, frozen_dir):
    """Freeze every finished checkpoint that falls on a milestone and has no .pb yet.
    TF lists a step in `checkpoint` only after its files are fully written."""
    state = workdir / "checkpoint"
    if not state.is_file():
        return
    every, last = cfg["training"]["freeze_every"], cfg["training"]["numb_steps"]
    for step in sorted({int(s) for s in re.findall(r'model\.ckpt-(\d+)"', state.read_text())}):
        out_pb = frozen_dir / f"model_step{step}.pb"
        if (step % every == 0 or step == last) and not out_pb.exists():
            freeze_checkpoint(cfg, workdir, step, out_pb)


def plot_lcurve(workdir):
    """Plot lcurve.out (energy and force RMSE, train vs validation)."""
    lines = (workdir / "lcurve.out").read_text().splitlines()
    header = [l.lstrip("# ").split() for l in lines if l.startswith("#")][-1]  # last restart
    rows = [r for r in (l.split() for l in lines if l.strip() and not l.startswith("#"))
            if len(r) == len(header)]
    cols = dict(zip(header, np.array(rows, dtype=float).T))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; keeping lcurve.out only")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in zip(axes, "ef", ("Energy RMSE (eV/atom)", "Force RMSE (eV/Å)")):
        for split, label in (("val", "validation"), ("trn", "training")):
            if f"rmse_{key}_{split}" in cols:
                ax.loglog(cols["step"] + 1, cols[f"rmse_{key}_{split}"], label=label)
        ax.set(xlabel="step", title=title)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(workdir / "lcurve.png", dpi=150)
    plt.close(fig)
    log.info("learning curve saved to %s", workdir / "lcurve.png")


def evaluate_on_test(cfg, workdir, model_pb, test_systems):
    """`dp test` on every test system; the summaries go to test_results.txt."""
    with (workdir / "test_results.txt").open("w") as out:
        for i, system in enumerate(test_systems):
            out.write(f"### {system}\n")
            out.flush()
            res = run_dp(cfg, "test", "-m", model_pb, "-s", system,
                         "-d", workdir / f"test_sys{i:03d}", "-n", 0,
                         cwd=workdir, stdout=out, stderr=subprocess.STDOUT)
            if res.returncode:
                log.warning("dp test failed on %s (exit %d)", system, res.returncode)
    log.info("test results -> %s", workdir / "test_results.txt")


def stage_train(cfg, idx, restart=False, fresh=False):
    workdir = model_dir(cfg, idx)
    if fresh:
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    setup_logging(f"[model {idx}] ", workdir / "mlff.log")

    systems_json = data_dir(cfg) / "systems.json"
    if not systems_json.is_file():
        raise FileNotFoundError(f"{systems_json} missing -- run --stage prepare first")
    systems = json.loads(systems_json.read_text())

    frozen_dir = workdir / "frozen_models"
    final_pb = frozen_dir / "model_final.pb"
    if final_pb.exists() and not restart:
        log.info("%s already exists, skipping", final_pb)
        return

    seed = cfg["seed_base"] + idx
    write_input(cfg, workdir, systems, seed)
    resume = ["--restart", "model.ckpt"] if restart and (workdir / "checkpoint").exists() else []
    log.info("training: seed %d, workdir %s%s", seed, workdir, " (resuming)" if resume else "")

    # dp train runs in the background; meanwhile freeze checkpoints as they appear
    with (workdir / "train.log").open("a") as dp_log:
        proc = subprocess.Popen(
            dp_cmd(cfg, "train", *resume, "input.json"), cwd=workdir,
            env={**os.environ, "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "-1"},
            stdout=dp_log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            freeze_pending(cfg, workdir, frozen_dir)
            time.sleep(cfg["poll_seconds"])
    if proc.returncode:
        raise RuntimeError(f"dp train failed (exit {proc.returncode}); see {workdir / 'train.log'}")

    freeze_pending(cfg, workdir, frozen_dir)  # the final checkpoint
    last_pb = frozen_dir / f"model_step{cfg['training']['numb_steps']}.pb"
    if not last_pb.exists():
        raise RuntimeError(f"expected final frozen model {last_pb} not found")
    shutil.copyfile(last_pb, final_pb)

    plot_lcurve(workdir)
    evaluate_on_test(cfg, workdir, final_pb, systems["test"])

    if cfg["cleanup_checkpoints"]:
        for f in workdir.glob("model.ckpt*"):
            f.unlink()
        log.info("checkpoints removed")
    log.info("done: %d frozen model(s) in %s", len(list(frozen_dir.glob("model_step*.pb"))), frozen_dir)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="JSON config file")
    p.add_argument("--stage", choices=["check", "prepare", "train", "all"], default="all")
    p.add_argument("--model_idx", type=int,
                   default=int(os.environ.get("SLURM_PROCID", "0")),
                   help="Model index (default: $SLURM_PROCID, else 0)")
    p.add_argument("--restart", action="store_true", help="Resume from model.ckpt if present")
    p.add_argument("--fresh", action="store_true", help="Delete this model's directory first")
    args = p.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    if args.stage in ("check", "all"):
        stage_check(cfg)
    if args.stage in ("prepare", "all"):
        stage_prepare(cfg)
    if args.stage in ("train", "all"):
        stage_train(cfg, args.model_idx, args.restart, args.fresh)


if __name__ == "__main__":
    main()
