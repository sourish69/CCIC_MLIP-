#!/usr/bin/env python
"""MLFFMD.py -- run MLIP molecular dynamics in LAMMPS and render a movie.

    POSCAR + INCAR  ->  data.lmp + in.lammps  ->  LAMMPS  ->  dump  ->  frames/movie
                              ^
                        DeePMD committee

"""
import os
#import re
#import sys
import glob
import shutil
import argparse
import subprocess
import numpy as np

# ----------------------------------------------------------------- system: HCN
TYPE_MAP = ["C", "N", "H"]                  # LAMMPS type 1,2,3 in this order
MASSES = {"C": 12.011, "N": 14.007, "H": 1.008, "O": 15.999, "Cu": 63.546}
BOX_L = 20.0                                # vacuum cube edge (Angstrom)

ROOT = os.environ.get("HCN_AL_ROOT", os.path.expanduser("~/hcn-al-mlip"))
MODELS_DIR = os.path.join(ROOT, "models")
MD_DIR = os.path.join(ROOT, "lammps")


# ============================================================ 1. READ VASP INPUTS
def parse_incar(path):
    """Extract only the MD-relevant INCAR tags; report what was ignored."""
    used, ignored = {}, {}
    if not path or not os.path.exists(path):
        print("[incar] no INCAR given -- falling back to CLI/default MD settings")
        return used, ignored

    md_tags = {"POTIM", "NSW", "TEBEG", "TEEND", "SMASS", "MDALGO", "IBRION",
               "ISIF", "NBLOCK"}
    for raw in open(path):
        line = raw.split("#")[0].split("!")[0].strip()   # strip both comment styles
        if not line or "=" not in line:
            continue
        # one line may hold several "TAG = VALUE" pairs separated by ';'
        for chunk in line.split(";"):
            if "=" not in chunk:
                continue
            k, v = chunk.split("=", 1)
            k, v = k.strip().upper(), v.strip().split()[0] if v.strip() else ""
            (used if k in md_tags else ignored)[k] = v

    if ignored:
        print(f"[incar] ignored {len(ignored)} electronic-structure tag(s): "
              f"{', '.join(sorted(ignored))}")
    if used:
        print(f"[incar] using: {used}")
    # IBRION!=0 means this INCAR describes a relaxation, not MD -- worth saying
    if used.get("IBRION") not in (None, "0"):
        print(f"[incar] WARNING: IBRION={used.get('IBRION')} is not MD (IBRION=0). "
              "The MD settings below are being taken anyway; check they make sense.")
    return used, ignored


def incar_to_md(used, defaults):
    """Map INCAR MD tags onto LAMMPS settings, with CLI/defaults as fallback."""
    md = dict(defaults)
    if "POTIM" in used:
        md["timestep_fs"] = float(used["POTIM"])
    if "NSW" in used:
        md["nsteps"] = int(float(used["NSW"]))
    if "TEBEG" in used:
        md["temp_start"] = float(used["TEBEG"])
        md["temp_end"] = float(used["TEBEG"])       # default: isothermal
    if "TEEND" in used:
        md["temp_end"] = float(used["TEEND"])       # a ramp, if different
    if "MDALGO" in used:
        # 3 = Langevin in VASP; anything else here is treated as Nose-Hoover
        md["thermostat"] = "langevin" if used["MDALGO"].strip() == "3" else "nvt"
    if "SMASS" in used:
        try:
            # SMASS<0 in VASP means no thermostat (microcanonical)
            if float(used["SMASS"]) < 0:
                md["thermostat"] = "nve"
        except ValueError:
            pass
    if "NBLOCK" in used:
        md["dump_every"] = max(1, int(float(used["NBLOCK"])))
    return md


def parse_poscar(path):
    """Read POSCAR/CONTCAR -> (symbols, positions, cell). Handles Direct/Cartesian."""
    lines = [l.rstrip("\n") for l in open(path)]
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale

    # VASP 5+ names species on line 5; VASP 4 omits it and jumps straight to counts
    if lines[5].split() and not lines[5].split()[0].lstrip("-").isdigit():
        species = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        i = 7
    else:
        raise ValueError("POSCAR has no species line (VASP 4 format) -- add one, "
                         "e.g. 'C N H' above the counts")

    if lines[i].strip().lower().startswith("s"):      # optional Selective dynamics
        i += 1
    mode = lines[i].strip().lower()
    i += 1

    natoms = sum(counts)
    coords = np.array([[float(x) for x in lines[i + k].split()[:3]] for k in range(natoms)])
    if mode.startswith(("d", "f")):                   # Direct/Fractional -> Cartesian
        coords = coords @ cell
    else:
        coords = coords * scale

    symbols = [s for s, c in zip(species, counts) for _ in range(c)]
    return symbols, coords, cell


# ============================================================ 2. WRITE LAMMPS INPUTS
def write_lammps_data(path, symbols, positions, type_map=TYPE_MAP, box_l=BOX_L):
    """Write data.lmp: atoms centred in a vacuum cube, types ordered by type_map."""
    positions = np.asarray(positions, dtype=float)
    # centre the molecule so it sits in the middle of the box, away from walls
    positions = positions - positions.mean(axis=0) + box_l / 2.0

    unknown = sorted(set(symbols) - set(type_map))
    if unknown:
        raise ValueError(f"species {unknown} are not in TYPE_MAP {type_map}; "
                         "the LAMMPS type indices would not match the model")

    with open(path, "w") as f:
        f.write("LAMMPS data file -- MLIP MD\n\n")
        f.write(f"{len(symbols)} atoms\n{len(type_map)} atom types\n\n")
        f.write(f"0.0 {box_l} xlo xhi\n0.0 {box_l} ylo yhi\n0.0 {box_l} zlo zhi\n\n")
        f.write("Masses\n\n")
        for t, s in enumerate(type_map, start=1):
            f.write(f"{t} {MASSES.get(s, 1.0)}   # {s}\n")
        f.write("\nAtoms # atomic\n\n")
        for i, (s, p) in enumerate(zip(symbols, positions), start=1):
            t = type_map.index(s) + 1              # LAMMPS types are 1-based
            f.write(f"{i} {t} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
    return path


def write_lammps_input(path, models, md, out_freq=20, seed=12345):
    """Write in.lammps: DeePMD pair style + thermostat + dump."""
    # An empty or unsamplable run is a silent failure: LAMMPS exits 0 and you get
    # a movie with no frames. Catch it here rather than after the run.
    if md["nsteps"] <= 0:
        raise ValueError(f"nsteps={md['nsteps']}: nothing would be simulated "
                         "(check NSW in the INCAR, or pass --nsteps)")
    if md["dump_every"] > md["nsteps"]:
        raise ValueError(f"dump_every={md['dump_every']} exceeds nsteps={md['nsteps']}: "
                         "no frames would be written (lower --dump_every or NBLOCK)")
    if not models:
        raise ValueError("no models given -- LAMMPS has no potential to run with")
    missing = [m for m in models if not os.path.exists(m)]
    if missing:
        raise FileNotFoundError(f"model file(s) not found: {missing}")

    model_list = " ".join(os.path.abspath(m) for m in models)
    ts_ps = md["timestep_fs"] / 1000.0             # LAMMPS 'metal' units use ps
    t0, t1 = md["temp_start"], md["temp_end"]

    lines = [
        "units metal",                              # eV, Angstrom, ps -- DeePMD's units
        "atom_style atomic",
        "boundary s s s",                           # shrink-wrapped: isolated molecule
        "read_data data.lmp",
        "",
        "# First model drives the dynamics; the rest are evaluated every out_freq",
        "# steps only, to write the committee force deviation to model_devi.out.",
        f"pair_style deepmd {model_list} out_freq {out_freq} out_file model_devi.out",
        "pair_coeff * *",
        "",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
        "",
        # mom/rot yes removes net translation and rotation, which for an isolated
        # molecule would otherwise absorb energy and depress the internal temperature
        f"velocity all create {t0} {seed} mom yes rot yes dist gaussian",
        f"timestep {ts_ps}",
        "",
    ]

    # thermostat: fix ids are kept distinct so unfixing later is unambiguous
    if md["thermostat"] == "langevin":
        lines += [f"fix 1 all langevin {t0} {t1} 0.1 {seed}", "fix 2 all nve"]
    elif md["thermostat"] == "nve":
        # microcanonical: temperature is set once by the initial velocities and
        # then free to drift, so any TEEND ramp is meaningless here
        if abs(t1 - t0) > 1e-9:
            print(f"[lammps] NVE requested but TEBEG={t0} != TEEND={t1}; "
                  "the ramp is ignored (no thermostat to enforce it)")
        lines += ["fix 1 all nve"]
    else:
        lines += [f"fix 1 all nvt temp {t0} {t1} 0.1"]

    lines += [
        "",
        f"dump 1 all custom {md['dump_every']} dump.lammpstrj id type x y z fx fy fz",
        "dump_modify 1 sort id",                    # stable atom order for parsing
        "",
        f"thermo {max(1, md['dump_every'])}",
        "thermo_style custom step temp pe ke etotal",
        "",
        f"run {md['nsteps']}",
    ]
    open(path, "w").write("\n".join(lines) + "\n")
    return path


# ============================================================ 3. RUN
def lmp_bin():
    """Locate the LAMMPS binary; LMP_BIN overrides the search."""
    if os.environ.get("LMP_BIN"):
        return os.environ["LMP_BIN"]
    for n in ("lmp", "lmp_mpi", "lmp_serial"):
        p = shutil.which(n)
        if p:
            return p
    return None


def check_deepmd_pair(binary):
    """Verify this LAMMPS build actually has pair_style deepmd before running."""
    try:
        out = subprocess.run(binary.split() + ["-h"], capture_output=True,
                             text=True, timeout=60)
    except Exception as exc:
        print(f"[lammps] could not query {binary}: {exc}")
        return False
    return "deepmd" in (out.stdout + out.stderr).lower()


def run_lammps(workdir, binary=None, np_ranks=1):
    """Execute LAMMPS in workdir; returns the dump path."""
    binary = binary or lmp_bin()
    if binary is None:
        raise RuntimeError("no LAMMPS binary found -- set $LMP_BIN")
    if not check_deepmd_pair(binary):
        raise RuntimeError(
            f"{binary} does not list pair_style deepmd. This is the known "
            "conda-forge packaging issue: a plain LAMMPS gets resolved instead of "
            "the deepmd-enabled one, and `lmp -h` runs fine either way. See "
            "install_env.sh for the two fixes (clean-room reinstall, or "
            "plugin mode via LAMMPS_PLUGIN_PATH).")

    # a few atoms in a vacuum box: MPI communication costs more than it saves
    cmd = (["mpirun", "-np", str(np_ranks)] if np_ranks > 1 else []) + \
          binary.split() + ["-in", "in.lammps"]
    log = open(os.path.join(workdir, "lammps.log"), "w")
    r = subprocess.run(cmd, cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    if r.returncode != 0:
        raise RuntimeError(f"LAMMPS exited {r.returncode} -- see {workdir}/lammps.log")
    return os.path.join(workdir, "dump.lammpstrj")


# ============================================================ 4. READ TRAJECTORY
def parse_dump(path, type_map=TYPE_MAP):
    """Read a LAMMPS custom dump -> (positions (nframes,natoms,3), symbols)."""
    frames, symbols = [], None
    if not os.path.exists(path):
        print(f"[dump] {path} does not exist -- LAMMPS wrote no trajectory")
        return np.zeros((0, 0, 3)), []
    lines = open(path).readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            n = int(lines[i + 1])
            i += 2
        elif lines[i].startswith("ITEM: ATOMS"):
            cols = lines[i].split()[2:]                  # column names after 'ITEM: ATOMS'
            ci = {c: k for k, c in enumerate(cols)}
            block = lines[i + 1:i + 1 + n]
            # sort by atom id: dump_modify sort id should do this, but a dump
            # written without it would otherwise scramble the atom order
            rows = sorted((int(r.split()[ci["id"]]), r.split()) for r in block)
            frames.append(np.array([[float(r[ci["x"]]), float(r[ci["y"]]),
                                     float(r[ci["z"]])] for _, r in rows]))
            if symbols is None:
                symbols = [type_map[int(r[ci["type"]]) - 1] for _, r in rows]
            i += 1 + n
        else:
            i += 1
    if not frames:
        print(f"[dump] {path} exists but contains no frames -- check nsteps/dump_every")
        return np.zeros((0, 0, 3)), []
    return np.array(frames), (symbols or [])


def parse_model_devi(path):
    """Read model_devi.out -> (array, column index map); (None, None) if absent."""
    if not os.path.exists(path):
        return None, None
    header = open(path).readline().lstrip("#").split()
    d = np.loadtxt(path, comments="#")
    d = d[None, :] if d.ndim == 1 else d              # single-row file guard
    return d, {name: i for i, name in enumerate(header)}


# ============================================================ 5. RENDER
_ATOM_STYLE = {"C": ("#404040", 260), "N": ("#3050F8", 280), "H": ("#E8E8E8", 130),
               "O": ("#FF0D0D", 280), "Cu": ("#C88033", 400)}
_BOND_CUTOFF = {("C", "H"): 1.35, ("N", "H"): 1.30, ("C", "N"): 1.45,
                ("C", "C"): 1.70, ("N", "N"): 1.50}


def _bonded(sa, sb, r):
    """Draw a bond only if the pair is closer than a per-element-pair cutoff."""
    cut = _BOND_CUTOFF.get((sa, sb)) or _BOND_CUTOFF.get((sb, sa)) or 1.6
    return r < cut


def render_movie(frames, symbols, out_dir, stride=1, fps=15, elev=20, azim=-60,
                 pad=0.9, devi=None, make_gif=True, zoom=1.0):
    """Render frames with matplotlib and stitch a GIF.

    The viewport is fixed across the whole trajectory, computed once from the
    global coordinate extent. Autoscaling per frame makes the molecule appear to
    jump and zoom, which looks like motion that is not there.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401 (registers 3d)

    os.makedirs(out_dir, exist_ok=True)
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[0] == 0:
        print("[render] no frames to render")
        return None
    if len(symbols) != frames.shape[1]:
        print(f"[render] {len(symbols)} symbols but {frames.shape[1]} atoms per "
              "frame -- refusing to render a mislabelled molecule")
        return None
    sub = frames[::stride]
    if len(sub) < 2:
        # a one-frame 'movie' is almost always an accident (stride too large)
        print(f"[render] stride={stride} leaves {len(sub)} frame(s) of "
              f"{len(frames)}; lower --stride for an actual movie")

    # one fixed cubic viewport for every frame
    allp = frames.reshape(-1, 3)
    ctr = allp.mean(axis=0)
    # zoom<1 tightens the crop; the cube stays FIXED, so nothing jitters
    half = (max(np.abs(allp - ctr).max(), 1.0) + pad) * zoom
    lims = [(ctr[k] - half, ctr[k] + half) for k in range(3)]

    png = []
    for fi, pos in enumerate(sub):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)   # use the canvas
        # bonds first so spheres draw on top of them
        for a in range(len(symbols)):
            for b in range(a + 1, len(symbols)):
                r = np.linalg.norm(pos[a] - pos[b])
                if _bonded(symbols[a], symbols[b], r):
                    ax.plot(*zip(pos[a], pos[b]), color="0.45", lw=2.5, zorder=1)
        for s, p in zip(symbols, pos):
            c, size = _ATOM_STYLE.get(s, ("#AAAAAA", 200))
            ax.scatter(*p, s=size, c=c, edgecolors="k", linewidths=.6,
                       depthshade=False, zorder=2)
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        title = f"step {fi * stride}"
        if devi is not None and fi * stride < len(devi):
            title += f"    $\\epsilon$ = {devi[fi * stride]:.3f} eV/$\\AA$"
        ax.set_title(title, fontsize=10)
        f = os.path.join(out_dir, f"frame_{fi:05d}.png")
        fig.savefig(f, dpi=110)   # NOT bbox_inches="tight": that re-crops per frame
        plt.close(fig)
        png.append(f)

    print(f"[render] {len(png)} PNG frames -> {out_dir}")
    if not make_gif:
        return None

    # Pillow ships with matplotlib, so GIF needs no extra dependency
    try:
        from PIL import Image
        imgs = [Image.open(f) for f in png]
        gif = os.path.join(out_dir, "movie.gif")
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0)
        print(f"[render] movie -> {gif}")
        return gif
    except Exception as exc:
        print(f"[render] GIF assembly failed ({exc}); PNG frames are still in {out_dir}")
        return None


# ============================================================ 6. ORCHESTRATOR
def run_md(poscar, incar=None, models=None, workdir=None, tag="md",
           timestep_fs=0.25, nsteps=20000, temp=1000.0, dump_every=20,
           out_freq=20, thermostat="nvt", seed=12345, np_ranks=1,
           render=True, stride=5, zoom=1.0, dry_run=False):
    """POSCAR/INCAR -> LAMMPS inputs -> MD -> trajectory (+ optional movie)."""
    workdir = workdir or os.path.join(MD_DIR, tag)
    os.makedirs(workdir, exist_ok=True)

    models = models or sorted(glob.glob(os.path.join(MODELS_DIR, "iter_*", "model_*",
                                                     "frozen_model.pb")))
    if not models:
        raise RuntimeError(f"no frozen models found under {MODELS_DIR} -- "
                           "train a committee first, or pass --models")
    # non-empty check: an interrupted freeze leaves a 0-byte .pb that LAMMPS
    # accepts at parse time and then fails on mid-run
    empty = [m for m in models if os.path.getsize(m) == 0]
    if empty:
        raise RuntimeError(f"model file(s) are empty (interrupted `dp freeze`?): {empty}")
    print(f"[md] {len(models)} committee model(s); model[0] drives the dynamics")

    # INCAR overrides the defaults; CLI values are the fallback
    used, _ = parse_incar(incar)
    md = incar_to_md(used, dict(timestep_fs=timestep_fs, nsteps=nsteps,
                                temp_start=temp, temp_end=temp,
                                dump_every=dump_every, thermostat=thermostat))
    print(f"[md] {md['nsteps']} steps x {md['timestep_fs']} fs = "
          f"{md['nsteps'] * md['timestep_fs'] / 1000:.1f} ps, "
          f"T {md['temp_start']}->{md['temp_end']} K, {md['thermostat']}")

    symbols, positions, _ = parse_poscar(poscar)
    print(f"[md] {len(symbols)} atoms: {' '.join(symbols)}")
    write_lammps_data(os.path.join(workdir, "data.lmp"), symbols, positions)
    write_lammps_input(os.path.join(workdir, "in.lammps"), models, md,
                       out_freq=out_freq, seed=seed)
    print(f"[md] wrote data.lmp + in.lammps -> {workdir}")
    if dry_run:
        print("[md] --dry_run: inputs written, LAMMPS not launched")
        return {"workdir": workdir, "dump": None, "movie": None}

    dump = run_lammps(workdir, np_ranks=np_ranks)
    frames, syms = parse_dump(dump)
    print(f"GATE MD: {len(frames)} frames -> {dump}")

    # committee deviation, if the pair style wrote it
    d, col = parse_model_devi(os.path.join(workdir, "model_devi.out"))
    eps = None
    if d is not None and col and "max_devi_f" in col:
        eps_rows = d[:, col["max_devi_f"]]
        print(f"[md] committee deviation: {eps_rows.min():.3f} .. {eps_rows.max():.3f} eV/A")
        # model_devi rows are written every out_freq steps, dump every dump_every:
        # stretch onto the dump frame index so the movie caption lines up
        eps = np.interp(np.arange(len(frames)),
                        np.linspace(0, len(frames) - 1, len(eps_rows)), eps_rows)

    movie = None
    if render and len(frames):
        movie = render_movie(frames, syms, os.path.join(workdir, "render"),
                             stride=stride, devi=eps, zoom=zoom)
    return {"workdir": workdir, "dump": dump, "frames": frames, "movie": movie}


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="POSCAR/INCAR -> LAMMPS MD with a DeePMD committee -> movie.")
    ap.add_argument("--poscar", required=True)
    ap.add_argument("--incar", default=None, help="MD tags only; DFT tags ignored")
    ap.add_argument("--models", nargs="+", default=None,
                    help="frozen_model.pb files; default: every model under models/")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--tag", default="md")
    ap.add_argument("--timestep_fs", type=float, default=0.25)
    ap.add_argument("--nsteps", type=int, default=20000)
    ap.add_argument("--temp", type=float, default=1000.0)
    ap.add_argument("--dump_every", type=int, default=20)
    ap.add_argument("--out_freq", type=int, default=20,
                    help="committee deviation logging interval")
    ap.add_argument("--thermostat", default="nvt", choices=["nvt", "langevin", "nve"])
    ap.add_argument("--np", dest="np_ranks", type=int, default=1,
                    help="MPI ranks; >1 rarely helps for small molecules")
    ap.add_argument("--stride", type=int, default=5, help="render every Nth frame")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="<1 tightens the fixed viewport crop")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="write LAMMPS inputs without running")
    args = ap.parse_args()

    run_md(args.poscar, args.incar, args.models, args.workdir, args.tag,
           args.timestep_fs, args.nsteps, args.temp, args.dump_every,
           args.out_freq, args.thermostat, np_ranks=args.np_ranks,
           render=not args.no_render, stride=args.stride, zoom=args.zoom,
           dry_run=args.dry_run)


if __name__ == "__main__":
    main()
