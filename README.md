# CCIC_MLIP-Reactive & Condensed-Phase Machine-Learned Interatomic Potentials
A working pipeline for training DeePMD-kit interatomic potentials for reactive and condensed-phase chemistry -built toward a molten Cu-In machine-learned potential for catalytic carbon-nanotube growth, and validated along the way on a sequence of progressively harder test systems.
This repo documents both the tools (converters, active-learning scripts, notebooks) and the methodology (what worked, what didn't, and why) -including a documented dead end that turned out to be a genuine, field-wide hard problem rather than a bug.
________________________________________
TL;DR
•	Validated pipeline on HCN ⇌ HNC isomerization: DFT labeling → DeePMD training → PLUMED/OPES enhanced sampling → free-energy reweighting → NEB cross-check. Barrier matches literature (~187 kJ/mol vs. ~125-190 reported).
•	CH₄ → CH₃• + H• homolysis abandoned -diagnosed as a genuine single-reference-DFT/spin limitation, not a pipeline bug.
•	VASP ⇄ DeePMD tooling for the metallic/condensed-phase phase of the project, including an active-learning handoff that writes VASP jobs directly from committee-flagged MLIP configurations.
•	Three-project roadmap toward Cu-In: pure liquid Cu → carbon in liquid metal → water autoionization, each scoped in a planning doc before any code was written.
________________________________________
Table of contents
•	Motivation
•	Environment setup
•	Part 1 -Reactive gas-phase MLIPs (validated)
•	Part 2 -VASP tooling for condensed-phase systems
•	Part 3 -Condensed-phase roadmap
•	Key methodological lessons
•	Roadmap / status
•	References
________________________________________
Motivation
Machine-learned interatomic potentials (MLIPs) promise DFT-level accuracy at a fraction of the cost, but they are only as good as their training data -and reactive configurations (stretched/breaking bonds, transition states) are exactly the geometries that standard MD-based sampling struggles to reach. This project builds and validates a full pipeline for generating that data deliberately, training a potential, and driving rare events with enhanced sampling, before scaling up to a genuinely hard condensed-phase target: carbon dissolution and reaction in a molten Cu-In catalyst, relevant to catalytic CNT growth.
The guiding discipline throughout: change one variable at a time, and validate every stage against a known answer before trusting it on the unknown system.
________________________________________
Environment setup
Local, CPU-only (WSL2/Ubuntu tested), conda-forge / Miniforge -chosen over Anaconda's defaults channel because the whole stack (psi4, deepmd-kit, plumed) lives on conda-forge.
The setup script installs Miniforge (if absent), creates a conda environment (python=3.11 psi4 deepmd-kit plumed py-plumed ase matplotlib numpy jupyterlab ipykernel), registers it as a Jupyter kernel, and -the one step that can't be conda installed -compiles PLUMED from source with --enable-modules=all, because the conda-forge PLUMED build does not ship the OPES module used throughout this project. It probes first and only builds if the module is genuinely missing.
Each session:
conda activate ch4mlip
jupyter lab --no-browser
Open a notebook and select the environment's kernel -the kernel is the environment, so no subprocess indirection is needed for imports (calls to the dp/plumed binaries resolve via sys.executable's directory).
WSL note: if dp train fails with libtensorflow_cc.so.2: cannot enable executable stack, the TensorFlow shared library needs its executable-stack ELF flag cleared (patchelf --clear-execstack, requires patchelf ≥0.18; older versions silently no-op). This is a WSL-kernel quirk, not a bug in the environment.
________________________________________
Part 1 -Reactive gas-phase MLIPs (validated)
Pipeline: Psi4 DFT labeling (resumable, per-frame) → DeePMD-kit se_e2_a training → ASE + PLUMED OPES_METAD enhanced sampling along a reaction coordinate → bias-reweighted free-energy surface → query-by-committee active learning → ASE NEB as an independent 0 K cross-check.
HCN ⇌ HNC -the validated case
HCN ⇌ HNC isomerization was chosen deliberately as a closed-shell calibration reaction after the CH₄ project (below) diagnosed a labeling failure specific to open-shell dissociation. Restricted Kohn-Sham DFT is correct for this system -no broken-symmetry gymnastics required -which makes it a clean test of everything downstream of labeling.
•	Reaction coordinate: d(C-H) − d(N-H), via a PLUMED COMBINE action.
•	Sampling: OPES metadynamics at 1000 K, with upper/lower walls to keep the bias in the chemically relevant region.
•	Reweighting: F(CV) = −kT ln P(CV) from the biased trajectory (exp(bias/kT) weights); reactant/product basins fixed from known chemistry, the barrier location found adaptively between them.
•	Cross-check: a reference NEB (ase.mep.NEB, method='improvedtangent', climbing image, IDPP-interpolated band) gives an independent 0 K barrier.
Result: forward barrier ≈ 187 kJ/mol (literature range ≈ 125-190 kJ/mol), reaction free energy ≈ 72 kJ/mol (HNC lit. ≈ 60 kJ/mol above HCN) -the pipeline reproduces known chemistry end-to-end.
CH₄ → CH₃• + H• -the abandoned case, and why
The original target was CH₄ → CH₃• + H• homolysis. After the free-energy surface repeatedly came out unphysical (barriers of a few kJ/mol, or wildly inconsistent depending on sampling parameters), a controlled bond-stretch scan with explicit ⟨S²⟩ monitoring showed the root cause: UKS + guess_mix collapses back to the restricted (closed-shell, ionic-dissociation) solution at many stretched geometries, rather than finding the broken-symmetry open-shell solution appropriate for two separating radicals. Energies across the "reaction" varied by only 2-5 kJ/mol -the labels never described bond breaking at all.
This is not a code bug: single-reference DFT is a known-hard case for homolytic bond dissociation (a genuinely multireference problem), and conventional MLIPs have no spin degree of freedom to represent the ambiguity even when the DFT reference is correct. The Reactive Active Learning paper (see References) this project draws on explicitly designs around the same limitation by choosing test systems (ammonia synthesis, methanimine hydrolysis, methane-on-carbide) that avoid open-shell products.
Practical implication for scaling up: the target Cu-In system is favourable on this axis -C-H activation on a metal surface has its radical character quenched by the metal's electrons, so it does not inherit this problem the way gas-phase homolysis does.
Active learning
A hand-rolled implementation of query-by-committee active learning, in the spirit of DP-GEN: four DeePMD models trained with different random seeds on the same data; per-atom force-prediction spread across the committee is used as an uncertainty proxy; explored configurations are binned into accurate / candidate / rejected by two deviation thresholds, and only the candidate band is sent for expensive DFT relabeling. This mirrors DP-GEN and the RAL paper's Step-7 filtering, with the same qualitative uncertainty threshold structure (~0.1-1.0 eV/Å band).
________________________________________
Part 2 -VASP tooling for condensed-phase systems
Moving to metallic and condensed-phase systems means moving to periodic VASP labeling (Psi4 has no role here). Two conversion tools bridge VASP and DeePMD in both directions.
VASP → training data (outcar2deepmd.py)
Parses one or more OUTCAR files into DeePMD-kit npy systems.
python outcar2deepmd.py OUTCAR --type_map Cu In C H --out deepmd_data \
    --skip 50 --stride 5 --virial
•	Unpacks every ionic step of an AIMD trajectory into a training frame.
•	Reads the lattice vectors per step and writes a genuine periodic box.npy (no nopbc -this is always a real condensed-phase cell).
•	No unit conversion needed: VASP's native units (eV, eV/Å, Å) already match DeePMD's.
•	A global, user-specified type_map (e.g. Cu In C H) keeps per-atom type indices consistent across OUTCARs that contain different subsets of elements -required for training one model across mixed systems.
•	--skip / --stride handle AIMD equilibration burn-in and frame decorrelation.
•	--virial extracts the stress tensor for bulk/surface training.
•	Frames are automatically grouped into DeePMD systems by composition.
A Colab-friendly wrapper adds file upload / Google Drive input handling and an optional cross-check against the community-standard dpdata package.
Metals note: the force-consistent energy for a system with electronic smearing is the electronic free energy (TOTEN), not the energy(sigma→0) extrapolation appropriate for insulators/molecules. Use --energy free for metallic systems.
MLIP-flagged configurations → VASP jobs (traj2vasp_al.py)
Closes the active-learning loop in the other direction: given an MLIP-explored ASE trajectory and a trained DeePMD committee, this script identifies the configurations the committee disagrees about, and writes them out as ready-to-submit VASP single-point jobs.
python traj2vasp_al.py explore.traj \
    --models cu-mlip/models --type_map Cu In \
    --lo 0.05 --hi 0.30 --max_select 40 --min_gap 5 \
    --out al_labels --encut 400 --sigma 0.15 --kmesh 2 2 2 --ispin 1
•	Computes per-frame committee force deviation and selects the band between "already known" (low deviation) and "probably unphysical" (very high deviation) -the same query-by-committee logic as Part 1's active learning.
•	Decorrelates picks by trajectory-index spacing (--min_gap), since a liquid has no single reaction coordinate to thin against.
•	Writes one folder per selected frame with POSCAR (species-sorted, direct coordinates), a metallic-system INCAR (Methfessel-Paxton smearing, ISPIN configurable for magnetic vs. non-magnetic hosts), and KPOINTS.
•	Ready for outcar2deepmd.py --energy free once VASP has run.
explore (MLIP committee) → traj2vasp_al.py → VASP labels → outcar2deepmd.py → retrain
________________________________________
Part 3 -Condensed-phase roadmap
Before writing any condensed-phase code, three candidate systems were scoped in detail -feasibility, DFT methodology, ranked bottlenecks, validation criteria, and relevance to the Cu-In target. The sequencing is deliberate: each step adds exactly one new axis of difficulty.
#	System	Purpose	Status
1	Pure liquid Cu	Validate the metallic condensed phase itself (structure, diffusion, density vs. AIMD/experiment) -no reactive chemistry yet	In progress (Δ-learning notebook)
2	Carbon in liquid metal	Add the reactive solute: dissolution, diffusion, C-C coupling -the direct methodological ancestor of Cu-In	Planned
3	Water autoionization	Stress-test reactive free-energy sampling with a genuinely hard collective variable, in a molecular (non-metallic) bath	Planned, lower priority
Why liquid Cu first, and why Δ-learning
Pure liquid Cu isolates the metallic periodic condensed phase as a single new variable -no reaction, no collective variable, no rare-event sampling -analogous to how HCN isolated "does the reactive pipeline work at all" before CH₄'s spin physics was allowed to complicate the picture.
Full ab initio MD of even a 32-atom metallic cell is not CPU-feasible for this project (metallic SCF with k-point sampling and electronic smearing, forces at every timestep, for the tens of thousands of steps liquid sampling requires). The adopted strategy:
1.	Explore liquid configurations cheaply with EMT (ASE's built-in effective-medium potential -essentially free, and a reasonable qualitative description of FCC Cu).
2.	Label a sparse, decorrelated subset with CP2K single-points at a proper metallic DFT level (PBE, GTH pseudopotentials, MOLOPT basis, Fermi-Dirac smearing).
3.	Train on the residual, Δ = E<sub>CP2K</sub> − E<sub>EMT</sub> (and the corresponding force difference) -a Δ-learning setup. Because EMT is both the (free, always-available) exploration engine and the baseline, this is close to the best case for Δ-learning: the correction the network has to learn is small and smooth, and the physical baseline keeps molecular dynamics stable even where the network is extrapolating.
4.	Production MD runs EMT + Δ-network on a replicated, larger box (train small, run big), and is validated against literature/AIMD radial distribution functions and self-diffusion coefficients.
This is explicitly framed as a methodology validation, not a production-accuracy potential -the real production data source for Cu-In is VASP AIMD, for which the Part 2 tooling is built.
________________________________________
Key methodological lessons
Collected here because they cost real debugging time and are easy to reintroduce by accident.
•	PLUMED unit consistency. With UNITS ENERGY={units.mol/units.kJ}, PLUMED's internal energy unit is eV (matching ASE), not kJ/mol. Free- energy reweighting must use kT = 8.617×10⁻⁵ × T (eV) and convert to kJ/mol only for display -using the kJ/mol Boltzmann constant here silently flattens the reconstructed free-energy surface.
•	One temperature, everywhere. The Maxwell-Boltzmann initialization, the Langevin thermostat, PLUMED's TEMP/kT, and the reweighting temperature must all agree, or the bias-reweighting is formally invalid.
•	OPES_METAD BARRIER should reflect the actual physical barrier height (in eV) -setting it far too low prevents the bias from ever pushing the system across a real barrier.
•	Free-energy barrier extraction: fix the reactant/product basin locations from known chemistry, but locate the barrier adaptively between them rather than assuming its position -the saddle is the quantity being measured, not an input.
•	Metals need different DFT settings than molecules: electronic smearing (not Gaussian/none), real k-point sampling (not Γ-only), and training on the free energy rather than the sigma→0-extrapolated energy.
•	Committee query-by-committee and Δ-learning compose cleanly: if every committee member is trained on the same Δ-baseline, the baseline cancels in the inter-model force spread, so uncertainty quantification is unaffected by adding Δ-learning.
________________________________________
Roadmap / status
•	[x] Reactive pipeline built and validated (HCN ⇌ HNC + reference NEB)
•	[x] CH₄ homolysis diagnosed as a DFT/spin limitation
•	[x] VASP → DeePMD converter, local + Colab
•	[x] MLIP → VASP active-learning handoff
•	[x] Condensed-phase roadmap scoped
•	[ ] Liquid Cu Δ-learning notebook (CP2K labeling + training + validation)
•	[ ] Cu-In alloy melt
•	[ ] Carbon in liquid metal
•	[ ] Water autoionization
•	[ ] Cu-In + carbon (production target)
•	[ ] Migration to HPC / SLURM for VASP-scale production runs
________________________________________
References
•	Achar, S. K. et al. "Reactive Active Learning: An Efficient Approach for Training Machine Learning Interatomic Potentials for Reacting Systems." J. Chem. Theory Comput. 2025, 21, 8889-8906.
•	Zhang, Y. et al. "DP-GEN: A concurrent learning platform for the generation of reliable deep learning based potential energy models." Comput. Phys. Commun. 2020, 253, 107206.
•	Invernizzi, M.; Parrinello, M. "Rethinking Metadynamics: From Bias Potentials to Probability Distributions." J. Phys. Chem. Lett. 2020, 11, 2731-2736. (OPES)
•	Wang, H.; Zhang, L.; Han, J.; E, W. "DeePMD-Kit: A Deep Learning Package for Many-Body Potential Energy Representation and Molecular Dynamics." Comput. Phys. Commun. 2018, 228, 178-184.
•	Larsen, A. H. et al. "The Atomic Simulation Environment -a Python library for working with atoms." J. Phys.: Condens. Matter 2017, 29, 273002.
________________________________________
License / status
Research code in active development; interfaces may change.

