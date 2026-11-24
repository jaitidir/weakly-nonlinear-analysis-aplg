# Weakly Nonlinear Analysis of the Active–Passive Lattice Gas

This repository contains Julia code for the weakly nonlinear analysis of the hydrodynamic equations of the
active–passive lattice gas (APLG). It covers the stationary and oscillatory instabilities of the homogeneous state, the
Bogdanov–Takens point where they meet, Newton continuation of the exact solutions, and time stepping of the full
equations. The same weakly nonlinear analysis is done for the Cahn–Hilliard equation as a reference case.

The solution types in the code and the figures are:

| name | solution |
|---|---|
| H | homogeneous state |
| PS | stationary pattern |
| T | travelling wave |
| CP | counter-propagating pattern |
| MPS | modulated stationary pattern (near the Bogdanov–Takens point) |

## 1. System Requirements

### Software Dependencies
- Operating System: macOS, Linux or Windows
- Julia: version 1.12 or later
- Jupyter with the IJulia kernel, to run the notebooks
- Required packages: see `Project.toml`

### Tested Versions
- Julia 1.12.4 on macOS

### Hardware Requirements
- A standard desktop or laptop computer
- No specialised hardware is needed

## 2. Installation Guide

Clone this repository:

```
git clone https://github.com/jaitidir/weakly-nonlinear-analysis-aplg
cd weakly-nonlinear-analysis-aplg
```

Install the dependencies from Julia, in the project folder:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Installation takes a few minutes; most of it is the precompilation of the packages.

## 3. Demo

The notebook `notebooks/examples.ipynb` shows the main features of the code. Open it in Jupyter (IJulia kernel) and
run all cells.

### Expected Output
A PS, a CP, a T and an MPS from time stepping, each compared with its weakly nonlinear solution: profiles of the fields
at several times and kymographs (`images/example_*.pdf`), and the videos 01 and 06 in `videos/`.

### Expected Run Time
With the saved data (the default), the notebook only makes the figures. Recomputing the data takes:
- PS, CP and T fields: about 10 minutes
- MPS fields: a few minutes
- videos: a few minutes

## 4. Instructions for Use

Start Julia in the project folder (`julia --project=.`) and load the code:

```julia
include("src/common.jl")
load_src("weakly_nonlinear.jl"); load_src("newton.jl"); load_src("timestepping.jl"); load_src("bt_amplitude.jl")
```

The parameters are the total density `ϕ`, the active density `ϕa`, the Péclet number `Pe` and the box length `Lx`
(`Lx = 10` throughout).

### Onset of Instability
Critical wavenumber, onset `ϕac`, frequency at onset and type of instability (`:stationary` or `:oscillatory`):

```julia
qc, ϕac, wc, kind = find_qc_phiac_wc(0.81, 4.5, 10.0)
```

### Time Stepping
Solve the full equations (pseudospectral, SBDF2) from a perturbed homogeneous state, with `Nx` grid points, until the
solution converges or `T` is reached:

```julia
r = run_spectral(0.81, 0.782, 4.5, 10.0, 256; T=5000.0, pert_type="rand")
r.sol_type, r.converged, r.drho_eff     # (:PS, true, 0.159)
```

The final fields are `r.rho`, `r.rhoa` and `r.m`. Other initial perturbations are `"non-symmetric"`, `"double"` and
`"biased"` (travelling modes; see `run_spectral` in `src/timestepping.jl`).

### Newton Continuation
Steady branch near a stationary onset, started from the weakly nonlinear solution and compared with it:

```julia
qc, ϕac = find_qc_and_phiac(0.81, 4.5, 10.0)
b = bifurcation_scan(0.81, 4.5, 10.0, 400, qc, ϕac, 1.0; eps_max=0.01)
b.phia, b.drho          # Newton branch
b.phia_wnl, b.drho_wnl  # weakly nonlinear branch
```

### Bogdanov–Takens Point
Coefficients of the amplitude equation at the Bogdanov–Takens point for a given `ϕ`:

```julia
c = bt_coefficients(0.88998)
c.Pe, c.ϕas, c.Γ, c.K2, c.K3
```

### Saved Data
Every dataset is saved in `data/<notebook>/` and can be loaded with `loaddata`, for example

```julia
d = loaddata("oscillatory/runs/phi0.88/rand/phia=0.666500.jld2")
d.type, d.t, d.rho
```

## 5. Reproduction Instructions

The notebooks in `notebooks/` reproduce all the figures (`images/`) and videos (`videos/`):

| notebook | contents |
|---|---|
| `stationary.ipynb` | stationary spinodals and bifurcation diagrams (Pe = 4.5) |
| `oscillatory.ipynb` | oscillatory spinodals, speed of T and bifurcation diagrams (Pe = 6) |
| `examples.ipynb` | PS, CP, T and MPS against their weakly nonlinear solutions |
| `hysteresis.ipynb` | hysteresis by time-stepping continuation |
| `bogdanov_takens.ipynb` | coefficients, bifurcation diagrams and the MPS near the Bogdanov–Takens point |
| `cahn_hilliard.ipynb` | weakly nonlinear analysis of the Cahn–Hilliard equation |

Each notebook takes its data from `data/`. The first cell has flags such as `COMPUTE = (branches=false, ...)`: `false`
loads the saved data and `true` computes it again. The cell also gives the approximate run time of each computation;
the time-stepping sweeps take several hours.

A notebook can also be run from the terminal:

```
jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
```

## Contributing

This project was developed by Jesus Ait Idir. The binodal code (`src/binodal.jl`) is adapted from the code of
[Mason, Jack and Bruna](https://github.com/mbruna/Nonreciprocal_APLG).

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT No Attribution License. See the [LICENSE](LICENSE.md) file for details.
