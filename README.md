# Active-passive lattice gas: weakly nonlinear analysis, bifurcations and Bogdanov–Takens point

Julia code for the hydrodynamic equations of the active-passive lattice gas (APLG): linear and weakly nonlinear
analysis of the homogeneous state, Newton continuation of stationary patterns (PS) and travelling waves (T),
pseudospectral time stepping (PS, T, counter-propagating patterns CP, modulated PS), hysteresis by time-stepping
continuation, and the amplitude equation near the Bogdanov–Takens point. Also the same analysis for the Cahn–Hilliard
equation (time stepping, bifurcation diagrams and the fold near the tricritical point).

## Layout

```
notebooks/   one notebook per topic; each makes its figures and videos
src/         all the code, included by the notebooks
data/        the data of every figure, one folder per notebook
images/      every figure, as pdf (written by the notebooks)
videos/      every video, as mp4 (written by the notebooks)
```

| notebook | figures (`images/`) | videos (`videos/`) |
|---|---|---|
| `stationary.ipynb` (Pe = 4.5) | `stationary_spinodals`, `stationary_bifurcation_phi0.81` (+ `_rho`), `stationary_bifurcation_phi0.7` (+ `_rho`) | |
| `oscillatory.ipynb` (Pe = 6) | `oscillatory_speed_phi0.88`, `oscillatory_spinodals`, `oscillatory_bifurcation_phi<φ>` (+ `_rho`) for φ = 0.7, 0.805, 0.88, 0.905, 0.915, `oscillatory_spinodals_shaded` | |
| `examples.ipynb` | `example_PS_phi0.81`, `example_CP_phi0.88`, `example_T_phi0.88`, `example_MPS_phi0.88998` | 01, 06 |
| `hysteresis.ipynb` | `hysteresis_kymograph` | 02–05 |
| `bogdanov_takens.ipynb` | `bt_K2K3_ratio`, `bt_two_parameter_schematic`, `bt_two_parameter`, `bt_paths`, `bt_homoclinic_phase` | 07 |
| `cahn_hilliard.ipynb` | `cahn_hilliard_profiles`, `cahn_hilliard_bifurcation`, `cahn_hilliard_fold` | |

`src/`: `common.jl` (folders, loading/saving data, figure style, legends, videos), `weakly_nonlinear.jl` (onset and
weakly nonlinear solutions), `newton.jl` (Newton solvers and continuation), `timestepping.jl` (pseudospectral time
stepping with the second-order IMEX scheme SBDF2, and the classification of its final state), `binodal.jl`,
`bt_amplitude.jl` (amplitude equation at the Bogdanov–Takens point), and one file per notebook with its data functions
and figures (`stationary.jl`, `oscillatory.jl`, `examples.jl`, `hysteresis.jl`, `bogdanov_takens.jl`).
`cahn_hilliard.jl` has all the Cahn–Hilliard code (its own solvers, weakly nonlinear estimates, fold continuation and
figures).

## Running

Install the packages once (Julia 1.12), from the project folder:

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
```

Open a notebook (Jupyter with the IJulia kernel) and run all its cells. The first cell has the flags

```julia
COMPUTE = (name=false, ...)
```

With `false` a dataset is loaded from `data/`; with `true` it is computed again and saved there (the first cell says
roughly how long each takes; the time-stepping sweeps take hours). The hysteresis notebook uses the Newton branches of
the oscillatory notebook, and the examples notebook the modulated-PS run of the Bogdanov–Takens notebook.

When the time-stepping points are computed (stationary and oscillatory sweeps, hysteresis chains), the whole history of
every run is also saved, in `data/stationary/runs/phi<φ>/phia=<φa>.jld2`,
`data/oscillatory/runs/phi<φ>/<perturbation>/phia=<φa>.jld2` and `data/hysteresis/runs/phi<φ>/<down|up>/run<k>_phia=<φa>.jld2`
(rejected chain runs included): `t` every 5 time units from t = 0, the fields `rho`, `rhoa`, `m` at those times (one row
per time, Float32), the result of the run and its settings. The Cahn–Hilliard time stepping saves every time step in
its data file. Load one with `loaddata("oscillatory/runs/phi0.88/rand/phia=0.666500.jld2")`.

## License

MIT License, see [LICENSE](LICENSE.md). 
