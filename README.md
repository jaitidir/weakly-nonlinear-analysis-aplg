# Weakly Nonlinear Analysis of the Active–Passive Lattice Gas

This repository contains Julia code for the weakly nonlinear analysis of the hydrodynamic equations of the
active–passive lattice gas (APLG). It covers the stationary and oscillatory instabilities of the homogeneous state, the
Bogdanov–Takens point where they meet, Newton continuation of the exact solutions, and time stepping of the full
equations. The same weakly nonlinear analysis is done for the Cahn–Hilliard equation as a reference case.

<<<<<<< HEAD
At the codimension-two point where both instabilities happen together (a Bogdanov–Takens point with O(2) symmetry) a
modulated stationary pattern (MPS) also appears.

The code does the following:
- the linear analysis of the homogeneous state;
- an amplitude equation for each instability, including the one at the Bogdanov–Takens point, and the solutions and
  stabilities it predicts;
- Newton continuation of the exact PS and T solutions;
- time stepping of the full equations, used to find which states are reached;
- hysteresis by time-stepping continuation;
- the same weakly nonlinear analysis for the Cahn–Hilliard equation, as a reference case.

---

## 1. Model

The fields are:
- the total density $\rho$;
- the active density $\rho_a$;
- the polarisation of the active particles $m$.

The passive density is $\rho_0=\rho-\rho_a$. The equations on a periodic domain $[0,L_x)$ are

$$
\begin{aligned}
\partial_t \rho   &= \partial_x^2\rho - \mathrm{Pe}\,\partial_x\big[(1-\rho)\,m\big],\\
\partial_t \rho_a &= \partial_x\big[(1-\rho)\,\partial_x\rho_a + \rho_a\,\partial_x\rho\big] - \mathrm{Pe}\,\partial_x\big[(1-\rho)\,m\big],\\
\partial_t m      &= \partial_x\big[(1-\rho)\,\partial_x m + m\,\partial_x\rho\big] - 2m - \mathrm{Pe}\,\partial_x\big[(1-\rho)\,\rho_a\big].
\end{aligned}
$$

The parameters are:

| parameter | meaning |
|---|---|
| $\phi=\langle\rho\rangle$ | total density, conserved |
| $\phi_a=\langle\rho_a\rangle$ | active density, conserved |
| $\mathrm{Pe}$ | Péclet number (self-propulsion) |
| $L_x$ | box length, $L_x=10$ in every computation |

Physically, $0\le\rho\le1$, $0\le\rho_a\le\rho$ and $|m|\le\rho_a$. The homogeneous state is
$(\rho,\rho_a,m)=(\phi,\phi_a,0)$. The spinodal diagrams use $\nu=(\phi-\phi_a)/(1-\phi)$, the passive density
divided by the vacancy density.

Solution types used throughout:
=======
The solution types in the code and the figures are:
>>>>>>> dda6c9b (Latest version)

| name | solution |
|---|---|
| H | homogeneous state |
| PS | stationary pattern |
| T | travelling wave |
| CP | counter-propagating pattern (standing wave) |
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

<<<<<<< HEAD
- **Stationary** ($\det L=0$, a real eigenvalue crosses zero):
  $$\phi_a^{\rm st}(q)=\frac{2+(1-\phi)\mathrm{Pe}^2+(1-\phi)q^2}{2\,\mathrm{Pe}^2(1-\phi)}.$$
- **Oscillatory** (Routh–Hurwitz, a pair $\pm i\omega$ crosses). This threshold is meaningful only where $\omega^2>0$:
  $$\phi_a^{\rm osc}(q)=\frac{2\,(1+(1-\phi)q^2)\,\big(2+2(1-\phi)+(1-\phi)^2\mathrm{Pe}^2+(2-\phi)^2q^2\big)}{(1-\phi)\,\mathrm{Pe}^2\,(2+q^2)}.$$

Only the box modes $q=2\pi n/L_x$ are allowed. The onset $\phi_{ac}$ is the lowest threshold over them. In the cases
studied this is $n=1$, so $q_c=2\pi/L_x$, with frequency $\omega_c$ in the oscillatory case.

The two thresholds meet at $q_c$ when $L(q_c)$ has a double zero eigenvalue (a Jordan block). This happens at

$$
\mathrm{Pe}_*^2=\frac{2(1-\phi)^2q^2+3(1-\phi)q^2+4(1-\phi)+2}{(1-\phi)(2\phi-1)},\qquad
\phi_{a*}=\frac{2\,(1+(1-\phi)q^2)}{\mathrm{Pe}_*^2(1-\phi)(2\phi-1)},
$$

which exists for $\phi>1/2$. This is the Bogdanov–Takens (BT) point. The code is `bt_point` in
[src/bt_amplitude.jl](src/bt_amplitude.jl).

---

## 3. Amplitude equations

Each expansion is about the onset. The distance to onset is $\delta$, so $\phi_a=\phi_{ac}+\delta^2\phi_a^1$, and the
fields are written as

$$
(\rho,\rho_a,m)=(\phi,\phi_a,0)+\delta\big(A\,\mathbf v\,e^{iq_cx}+\text{c.c.}\big)+O(\delta^2).
$$

Solvability at $O(\delta^3)$ gives the amplitude equation. In the figures the amplitude is plotted as the peak-to-peak
value, $\Delta\rho=\max\rho-\min\rho$, and likewise $\Delta\rho_a$ and $\Delta m$.

### 3.1 Stationary instability (`coefs_stationary`)

Time is slow, $T=\delta^2t$, and the critical eigenvector is $\mathbf v=(1,\alpha,i\beta)$. The amplitude obeys

$$
\frac{dA}{dT}=\phi_a^1\,\Pi\,A+\Gamma\,|A|^2A .
$$

- $\Pi=d\sigma/d\phi_a$ at onset.
- $\Gamma$ comes from the second harmonic $e^{2iq_cx}$ generated at $O(\delta^2)$.
- The PS branch is $|A|^2=-\phi_a^1\Pi/\Gamma$, so $\Delta\rho=4\delta|A|$.
- It is supercritical, stable and above onset, when $\Pi/\Gamma<0$; otherwise it is subcritical and unstable.

The higher branches with $n=2,3$ wavelengths use the same formulas with $q=nq_c$ and their own onset.

### 3.2 Oscillatory instability (`coefs_osc`)

Two waves travel in opposite directions, $A^\pm\,\mathbf v^\pm e^{i(q_cx\pm\omega_ct)}$. With $T=\delta^2t$,

$$
\frac{dA^\pm}{dT}=\phi_a^1\,\Pi^\pm A^\pm+\Gamma^\pm|A^\pm|^2A^\pm+\Sigma^\pm|A^\mp|^2A^\pm ,
$$

where $\Pi^-=\overline{\Pi^+}$, $\Gamma^-=\overline{\Gamma^+}$ and $\Sigma^-=\overline{\Sigma^+}$. The cross-coupling
$\Sigma$ comes from the zero-frequency harmonic $A^+A^-e^{2iq_cx}$.

With $A^\pm=a_\pm e^{i\theta_\pm}$, the moduli obey
$(a_\pm^2)'=2a_\pm^2\big(\phi_a^1\Pi_r^\pm+\Gamma_r^\pm a_\pm^2+\Sigma_r^\pm a_\mp^2\big)$, where the subscript $r$
means the real part. The fixed points are:
- **T**, one wave only: $a_+^2=-\phi_a^1\Pi_r^+/\Gamma_r^+$ and $a_-=0$, or the mirror image;
- **CP**, both waves: $a_\pm^2$ solve $\Gamma_r^\pm a_\pm^2+\Sigma_r^\pm a_\mp^2=-\phi_a^1\Pi_r^\pm$, with both positive.

The imaginary parts give the nonlinear frequency shift. Stability comes from the two real eigenvalues of this system
at each fixed point, and each branch is classified as stable, saddle or unstable.

The oscillatory spinodal figures colour each point of the spinodal by this classification of T and CP.

### 3.3 Bogdanov–Takens point (`bt_coefficients`, [src/bt_amplitude.jl](src/bt_amplitude.jl))

At the BT point both small eigenvalues are $O(\delta)$, so the slow time is $T=\delta t$ and the amplitude equation is
second order in time. Both $\phi_a$ and $\mathrm{Pe}$ are unfolded:

$$
\phi_a=\phi_{a*}+\delta^2\phi_a^1,\qquad \mathrm{Pe}=\mathrm{Pe}_*+\delta^2\mu^1 .
$$

The amplitude equation is

$$
A''=\eta^1A+\Gamma|A|^2A+\delta\big(K_1A'+K_2|A|^2A'+K_3A^2\overline{A'}\big),
\qquad \eta^1=\Pi\,\phi_a^1+\Omega\,\mu^1,\quad K_1=k_{1,\phi_a}\phi_a^1+k_{1,\mu}\mu^1 .
$$

- $\eta^1$ is minus the product of the two small eigenvalues of $L(q_c)$ (divided by $\delta^2$); $\delta K_1$ is
  their sum.
- $\Gamma$ is the same as in the stationary equation.
- $K_2$ and $K_3$ need the $O(\delta)$ harmonics; in the code these are the $D$, $D^*$ and $W$ vectors.
- The code checks that all coefficients are real.

With $A=a\,e^{i\theta}$ and the angular momentum $J=a^2\theta'$, the equation becomes a slowly damped Hamiltonian
system:

$$
a''+\partial_aV=\delta\big(K_1+(K_2+K_3)a^2\big)a',\qquad J'=\delta\big(K_1+(K_2-K_3)a^2\big)J,\qquad
V=\tfrac14\big(2J^2/a^2-2\eta^1a^2-\Gamma a^4\big).
$$

Its solutions, with existence and stability as computed in the code:

| solution | in the amplitude equation | condition |
|---|---|---|
| H | $A=0$ | stable iff $\eta^1<0$ and $K_1<0$: below the pitchfork (D&K 3.11) and the Hopf bifurcation (D&K 3.12) |
| PS | $A=r$, $r^2=-\eta^1/\Gamma$ (D&K 3.13) | stable ($\Gamma<0$, $\eta^1>0$) where $\Gamma K_1>M\eta^1$ and $\Gamma K_1>D\eta^1$: beyond its Hopf bifurcation (D&K 3.17) and its drift instability (D&K 3.16) |
| T (TW) | $A=re^{i\Omega T}$, $r^2=-K_1/(K_2-K_3)$ (D&K 3.18) | exists where $r^2>0$ and $\Omega^2=-\eta^1-\Gamma r^2>0$|
| CP and MPS | periodic orbits with $J=0$ | see below |

The solutions, their stability and the bifurcation conditions are those of Dangelmayr & Knobloch (D&K), *The
Takens–Bogdanov bifurcation with O(2)-symmetry*, Phil. Trans. R. Soc. Lond. A **322** (1987), with $M=K_2+K_3$ and
$D=K_2-K_3$. CP and MPS are their SW2 and SW3.

**CP and MPS** (D&K §5, $\Gamma<0$). Each orbit is labelled by an elliptic modulus $k$, with $k'^2=1-k^2$ and
$\Phi=\mathcal{E}(k)/\mathcal{K}(k)$:
- **CP (SW2):** $r^2=s_1\,\mathrm{cn}^2\theta$ with $\Gamma s_1=-2\eta^1k^2/(k^2-k'^2)$ (D&K 5.12). It lies on the line
  $5\Gamma K_1=-2M\eta^1R_3(k)$ (D&K 5.13–5.14). It branches off H at its Hopf bifurcation ($k\to0$), has $\eta^1<0$ for
  $k^2<\tfrac12$ and $\eta^1>0$ for $k^2>\tfrac12$, and ends on the homoclinic line ($k\to1$).
- **MPS (SW3):** $r^2=s_1\,\mathrm{dn}^2\theta$ with $\Gamma s_1=-2\eta^1/(1+k'^2)$ (D&K 5.20). It lies on the line
  $5\Gamma K_1=2M\eta^1R_5(k)$ (D&K 5.21–5.22), from the Hopf bifurcation of PS ($k\to0$) to the homoclinic line ($k\to1$).

The plotted amplitude is the peak-to-peak value, $\Delta\rho=\max\rho-\min\rho=4\delta r$, which oscillates over a
period between $4\delta\sqrt{s_2}$ (MPS; $0$ for CP) and $4\delta\sqrt{s_1}$. As for the PDE runs, its largest value
over the period, $4\delta\sqrt{s_1}$, is plotted. For $k^2<10^{-4}$ the code uses D&K's expansions $R_3\sim\tfrac54k^2$ and
$R_5\sim\tfrac52-\tfrac5{64}k^4$, because the exact fractions are $0/0$ there. No stability is computed for any
branch; it is all taken from D&K (table above, and for the periodic orbits: MPS stable, and CP stable only between the
homoclinic and fold points of a path).

**Bifurcation sets.** H has a pitchfork bifurcation at $\eta^1=0$ (D&K 3.11) and a Hopf bifurcation at $K_1=0$,
$\eta^1<0$ (D&K 3.12). D&K also give the following rays with $\eta^1>0$ (here $\Gamma<0$):
- $\Gamma K_1=M\eta^1$ (D&K 3.17): Hopf bifurcation of PS, where the MPS is born;
- $\Gamma K_1=D\eta^1$ (D&K 3.16): drift instability of PS, where T branches off;
- $5\Gamma K_1=4M\eta^1$ (D&K 5.19): homoclinic orbit, where the MPS glues into a CP;
- $\Gamma K_1\approx0.74\,M\eta^1$ (D&K 5.18): fold of CP.

On each path, a bifurcation point is where the path crosses one of these lines. Its marker in `bt_paths` sits at the
amplitude of the branch that starts or ends there. The existence and stability arcs of the schematic
`bt_two_parameter_schematic` are written by hand from D&K.

In the code each ray is stored as its value of $\kappa=K_1|\Gamma|/\big((K_2+K_3)|\eta^1|\big)$, so $\Gamma K_1=cM\eta^1$
is $\kappa=-c$.

The classification of the coefficients, and therefore which of these scenarios occurs at each $\phi$, also follows
D&K. The figure `bt_K2K3_ratio` shows $D/M=(K_2-K_3)/(K_2+K_3)$ and the sign of $\Gamma$ against $\phi$, with the
regions of that classification.

The PDE runs are at $\phi=0.88998$, where $K_2+K_3>0$ and $\Gamma<0$, so the MPS exists, with $\delta=0.01$.
- **Paths:** two straight paths in $(\phi_a^1,\mu^1)$, with $\phi_a^1=s$ varying and $\mu^1$ fixed at $\pm\bar\mu$,
  where $\bar\mu$ is the $\mu^1$ of the point $(K_1,\eta^1)=(4.9,0.35)$.
- **MPS from noise:** on the path $\mu^1<0$ at $s=1.84$, the MPS grows from small noise (amplitude $10^{-4}$).
- **Gluing:** raising $s$ slowly, it glues into a CP at $s\approx1.8917$. The figure `bt_homoclinic_phase` and video 07
  compare this with the leading-order homoclinic orbit $a=A\,\mathrm{sech}(\sqrt{\eta^1}\,T)$.

### 3.4 Cahn–Hilliard ([src/cahn_hilliard.jl](src/cahn_hilliard.jl))

The reference case is $\partial_t\varphi=\partial_x^2\big(\varphi^3-\gamma\varphi-\partial_x^2\varphi\big)$ with mean
$\varphi_0$. Its onset is at $\gamma_c=3\varphi_0^2+k_C^2$ with $k_C=2\pi/L$.

- **Cubic amplitude equation** ($\gamma=\gamma_c+\delta^2\gamma_1$): the steady state is
  $\varphi=\varphi_0-2\delta\sqrt{\gamma_1k_C^2/(3k_C^2-6\varphi_0^2)}\cos k_Cx$. It is supercritical when
  $\varphi_0^2<k_C^2/2$ and subcritical otherwise.
- **Quintic amplitude equation** near the tricritical point $\varphi_0^c=k_C/\sqrt2$, with
  $\varphi_0^2=(3k_C^2+\mu_1\delta^2)/6$ and $\gamma=\gamma_c+\delta^4\gamma_1$: $15X^2-2\mu_1X-2\gamma_1k_C^2=0$ with
  $X=|A|^2$. It predicts the fold of the subcritical branch.

These are compared with the steady states found by Newton:
- Newton with deflation, sweeping in $\gamma$, to find every branch;
- pseudo-arclength continuation around the fold;
- stability from the smallest eigenvalue of the linearised operator.

---

## 4. Numerical methods

### Newton continuation ([src/newton.jl](src/newton.jl))

- **Discretisation:** second-order periodic finite differences: $N_x=2000$ for the stationary branches (Pe = 4.5),
  400 for the T and PS branches at Pe = 6 and 1000 for the speed of T.
- **Linear algebra:** sparse LU; the dense rank-one term of the deflation is handled by the Sherman–Morrison formula,
  and the bordered arclength system of the PS is factorised as a sparse matrix.
- **PS unknowns:** the fields plus Lagrange multipliers for the masses, and a pin $m[1]=m_{\rm fix}$ that removes
  translations.
- **T unknowns:** in the frame $\xi=x+ct$, the fields, three multipliers, a phase pin and the period $T=2\pi/(q_cc)$.
- **Starting point:** the weakly nonlinear solution, with deflation away from H.
- **Continuation:** natural continuation in $\phi_a$ (or $\delta$), and pseudo-arclength around folds.

### Time stepping ([src/timestepping.jl](src/timestepping.jl), `run_spectral`)

- **Spatial discretisation:** pseudospectral (FFT) with the 2/3 dealiasing rule.
- **Time stepping:** SBDF2 (second-order IMEX).
  - Implicit: $\partial_x^2\rho$, the constant part $(1-\phi)\partial_x^2$ of the $\rho_a$ and $m$ diffusion, and $-2m$.
  - Explicit: everything else.
  - The first step is IMEX Euler.
- **Conservation:** the masses of $\rho$ and $\rho_a$ are conserved exactly.
- **Retries:** a run that blows up or leaves the physical bounds is restarted with $dt/2$.
- **Grid:** $N_x=256$ and $dt=0.01$, except $N_x=100$ for the MPS runs.

Initial conditions are H plus a zero-mean perturbation with joint RMS `pert_amp`, clipped to stay at least
$\varepsilon=10^{-3}$ inside the physical bounds ($2\varepsilon\le\rho\le1-\varepsilon$,
$\varepsilon\le\rho_a\le\rho-\varepsilon$, $|m|\le\rho_a-\varepsilon$, so no point starts on a bound) with the masses
kept:

| `pert_type` | perturbation |
|---|---|
| `rand` | random noise |
| `non-symmetric` | $\mathrm{Re}[(A^+\mathbf v^++A^-\mathbf v^-)e^{iq_cx}]$ with $(A^+,A^-)=(1,0)$: one travelling mode |
| `double` | $(1,1)$: both modes equally, reflection symmetric |
| `biased` | $(2,1)$ |

**Classifying the final state.** A converged solution repeats itself:
- $U(x,t)=U(x-c\tau,t-\tau)$ (compared up to a shift) means PS when $c=0$ and T when $c\ne0$; as for H, a PS also
  needs $|c|$ to be still decreasing (near a drift instability an unstable PS drifts away only slowly, from a very
  small speed, and would otherwise be accepted);
- $U(t)=U(t-P)$ (compared without shift, periods up to 750) means CP if the profile half a period earlier matches it
  after a translation ($\geq10^{-3}$, mismatch $\leq10^{-2}$) and a quarter period earlier matches no translate
  (mismatch $\geq0.5$; a T is periodic without shift too, with $P=L_x/|c|$, and a T still carrying a weak
  counter-propagating wave is otherwise taken for a CP), and MPS if it matches best without translation and the
  profile changes over half a period (mismatch $\geq10^{-4}$; otherwise unknown, as it could be a PS).

Anything else (still changing, modulated or quasi-periodic) is `:unknown`, and the run goes on. $\rho$, $\rho_a$ and
$m$ are classified separately, every 100 time units, and a run stops only when all three have converged at the same
check (a field that converged earlier keeps being checked); the type of the run is that of $\rho$, and the types of all
three are stored. The amplitudes $\Delta\rho$, $\Delta\rho_a$, $\Delta m$ are the largest peak-to-peak values of each
field over the snapshots on which it was classified: the last one for H, the last 100 time units for PS and T, the last
period for CP and MPS.

### Hysteresis by time-stepping continuation ([src/hysteresis.jl](src/hysteresis.jl))

A chain of runs in $\phi_a$, decreasing (down) and increasing (up):
1. **First run:** starts from random noise.
2. **Next run:** starts from the final state of the last accepted run, with:
   - $\rho_a$ rescaled to the new mass;
   - a zero-mean random kick of size $10^{-4}$, which breaks the reflection symmetry so that, for example, PS can turn
     into T;
   - no classification before $t=10^4$.
3. **Different branch:** a run that ends on a different branch (type, number of wavelengths) is rejected and retried
   with half the step, down to $h_0/8$. Then the jump is accepted. The number of wavelengths is the dominant Fourier
   mode of $\rho$, for a CP averaged over its last period (a single snapshot can catch its first mode passing through
   zero).

The saved histories of the accepted runs, joined in order, give the kymographs and videos 02–05.

### Binodal ([src/binodal.jl](src/binodal.jl))

The coexisting densities of the phase-separated state, drawn with the stationary spinodal. They are adapted from the
code of Mason et al., *Dynamical patterns and nonreciprocal effective interactions in an active–passive mixture
through exact hydrodynamic analysis*, Nat. Commun. **16** (2025), for $d_s(\rho)=1-\rho$.

---

## 5. Notebooks, figures and videos
=======
Clone this repository:
>>>>>>> dda6c9b (Latest version)

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
