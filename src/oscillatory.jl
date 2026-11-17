# Oscillatory notebook (Pe = 6, Lx = 10): speed of the travelling wave T, spinodals classified by the weakly nonlinear
# analysis of T and CP, the bifurcation diagrams at φ = 0.7, 0.805, 0.88, 0.905, 0.915, and the shaded spinodal.
include(joinpath(@__DIR__, "common.jl"))
load_src("weakly_nonlinear.jl"); load_src("newton.jl"); load_src("timestepping.jl")

const PE = 6.0
const LX = 10.0
const PHIS = (0.7, 0.805, 0.88, 0.905, 0.915)
const FS = 18      # ticks, legends, curve labels
const GFS = 24     # axis labels, panel labels

# ==================================================================================================================
# Data
# ==================================================================================================================

# Speed of T at φ = 0.88: Newton (Nx = 1000) from the weakly nonlinear solution at δ = 0.1 (φa = ϕac + δ²), continued
# in δ down to 0.01 and up to 0.19 (steps 0.001). Returns δ, φa and c of [down continuation; up continuation].
function travelling_speed(; ϕ=0.88, δ=0.1, Nx=1000)
    qc, ϕac, wc, _ = find_qc_phiac_wc(ϕ, PE, LX)  # critical wavenumber, onset, frequency at onset
    ϕa = ϕac + δ^2
    p = Params(Nx, LX, PE, ϕ, ϕa; pin_idx=1)
    # period guess T0 = 2π/ω from the weakly nonlinear frequency (ϕ1 = +1)
    T0 = 2π / wnl_effective_frequency(PE, qc, ϕ, ϕac, δ, 1.0, wc; mode=:plus)
    kw = (tol=1e-10, maxiter=50, defl_power=1.0, defl_shift=0.0)
    # seed: Newton from the weakly nonlinear T at δ = 0.1
    res = solve_tw(build_initial_guess_tw(p, PE, qc, ϕ, ϕac, ϕa, δ, 1.0, wc, T0; mode=:plus), p; kw...)
    # down: natural continuation in δ (bisects the step on failure); its first entry is the seed
    down = continue_in_delta(res, p, qc, ϕ, ϕac, 1.0; delta_start=δ, delta_target=0.01, step0=0.001, min_step=1e-6, kw...)
    # up: fixed steps of 0.001 in δ, each from the last solution, until the first Newton failure
    up = Any[(delta=δ, phia=ϕa, res=res)]
    for d in δ+0.001:0.001:0.19
        pd = Params(Nx, LX, PE, ϕ, ϕac + d^2; pin_idx=1)
        r = try solve_tw(copy(up[end].res.Z), pd; q_override=qc, kw...) catch; nothing end
        (r === nothing || !r.conv) && break
        push!(up, (delta=d, phia=pd.phia, res=r))
    end
    sols = vcat(down, up[2:end])  # up[1] is the seed, already in down
    return (delta=[s.delta for s in sols], phia=[s.phia for s in sols], c=[s.res.c for s in sols])
end

# Along the spinodal (n values of φ in [0.5, 0.98]): onset ν = (φ - ϕac)/(1 - φ), whether it is stationary (and then
# subcritical or not) and, for an oscillatory onset, existence and stability of T and CP. Also the threshold of the
# stationary instability alone at q = 2π/Lx (ν_st) and whether that bifurcation is subcritical.
function spinodal_classes(; n=100_000)
    phi = Float64[]; nu = Float64[]; is_stationary = Bool[]; stat_sub = Bool[]
    has_T = Bool[]; has_CP = Bool[]; stab_T = String[]; stab_CP = String[]
    phi_st = Float64[]; nu_st = Float64[]; st_sub = Bool[]
    q1 = 2π / LX
    for ϕ in range(0.5, 0.98, length=n)
        qc, ϕa, wc, inst = find_qc_phiac_wc(ϕ, PE, LX)  # onset over the box wavenumbers and its type
        ϕa_st = (2 + (1 - ϕ)*(PE^2 + q1^2))/(2PE^2*(1 - ϕ))  # = phia_stationary(q1, ϕ, PE)
        if ϕa_st < ϕ  # stationary threshold below φa = φ, i.e. ν_st > 0
            push!(phi_st, ϕ); push!(nu_st, (ϕ - ϕa_st) / (1 - ϕ)); push!(st_sub, stationary_subcritical(ϕ, PE, q1, ϕa_st))
        end
        ϕa >= ϕ && continue  # onset at or above φa = φ (ν ≤ 0): left out
        stat = inst == :stationary
        # oscillatory onset: existence and stability of T and CP for ϕ1 = +1 (an error counts as no branch)
        h, c, st, sc = stat ? (false, false, :none, :none) :
                       (try osc_classes(PE, qc, ϕ, ϕa, wc) catch; (false, false, :none, :none) end)
        push!(phi, ϕ); push!(nu, (ϕ - ϕa) / (1 - ϕ)); push!(is_stationary, stat)
        push!(stat_sub, stat && stationary_subcritical(ϕ, PE, qc, ϕa))  # stationary onset: subcritical?
        push!(has_T, h); push!(has_CP, c); push!(stab_T, string(st)); push!(stab_CP, string(sc))
    end
    return (phi=phi, nu=nu, is_stationary=is_stationary, stat_sub=stat_sub, has_T=has_T, has_CP=has_CP, stab_T=stab_T,
            stab_CP=stab_CP, phi_st=phi_st, nu_st=nu_st, st_sub=st_sub)
end

# Seeds (δ of the two first Newton solutions), arclength step and its budget of the T branch at each φ
const T_BRANCH_SETTINGS = Dict(0.7 => (seeds=(0.1, 0.11), ds=0.05, steps=8000), 0.805 => (seeds=(0.01, 0.015), ds=0.05, steps=8000),
                               0.88 => (seeds=(0.1, 0.11), ds=0.05, steps=8000), 0.905 => (seeds=(0.1, 0.11), ds=0.05, steps=8000),
                               0.915 => (seeds=(0.1, 0.11), ds=0.05, steps=3000))

# T branch at φ (Newton, Nx = 400): two seeds from the weakly nonlinear solution, then pseudo-arclength continuation
# outwards (up to the plot window or a branch point) and inwards (back to the onset); the plot window (up to the second
# oscillatory instability); and the weakly nonlinear amplitudes of T and CP over the window (analytic envelope)
function travelling_branch(ϕ; Nx=400, target_width=0.08, buffer_next_inst=0.002)
    S = T_BRANCH_SETTINGS[ϕ]
    qc, ϕac, wc, _ = find_qc_phiac_wc(ϕ, PE, LX)
    ϕ1T = branch_sign_osc(PE, qc, ϕ, ϕac, 1.0, wc; mode=:plus)  # side of onset where T exists: +1 φa > ϕac
    ϕ1CP = branch_sign_osc_cp(PE, qc, ϕ, ϕac, 1.0, wc)  # the same for CP
    # stability of T and CP from the amplitude equations
    stabT = classify_branch_stability(PE, qc, ϕ, ϕac, ϕ1T, wc, :plus)
    stabCP = classify_branch_stability(PE, qc, ϕ, ϕac, ϕ1CP, wc, :coexistence)
    ϕ_osc2 = phia_oscillatory(2qc, ϕ, PE)  # second oscillatory instability (2qc)
    # Newton solution at φa = ϕac + δ²ϕ1T from the weakly nonlinear T, or from Z_warm with q fixed to qc
    function seed(δ, Z_warm=nothing)
        ϕa = ϕac + δ^2 * ϕ1T
        p = Params(Nx, LX, PE, ϕ, ϕa; pin_idx=1)
        T0 = 2π / abs(wnl_effective_frequency(PE, qc, ϕ, ϕac, δ, ϕ1T, wc; mode=:plus))
        Z0 = Z_warm === nothing ? build_initial_guess_tw(p, PE, qc, ϕ, ϕac, ϕa, δ, ϕ1T, wc, T0; mode=:plus) : copy(Z_warm)
        res = solve_tw(Z0, p; tol=1e-10, maxiter=50, defl_power=1.0, defl_shift=0.0, q_override=(Z_warm === nothing ? nothing : qc))
        return res, p, ϕa
    end
    res0, p0, ϕa0 = seed(S.seeds[1])  # the two seeds: the second is warm-started
    res1, p1, ϕa1 = seed(S.seeds[2], res0.Z)  # from the first; together they give the first secant
    # One arclength step at a time, trying other step sizes if it fails. Stops at a branch point, past the plot range,
    # at the onset (inward walk), or if the speed stays ~0 for several steps (the walk has slipped onto the PS branch,
    # which also has c = 0: those points are removed).
    function walk(Zprev, ϕaprev, Zcur, ϕacur, max_steps; stop_near_onset=false)
        out = Any[]; reason = "step budget"; tiny = 0
        for step in 1:max_steps
            newsol = nothing
            for mult in (1.0, 0.3, 0.1, 0.03, 0.01, 3.0, 10.0)  # step multipliers tried in turn
                batch = continue_arclength_tw((phia=ϕaprev, res=(Z=Zprev, phia=ϕaprev)), (phia=ϕacur, res=(Z=Zcur, phia=ϕacur)),
                                              p1, ϕ; qc=qc, ds=S.ds*mult, n_steps=1, tol=1e-10, maxiter=50)
                length(batch) >= 3 && (newsol = batch[end]; break)  # [sol0, sol1, new] on success
            end
            newsol === nothing && (reason = "branch point at ϕa = $ϕacur"; break)
            tiny = abs(newsol.res.c) < 1e-6 ? tiny + 1 : 0  # consecutive steps with |c| < 1e-6
            if tiny > 4
                for _ in 1:4; isempty(out) || pop!(out); end  # drop the last 4 (c ≈ 0) points
                reason = "c ≈ 0 for 4+ steps at ϕa = $ϕacur (onto the PS branch): rolled back"
                break
            end
            # the new point becomes the current one of the secant pair
            Zprev, ϕaprev = Zcur, ϕacur
            Zcur, ϕacur = newsol.res.Z, newsol.phia
            push!(out, newsol)
            # inward walk: stop once within 1e-4 of the onset (or past it)
            stop_near_onset && (ϕ1T > 0 ? ϕacur - ϕac : ϕac - ϕacur) < 1e-4 && (reason = "reached onset"; break)
            # stop 1.5 window widths away from the onset
            (ϕacur < ϕac - target_width*1.5 || ϕacur > ϕac + target_width*1.5) && (reason = "left the plot range"; break)
        end
        return out, reason
    end
    sols_out, stop_reason = walk(res0.Z, ϕa0, res1.Z, ϕa1, S.steps)  # outwards, budget S.steps
    sols_in, _ = walk(res1.Z, ϕa1, res0.Z, ϕa0, 200; stop_near_onset=true)  # inwards: the seeds swapped
    # the branch in order: inward points (reversed: onset first), the two seeds, outward points
    sols = vcat(reverse([(phia=s.phia, res=s.res) for s in sols_in]), [(phia=ϕa0, res=res0), (phia=ϕa1, res=res1)],
                [(phia=s.phia, res=s.res) for s in sols_out])
    # plot window: to the right up to the second oscillatory instability where a branch or a stationary instability
    # lies there, to the left where a branch is subcritical
    has_right = (ϕ1T > 0) || (ϕ1CP > 0) || (phia_stationary(qc, ϕ, PE) < ϕ_osc2) || (phia_stationary(2qc, ϕ, PE) < ϕ_osc2)
    # lo: target_width below the onset if T or CP is subcritical, else 0.02; hi: target_width above it, but
    # buffer_next_inst short of the second oscillatory instability (0.02 if nothing lies to the right)
    lo = ϕac - ((ϕ1T < 0) || (ϕ1CP < 0) ? target_width : 0.02)
    hi = ϕac + (has_right ? min(target_width, ϕ_osc2 - ϕac - buffer_next_inst) : 0.02)
    # the branch up to its first point outside [lo, hi] (anything after it is dropped)
    kept = sols[1:something(findfirst(s -> s.phia < lo || s.phia > hi, sols), length(sols) + 1) - 1]
    amp = [amplitudes(s.res) for s in kept]  # (Δρ, Δρa, Δm), peak to peak
    # weakly nonlinear T and CP over the window
    # δ grid reaching the farther window edge (φa = ϕac + ϕ1 δ²)
    deltas = range(0.0, sqrt(max(abs(lo - ϕac), abs(hi - ϕac))); length=80)
    function wnl_curve(ϕ1, mode)
        # (φa, δ) of the grid points inside the window, on the side of onset given by ϕ1
        pts = [(ϕac + ϕ1 * δ^2, δ) for δ in deltas if lo <= ϕac + ϕ1 * δ^2 <= hi]
        sort!(pts)
        # analytic peak-to-peak envelope (Δρ, Δρa, Δm) at each point
        a = [wnl_amplitude_envelope(PE, qc, ϕ, ϕac, pa, δ, ϕ1, wc, mode) for (pa, δ) in pts]
        return first.(pts), getindex.(a, 1), getindex.(a, 2), getindex.(a, 3)  # φa, Δρ, Δρa, Δm
    end
    T = wnl_curve(ϕ1T, :plus); CP = wnl_curve(ϕ1CP, :coexistence)
    return (qc=qc, ϕac=ϕac, wc=wc, ϕ1T=ϕ1T, ϕ1CP=ϕ1CP, stabT=stabT, stabCP=stabCP, phia_lo_plot=lo, phia_hi_plot=hi,
            phia=[s.phia for s in kept], drho=getindex.(amp, 1), drhoa=getindex.(amp, 2), dm=getindex.(amp, 3), stop_reason=stop_reason,
            phia_wnl_T=T[1], wnl_T_rho=T[2], wnl_T_rhoa=T[3], wnl_T_m=T[4],
            phia_wnl_CP=CP[1], wnl_CP_rho=CP[2], wnl_CP_rhoa=CP[3], wnl_CP_m=CP[4])
end

# Stationary (PS) branches with n wavelengths shown in the diagrams: how far past onset (eps_max), the seed δ of a
# subcritical branch and the upper end of the window kept (default: that of the T branch; at φ = 0.915 the branch is
# followed to just below φa = φ for the hysteresis figure)
const PS_BRANCH_SETTINGS = Dict(0.7 => [(n=1, eps_max=0.0447, delta_seed=0.01), (n=2, eps_max=0.02826, delta_seed=0.01)],
                                0.905 => [(n=1, eps_max=0.0285), (n=2, eps_max=0.012)],
                                0.915 => [(n=1, eps_max=0.0827, delta_seed=0.001, phia_hi=0.9149), (n=2, eps_max=0.0662)])

# PS branch with n wavelengths at φ (Newton, Nx = 400): supercritical: bifurcation_scan from ϕac + eps_max down to the
# onset; subcritical: from a weakly nonlinear seed along φa to the fold, then by arclength around it, kept in the window
function ps_branch(ϕ, n, window; eps_max, delta_seed=0.001, phia_hi=window[2], Nx=400)
    qn = n * 2π / LX
    ϕac_n = phia_stationary(qn, ϕ, PE)  # onset of the PS mode with n wavelengths
    bs = branch_sign(ϕ, PE, qn, ϕac_n, 1.0)  # +1 supercritical, -1 subcritical
    if bs > 0
        b = bifurcation_scan(ϕ, PE, LX, Nx, qn, ϕac_n, 1.0; eps_max=eps_max, min_step=1e-8)
        return (n=n, ϕac_n=ϕac_n, bs=bs, phia=b.phia, drho=b.drho, drhoa=b.drhoa, dm=b.dm)
    end
    x = 0:LX/Nx:(LX - LX/Nx)  # grid of Nx points
    ϕa_seed = ϕac_n + bs * delta_seed^2  # seed on the subcritical side
    z = [weaklynonlinear_stat(xi, PE, qn, ϕ, ϕac_n, ϕa_seed, delta_seed, bs) for xi in x]
    rho0 = real([zi[1] for zi in z]); rhoa0 = real([zi[2] for zi in z]); m0 = real([zi[3] for zi in z])
    enforce_masses!(rho0, rhoa0, ϕ, ϕa_seed)  # exact masses φ, φa
    p_seed = Params(Nx, LX, PE, ϕ, ϕa_seed; pin_idx=1)
    # Newton from the weakly nonlinear seed
    res_seed = converge_guess(pack_state(rho0, rhoa0, m0), p_seed; tol=1e-10, maxiter=80)
    # natural continuation away from the onset until Newton fails (the fold, before the target 0.3²)
    unstable = continue_in_phia(res_seed, p_seed, ϕ; phia_target=ϕac_n + bs * 0.3^2, step0=0.00005, min_step=1e-6)
    # arclength from the last two points (pin m[1] at its last value) around the fold and up the stable branch
    stable = continue_arclength(unstable[end-1], unstable[end], p_seed, ϕ, unstable[end].res.m[1]; ds=0.01, n_steps=2000)
    # the arclength part starts with the last two points of the first; the seed is kept only if its Newton converged
    sols = vcat(filter(s -> s.res.ok, unstable[1:end-2]), stable)
    # points in the window: above max(lo, ϕac_n - 0.2), below min(phia_hi, ϕac_n + eps_max)
    kept = [s for s in sols if max(window[1], ϕac_n - 0.2) <= s.phia <= min(phia_hi, ϕac_n + eps_max)]
    amp = [amplitudes(s.res) for s in kept]
    return (n=n, ϕac_n=ϕac_n, bs=bs, phia=[s.phia for s in kept], drho=getindex.(amp, 1), drhoa=getindex.(amp, 2), dm=getindex.(amp, 3))
end

# The PS branches of the diagram at φ (data/oscillatory/ps_phi<φ>_n<n>.jld2), computed again if `compute`
# (the settings other than n are passed as keywords of ps_branch)
ps_branches(ϕ, tb, compute) = [getdata(() -> ps_branch(ϕ, s.n, (tb.phia_lo_plot, tb.phia_hi_plot); Base.structdiff(s, (n=0,))...),
                                       "oscillatory/ps_phi$(ϕ)_n$(s.n).jld2", compute) for s in get(PS_BRANCH_SETTINGS, ϕ, [])]

# φa of the time-stepping sweep at each φ: n points across the plot window, with points added or removed by hand
const SWEEP_POINTS = Dict(0.7 => (n=10, add=[0.5645, 0.567], remove=Float64[]),
                          0.805 => (n=10, add=[0.5645, 0.569], remove=Float64[]),
                          0.88 => (n=10, add=[0.6605, 0.6615, 0.664, 0.6665], remove=Float64[]),
                          0.905 => (n=30, add=Float64[], remove=Float64[]),
                          0.915 => (n=10, add=[0.819, 0.8215, 0.825, 0.83, 0.84], remove=Float64[]))

# Time stepping (Nx = 256, dt = 0.01) at every φa of the sweep from four initial conditions: random noise and the
# travelling-mode perturbations "non-symmetric" (A⁺, A⁻) = (1, 0), "double" (1, 1) and "biased" (2, 1). Each run stops
# once the recurrence classifier has converged (budget 2·10⁵: near onset at φ = 0.88 the runs converge to T only after
# up to ~1.3·10⁵ time units).
function timestepping_sweep(ϕ, window)
    S = SWEEP_POINTS[ϕ]
    phia_ts = collect(range(window...; length=S.n))  # n evenly spaced φa
    # remove: the grid point nearest each listed φa is dropped before any run (it is never computed)
    for r in S.remove; deleteat!(phia_ts, argmin(abs.(phia_ts .- r))); end
    phia_ts = sort(unique(vcat(phia_ts, S.add)))  # plus the added points, sorted
    # one entry per run: initial condition, φa, final type, converged flag, Δρ, Δρa, Δm
    out = (pert=String[], phia=Float64[], type=Symbol[], converged=Bool[], drho=Float64[], drhoa=Float64[], dm=Float64[])
    for pert in ("rand", "non-symmetric", "double", "biased"), ϕa in phia_ts  # every IC at every φa
        r = sweep_run(ϕ, ϕa, pert)
        push!(out.pert, pert); push!(out.phia, ϕa); push!(out.type, r.sol_type); push!(out.converged, r.converged)
        push!(out.drho, r.drho_eff); push!(out.drhoa, r.drhoa_eff); push!(out.dm, r.dm_eff)
    end
    return out
end

# One run of the sweep at (φ, φa) from initial condition pert, its whole history saved in data/oscillatory/runs
function sweep_run(ϕ, ϕa, pert)
    qc, ϕac, wc, _ = find_qc_phiac_wc(ϕ, PE, LX)
    vp, vm = travelling_eigenvectors(ϕ, ϕac, qc, wc, PE)  # critical travelling eigenvectors (for the mode ICs)
    # pert_Ap, pert_Am: (A⁺, A⁻) of the biased IC only; run stops when the classifier converges
    r = run_spectral(ϕ, ϕa, PE, LX, 256; T=200000.0, dt=0.01, pert_type=pert, pert_amp=0.1,
                     pert_qc=qc, pert_vp=vp, pert_vm=vm, pert_Ap=2.0, pert_Am=1.0, seed=1, save_interval=20.0,
                     record_interval=HISTORY)
    save_history(@sprintf("oscillatory/runs/phi%s/%s/phia=%.6f.jld2", ϕ, pert, ϕa), r; ϕ, ϕa, Pe=PE, Lx=LX, pert)
    @printf("  %-13s φa = %.5f: %s (converged %s, t = %.0f)\n", pert, ϕa, r.sol_type, r.converged, r.t)
    return r
end

# ==================================================================================================================
# Figures
# ==================================================================================================================

# c(φa) of the continuation against the weakly nonlinear speed (its first point, the seed at δ = 0.1, is left out)
function speed_figure(d; ϕ=0.88)
    qc, ϕac, wc, _ = find_qc_phiac_wc(ϕ, PE, LX)
    # stable or not: the weakly nonlinear T with ϕ1 = +1
    stab = classify_branch_stability(PE, qc, ϕ, ϕac, branch_sign_osc(PE, qc, ϕ, ϕac, 1.0, wc; mode=:plus), wc, :plus) == :stable
    i = sortperm(d.phia[2:end]) .+ 1  # all but the seed, by increasing φa
    # weakly nonlinear speed c = ω(δ)/qc at the same δ
    c_wnl = [wnl_effective_frequency(PE, qc, ϕ, ϕac, δ, 1.0, wc; mode=:plus) / qc for δ in d.delta[i]]
    xt = sort(vcat(collect(range(ϕac, ϕac + 0.03, length=4)), ϕac))  # every 0.01 from the onset (φa*)
    labels = [isapprox(x, ϕac; atol=1e-8) ? L"\phi_a^{*}" : @sprintf("%.3f", x) for x in xt]
    p = plot(d.phia[i], d.c[i]; xlabel=L"\phi_a", ylabel=L"c", color=:blue, lw=2.8, legend=false, xticks=(xt, labels),
             xlims=(ϕac - 0.003, ϕac + 0.035), ylims=(0.07, 0.12), guidefontsize=GFS, tickfontsize=FS)
    lab = stab ? "weakly nonlinear estimation (stable)" : "weakly nonlinear estimation (unstable)"
    plot!(p, d.phia[i], c_wnl; color=:black, lw=2.8, ls=stab ? :dashdot : :dash)
    entries = [legend_entry("numerical continuation"; color=:blue, linewidth=2.8),
               legend_entry(lab; linestyle=stab ? :dashdot : :dash, linewidth=2.8)]
    return with_legend(p, entries; size=(900, 600), legend_height=0.2, columns=1, fontsize=FS, sample=0.07, margin=8Plots.mm)
end

const STAB_COLOR = Dict("stable" => :blue, "unstable" => :yellow, "saddle" => :purple)
const SPIN_MARKS = [0.7, 0.805, 0.88, 0.905, 0.915]

# Spinodal coloured by the expected solution of one branch (T or CP): blue stable, purple saddle, yellow unstable,
# red subcritical, gray stationary onset; black squares at the φ of the bifurcation diagrams
function spinodal_panel(d, has, stab, tag)
    # gray: stationary onset; red: branch not on the supercritical side (subcritical); else by its stability
    colors = [d.is_stationary[i] ? :gray : has[i] ? STAB_COLOR[stab[i]] : :red for i in eachindex(d.phi)]
    p = plot(; xlabel=L"\phi", ylabel=L"\nu", title=tag, titlelocation=:left, titlefontsize=GFS, legend=:topleft,
             guidefontsize=GFS, tickfontsize=FS, legendfontsize=FS)
    starts = [1; findall(i -> colors[i] != colors[i-1], 2:length(colors)) .+ 1]   # one line per run of equal colour
    for (k, i0) in enumerate(starts)
        i1 = k < length(starts) ? starts[k+1] : length(colors)  # runs share their end points
        plot!(p, d.phi[i0:i1], d.nu[i0:i1]; color=colors[i0], lw=4.2, label="")
    end
    for (c, lab) in ((:blue, "Supercritical region (stable)"), (:purple, "Supercritical region (saddle)"),
                     (:yellow, "Supercritical region (unstable)"), (:red, "Subcritical region"), (:gray, "Stationary region"))
        # a legend entry for each colour present
        any(==(c), colors) && scatter!(p, [NaN], [NaN]; color=c, label=lab, ms=5, msw=0.5)
    end
    # ν of the marked φ
    nu = [(ϕ - find_qc_phiac_wc(ϕ, PE, LX)[2]) / (1 - ϕ) for ϕ in SPIN_MARKS]
    scatter!(p, SPIN_MARKS, nu; color=:black, marker=:square, ms=6, msw=0, label="")
    return p
end

spinodals_figure(d) = plot(spinodal_panel(d, d.has_T, d.stab_T, L"\mathrm{(a)}"), spinodal_panel(d, d.has_CP, d.stab_CP, L"\mathrm{(b)}");
                           layout=(1, 2), size=(1800, 600), margin=8Plots.mm, bottom_margin=14Plots.mm)

# Shaded spinodal: the plane below (and above) the spinodal coloured by the solution expected there, (a) for T and
# (b) for CP, fading away from the spinodal; dashed: the stationary instability alone at q = 2π/Lx, which adds its PS
# shading where it is a second instability. Markers: final states of the sweeps, (a) non-symmetric, (b) double IC.
const SHADE_SETTINGS = (size=(1400, 560), legend_height=0.14, xlims=(0.55, 0.95), ylims=(0.0, 2.25),
    colors=Dict("PS" => (1.0, 0.0, 0.0), "T" => (0.5, 0.0, 0.5), "CP" => (0.0, 0.0, 1.0)),
    shade_max=0.85, shade_length=0.4, shade_max_above=0.5, shade_length_above=0.2, shade_res=(1600, 800),
    # (class of T, class of CP) along the spinodal => (colour in (a), (b)), (shaded above the spinodal in (a), (b))
    # shaded above (beyond the spinodal, where H is stable) only if the solution coloured in that panel is subcritical
    # (it exists there); a supercritical T or CP (stable or saddle) exists only below the spinodal
    classes=Dict(("stat", "stat") => (("PS", "PS"), (true, true)), ("sub", "sub") => (("T", "none"), (true, true)),
                 ("sub", "saddle") => (("T", "CP"), (true, false)), ("stable", "saddle") => (("T", "CP"), (false, false)),
                 ("saddle", "stable") => (("CP", "CP"), (false, false)), ("saddle", "sub") => (("CP", "CP"), (true, true))),
    second="PS", second_length_alone=0.45, ts_ics=("non-symmetric", "double"),
    markers=[(:T, :dtriangle, :purple, 7), (:H, :x, :white, 10), (:PS, :utriangle, :red, 7), (:CP, :circle, :blue, 5)])

function shaded_spinodal_figure(d, sweeps)
    S = SHADE_SETTINGS
    cls(h, s, st) = st ? "stat" : (h ? s : "sub")  # stat, the stability, or sub
    pair = [(cls(d.has_T[i], d.stab_T[i], d.is_stationary[i]), cls(d.has_CP[i], d.stab_CP[i], d.is_stationary[i])) for i in eachindex(d.phi)]
    nx, ny = S.shade_res
    xs = range(S.xlims..., length=nx); ys = range(S.ylims..., length=ny)
    # shading weight at ν = y for a line at y0: decays over shade_length below, shade_length_above above
    fade(y, y0) = y > y0 ? S.shade_max_above*exp(-(y - y0)/S.shade_length_above) : S.shade_max*exp(-(y0 - y)/S.shade_length)
    # blend colour c into pixel (k, j) with weight t
    paint!(img, k, j, t, c) = (img[k, j, :] .= (1 - t) .* img[k, j, :] .+ t .* c)
    function shade(panel)
        img = ones(ny, nx, 3)
        for (j, x) in enumerate(xs)
            d.phi[1] <= x <= d.phi[end] || continue  # only within the computed φ
            i = clamp(searchsortedfirst(d.phi, x), 1, length(d.phi))  # spinodal point of this column
            (name, above) = S.classes[pair[i]]  # its colours and above-flags
            if name[panel] != "none"
                for (k, y) in enumerate(ys)
                    y > d.nu[i] && !above[panel] && break  # stop at the line unless shaded above
                    paint!(img, k, j, fade(y, d.nu[i]), S.colors[name[panel]])
                end
            end
            d.is_stationary[i] && continue  # no second instability here
            i2 = clamp(searchsortedfirst(d.phi_st, x), 1, length(d.phi_st))  # point of the ν_st line
            d.phi_st[1] <= x <= d.phi_st[end] || continue
            for (k, y) in enumerate(ys)
                y > d.nu_st[i2] && !d.st_sub[i2] && break  # above ν_st only if subcritical
                # no colour of the panel's own here: the PS shading above ν_st fades over second_length_alone instead
                t = name[panel] == "none" && y > d.nu_st[i2] ? S.shade_max_above*exp(-(y - d.nu_st[i2])/S.second_length_alone) : fade(y, d.nu_st[i2])
                paint!(img, k, j, t, S.colors[S.second])
            end
        end
        return [RGB(img[k, j, 1], img[k, j, 2], img[k, j, 3]) for k in 1:ny, j in 1:nx]
    end
    w = 0.3        # thick ×: a plus of arm half-width w turned by 45°
    plus = [(w, 1), (w, w), (1, w), (1, -w), (w, -w), (w, -1), (-w, -1), (-w, -w), (-1, -w), (-1, w), (-w, w), (-w, 1)]
    xmark = Shape([((x - y)/√2, (x + y)/√2) for (x, y) in plus])
    panels = map(enumerate((L"\mathrm{(a)}", L"\mathrm{(b)}"))) do (k, tag)
        p = plot(; xlims=S.xlims, ylims=S.ylims, xlabel=L"\phi", ylabel=L"\nu", legend=false, grid=false,
                 guidefontsize=GFS, tickfontsize=FS, xticks=0.6:0.1:0.9, yticks=0:0.5:2.0)
        plot!(p, collect(xs), collect(ys), shade(k); label="")
        plot!(p; yflip=false, aspect_ratio=:none, xlims=S.xlims, ylims=S.ylims)     # (an image sets an equal aspect ratio)
        for x in 0.6:0.1:0.9; plot!(p, [x, x], collect(S.ylims); color=:lightgray, lw=1, label=""); end   # grid over the shading
        for y in 0.5:0.5:2.0; plot!(p, collect(S.xlims), [y, y]; color=:lightgray, lw=1, label=""); end
        plot!(p, d.phi, d.nu; color=:black, lw=4.2, label="")
        plot!(p, d.phi_st, d.nu_st; color=:black, lw=2.8, ls=:dash, label="")
        for (type, shape, color, ms) in S.markers
            # (φ, ν) of the converged runs from this panel's IC that ended in `type`
            pts = [(ϕt, (ϕt - s.phia[i])/(1 - ϕt)) for (ϕt, s) in sweeps for i in eachindex(s.phia)
                   if s.converged[i] && s.pert[i] == S.ts_ics[k] && s.type[i] == type]
            scatter!(p, first.(pts), last.(pts); markershape=shape === :x ? xmark : shape, markercolor=color,
                     markerstrokecolor=:black, markerstrokewidth=1.0, ms=ms, label=string(type))
        end
        annotate!(p, S.xlims[1], S.ylims[2] + 0.04(S.ylims[2] - S.ylims[1]), txt(tag, GFS, :black, :left, :bottom))
        p
    end
    return with_legend(plot(panels...; layout=(1, 2)), legend_entries(panels...); size=S.size, legend_height=S.legend_height,
                       columns=4, fontsize=FS, margin=8Plots.mm, legend_bottom=6Plots.mm)
end

# ------------------------------------------------------------------------------------------------------------------
# Bifurcation diagrams
# ------------------------------------------------------------------------------------------------------------------

const NEWTON_LABEL = "numerical continuation"
const NEWTON_STYLE = (lw=2.8, ls=:solid, color=:blue)
wnl_label(stab) = stab == :stable ? "weakly nonlinear estimation (stable)" : "weakly nonlinear estimation (unstable)"
wnl_style(stab) = stab == :stable ? :dashdot : :dash          # dash-dot stable, dashed unstable (or saddle)
# the sweeps' markers: hollow, so that overlapping points stay visible; biased small and filled, drawn last
const TS_MARKERS = [("rand", "time stepping (rand)", (markercolor=:white, markerstrokecolor=:purple, marker=:square, ms=7, msw=2.8)),
                    ("double", "time stepping (double)", (markercolor=:white, markerstrokecolor=:red, marker=:circle, ms=6, msw=3.5)),
                    ("non-symmetric", "time stepping (non-symmetric)", (markercolor=:white, markerstrokecolor=:blue, marker=:utriangle, ms=5, msw=2.1)),
                    ("biased", "time stepping (biased)", (markercolor=:darkgreen, markerstrokecolor=:white, marker=:circle, ms=3.2, msw=0.9))]
marker_entry(lab, m) = legend_entry(lab; marker=m.marker, markercolor=m.markercolor, markerstrokecolor=m.markerstrokecolor,
                                    markersize=m.ms, markerstrokewidth=m.msw)
const BIF_LEGEND = [legend_entry(NEWTON_LABEL; color=:blue, linewidth=2.8), legend_entry(wnl_label(:stable); linestyle=:dashdot, linewidth=2.8),
                    legend_entry(wnl_label(:unstable); linestyle=:dash, linewidth=2.8), (marker_entry(l, m) for (_, l, m) in TS_MARKERS)...]
# BIF_LEGEND without the stable weakly nonlinear entry when none of the panels has a stable (dash-dot) curve
has_ls(ls, panels...) = any(p -> any(s -> s[:linestyle] == ls, p.series_list), panels)
bif_legend(panels...) = has_ls(wnl_style(:stable), panels...) ? BIF_LEGEND : [e for (i, e) in enumerate(BIF_LEGEND) if i != 2]

# Weakly nonlinear amplitude of the stationary branch with n wavelengths, field k (stable if supercritical)
function ps_wnl_curve(ϕ, b, k, lo, hi; Nx_wnl=1000)
    qn = b.n * 2π / LX
    x = (0:Nx_wnl-1) .* (LX / Nx_wnl)
    pts = [(b.ϕac_n + b.bs * e, sqrt(e)) for e in range(0.0, abs(hi - b.ϕac_n) * 1.3; length=200) if lo <= b.ϕac_n + b.bs * e <= hi]
    return first.(pts), [wnl_stat_amplitudes(x, PE, qn, ϕ, b.ϕac_n, pa, δ, b.bs)[k] for (pa, δ) in pts]
end

"""
One panel (field k = 1, 2, 3: Δρ, Δρa, Δm) of the bifurcation diagram at φ: numerical continuation of T, of the
homogeneous state and of the PS branches (blue), the weakly nonlinear curves of T, CP and PS (`wnl`), the sweep's final
states (`ts`), and time-stepping continuation chains (`chains`: (results, label, colour, marker, ±1), points joined by
arrows, for the hysteresis figure). Returns the panel and what is needed to place the curve labels.
"""
function bifurcation_panel(ϕ, tb, ps, ts, k; xlims=nothing, ylims=nothing, wnl=true, chains=(), arrow_offset=0.02,
                           arrow_px=(1000, 420), xhi_curves=tb.phia_hi_plot, title="")
    key = (:drho, :drhoa, :dm)[k]  # field of panel k
    lo, hi = tb.phia_lo_plot, tb.phia_hi_plot
    wT = (tb.phia_wnl_T, getfield(tb, (:wnl_T_rho, :wnl_T_rhoa, :wnl_T_m)[k]))
    wCP = (tb.phia_wnl_CP, getfield(tb, (:wnl_CP_rho, :wnl_CP_rhoa, :wnl_CP_m)[k]))
    wPS = wnl ? [ps_wnl_curve(ϕ, b, k, lo, xhi_curves) for b in ps] : []  # weakly nonlinear PS curves
    if ylims === nothing    # from the data; the weakly nonlinear curves may extend it by at most 1.5x
        # Δ of the converged runs
        tsv = ts === nothing ? Float64[] : [getfield(ts, key)[i] for i in eachindex(ts.phia) if ts.converged[i]]
        ymax_data = maximum(vcat([0.0], getfield(tb, key), tsv, [v for b in ps for v in getfield(b, key)]))
        ymax_wnl = maximum(vcat([0.0], wT[2], wCP[2], [v for c in wPS for v in c[2]]))
        ymax = max(ymax_data, min(ymax_wnl, 1.5ymax_data))
        ylims = (-0.03ymax, 1.08ymax)
    end
    xlims === nothing && (xlims = (lo - 0.01(hi - lo), hi + 0.01(hi - lo)))  # the window, 1% margins
    p = plot(; xlabel=L"\phi_a", ylabel=(L"\Delta\rho", L"\Delta\rho_a", L"\Delta m")[k], title=title, titlelocation=:left,
             titlefontsize=GFS, legend=false, xlims=xlims, ylims=ylims, guidefontsize=GFS, tickfontsize=FS)
    drawn = Tuple{Vector{Float64},Vector{Float64},Bool}[]     # (x, y, is_line) of everything drawn, for the labels
    # draw a line / markers and keep their points for the labels (empty data skipped)
    line!(x, y; kw...) = (isempty(x) || (push!(drawn, (collect(Float64, x), collect(Float64, y), true)); plot!(p, x, y; label="", kw...)))
    pts!(x, y; kw...) = (isempty(x) || (push!(drawn, (collect(Float64, x), collect(Float64, y), false)); scatter!(p, x, y; label="", kw...)))
    line!(tb.phia, getfield(tb, key); NEWTON_STYLE...)  # T branch (Newton)
    line!([lo, xhi_curves], [0.0, 0.0]; NEWTON_STYLE...)  # homogeneous state
    for b in ps; line!(b.phia, getfield(b, key); NEWTON_STYLE...); end  # PS branches (Newton)
    labels = Tuple{String,Vector{Float64},Vector{Float64}}[]  # (name, x, y) of the curves to label
    if wnl
        line!(wT...; color=:black, lw=2.8, ls=wnl_style(tb.stabT)); push!(labels, ("T", wT...))
        line!(wCP...; color=:black, lw=2.8, ls=wnl_style(tb.stabCP)); push!(labels, ("CP", wCP...))
        for (b, c) in zip(ps, wPS)
            line!(c...; color=:black, lw=2.8, ls=wnl_style(b.bs > 0 ? :stable : :unstable)); push!(labels, ("PS", c...))
        end
    end
    if ts !== nothing
        for (pert, _, m) in TS_MARKERS
            # the converged runs from this IC
            i = [j for j in eachindex(ts.phia) if ts.pert[j] == pert && ts.converged[j]]
            pts!(ts.phia[i], getfield(ts, key)[i]; m...)
        end
    end
    for (r, _, col, mk, sgn) in chains        # accepted points in order, arrows (shifted up or down) between them
        xs = r.phia; ys = getfield(r, key)
        off = arrow_offset * (ylims[2] - ylims[1]) * sgn  # arrows shifted by a fraction of the y range
        for j in 1:length(xs)-1
            arrow!(p, xs[j] + 0.2(xs[j+1] - xs[j]), ys[j] + 0.2(ys[j+1] - ys[j]) + off, xs[j] + 0.8(xs[j+1] - xs[j]),
                   ys[j] + 0.8(ys[j+1] - ys[j]) + off, xlims, ylims, arrow_px; color=col)
        end
        pts!(xs, ys; chain_style(col, mk)...)
    end
    return (p=p, xl=xlims, yl=ylims, drawn=drawn, labels=labels)
end

# Markers of the hysteresis chains (sized for the 2600 × 2050 px hysteresis figure)
chain_style(col, mk) = (markercolor=:white, markerstrokecolor=col, marker=mk, ms=(mk == :square ? 7.1 : 6.8), msw=0.73)

# Label of each weakly nonlinear curve (T, CP, PS) as close to its onset as possible, inside the panel, clear of every
# curve and marker, nearer its own curve than any other, and away from the other labels; computed in pixels of the
# panel's plot area (W × H). at: labels placed by hand, name => φa, centred just above their curve at that φa
function place_wnl_labels!(sp, pan, W, H; at=Dict{String,Float64}())
    xl, yl = pan.xl, pan.yl
    # data -> px of the plot area
    topx(x, y) = ((x - xl[1]) / (xl[2] - xl[1]) * W, (y - yl[1]) / (yl[2] - yl[1]) * H)
    function dense(xs, ys)          # a point every ~3 px along a curve
        P = [topx(x, y) for (x, y) in zip(xs, ys)]
        out = Tuple{Float64,Float64}[]
        for k in 1:length(P)-1
            n = max(1, ceil(Int, hypot(P[k+1][1]-P[k][1], P[k+1][2]-P[k][2]) / 3))
            append!(out, [(P[k][1] + t*(P[k+1][1]-P[k][1]), P[k][2] + t*(P[k+1][2]-P[k][2])) for t in range(0, 1; length=n+1)])
        end
        return out
    end
    obst = Tuple{Float64,Float64,Float64}[]             # (x, y, radius) in px
    for (xs, ys, is_line) in pan.drawn
        is_line ? append!(obst, [(q..., 2.0) for q in dense(xs, ys)]) : append!(obst, [(topx(x, y)..., 8.0) for (x, y) in zip(xs, ys)])
    end
    filter!(q -> -50 <= q[1] <= W + 50 && -50 <= q[2] <= H + 50, obst)
    # number of obstacles inside box b = (x0, x1, y0, y1) padded by pad
    hits(b, pad) = count(o -> b[1] - pad - o[3] < o[1] < b[2] + pad + o[3] && b[3] - pad - o[3] < o[2] < b[4] + pad + o[3], obst)
    # do boxes b and c overlap (with padding)
    overlap(b, c, pad) = b[1] - pad < c[2] && c[1] - pad < b[2] && b[3] - pad < c[4] && c[3] - pad < b[4]
    boxes = NTuple{4,Float64}[]
    th = FS * 100 / 72 * 1.05                          # text height in px
    # points of each labelled curve near the panel
    curves = [filter(q -> -20 <= q[1] <= W + 20 && -20 <= q[2] <= H + 20, dense(xs, ys)) for (_, xs, ys) in pan.labels]
    # distance from box b to a point q, and to the nearest point of a set Q
    boxdist(b, q) = hypot(max(b[1] - q[1], 0, q[1] - b[2]), max(b[3] - q[2], 0, q[2] - b[4]))
    mindist(b, Q) = isempty(Q) ? Inf : minimum(q -> boxdist(b, q), Q)
    for (j, (lab, xs, ys)) in enumerate(pan.labels)
        length(xs) < 2 && continue
        if haskey(at, lab)                     # by hand: at φa = at[lab], just above the curve (below if that is clearer)
            o = sortperm(xs); x0 = at[lab]
            cx, cy = topx(x0, interp1(xs[o], ys[o], x0))
            tw = 0.62 * FS * 100 / 72 * length(lab)
            up = (cx - tw/2, cx + tw/2, cy + 8.0, cy + 8.0 + th); dn = (cx - tw/2, cx + tw/2, cy - 8.0 - th, cy - 8.0)
            b = hits(dn, 3) < hits(up, 3) ? dn : up; push!(boxes, b)
            annotate!(sp, x0, yl[1] + (b[3] + b[4]) / 2 / H * (yl[2] - yl[1]), txt(lab, FS, :black, :center))
            continue
        end
        # the points of the other labelled curves
        others = reduce(vcat, [curves[i] for i in eachindex(curves) if i != j]; init=Tuple{Float64,Float64}[])
        ord = ys[1] <= ys[end] ? (1:length(xs)) : (length(xs):-1:1)     # walk from the onset outwards
        P = [topx(xs[i], ys[i]) for i in ord]
        keep = findall(q -> 0 <= q[1] <= W && 0 <= q[2] <= H, P)
        isempty(keep) && continue
        P = P[keep[1]:keep[end]]  # first to last point inside the panel
        # arclength in px along the curve
        s = [0.0; cumsum([hypot(P[k+1][1]-P[k][1], P[k+1][2]-P[k][2]) for k in 1:length(P)-1])]
        tw = 0.62 * FS * 100 / 72 * length(lab)          # text width in px
        best = nothing; bestscore = Inf
        for dd in 20:6:max(20, min(s[end], 400))  # positions 20, 26, ... px from the onset
            k = clamp(searchsortedfirst(s, dd), 2, length(P))
            t = (dd - s[k-1]) / max(s[k] - s[k-1], 1e-9)
            cx, cy = P[k-1][1] + t*(P[k][1]-P[k-1][1]), P[k-1][2] + t*(P[k][2]-P[k-1][2])
            g = 8.0
            # 8 candidate boxes: left and right of the point (level, below, above) and centred below and above it
            for b in ((cx - g - tw, cx - g, cy - th/2, cy + th/2), (cx + g, cx + g + tw, cy - th/2, cy + th/2),
                      (cx - g - tw, cx - g, cy + g/2, cy + g/2 + th), (cx + g, cx + g + tw, cy + g/2, cy + g/2 + th),
                      (cx - g - tw, cx - g, cy - g/2 - th, cy - g/2), (cx + g, cx + g + tw, cy - g/2 - th, cy - g/2),
                      (cx - tw/2, cx + tw/2, cy + g, cy + g + th), (cx - tw/2, cx + tw/2, cy - g - th, cy - g))
                (b[1] < 4 || b[2] > W - 4 || b[3] < 4 || b[4] > H - 4) && continue  # inside the panel
                any(c -> overlap(b, c, 40), boxes) && continue  # away from the labels placed
                # score: obstacles hit, nearer another curve than its own, distance from the onset
                score = 1000 * hits(b, 3) + 500 * (mindist(b, others) < mindist(b, curves[j]) + 20) + dd
                if score < bestscore
                    best, bestscore = b, score
                end
            end
            bestscore < 500 && break  # no hits and nearest its own curve
        end
        best === nothing && continue
        push!(boxes, best)
        # centre of the box back to data coordinates
        annotate!(sp, xl[1] + (best[1] + best[2]) / 2 / W * (xl[2] - xl[1]), yl[1] + (best[3] + best[4]) / 2 / H * (yl[2] - yl[1]),
                  txt(lab, FS, :black, :center))
    end
end

# linear interpolation of y(x) at x0 (x increasing)
interp1(x, y, x0) = (k = clamp(searchsortedfirst(x, x0), 2, length(x)); y[k-1] + (x0 - x[k-1]) * (y[k] - y[k-1]) / (x[k] - x[k-1]))

# labels of the weakly nonlinear curves placed by hand (φ => name => φa), the others placed automatically
const WNL_LABEL_AT = Dict(0.915 => Dict("T" => 0.87))

# Panel size in px of subplot sp of a laid-out figure
panel_px(sp) = (pa = Plots.plotarea(sp); (px(Plots.width(pa)), px(Plots.height(pa))))

# The bifurcation diagram at φ (three panels, legend above in three columns) and its Δρ panel alone (legend in one
# column), with the labels of the weakly nonlinear curves
function bifurcation_figures(ϕ, tb, ps, ts)
    pans = [bifurcation_panel(ϕ, tb, ps, ts, k; title=k == 1 ? L"\mathrm{(a)}" : "") for k in 1:3]
    legend = bif_legend((q.p for q in pans)...)
    fig = with_legend(plot([q.p for q in pans]...; layout=(1, 3)), legend; size=(1800, 550), legend_height=0.24, columns=3, fontsize=FS)
    # subplot 1 is the legend slot: the panels are subplots 2-4
    at = get(WNL_LABEL_AT, ϕ, Dict{String,Float64}())
    for k in 1:3; place_wnl_labels!(fig[k + 1], pans[k], panel_px(fig[k + 1])...; at=at); end
    pan = bifurcation_panel(ϕ, tb, ps, ts, 1; title=L"\mathrm{(a)}")
    legend = bif_legend(pan.p)
    leg_px = 36 * length(legend) / 0.76 + 47  # legend height in px
    rho = with_legend(pan.p, legend; size=(680, 550), legend_height=leg_px / (550 + leg_px), columns=1, fontsize=FS,
                      sample=0.07, pad=0.05)
    place_wnl_labels!(rho[2], pan, panel_px(rho[2])...; at=at)
    return fig, rho
end
