# Stationary notebook (Pe = 4.5): spinodals with the binodal, and the bifurcation diagrams of the stationary
# instability at φ = 0.81 (supercritical) and φ = 0.7 (subcritical).
include(joinpath(@__DIR__, "common.jl"))
load_src("weakly_nonlinear.jl"); load_src("newton.jl"); load_src("timestepping.jl"); load_src("binodal.jl")

const PE_STAT = 4.5
const LX = 10.0

# ==================================================================================================================
# Data
# ==================================================================================================================

# Spinodal ν(φ) = (φ - ϕac)/(1 - φ) for n values of φ in [0.5, 0.98] (where ϕac < φ), and whether the stationary
# bifurcation there is subcritical
function spinodal_scan(Pe, Lx; n)
    phi = Float64[]; nu = Float64[]; phia = Float64[]; sub = Bool[]
    for ϕ in range(0.5, 0.98, length=n)
        qc, ϕac = find_qc_and_phiac(ϕ, Pe, Lx)  # ϕac: the lower of the stationary (q = 2π/Lx) and oscillatory thresholds
        ϕac >= ϕ && continue  # unstable only for φa > φ (no passive particles): no point
        # ν = (φ - ϕac)/(1 - φ)
        push!(phi, ϕ); push!(nu, (ϕ - ϕac) / (1 - ϕ)); push!(phia, ϕac)
        push!(sub, stationary_subcritical(ϕ, Pe, qc, ϕac))  # criticality from the sign of Π/Γ at (qc, ϕac)
    end
    return (phi=phi, nu=nu, phia=phia, sub=sub)
end

# Onset (qc, ϕac) at φ and the onsets of the branches with 2 and 3 wavelengths in the box
function onsets(ϕ; Pe=PE_STAT, Lx=LX)
    qc, ϕac = find_qc_and_phiac(ϕ, Pe, Lx)
    # thresholds of the modes 2qc and 3qc (2 and 3 wavelengths)
    return qc, ϕac, phia_stationary(2qc, ϕ, Pe), phia_stationary(3qc, ϕ, Pe)
end

# Branch with n wavelengths (n = 2, 3; and n = 1 at φ = 0.81) of the supercritical-type scan: Newton continuation
# from ϕac + eps_max down to the onset, and its weakly nonlinear amplitudes. tol: Newton tolerance on the residual norm
# (round-off sets a floor that grows with Nx and near the onset: about 1e-10 to 3e-10 at Nx = 2000)
function stationary_branch(ϕ, n; Nx, eps_max=0.035, eps_min=0.0, min_step=1e-10, tol=1e-10)
    qc, ϕac, ϕ2ac, ϕ3ac = onsets(ϕ)
    onset = (ϕac, ϕ2ac, ϕ3ac)[n]  # onset of the branch with n wavelengths
    # Newton continuation at wavenumber n qc, φa = onset + ε for ε from eps_max down to eps_min
    b = bifurcation_scan(ϕ, PE_STAT, LX, Nx, n * qc, onset, 1.0; eps_max=eps_max, eps_min=eps_min, min_step=min_step,
                         tol=tol)
    return merge(b, (onset=onset,))  # fields of bifurcation_scan plus the onset
end

# Subcritical branch with one wavelength at φ: from a weakly nonlinear seed near onset along φa to the fold (the
# unstable part), then by arclength continuation around the fold (the stable part). Peak-to-peak amplitudes of both.
# The arclength step is 0.005 at Nx = 200 and scales as √Nx, since the norm of the state grows as √Nx: the same step
# along the branch for every Nx. tol: as in stationary_branch.
function subcritical_branch(ϕ; Nx=200, tol=1e-10)
    qc, ϕac = find_qc_and_phiac(ϕ, PE_STAT, LX)
    bd = branch_sign(ϕ, PE_STAT, qc, ϕac, 1.0)  # -1: the branch exists for φa < ϕac (subcritical)
    @assert bd == -1.0 "the bifurcation at φ = $ϕ is not subcritical"
    δ = 0.001  # φa of the seed: ϕac - δ²
    ϕa_seed = ϕac + bd * δ^2
    x = 0:LX/Nx:(LX - LX/Nx)  # grid
    z = [weaklynonlinear_stat(xi, PE_STAT, qc, ϕ, ϕac, ϕa_seed, δ, bd) for xi in x]  # weakly nonlinear seed
    rho0 = real([zi[1] for zi in z]); rhoa0 = real([zi[2] for zi in z]); m0 = real([zi[3] for zi in z])
    enforce_masses!(rho0, rhoa0, ϕ, ϕa_seed)  # exact masses φ, φa
    p_seed = Params(Nx, LX, PE_STAT, ϕ, ϕa_seed; pin_idx=1)  # m[1] pinned
    res_seed = converge_guess(pack_state(rho0, rhoa0, m0), p_seed; tol=tol, maxiter=80)  # Newton on the seed
    # natural continuation in φa towards ϕac - 0.09; it stops where Newton fails (the fold)
    unstable = continue_in_phia(res_seed, p_seed, ϕ; phia_target=ϕac + bd * 0.3^2, step0=0.00005, min_step=1e-6, tol=tol)
    # pseudo-arclength from the last two solutions, around the fold
    stable = continue_arclength(unstable[end-1], unstable[end], p_seed, ϕ, unstable[end].res.m[1]; ds=0.005 * sqrt(Nx / 200),
                                n_steps=1500, tol=tol)
    # peak-to-peak amplitudes (Δρ, Δρa, Δm) of each solution
    amps(sols) = (phia=[s.phia for s in sols], drho=[amplitudes(s.res)[1] for s in sols],
                  drhoa=[amplitudes(s.res)[2] for s in sols], dm=[amplitudes(s.res)[3] for s in sols])
    return (onset=ϕac, unstable=amps(unstable), stable=amps(stable))
end

# Time stepping (Nx = 256, dt = 0.01) from random initial conditions at 10 values of φa from phia_start to φ (the last
# one, φa = φ, has no passive particles and is skipped): final state type and amplitudes
function stationary_timestepping(ϕ, phia_start; T)
    out = (phia=Float64[], type=Symbol[], converged=Bool[], drho=Float64[], drhoa=Float64[], dm=Float64[],
           types=NTuple{3,Symbol}[])
    for ϕa in range(phia_start, ϕ; length=10)  # 10 values of φa from phia_start to φ
        (ϕa <= 0 || ϕ - ϕa <= 0 || ϕa >= 1) && continue  # outside 0 < φa < φ, φa < 1: skipped
        # classifier run from random noise, stopped once converged
        r = run_spectral(ϕ, ϕa, PE_STAT, LX, 256; T=T, dt=0.01, pert_type="rand", pert_amp=0.1, seed=1, save_interval=5.0,
                         record_interval=HISTORY)
        # whole history in data/stationary/runs
        save_history(@sprintf("stationary/runs/phi%s/phia=%.6f.jld2", ϕ, ϕa), r; ϕ, ϕa, Pe=PE_STAT, Lx=LX, pert="rand")
        @printf("  φa = %.5f: %s (converged %s), Δρ = %.4f%s\n", ϕa, r.sol_type, r.converged, r.drho_eff,
                allequal(r.sol_types) ? "" : "  types of ρ, ρa, m differ: $(r.sol_types)")
        push!(out.phia, ϕa); push!(out.type, r.sol_type); push!(out.converged, r.converged)
        # Δρ, Δρa, Δm of the classifier and the types of ρ, ρa, m
        push!(out.drho, r.drho_eff); push!(out.drhoa, r.drhoa_eff); push!(out.dm, r.dm_eff); push!(out.types, r.sol_types)
    end
    return out
end

# ==================================================================================================================
# Figures
# ==================================================================================================================

const FS = 18      # ticks, legends, annotations
const GFS = 24     # axis labels, panel labels

# The spinodal coloured by criticality: red where the bifurcation is subcritical, blue where supercritical (each run
# extended by one point so that the colours join)
function spinodal_lines!(p, s; subplot=1, lw=4.2)
    run_of(flag) = [s.sub[i] == flag || (i > 1 && s.sub[i-1] == flag) for i in eachindex(s.sub)]
    # subcritical runs
    plot!(p, ifelse.(run_of(true), s.phi, NaN), ifelse.(run_of(true), s.nu, NaN); color=:red, lw=lw, label="", subplot=subplot)
    # supercritical runs
    plot!(p, ifelse.(run_of(false), s.phi, NaN), ifelse.(run_of(false), s.nu, NaN); color=:blue, lw=lw, label="", subplot=subplot)
end

fmt(f) = x -> Printf.format(Printf.Format(f), x)          # tick labels with a fixed number of decimals

# (a) Lx = 10, with the bounds 1 - r± of the supercritical interval and the spinodal points at φ = 0.7 and 0.81;
# (b) Lx = 1000 with the binodal and a zoom near φ = 1 - 4/Pe²
function spinodals_figure(s10, s1000, bin; Pe=PE_STAT)
    kw = (xlabel=L"\phi", ylabel=L"\nu", legend=:topleft, guidefontsize=GFS, tickfontsize=FS, legendfontsize=FS,
          titlefontsize=GFS, titlelocation=:left, framestyle=:axes)
    # (a): r± are the roots of the cubic coefficient for q = 2π/Lx
    qc = 2π / LX
    denom = Pe^4 + 2Pe^2 * qc^2 - 23qc^4
    sq = 2sqrt(3qc^2 * (4Pe^2 + 39qc^2))
    # 1 - r± are values of φ
    r_plus, r_minus = (4Pe^2 + 10qc^2 + sq) / denom, (4Pe^2 + 10qc^2 - sq) / denom
    pa = plot(; title=L"\mathrm{(a)}", xformatter=fmt("%.2f"), yformatter=fmt("%.2f"), kw...)
    spinodal_lines!(pa, s10)
    for (r, lab) in ((r_plus, L"1 - r_+"), (r_minus, L"1 - r_-"))
        vline!(pa, [1 - r]; color=:black, ls=:dot, lw=2.8, label="")
        annotate!(pa, 1 - r - 0.003, 0.03, txt(lab, FS, :black, :right))
    end
    # legend entries only
    scatter!(pa, [NaN], [NaN]; color=:red, ms=5, msw=0, label="Subcritical region")
    scatter!(pa, [NaN], [NaN]; color=:blue, ms=5, msw=0, label="Supercritical region")
    marks = [0.7, 0.81]  # φ of the two bifurcation diagrams
    numarks = [(ϕ - find_qc_and_phiac(ϕ, Pe, LX)[2]) / (1 - ϕ) for ϕ in marks]  # their ν on the Lx = 10 spinodal
    scatter!(pa, marks, numarks; color=:black, marker=:square, ms=6, msw=0, label="")
    # (b)
    ϕ0 = 1 - 4 / Pe^2  # φ = 1 - 4/Pe²
    pb = plot(; title=L"\mathrm{(b)}", xformatter=fmt("%.1f"), yformatter=fmt("%.2f"), kw...)
    spinodal_lines!(pb, s1000)
    vline!(pb, [ϕ0]; color=:black, ls=:dot, lw=2.8, label="")
    annotate!(pb, ϕ0 - 0.004, 0.165, txt(L"1-4/\mathrm{Pe}^2", FS, :black, :right))
    scatter!(pb, [NaN], [NaN]; color=:red, ms=5, msw=0, label="Subcritical region")
    scatter!(pb, [NaN], [NaN]; color=:blue, ms=5, msw=0, label="Supercritical region")
    # binodal as ν: passive fraction (φ - φa)/(1 - φ) of both coexisting densities
    plot!(pb, bin.ϕ1, bin.ϕp1 ./ (1 .- bin.ϕ1); color=:purple, lw=4.2, label="")
    plot!(pb, bin.ϕ2, bin.ϕp2 ./ (1 .- bin.ϕ2); color=:purple, lw=4.2, label="Binodal")
    sup = .!s1000.sub  # supercritical points of the Lx = 1000 spinodal
    scatter!(pb, s1000.phi[sup], s1000.nu[sup]; color=:blue, ms=4, msw=0, label="")   # the short supercritical stretch
    # zoom
    near = abs.(s1000.phi .- ϕ0) .<= 0.05  # zoom: points within 0.05 of φ0
    ylo, yhi = maximum(s1000.nu[near]) - 0.2, maximum(s1000.nu[near]) + 0.2  # y range about the largest ν there
    plot!(pb; inset=(1, bbox(0.4, 0.5, 0.3, 0.40)), subplot=2, xlims=(ϕ0 - 0.05, ϕ0 + 0.05), ylims=(ylo, yhi),
          xticks=[0.76, 0.80, 0.84], yticks=0.1:0.1:0.4, xformatter=fmt("%.2f"), yformatter=fmt("%.1f"), tickfontsize=FS,
          legend=false, framestyle=:axes)
    spinodal_lines!(pb, s1000; subplot=2)
    vline!(pb, [ϕ0]; color=:black, ls=:dot, lw=2.1, label="", subplot=2)
    plot!(pb, bin.ϕ1, bin.ϕp1 ./ (1 .- bin.ϕ1); color=:purple, lw=4.2, label="", subplot=2)
    plot!(pb, bin.ϕ2, bin.ϕp2 ./ (1 .- bin.ϕ2); color=:purple, lw=4.2, label="", subplot=2)
    scatter!(pb, s1000.phi[sup], s1000.nu[sup]; color=:blue, ms=6, msw=0, label="", subplot=2)
    return plot(pa, pb; layout=(1, 2), size=(1800, 600), margin=8Plots.mm, bottom_margin=14Plots.mm)
end

const NC_LABEL = "numerical continuation"
const WNL_STABLE_LABEL = "weakly nonlinear estimation (stable)"
const WNL_UNSTABLE_LABEL = "weakly nonlinear estimation (unstable)"
const TS_LABEL = "time stepping"
const NC = (color=:blue, lw=2.8)                                   # numerical continuation

# The three panels (Δρ, Δρa, Δm) of a stationary bifurcation diagram: the shaded region φa > φ (no passive particles),
# the n = 1 branch (supercritical: from bifurcation_scan; subcritical: unstable and stable parts, with the weakly
# nonlinear curve of the unstable one), the branches with n = 2, 3 wavelengths, the homogeneous state and the
# time-stepping points. Stability of a weakly nonlinear curve is that of its amplitude equation, i.e. against the
# perturbations it contains (for the n-wavelength branch the modes n q, 2n q, ..., not the critical q): the
# supercritical n = 2, 3 branches are stable in that sense (dash-dot), although unstable to the mode q in the full box
function stationary_panels(ϕ, n1, n2, n3, ts; xlims)
    qc, ϕac = find_qc_and_phiac(ϕ, PE_STAT, LX)
    # (fields of the time stepping and of the branches, fields of the weakly nonlinear curves, label)
    keys_ = ((:drho, :drho_wnl, L"\Delta\rho"), (:drhoa, :drhoa_wnl, L"\Delta\rho_a"), (:dm, :dm_wnl, L"\Delta m"))
    sub = haskey(n1, :unstable)  # subcritical_branch data: unstable and stable parts
    if sub   # unstable part sorted in φa and joined to the onset; weakly nonlinear curve from ϕac - 0.02 to ϕac
        iu = sortperm(n1.unstable.phia); is = sortperm(n1.stable.phia)  # orders in φa
        Nx = 200; x = 0:LX/Nx:(LX - LX/Nx)  # grid of the weakly nonlinear amplitudes
        phia_w = collect(range(ϕac - 0.02, ϕac; length=200))
        wnl = [wnl_stat_amplitudes(x, PE_STAT, qc, ϕ, ϕac, ϕa, sqrt(abs(ϕa - ϕac)), -1.0) for ϕa in phia_w]
        phia_hi = max(maximum(n1.unstable.phia), maximum(n1.stable.phia), ϕac)  # right end of the H line
    end
    return map(enumerate(keys_)) do (k, (key, wkey, ylab))
        p = plot(; xlabel=L"\phi_a", ylabel=ylab, xlims=xlims, legend=false, guidefontsize=GFS, tickfontsize=FS)
        vspan!(p, [ϕ, xlims[2] + 1]; color=:gray85, alpha=1.0, linealpha=0, label="")  # φa > φ: no passive particles
        if sub
            u = getfield(n1.unstable, key)[iu]; pu = n1.unstable.phia[iu]  # unstable part sorted in φa
            ends = abs(pu[1] - ϕac) < abs(pu[end] - ϕac)  # end nearest to the onset, joined there to (ϕac, 0)
            plot!(p, ends ? [ϕac; pu] : [pu; ϕac], ends ? [0.0; u] : [u; 0.0]; NC..., label=NC_LABEL)
            # empty series: legend entry only
            plot!(p, Float64[], Float64[]; color=:black, lw=2.8, ls=:dashdot, label=WNL_STABLE_LABEL)
            # weakly nonlinear curve of the unstable part
            plot!(p, phia_w, getindex.(wnl, k); color=:black, lw=2.8, ls=:dash, label=WNL_UNSTABLE_LABEL)
            plot!(p, n1.stable.phia[is], getfield(n1.stable, key)[is]; NC..., label=NC_LABEL)  # stable part
            plot!(p, [ϕac - 0.02, ϕac], [0.0, 0.0]; NC..., label="")  # H stable (φa < ϕac)
            plot!(p, [ϕac, phia_hi], [0.0, 0.0]; NC..., label="")  # H unstable (φa > ϕac)
        else
            plot!(p, [n1.onset; n1.phia], [0.0; getfield(n1, key)]; NC..., label=NC_LABEL)  # n = 1 branch from (onset, 0)
            plot!(p, [xlims[1], n1.onset], [0.0, 0.0]; NC..., label="")  # H from the left edge to the onset
            plot!(p, [n1.onset, n1.phia[end]], [0.0, 0.0]; NC..., label="")  # H beyond the onset
            # weakly nonlinear curve
            plot!(p, n1.phia_wnl, getfield(n1, wkey); color=:black, lw=2.8, ls=:dashdot, label=WNL_STABLE_LABEL)
        end
        for b in (n2, n3)  # branches with n = 2, 3 joined to their onsets, and their (supercritical) weakly nonlinear curves
            plot!(p, [b.onset; b.phia], [0.0; getfield(b, key)]; NC..., label="")
            plot!(p, b.phia_wnl, getfield(b, wkey); color=:black, lw=2.8, ls=:dashdot, label=WNL_STABLE_LABEL)
        end
        # time stepping: every run, as the classifier left it
        scatter!(p, ts.phia, getfield(ts, key); color=:purple, marker=:square, ms=4, label=TS_LABEL)
        p
    end
end

const STAT_LEGEND = [legend_entry(NC_LABEL; color=:blue, linewidth=2.8),
                     legend_entry(TS_LABEL; marker=:square, markercolor=:purple, markerstrokecolor=:black, markersize=4),
                     legend_entry(WNL_STABLE_LABEL; linestyle=:dashdot, linewidth=2.8),
                     legend_entry(WNL_UNSTABLE_LABEL; linestyle=:dash, linewidth=2.8)]

# The legend entries for these panels: the "unstable" one only if a dashed (unstable) weakly nonlinear curve is drawn
stat_legend(panels...) = any(p -> any(s -> s[:linestyle] == :dash, p.series_list), panels) ? STAT_LEGEND : STAT_LEGEND[1:3]

# The diagram at one φ: three panels, legend in two columns above
stationary_bifurcation_figure(panels) =
    with_legend(plot(panels...; layout=(1, 3)), stat_legend(panels[1]); size=(1700, 500), legend_height=0.2, columns=2,
                fontsize=FS)

# The Δρ panels of two diagrams side by side, tagged (a) and (b) at the top left, legend in two columns above
function stationary_rho_figure(pa, pb)
    pa, pb = deepcopy(pa), deepcopy(pb)  # (the panels of the full figures are left untouched)
    for (p, tag) in ((pa, L"\mathrm{(a)}"), (pb, L"\mathrm{(b)}"))
        plot!(p; title=tag, titlelocation=:left, titlefontsize=GFS)
    end
    return with_legend(plot(pa, pb; layout=(1, 2)), stat_legend(pa, pb); size=(1250, 500), legend_height=0.2, columns=2,
                       fontsize=FS, sample=0.05, gap=0.012, pad=0.025)
end
