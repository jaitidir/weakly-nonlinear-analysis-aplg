# Hysteresis notebook (Pe = 6, φ = 0.905 and 0.915): continuation by time stepping along φa, decreasing and increasing,
# its recording as one continuous run, the figure with the Δρ diagrams and the kymographs, and the videos.
include(joinpath(@__DIR__, "oscillatory.jl"))

const PHIS_H = (0.905, 0.915)

# ==================================================================================================================
# Data
# ==================================================================================================================

# Chains: start of the decreasing (down) and increasing (up) chains, normal step h0, lowest/highest φa, and the branch
# (type, dominant mode n) where the up chain ends
const CHAIN_SETTINGS = Dict(0.905 => (start=0.81094, up_start=0.80381, h0=0.0015, floor=0.795, ceiling=0.8165, up_target=(:PS, 1)),
                            0.915 => (start=0.8800, up_start=0.82225, h0=0.0060, floor=0.815, ceiling=0.9100, up_target=(:PS, 2)))
const KICK_AMP = 1e-4

# New φa: ρa rescaled to the new mass (ρ and m unchanged)
rescale_mass(rho, rhoa, m, ϕa_new) = (copy(rho), rhoa .* (ϕa_new / mean(rhoa)), copy(m))
# Zero-mean random perturbation of size KICK_AMP (seed 1000 + k), so that an unstable state can leave
function kick(rho, rhoa, m, k)
    rng = seeded_rng(1000 + k)
    n = [rand(rng, length(rho)) .- 0.5 for _ in 1:3]  # uniform noise for each field
    # zero mean, largest |value| = KICK_AMP
    n = [(v .- mean(v)) .* (KICK_AMP / maximum(abs, v .- mean(v))) for v in n]
    return (rho .+ n[1], rhoa .+ n[2], m .+ n[3])
end
# Number of wavelengths n ≥ 1 of a run: the dominant Fourier mode of ρ in its final state or, for a CP or MPS, of the
# power |ρ̂_k|² averaged over its last period (records every HISTORY): a single snapshot can catch the first mode of a
# standing wave passing through zero, when the second dominates
function dominant_n(res)
    rows = res.sol_type in (:CP, :MPS) ?
        [Float64.(r) for (s, r) in zip(res.rec.t, res.rec.rho) if s >= res.rec.t[end] - res.period] : [res.rho]
    return argmax(sum(abs2.(fft(r .- mean(r))[2:end÷2]) for r in rows))
end

"""
Continuation chain by time stepping (Nx = 256, dt = 0.01): the first run from random noise, every later run from the
final state of the last accepted one with φa changed by a step h (mass rescaled, kick added, no classification before
t = 10⁴). A run on a different branch (type, n) is rejected and retried with h/2, down to h0/8; then the
jump is accepted and h goes back to h0. The down chain stops 2 accepted points after reaching CP, the up chain 2 after
reaching its target branch. The up chain's φa are half a step off the down chain's (run the down chain first).
"""
function hysteresis_chain(ϕ, direction; down=nothing)
    cfg = CHAIN_SETTINGS[ϕ]; h_min = cfg.h0 / 8  # smallest step: h0/8
    seed_offset = direction == "down" ? 0 : 5000  # different kick seeds for the two chains
    common = (T=60000.0, dt=0.01, seed=1, save_interval=20.0, record_interval=HISTORY)
    # one entry per run (accepted or not): label, φa, type (of ρ), converged (all three fields), Δρ, Δρa, Δm, run, n,
    # stop time, accepted, h, types of ρ, ρa, m
    out = (pert=String[], phia=Float64[], type=Symbol[], converged=Bool[], drho=Float64[], drhoa=Float64[], dm=Float64[],
           step=Int[], n=Int[], t_stop=Float64[], accepted=Bool[], h=Float64[], types=NTuple{3,Symbol}[])
    # print, store and save the history of run k
    function record!(k, ϕa, res, acc, h)
        @printf("[run %2d] φa = %.5f  h = %.5f  %-3s n = %d  t = %.0f  %s%s\n", k, ϕa, h, res.sol_type, dominant_n(res), res.t,
                acc ? "accepted" : "rejected", allequal(res.sol_types) ? "" : "  types of ρ, ρa, m differ: $(res.sol_types)")
        # values in the order of the fields of out
        for (f, v) in zip(keys(out), (direction == "down" ? "continuation" : "continuation up", ϕa, res.sol_type, res.converged,
                                      res.drho_eff, res.drhoa_eff, res.dm_eff, k, dominant_n(res), res.t, acc, h,
                                      res.sol_types))
            push!(getfield(out, f), v)
        end
        save_history(@sprintf("hysteresis/runs/phi%s/%s/run%02d_phia=%.6f.jld2", ϕ, direction, k, ϕa), res;
                     ϕ, ϕa, Pe=PE, Lx=LX, step=k, accepted=acc, h)
    end
    branch(res) = (res.sol_type, dominant_n(res))  # (type, number of wavelengths)
    down_phia = direction == "up" ? down.phia[down.accepted] : Float64[]  # φa of the accepted down runs
    # φa of the next run: down cur - h; up cur + h, or with the normal step the next half-grid point of the down chain
    function next_trial(cur, h)
        direction == "down" && return round(cur - h; digits=6)
        t = cur + h
        if h == cfg.h0      # half a step off the down chain's grid, at least h_min above
            t = cfg.start - (floor((cfg.start - cur - h_min) / cfg.h0 - 0.5) + 0.5) * cfg.h0
        end
        any(abs(t - d) < h_min / 2 for d in down_phia) && (t += h_min / 2)  # never repeat a φa of the down chain
        return round(t; digits=6)
    end
    k = 1
    cur_ϕa = direction == "down" ? cfg.start : cfg.up_start
    # first run: from random noise
    cur = run_spectral(ϕ, cur_ϕa, PE, LX, 256; pert_type="rand", pert_amp=0.1, common...)
    record!(k, cur_ϕa, cur, true, 0.0)
    h = cfg.h0; n_after = -1  # n_after: accepted runs since the end branch was reached (-1: not yet)
    while true
        trial = next_trial(cur_ϕa, h)
        (trial < cfg.floor || trial > cfg.ceiling) && break  # out of the range: stop
        k += 1
        # from the last accepted state, mass rescaled to the new φa and kicked; no classification before t = 10⁴
        res = run_spectral(ϕ, trial, PE, LX, 256; init_fields=kick(rescale_mass(cur.rho, cur.rhoa, cur.m, trial)..., k + seed_offset),
                           check_after=10000.0, common...)
        same = res.converged && branch(res) == branch(cur)  # converged on the same branch
        if same || h / 2 < h_min - 1e-12  # accepted; at the smallest step any result is accepted
            record!(k, trial, res, true, h)
            same || (h = cfg.h0)  # after a jump: back to the normal step
            cur, cur_ϕa = res, trial
            if n_after >= 0  # count the accepted runs after the end branch
                n_after += 1
            # end branch reached: CP (down) or up_target (up), converged
            elseif (direction == "down" && res.sol_type == :CP && res.converged) ||
                   (direction == "up" && res.converged && branch(res) == cfg.up_target)
                n_after = 0
            end
            n_after >= 2 && break  # stop 2 accepted runs after it
        else
            record!(k, trial, res, false, h)
            h /= 2  # rejected: retry from the same state with half the step
        end
    end
    return out
end

# Accepted, converged points of a chain in order
function accepted_points(ch)
    i = sort([j for j in eachindex(ch.phia) if ch.accepted[j] && ch.converged[j]], by=j -> ch.step[j])
    return (; (f => getfield(ch, f)[i] for f in keys(ch))...)
end

"""
    chain_record(ϕ, direction, chain)

The chain as one continuous run: the saved histories (ρ, ρa, m every HISTORY time units, data/hysteresis/runs) of its
accepted runs joined in order. Each accepted run starts from the final state of the previous accepted one (mass
rescaled to the new φa, kick added) and stops when the recurrence classifier has converged, so this is exactly the
simulation of the chain; the rejected trials are left out. Segments: per accepted run its φa, start and end time,
run number, final type, dominant mode n, converged flag, and whether the branch changed.
"""
function chain_record(ϕ, direction, chain)
    acc = sort(findall(chain.accepted), by=j -> chain.step[j])        # accepted runs, in order
    t_all = Float64[]; R = Matrix{Float32}[]; RA = Matrix{Float32}[]; MM = Matrix{Float32}[]
    seg = (phia=Float64[], t_start=Float64[], t_end=Float64[], step=Int[], type=Symbol[], n=Int[], converged=Bool[], jump=Bool[])
    t0 = 0.0
    for (q, j) in enumerate(acc)
        h = loaddata(@sprintf("hysteresis/runs/phi%s/%s/run%02d_phia=%.6f.jld2", ϕ, direction, chain.step[j], chain.phia[j]))
        k0 = q == 1 ? 1 : 2                     # the first record of a later run repeats the end of the previous one
        append!(t_all, t0 .+ h.t[k0:end]); push!(R, h.rho[k0:end, :]); push!(RA, h.rhoa[k0:end, :]); push!(MM, h.m[k0:end, :])
        jump = q > 1 && (chain.type[j], chain.n[j]) != (seg.type[end], seg.n[end])
        for (f, v) in zip(keys(seg), (chain.phia[j], t0, t0 + h.t_stop, chain.step[j], chain.type[j], chain.n[j], chain.converged[j], jump))
            push!(getfield(seg, f), v)
        end
        t0 += h.t_stop
    end
    return (t=t_all, rho=reduce(vcat, R), rhoa=reduce(vcat, RA), m=reduce(vcat, MM), segments=seg, ϕ=ϕ, Lx=LX, rint=HISTORY)
end

# ==================================================================================================================
# Transitions: changes of branch between consecutive runs of a recording (the run, i.e. the φa, where the branch
# changes), numbered first along the increasing chain, then along the decreasing one
# ==================================================================================================================

# runs where the branch (type, n) changes
jump_runs(seg) = [j for j in 2:length(seg.phia) if (seg.type[j], seg.n[j]) != (seg.type[j-1], seg.n[j-1])]

# (chain, run) of every transition; recs = (increasing recording, decreasing recording)
hysteresis_transitions(recs) = [(c, j) for (c, rec) in enumerate(recs) for j in jump_runs(rec.segments)]

# ==================================================================================================================
# Figure: (a), (b) the Δρ diagram at φ = 0.905, 0.915 with both chains; (c), (d) kymograph of ρ along the increasing
# chain with φa(t) below; (e), (f) the same for the decreasing chain
# ==================================================================================================================

# Zoom, arrow offset and labels (transition numbers, branch names) of the Δρ panels: ("text", φa, Δρ, alignment)
const HYST_PANELS = Dict(
    0.905 => (xlims=(0.80286, 0.81264), ylims=(0.082114, 0.1835), arrow_offset=0.02,
              labels=[("T", 0.8075, 0.0975, :left), ("PS", 0.81128, 0.10852, :right), ("CP", 0.80576, 0.1765, :center)]),
    0.915 => (xlims=(0.81892, 0.85), ylims=(0.065095, 0.1548), arrow_offset=0.02,
              labels=[("PS", 0.84535, 0.1287, :right), ("CP", 0.8347, 0.1426, :center)]))

const CHAIN_LEGEND = [legend_entry(NEWTON_LABEL; color=:blue, linewidth=2.8),
                      legend_entry(L"\mathrm{time\ stepping,}\ \phi_a\ \mathrm{decreasing}"; marker=:square, markercolor=:white, markerstrokecolor=:purple, markersize=7.1, markerstrokewidth=0.73),
                      legend_entry(L"\mathrm{time\ stepping,}\ \phi_a\ \mathrm{increasing}"; marker=:utriangle, markercolor=:white, markerstrokecolor=:blue, markersize=6.8, markerstrokewidth=0.73)]

# Time axis and kymographs: settings. heights: (a)/(b), kymographs, φa(t), kymographs, φa(t) (the legend is added on top);
# ncol: columns of a kymograph; w_*, c_*: the time axis (fine scale: the last w_end of every run (w_end_ps for a PS,
# w_end_last for the last run) and the first w_start; the rest of a jump run linear up to w_relax; any other stretch
# compressed to c_len(1 + c_log log(length/c_len)), c_ps times that in a PS); tick_min: least distance between time
# ticks (fraction of the axis); num_dy, type_dy: transition numbers above and state names below φa(t)
const HKS = (size=(2600, 2050), legend_height=0.045, heights=[0.235, 0.2, 0.1, 0.2, 0.1] ./ 0.835, ncol=1600, lw=5.6,
             num_dy=0.12, type_dy=0.30, w_end=700.0, w_end_ps=100.0, w_end_last=800.0, w_start=300.0,
             w_relax=1800.0, c_len=120.0, c_log=0.35, c_ps=0.35, tick_min=0.11, cbar_width=0.035)

# Piecewise linear display coordinate d = D(t) of the time axis, with knots (t_k, d_k)
struct TimeAxis
    t::Vector{Float64}; d::Vector{Float64}; Dmax::Float64
end
function TimeAxis(seg)
    T = seg.t_end[end]; ns = length(seg.phia)  # end time, number of runs
    # fine (true-scale) intervals: the end of every run and the start
    F = [(max(seg.t_end[j] - (j == ns ? HKS.w_end_last : seg.type[j] == :PS ? HKS.w_end_ps : HKS.w_end), seg.t_start[j]), seg.t_end[j]) for j in 1:ns]
    push!(F, (0.0, min(HKS.w_start, T)))
    sort!(F); M = [F[1]]                                        # union of the fine intervals
    for f in F[2:end]
        f[1] <= M[end][2] + 1e-6 ? (M[end] = (M[end][1], max(M[end][2], f[2]))) : push!(M, f)
    end
    # display width of a stretch (a, b) between fine intervals
    function width(a, b)
        len = b - a
        j = clamp(searchsortedlast(seg.t_start, (a + b) / 2), 1, ns)  # run containing the stretch
        seg.jump[j] && return min(len, HKS.w_relax)  # jump run: linear up to w_relax
        # logarithmically compressed beyond c_len
        w = len <= HKS.c_len ? len : HKS.c_len * (1 + HKS.c_log * log(len / HKS.c_len))
        return seg.type[j] == :PS ? HKS.c_ps * w : w
    end
    # knots: compressed gaps and fine intervals at true scale, in order
    tk = [0.0]; dk = [0.0]
    for (a, b) in M
        a > tk[end] + 1e-6 && (push!(dk, dk[end] + width(tk[end], a)); push!(tk, a))
        push!(dk, dk[end] + (b - a)); push!(tk, b)
    end
    T > tk[end] + 1e-6 && (push!(dk, dk[end] + width(tk[end], T)); push!(tk, T))
    return TimeAxis(tk, dk, dk[end])
end
# piecewise linear interpolation of Y(X) at v (linear extrapolation beyond the ends)
interp(X, Y, v) = (k = clamp(searchsortedlast(X, v), 1, length(X) - 1); Y[k] + (Y[k+1] - Y[k]) * (v - X[k]) / (X[k+1] - X[k]))
D(ax::TimeAxis, t) = interp(ax.t, ax.d, t)  # display coordinate of time t
Tinv(ax::TimeAxis, d) = interp(ax.d, ax.t, d)  # time at display coordinate d

# a·10ᵇ with nd decimals in a
function sci(t, nd)
    t < 0.5 && return "0"
    e = floor(Int, log10(t)); m = round(t / 10.0^e; digits=nd)  # exponent and mantissa
    m >= 10 && (m /= 10; e += 1)  # rounded up to 10
    ms = isinteger(m) ? string(Int(m)) : string(m)
    return e == 0 ? ms : string(ms, "\\cdot 10^{", e, "}")
end
# time ticks at the changes of φa and at the end, at least tick_min apart
function tticks(ax::TimeAxis, seg)
    v = Float64[]; l = String[]
    for t in vcat(seg.t_start, seg.t_end[end])
        d = D(ax, t)
        if !isempty(v) && d - v[end] < HKS.tick_min * ax.Dmax
            # too close to the last tick: the end tick replaces it, any other is skipped
            t == seg.t_end[end] && length(v) > 1 ? (pop!(v); pop!(l)) : continue
        end
        # more decimals until the label differs from the previous one
        nd = 1; while !isempty(l) && sci(t, nd) == l[end] && nd < 4; nd += 1; end
        push!(v, d); push!(l, sci(t, nd))
    end
    return (v, latexstring.(l))
end

# Kymograph image: in each of ncol columns ρ at one recorded time taken at random (fixed seed) within the column's
# window and run, so that fast motion shows as texture instead of aliasing into false slow drifts
function kymo(rec, ax::TimeAxis)
    t = rec.t; ρ = rec.rho; seg = rec.segments; w = ax.Dmax / HKS.ncol  # w: display width of a column
    rng = seeded_rng(1); Z = zeros(size(ρ, 2) + 1, HKS.ncol)
    for c in 1:HKS.ncol
        # column c: its centre tc and its time window [t1, t2]
        tc = Tinv(ax, (c - 0.5) * w); t1 = Tinv(ax, (c - 1) * w); t2 = Tinv(ax, c * w)
        j = clamp(findlast(x -> x <= tc, seg.t_start), 1, length(seg.phia))  # run at the centre
        lo = max(t1, seg.t_start[j]); hi = min(t2, seg.t_end[j])  # window within that run
        # a recorded time drawn at random in the window
        τ = lo + rand(rng) * (hi - lo); i = clamp(searchsortedfirst(t, τ), 1, length(t))
        Z[1:end-1, c] .= ρ[i, :]; Z[end, c] = ρ[i, 1]             # periodic: closed at x = Lx
    end
    return Z
end

# The Δρ diagram at φ with both chains (no weakly nonlinear curves), as in the hysteresis figure
function chain_panel(ϕ, tb, ps, down, up, tag)
    S = HYST_PANELS[ϕ]
    chains = [(accepted_points(down), "", :purple, :square, -1), (accepted_points(up), "", :blue, :utriangle, +1)]
    pan = bifurcation_panel(ϕ, tb, ps, nothing, 1; xlims=S.xlims, ylims=S.ylims, wnl=false, chains=chains,
                            arrow_offset=S.arrow_offset, title=tag)
    for (s, x, y, align) in S.labels
        annotate!(pan.p, x, y, txt(align == :right ? s * " " : s, FS, :black, align))
    end
    # transition numbers from the chains, numbered as in the φa(t) panels (increasing chain first): beside the middle
    # of the jump's arrow, right of the increasing chain's arrows and left of the decreasing chain's
    xr = pan.xl[2] - pan.xl[1]; yr = pan.yl[2] - pan.yl[1]; i = 0
    for (r, sgn) in ((accepted_points(up), +1), (accepted_points(down), -1))
        off = S.arrow_offset * yr * sgn  # the arrows' shift (as in bifurcation_panel)
        for j in 2:length(r.phia)
            (r.type[j], r.n[j]) == (r.type[j-1], r.n[j-1]) && continue
            i += 1
            xm = (r.phia[j-1] + r.phia[j]) / 2; ym = (r.drho[j-1] + r.drho[j]) / 2 + off
            if abs(r.drho[j] - r.drho[j-1]) / yr > abs(r.phia[j] - r.phia[j-1]) / xr   # steep arrow: number beside it
                annotate!(pan.p, xm + sgn * 0.012xr, ym, txt(string(i), FS, :black, sgn > 0 ? :left : :right))
            else                                                                        # flat arrow: above / below it
                annotate!(pan.p, xm, ym + sgn * 0.035yr, txt(string(i), FS, :black, :center))
            end
        end
    end
    return pan.p
end

"""
    hysteresis_figure(diagrams, recs)

diagrams[φ] = (tb, ps, down, up) for the Δρ panels, recs[φ] = (increasing, decreasing) recordings.
"""
function hysteresis_figure(diagrams, recs)
    trs = Dict(ϕ => hysteresis_transitions(recs[ϕ]) for ϕ in PHIS_H)  # transitions at each φ
    axes_ = Dict((ϕ, c) => TimeAxis(rec.segments) for ϕ in PHIS_H for (c, rec) in enumerate(recs[ϕ]))  # time axes
    Zs = Dict(key => kymo(recs[key[1]][key[2]], ax) for (key, ax) in axes_)  # kymograph images
    ρlim = (minimum(minimum, values(Zs)), maximum(maximum, values(Zs)))     # one colour range for all the kymographs
    tag(s, ϕ) = latexstring("\\mathrm{($s)}\\quad \\phi = $(ϕ)")
    kw = (guidefontsize=GFS, titlefontsize=GFS, tickfontsize=FS, legend=false, titlelocation=:left)
    rows = Any[]
    for (k, ϕ) in enumerate(PHIS_H)
        tr = trs[ϕ]
        # empty cell next to the panels (colour-bar column)
        slot() = plot(framestyle=:none, grid=false, left_margin=1Plots.mm, right_margin=14Plots.mm)
        cells = Any[plot!(chain_panel(ϕ, diagrams[ϕ]..., tag("ab"[k], ϕ)); left_margin=16Plots.mm, right_margin=2Plots.mm,
                          top_margin=3Plots.mm, bottom_margin=16Plots.mm), slot()]
        for (c, rec) in enumerate(recs[ϕ])                # c = 1 increasing, 2 decreasing
            seg = rec.segments; mine = [x for x in tr if x[1] == c]
            ax = axes_[(ϕ, c)]; Z = Zs[(ϕ, c)]
            tk = tticks(ax, seg)  # time ticks
            pk = heatmap(range(0, ax.Dmax, length=HKS.ncol + 1), range(0, LX, length=size(Z, 1) + 1), Z; c=:viridis, clims=ρlim,
                         colorbar=false, xlims=(0, ax.Dmax), ylims=(0, LX), ylabel=L"x", title=tag("cdef"[2(c - 1) + k], ϕ),
                         xticks=(tk[1], fill("", length(tk[1]))), yticks=([0, LX/2, LX], ["0", "5", "10"]), grid=false,
                         framestyle=:box, left_margin=16Plots.mm, right_margin=2Plots.mm, top_margin=3Plots.mm, bottom_margin=0Plots.mm, kw...)
            for j in 2:length(seg.phia)                   # changes of φa
                d = D(ax, seg.t_start[j]); plot!(pk, [d, d], [0, LX]; color=:white, lw=1.7, ls=:dash, alpha=0.8)
            end
            # colour-bar ticks every 0.05 (every 0.1 if more than 6)
            tv = collect(ceil(ρlim[1] * 20) / 20:0.05:ρlim[2]); length(tv) > 6 && (tv = tv[1:2:end])
            cb = heatmap([0, 1], range(ρlim..., length=257), reshape(range(ρlim..., length=256), :, 1); c=:viridis, clims=ρlim,
                         colorbar=false, xticks=false, ylims=ρlim, yticks=(tv, [@sprintf("%.2f", v) for v in tv]), ymirror=true,
                         ylabel=L"\rho", framestyle=:box, guidefontsize=GFS, tickfontsize=FS, legend=false, left_margin=1Plots.mm,
                         right_margin=14Plots.mm, top_margin=3Plots.mm, bottom_margin=0Plots.mm)
            # φa(t) in display coordinates, the state between transitions below the line and the transition numbers above it
            xs = reduce(vcat, [[D(ax, seg.t_start[j]), D(ax, seg.t_end[j])] for j in eachindex(seg.phia)])
            ys = reduce(vcat, [[ϕa, ϕa] for ϕa in seg.phia])
            yr = extrema(seg.phia); dy = yr[2] - yr[1]  # φa range of the chain
            p2 = plot(xs, ys; xlims=(0, ax.Dmax), ylims=(yr[1] - 0.55dy, yr[2] + 0.7dy), color=:black, lw=HKS.lw, xlabel=L"t",
                      ylabel=L"\phi_a", xticks=tk, framestyle=:box, yformatter=v -> @sprintf("%.3f", v), left_margin=16Plots.mm,
                      right_margin=2Plots.mm, top_margin=1Plots.mm, bottom_margin=(c == 1 ? 14 : 4)Plots.mm, kw...)
            mid(j) = (D(ax, seg.t_start[j]) + D(ax, seg.t_end[j])) / 2  # middle of run j on the axis
            cuts = vcat(0.0, [mid(x[2]) for x in mine], ax.Dmax)  # transitions (middle of their run) split the axis
            names = vcat([seg.type[x[2] - 1] for x in mine], seg.type[end])  # state in each stretch
            for (q, name) in enumerate(names)
                dm = (cuts[q] + cuts[q+1]) / 2
                # run at the middle of the stretch: the name goes below its φa
                j = clamp(searchsortedlast(seg.t_start, Tinv(ax, dm)), 1, length(seg.phia))
                annotate!(p2, dm, seg.phia[j] - HKS.type_dy * dy, txt(string(name), FS, :black, :center, :top))
            end
            for (i, x) in enumerate(tr)
                x[1] == c || continue
                # the number just above the middle of the run where the branch changes
                annotate!(p2, mid(x[2]), seg.phia[x[2]] + HKS.num_dy * dy, txt(string(i), FS, :black, :center, :bottom))
            end
            append!(cells, [pk, cb, p2, slot()])
        end
        push!(rows, cells)
    end
    # columns: φ = 0.905 | φ = 0.915, each a panel and a narrow colour-bar slot; rows as in HKS.heights
    wcol = [(1 - HKS.cbar_width) / 2, HKS.cbar_width / 2, (1 - HKS.cbar_width) / 2, HKS.cbar_width / 2]
    # subplots in grid order: row r of both φ, each cell followed by its colour-bar cell
    order = [rows[k][2r - 1 + s] for r in 1:5 for k in 1:2 for s in 0:1]
    lay = grid(5, 4; heights=HKS.heights, widths=wcol)
    body = plot(order...; layout=lay)
    return with_legend(body, CHAIN_LEGEND; size=(HKS.size[1], round(Int, HKS.size[2] * (1 - HKS.legend_height))),
                       legend_height=HKS.legend_height, columns=3, fontsize=FS, margin=nothing)
end

# ==================================================================================================================
# Videos: ρ(x) along a whole chain, with φa(t) below (grey: the whole chain; black: up to now; red: now) and the
# transition numbers; in real detail (every record) for t_detail after each change of φa, fast forward elsewhere (at
# most n_skip frames per stretch, n_skip_jump in a run where the branch changes)
# ==================================================================================================================

const VIDEO_NUMBERS = Dict((0.905, "up") => 2, (0.905, "down") => 3, (0.915, "up") => 4, (0.915, "down") => 5)
const HV = (t_detail=600.0, n_skip=40, n_skip_jump=200, fps=25, size=(1100, 820))

video_name(ϕ, direction) = @sprintf("%02d Hysteresis Phi %s - Phi_a %s.mp4", VIDEO_NUMBERS[(ϕ, direction)], ϕ,
                                     direction == "up" ? "Increasing" : "Decreasing")

function hysteresis_video(ϕ, direction, recs)
    c = direction == "up" ? 1 : 2  # recs = (increasing, decreasing)
    rec = recs[c]; t = rec.t; ρ = rec.rho; seg = rec.segments; T = t[end]
    trans = [(i, x) for (i, x) in enumerate(hysteresis_transitions(recs)) if x[1] == c]   # global numbers
    jumps = Set(x[2] for (_, x) in trans)  # runs where the branch changes
    detail(τ) = any(s -> s <= τ <= s + HV.t_detail, seg.t_start)  # t_detail after each change of φa
    # frame indices: every record in detail, else at most n_skip frames up to the next detailed stretch
    frames = Int[]; i = 1
    while i <= length(t)
        if detail(t[i])
            push!(frames, i); i += 1
        else
            j = something(findnext(k -> detail(t[k]), eachindex(t), i), length(t) + 1)
            nf = clamp(searchsortedlast(seg.t_start, t[i]), 1, length(seg.phia)) in jumps ? HV.n_skip_jump : HV.n_skip
            append!(frames, unique(round.(Int, range(i, j - 1; length=min(nf, j - i)))))
            i = j
        end
    end
    x = range(0, rec.Lx; length=size(ρ, 2) + 1)  # closed at x = Lx
    yr = extrema(ρ); yr = (yr[1] - 0.08(yr[2] - yr[1]), yr[2] + 0.08(yr[2] - yr[1]))
    xs = reduce(vcat, [[a, b] for (a, b) in zip(seg.t_start, seg.t_end)]); ys = reduce(vcat, [[ϕa, ϕa] for ϕa in seg.phia])
    ar = extrema(ys); da = ar[2] - ar[1]
    segof(τ) = clamp(searchsortedlast(seg.t_start, τ), 1, length(seg.phia))  # run index at time τ
    dirname_ = direction == "up" ? "increasing" : "decreasing"
    anim = @animate for k in frames
        τ = t[k]; ϕa = seg.phia[segof(τ)]
        p1 = plot(x, vcat(ρ[k, :], ρ[k, 1]); color=:black, lw=3, xlims=(0, rec.Lx), ylims=yr, xlabel=L"x", ylabel=L"\rho",
                  legend=false, framestyle=:box, grid=false,
                  title=latexstring(@sprintf("\\phi = %.3f,\\ \\ \\phi_a\\ \\mathrm{%s}:\\ \\ \\phi_a = %.5f,\\ \\ t = %.0f", ϕ, dirname_, ϕa, τ)) *
                        (detail(τ) ? "" : "   (fast forward)"))
        hline!(p1, [ϕ]; color=:gray60, ls=:dot, lw=1.5)
        p2 = plot(xs, ys; color=:gray80, lw=4, xlims=(0, T), ylims=(ar[1] - 0.5da, ar[2] + 0.7da), xlabel=L"t", ylabel=L"\phi_a",
                  legend=false, framestyle=:box, grid=false, xformatter=v -> string(round(Int, v)),
                  yticks=([ar[1], ar[2]], [@sprintf("%.4f", ar[1]), @sprintf("%.4f", ar[2])]))
        m = searchsortedlast(xs, τ)  # knots of φa(t) up to τ
        plot!(p2, vcat(xs[1:m], τ), vcat(ys[1:m], ϕa); color=:black, lw=4)
        scatter!(p2, [τ], [ϕa]; color=:red, markerstrokewidth=0, ms=9)
        for (n, (_, jr)) in trans  # the number just above the middle of the run where the branch changes
            annotate!(p2, (seg.t_start[jr] + seg.t_end[jr]) / 2, seg.phia[jr] + 0.12da, Plots.text(string(n), 12, :black, :center, :bottom))
        end
        plot(p1, p2; layout=grid(2, 1, heights=[0.72, 0.28]), size=HV.size, dpi=VIDEO_DPI, guidefontsize=17, tickfontsize=12,
             titlefontsize=15, fontfamily="sans-serif", left_margin=6Plots.mm, right_margin=8Plots.mm, bottom_margin=5Plots.mm,
             top_margin=3Plots.mm)
    end
    return save_video(anim, video_name(ϕ, direction); fps=HV.fps)
end
