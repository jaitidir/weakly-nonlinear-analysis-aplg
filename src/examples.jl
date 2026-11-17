# Examples notebook: the fields of a PS (φ = 0.81, Pe = 4.5), a CP and a T (φ = 0.88, Pe = 6) and of the modulated PS
# (MPS, φ = 0.88998 near the Bogdanov–Takens point), their profiles and kymographs, and the videos.
include(joinpath(@__DIR__, "common.jl"))
load_src("weakly_nonlinear.jl"); load_src("timestepping.jl"); load_src("bt_amplitude.jl")

# ==================================================================================================================
# Data
# ==================================================================================================================

"""
    solution_fields(ϕ, ϕa, Pe, ic; warmup, Trec=600, rint=0.5)

The time-stepping run of the bifurcation diagrams at (φ, φa, Pe) from initial condition `ic` (Nx = 256, dt = 0.01; the
oscillatory sweep's settings for "double"/"non-symmetric", the stationary diagram's for "rand"), stopped by the
classifier, then continued for 'warmup' time to check for possible transients and then recorded (all fields every `rint`) for `Trec` after 
fully confirming convergence. The period is the return time of ρ(x, t) without shift (NaN for a PS).
"""
function solution_fields(ϕ, ϕa, Pe, ic; warmup, Trec=600.0, rint=0.5, Lx=10.0, Nx=256, dt=0.01)
    # classifier run from the perturbed homogeneous state, stopped once converged
    run0 = if ic == "rand"
        run_spectral(ϕ, ϕa, Pe, Lx, Nx; T=10000.0, dt=dt, pert_type="rand", pert_amp=0.1, seed=1, save_interval=5.0,
                     record_interval=HISTORY)
    else
        qc, ϕac, wc, _ = find_qc_phiac_wc(ϕ, Pe, Lx)  # onset: critical wavenumber, φa and frequency
        # eigenvectors of the two travelling modes (for the "double"/"non-symmetric" perturbations)
        vp, vm = travelling_eigenvectors(ϕ, ϕac, qc, wc, Pe)
        run_spectral(ϕ, ϕa, Pe, Lx, Nx; T=50000.0, dt=dt, pert_type=ic, pert_amp=0.1, pert_qc=qc, pert_vp=vp, pert_vm=vm,
                     pert_Ap=2.0, pert_Am=1.0, seed=1, save_interval=20.0, record_interval=HISTORY)
    end
    @printf("time stepping stopped at t = %.0f: %s\n", run0.t, run0.sol_type)
    # continue from a state, without the classifier
    run_(state, T; kw...) = run_spectral(ϕ, ϕa, Pe, Lx, Nx; init_fields=state, T=T, dt=dt, classify=false, save_interval=100.0, kw...)
    w = run_((run0.rho, run0.rhoa, run0.m), warmup; record_interval=HISTORY)  # warm-up after the classifier stopped
    res = run_((w.rho, w.rhoa, w.m), Trec; record_interval=rint)  # the recorded run (all fields every rint)
    M(v) = permutedims(Float64.(reduce(hcat, v)))  # records -> matrix, one row per time
    rho, rhoa, m = M(res.rec.rho), M(res.rec.rhoa), M(res.rec.m)
    H32(a, b) = Float32.(permutedims(reduce(hcat, vcat(a, b[2:end]))))      # classifier run + warmup (t = 0 repeats)
    # history: classifier run and warm-up, times continuous (the warm-up's t = 0 dropped)
    hist = (hist_t=vcat(run0.rec.t, run0.t .+ w.rec.t[2:end]), hist_rho=H32(run0.rec.rho, w.rec.rho),
            hist_rhoa=H32(run0.rec.rhoa, w.rec.rhoa), hist_m=H32(run0.rec.m, w.rec.m))
    lags = round(Int, 10/rint):size(rho, 1) ÷ 2  # lags (in rows) from 10 time units to half the record
    # mean squared difference between ρ(t + L rint) and ρ(t), without shift
    err = [sum(abs2, rho[1+L:end, :] .- rho[1:end-L, :]) / (size(rho, 1) - L) for L in lags]
    sol_type = string(run0.sol_type)
    # t: absolute times (classifier run + warm-up + record); period: the lag of the smallest difference (a multiple
    # of rint)
    return (t=res.rec.t .+ (run0.t + warmup), rho=rho, rhoa=rhoa, m=m, period=(sol_type == "PS" ? NaN : lags[argmin(err)] * rint),
            ϕ=ϕ, ϕa=ϕa, Pe=Pe, Lx=Lx, dt=dt, sol_type=sol_type, t_classified=run0.t, warmup=warmup, hist...)
end

"""
    mps_fields(nper)

The modulated PS: run rand_s1.84 of the Bogdanov–Takens notebook (data/bogdanov_takens/mps_runs; from small noise on
the path μ¹ < 0) continued from its final state for nper periods with the same Nx and dt, recording all fields every
time unit; period from the modulated-PS test.
"""
function mps_fields(nper)
    f = loaddata("bogdanov_takens/mps_runs/rand_s1.84.jld2")  # the MPS run from noise (bogdanov_takens notebook)
    # period from the modulated-PS test over its last 40000 time units
    P = classify_modulated_ps(f.t, Float64.(f.rho); window=40000.0, Lx=10.0).period
    # continue from its final state for nper periods
    res = run_spectral(f.ϕ, f.ϕa, f.Pe, 10.0, f.Nx; init_fields=(copy(f.rho_end), copy(f.rhoa_end), copy(f.m_end)),
                       T=nper*P, dt=f.dt, classify=false, save_interval=100.0, record_interval=1.0)
    M(v) = permutedims(reduce(hcat, v))
    # times continue from the end of rand_s1.84
    return (t=res.rec.t .+ f.t[end], rho=M(res.rec.rho), rhoa=M(res.rec.rhoa), m=M(res.rec.m), period=P, ϕ=f.ϕ, ϕa=f.ϕa,
            Pe=f.Pe, K1t=f.K1t, ηt=f.ηt, s=f.s, μ1=f.μ1, δ=f.δ, dt=f.dt, scheme=f.scheme, Lx=10.0, sol_type="MPS")
end

# ==================================================================================================================
# Figures
# ==================================================================================================================

# phases: times of the four profiles of a CP or MPS (fractions of the period from the largest |ρ̂₁|; one profile, at the
# first, for a PS or T); nper: periods in the kymographs (a PS spans ps_window); profile_ylims: nothing = from the data;
# size, heights: in px at 100 dpi and row heights (profiles, weakly nonlinear profiles, kymographs; without the second
# row for a PS, T or MPS); gap, side: widths of the gap and of the right-hand column (legend, colour bars, triangle);
# wnl_pde_times: the weakly nonlinear profiles at the same times as the time stepping (else at the same fractions of
# its own period); comp_centre, comp_reach: the composition (ρa, ρ0) is zoomed about its mean, placed at comp_centre of the colour triangle,
# so that its largest deviation reaches comp_reach; arrow: direction of travel of a T (x ends and height as fractions of
# the panel, headroom left above the curves)
const EXAMPLE_SETTINGS = (phases=(0.0, 0.25, 0.5, 0.75), nper=2, ps_window=200.0, profile_ylims=nothing, wnl_pde_times=false,
                          size=(1700, 1260), heights=[3.2, 3.2, 5.6], size_single=(1700, 900), heights_single=[3.2, 5.4],
                          FS=18, GFS=24, legend_fs=24, tri_fs=18, lw=2.8, gap=0.09, side=0.17,
                          comp_centre=(0.45, 0.45), comp_reach=0.35, arrow=(x=(0.40, 0.60), y=0.9, headroom=0.3))
const MPS_SETTINGS = merge(EXAMPLE_SETTINGS, (nper=1, wnl_pde_times=true))

# Composition colour of (ρa, ρ0): hue from blue (ρ0 only) to red (ρa only) through green, mixed with white by 1 - ρ
function tern(a, b)
    r = a + b; r <= 1e-12 && return (1.0, 1.0, 1.0)  # r: ρa + ρ0 (white if zero)
    h = (2/3)*(1 - a/r)  # hue: 0 (red) for ρa only, 2/3 (blue) for ρ0 only
    k(n) = mod(n + 6h, 6); c = [1 - max(0, min(k(n), 4 - k(n), 1)) for n in (5, 3, 1)]       # hsv(h, 1, 1) -> rgb
    return Tuple((1 - min(r, 1)) .+ min(r, 1) .* c)  # mixed with white by 1 - r
end

# Weakly nonlinear solution at the same (φ, φa, Pe), translated in x (and in t for a CP) so that its first Fourier mode
# is in phase with that of the time stepping at t = 0 (row j0; for a CP and an MPS, the largest |ρ̂₁|): returns the field
# (ρ, ρa, m) at time τ and its period
function aligned_wnl(d, j0)
    # grid without the end point; q1 = 2π/Lx
    t = d.t; ρ = d.rho; nx = size(ρ, 2); xg = range(0, d.Lx; length=nx + 1)[1:nx]; q1 = 2π/d.Lx
    mode1(v) = sum(v .* exp.(-2π*im .* (0:nx-1) ./ nx))/nx  # first Fourier mode (coefficient of e^{iq1x})
    a1 = [mode1(ρ[i, :]) for i in axes(ρ, 1)]  # ρ̂₁ of each row
    c_T = -angle(a1[j0 + 1]/a1[j0])/(t[j0 + 1] - t[j0])/q1      # direction of travel: ρ(x - ct) has ρ̂₁ ∝ e^{-iqct}
    if d.sol_type == "MPS"                                     # the orbit of the amplitude equation at the BT point,
        w = bt_mps_wnl(bt_coefficients(d.ϕ; Lx=d.Lx), d.s, d.μ1, d.δ)    # from its largest amplitude
        w === nothing && error("no stable weakly nonlinear MPS at φa¹ = $(d.s), μ¹ = $(d.μ1)")
        wnl = w.field; Pw = w.period  # its field (x, t) and period in t
    else
        qc, ϕac, wc, itype = find_qc_phiac_wc(d.ϕ, d.Pe, d.Lx)  # onset of the homogeneous state at (φ, Pe)
        δ = sqrt(abs(d.ϕa - ϕac)); s1 = sign(d.ϕa - ϕac)  # φa = ϕac + δ² s1, s1 = ±1
        if itype == :stationary  # stationary: the weakly nonlinear PS at q1 = 2π/Lx (where the stationary onset is taken)
            wnl = (xx, tt) -> weaklynonlinear_stat(xx, d.Pe, q1, d.ϕ, ϕac, d.ϕa, δ, s1)
            Pw = NaN  # no period
        else
            # CP: both travelling modes; T: the mode that travels like the time stepping (c > 0: A⁻)
            mode = d.sol_type == "CP" ? :coexistence : (c_T > 0 ? :minus : :plus)
            st = wnl_osc_setup(d.Pe, qc, d.ϕ, ϕac, d.ϕa, δ, s1, wc, mode)
            wnl = (xx, tt) -> wnl_osc_eval(xx, tt, qc, d.ϕ, d.ϕa, δ, wc, st)
            # period of that mode with its O(δ²) frequency correction (CP: from the + mode)
            Pw = 2π/abs(mode == :minus ? wc - δ^2*st.θm : wc + δ^2*st.θp)
        end
    end
    wfield(tt, sh) = reduce(hcat, [wnl(xx - sh, tt) for xx in xg])  # field on the grid shifted by sh in x: 3 × nx (ρ, ρa, m)
    # time origin: for a CP the largest |ρ̂₁| over one period; otherwise 0 (the MPS orbit starts at its maximum)
    t0w = d.sol_type == "CP" ? (tg = range(0, Pw; length=401)[1:end-1]; tg[argmax([abs(mode1(wfield(tt, 0.0)[1, :])) for tt in tg])]) : 0.0
    # x shift that puts its ρ̂₁ in phase with that of the time stepping at row j0
    xsh = (angle(mode1(wfield(t0w, 0.0)[1, :])) - angle(a1[j0]))/q1
    return (τ -> wfield(t0w + τ, xsh)), Pw, c_T  # τ -> field τ after the aligned origin
end

"""
    example_figure(d; S=EXAMPLE_SETTINGS, wnl=true)

(a) ρ, ρa, ρ0 = ρ - ρa and m minus their spatial mean at four times over one period (CP, MPS) or at one time (PS, T,
with the direction of travel of a T), t = 0 at the largest |ρ̂₁|; (b) the same for the weakly nonlinear solution (if
`wnl`: beside (a) for a PS or T, a second row for a CP or MPS); then kymographs of ρ, m and of the composition colour over
nper periods, with their colour bars and the colour triangle (its axes give the true ρa, ρ0).
"""
function example_figure(d; S=EXAMPLE_SETTINGS, wnl=true)
    t = d.t; Lx = d.Lx; typ = d.sol_type
    ρ = Float64.(d.rho); ρa = Float64.(d.rhoa); m = Float64.(d.m); ρ0 = ρ .- ρa  # ρ0 = ρ - ρa: passive density
    nx = size(ρ, 2); x = range(0, Lx; length=nx + 1)  # grid with the end point (closed profiles)
    per(v) = vcat(v, v[1:1])  # periodic closure of a profile
    # PS: no period, the kymograph spans ps_window
    stationary = isnan(d.period); P = stationary ? S.ps_window/S.nper : d.period
    # |ρ̂₁| of each row
    a1 = [abs(sum(ρ[i, :] .* exp.(-2π*im .* (0:nx-1) ./ nx))/nx) for i in axes(ρ, 1)]
    # j0: largest |ρ̂₁| in the first period (t = 0 of the figure); j: rows over nper periods from j0
    j0 = stationary ? 1 : argmax(a1[t .< t[1] + P]); j = findall(s -> t[j0] <= s <= t[j0] + S.nper*P, t)
    phases = typ in ("CP", "MPS") ? S.phases : S.phases[1:1]  # four phases for a CP or MPS, one for a PS or T
    idx = [j[argmin(abs.(t[j] .- (t[j0] + ph*P)))] for ph in phases]  # rows nearest to the phases
    sub(v) = v .- sum(v)/length(v)  # minus the spatial mean
    # time-stepping profiles (ρ, ρa, ρ0, m) at those rows
    curves = [[(sub(ρ[i, :]), sub(ρa[i, :]), sub(ρ0[i, :]), sub(m[i, :])) for i in idx]]
    times = [[t[i] - t[j0] for i in idx]]  # their times from t[j0]
    c_T = 0.0
    if wnl
        wf, Pw, c_T = aligned_wnl(d, j0)
        # weakly nonlinear times: those of the time stepping, or the same phases of its own period
        tw = S.wnl_pde_times ? times[1] : [ph * (isnan(Pw) ? 0.0 : Pw) for ph in phases]
        Fs = [wf(τ) for τ in tw]
        # weakly nonlinear profiles (ρ, ρa, ρ0 = ρ - ρa, m)
        push!(curves, [(sub(F[1, :]), sub(F[2, :]), sub(F[1, :] .- F[2, :]), sub(F[3, :])) for F in Fs])
        push!(times, tw)
    end
    lo, hi = extrema(reduce(vcat, [vcat(c...) for cs in curves for c in cs]))  # common y range of all the profiles
    typ == "T" && (hi += S.arrow.headroom*(hi - lo))  # room for the arrow of a T
    ylims = S.profile_ylims === nothing ? (lo - 0.08(hi - lo), hi + 0.08(hi - lo)) : S.profile_ylims
    # line styles of ρ, ρa, ρ0 and m
    styles = ((color=:black, ls=:solid, lw=S.lw), (color=RGB(0xe8/255, 0x31/255, 0x2a/255), ls=:dash, lw=S.lw),
              (color=RGB(0x1f/255, 0x1f/255, 0xd6/255), ls=:dot, lw=S.lw + 0.6), (color=RGB(0x2e/255, 0x7d/255, 0x1f/255), ls=:dashdot, lw=S.lw))
    single = length(phases) == 1
    W = (single ? S.size_single : S.size)[1]
    pw = W * (1 / (1 + S.gap + S.side)) / (single ? 2 : length(phases))     # width of a profile panel in px
    function profile(c, τ, k, tag)
        p = plot(; xlims=(0, Lx), ylims=ylims, xlabel=L"x", ylabel=latexstring(@sprintf("t = %.0f", τ)), legend=false,
                 guidefontsize=S.GFS, tickfontsize=S.FS, title=k == 1 ? tag : "", titlelocation=:left, titlefontsize=S.GFS,
                 yformatter=k == 1 || single ? :auto : (_ -> ""), xticks=[0, 5, 10], framestyle=:axes, left_margin=(k == 1 || single ? 14 : 10)Plots.mm,
                 right_margin=2Plots.mm, bottom_margin=12Plots.mm, top_margin=4Plots.mm)
        for (v, s) in zip(c, styles); plot!(p, x, per(v); s...); end
        if typ == "T"                                       # direction of travel
            # arrow along the direction of travel, at height arrow.y
            xa = c_T > 0 ? S.arrow.x : reverse(S.arrow.x); ya = ylims[1] + S.arrow.y*(ylims[2] - ylims[1])
            arrow!(p, Lx*xa[1], ya, Lx*xa[2], ya, (0, Lx), ylims, (pw, 280); lw=2.8, hl=20, hw=8)
        end
        return p
    end
    # empty cell, transparent so that it does not cover the labels of its neighbours (the colour bars' ρ and m)
    blank() = plot(framestyle=:none, grid=false, legend=false, ticks=false, background_color_subplot=RGBA(1, 1, 1, 0))
    # legend panel of the profiles
    legend_ = plot(framestyle=:none, grid=false, ticks=false, legend=false, xlims=(0, 1), ylims=(0, 1))
    for (k, (lab, s)) in enumerate(zip((L"\overline{\rho}", L"\overline{\rho_a}", L"\overline{\rho_0}", L"\overline{m}"), styles))
        y = 0.83 - 0.22(k - 1)
        plot!(legend_, [0.0, 0.4], [y, y]; s...)
        annotate!(legend_, 0.5, y, txt(lab, S.legend_fs, :black, :left))
    end
    # kymographs over nper periods (a PS: ps_window)
    tt = t[j] .- t[j0]; yc = ((1:nx+1) .- 0.5) .* (Lx / (nx + 1))         # rows of the closed profiles spread over [0, Lx]
    close(F) = permutedims(hcat(F, F[:, 1]))  # rows of F with the first column repeated, transposed (x along the vertical)
    ρlim = extrema(ρ[j, :]); mlim = maximum(abs.(m[j, :]))  # colour limits from the shown rows
    kymo_kw = (xlims=(tt[1], tt[end]), ylims=(0, Lx), ylabel=L"x", yticks=[0, Lx/2, Lx], framestyle=:box, grid=false, legend=false,
               guidefontsize=S.GFS, tickfontsize=S.FS, colorbar=false, left_margin=14Plots.mm, right_margin=2Plots.mm)
    # kymograph of ρ
    k1 = heatmap(tt, yc, close(ρ[j, :]); c=:viridis, clims=ρlim, xformatter=_ -> "", title=wnl && !single ? L"\mathrm{(c)}" :
                 (wnl ? L"\mathrm{(c)}" : L"\mathrm{(b)}"), titlelocation=:left, titlefontsize=S.GFS, top_margin=6Plots.mm,
                 bottom_margin=0Plots.mm, kymo_kw...)
    # kymograph of m (symmetric colour limits)
    k2 = heatmap(tt, yc, close(m[j, :]); c=:PRGn, clims=(-mlim, mlim), xformatter=_ -> "", top_margin=1Plots.mm,
                 bottom_margin=0Plots.mm, kymo_kw...)
    # means of ρa and ρ0 over the shown rows
    ā = sum(ρa[j, :])/length(ρa[j, :]); b̄ = sum(ρ0[j, :])/length(ρ0[j, :])
    # gain: the largest deviation from the means is drawn at comp_reach from comp_centre
    g = S.comp_reach/maximum(max.(abs.(ρa[j, :] .- ā), abs.(ρ0[j, :] .- b̄)))
    cA, cB = S.comp_centre
    # zoomed composition, scaled back into the triangle a + b ≤ 1 if outside it
    function zoom(a, b)
        za = max(cA + g*(a - ā), 0.0); zb = max(cB + g*(b - b̄), 0.0); s = za + zb
        return s > 1 ? (za/s, zb/s) : (za, zb)
    end
    # composition colour at each (x, t) (x = Lx repeats x = 0)
    img = [RGB(tern(zoom(ρa[i, ix == nx + 1 ? 1 : ix], ρ0[i, ix == nx + 1 ? 1 : ix])...)...) for ix in 1:nx+1, i in j]
    # kymograph of the composition
    k3 = plot(tt, range(0, Lx, length=nx + 1), img; xlabel=L"t", top_margin=1Plots.mm, bottom_margin=12Plots.mm, kymo_kw...)
    plot!(k3; yflip=false, aspect_ratio=:none, xlims=(tt[1], tt[end]), ylims=(0, Lx))
    # colour bars (narrow, at the left of the right-hand column) and the colour triangle
    function bar(lims, cmap, lab)
        tv = Plots.optimize_ticks(lims...; k_min=2, k_max=3)[1]
        heatmap([0, 1], range(lims..., length=257), reshape(range(lims..., length=256), :, 1); c=cmap, clims=lims, colorbar=false,
                xlims=(0, 1), ylims=lims, xticks=false, yticks=(tv, [@sprintf("%.3g", v) for v in tv]), ymirror=true, ylabel=lab,
                framestyle=:box, legend=false, guidefontsize=S.GFS, tickfontsize=S.FS, left_margin=1Plots.mm, right_margin=1Plots.mm,
                top_margin=1Plots.mm, bottom_margin=0Plots.mm)
    end
    n = 200
    # colour triangle: (ρa, ρ0) with ρa + ρ0 ≤ 1, white outside
    tri = [a + b <= 1 ? RGB(tern(a, b)...) : RGB(1, 1, 1) for b in ((1:n) .- 0.5) ./ n, a in ((1:n) .- 0.5) ./ n]
    # ticks: the true ρa or ρ0 at both ends of the zoomed range
    tk(c, mean) = [(c + g*(v - mean), v) for v in clamp.(mean .+ [-S.comp_reach, S.comp_reach]./g, 0, 1)]
    ta = tk(cA, ā); tb = tk(cB, b̄)
    # colour triangle (an inset, placed below): only its two axes, as a box would outline the empty upper triangle
    tri_kw = (xlims=(0, 1), ylims=(0, 1), yflip=false, aspect_ratio=:none,
              xticks=(first.(ta), [@sprintf("%.2f", v) for v in last.(ta)]), yticks=(first.(tb), [@sprintf("%.2f", v) for v in last.(tb)]),
              xrotation=45, xlabel=L"\rho_a", ylabel=L"\rho_0", guidefontsize=S.GFS - 2, tickfontsize=S.tri_fs, framestyle=:axes,
              grid=false, legend=false, margin=0Plots.mm)
    # layout: rows of profiles ([panels | gap | legend]) and the kymograph block ([kymograph | gap | side]) in three rows
    w1, wg, ws = [1, S.gap, S.side] ./ (1 + S.gap + S.side)
    cells = Any[]; rows = Any[]
    tags = (L"\mathrm{(a)}", L"\mathrm{(b)}")
    if single && wnl  # PS, T: (a) and (b) side by side
        push!(rows, grid(1, 4; widths=[w1/2, w1/2, wg, ws]))
        append!(cells, [profile(curves[1][1], times[1][1], 1, tags[1]), profile(curves[2][1], times[2][1], 1, tags[2]), blank(), legend_])
    else
        # CP, MPS, or no weakly nonlinear row: one row of profiles per solution
        for (r, cs) in enumerate(curves)
            push!(rows, grid(1, length(cs) + 2; widths=[fill(w1/length(cs), length(cs)); wg; ws]))
            append!(cells, [[profile(c, τ, k, tags[r]) for (k, (c, τ)) in enumerate(zip(cs, times[r]))]; blank(); r == 1 ? legend_ : blank()])
        end
    end
    kl = grid(3, 4; widths=[w1, wg, 0.015, ws - 0.015])   # kymographs | gap | colour bars | rest
    hs = single || !wnl ? S.heights_single : S.heights
    lay = grid(length(rows) + 1, 1; heights=hs ./ sum(hs))
    for (r, l) in enumerate(rows); lay[r, 1] = l; end
    lay[length(rows) + 1, 1] = kl
    n0 = length(cells)
    append!(cells, [k1, blank(), bar(ρlim, :viridis, L"\rho"), blank(), k2, blank(), bar((-mlim, mlim), :PRGn, L"m"), blank(),
                    k3, blank(), blank(), blank()])
    fig = plot(cells...; layout=lay, size=(single || !wnl ? S.size_single : S.size))
    # The colour triangle: an inset of the composition kymograph (subplot n0+9) whose square plot area spans the height
    # of that kymograph, with its ρ0 axis at the left edge of the colour bars (subplot n0+3 is the colour bar of ρ).
    # An inset's box includes its labels, so it is placed once, measured, and placed again with that padding removed.
    Plots.prepare_output(fig)
    pk = Plots.plotarea(fig[n0 + 9]); pb = Plots.plotarea(fig[n0 + 3])
    L(b) = Plots.left(b).value; T(b) = Plots.top(b).value; W(b) = Plots.width(b).value; H(b) = Plots.height(b).value
    target = (L(pb), T(pk), H(pk), H(pk))                     # left, top, width, height of its plot area (mm)
    function with_triangle(l, t, w, h)                        # the figure with the inset's box at (l, t, w, h) in mm
        f = deepcopy(fig)
        plot!(f, ((1:n) .- 0.5) ./ n, ((1:n) .- 0.5) ./ n, tri; tri_kw..., subplot=length(f.subplots) + 1,
              inset=(n0 + 9, bbox((l - L(pk))/W(pk), (t - T(pk))/H(pk), w/W(pk), h/H(pk))))
        Plots.prepare_output(f)
        return f, Plots.plotarea(f[length(f.subplots)])
    end
    _, pt = with_triangle(target...)
    pl = L(pt) - target[1]; ptop = T(pt) - target[2]                         # padding left and above the plot area
    pr = target[1] + target[3] - L(pt) - W(pt); pbot = target[2] + target[4] - T(pt) - H(pt)   # right and below
    f, _ = with_triangle(target[1] - pl, target[2] - ptop, target[3] + pl + pr, target[4] + ptop + pbot)
    return f
end

# ==================================================================================================================
# Videos
# ==================================================================================================================

# ρ(x) of the CP (above) and the T (below) against the weakly nonlinear solution aligned at t = 0, over nper periods in
# real time (one frame per dt_frame); thin grey: largest and smallest ρ at each x; dotted: ρ = φ
function cp_t_video(dCP, dT; nper=4, dt_frame=1.0, fps=25, size=(1100, 1000), name="01 CP and T vs Weakly Nonlinear Solution.mp4")
    function prep(d)
        # (size is a keyword here: size_ is Base.size)
        t = d.t; P = d.period; ρ = d.rho; nx = size_(ρ, 2)
        # |ρ̂₁| of each row
        a1 = [abs(sum(ρ[i, :] .* exp.(-2π*im .* (0:nx-1) ./ nx))/nx) for i in axes(ρ, 1)]
        j0 = argmax(a1[t .< t[1] + P])  # t = 0 at the largest |ρ̂₁| in the first period
        wf, Pw, _ = aligned_wnl(d, j0)
        j = findall(s -> t[j0] <= s <= t[j0] + nper*P + 1e-9, t)  # rows over nper periods
        # hi, lo: largest and smallest ρ at each x over those rows
        return (d=d, x=range(0, d.Lx; length=nx + 1), j0=j0, j=j, wnl=τ -> wf(τ)[1, :],
                hi=vec(maximum(ρ[j, :]; dims=1)), lo=vec(minimum(ρ[j, :]; dims=1)))
    end
    size_(a, k) = Base.size(a, k)
    S = [prep(dCP), prep(dT)]
    per(v) = vcat(v, v[1:1])
    τs = 0:dt_frame:minimum(s.d.t[s.j[end]] - s.d.t[s.j0] for s in S)  # frame times: up to the shorter of the two records
    W = [[s.wnl(τ) for τ in τs] for s in S]  # weakly nonlinear ρ at each frame
    # common y range
    lo, hi = extrema(vcat([vcat(vec(s.d.rho[s.j, :]), reduce(vcat, w)) for (s, w) in zip(S, W)]...))
    yr = (lo - 0.1(hi - lo), hi + 0.25(hi - lo))
    anim = @animate for (k, τ) in enumerate(τs)
        ps = map(enumerate(S)) do (n, s)
            i = s.j0 + round(Int, τ/(s.d.t[2] - s.d.t[1]))  # row τ after j0
            p = hline([s.d.ϕ]; color=:gray60, ls=:dot, lw=1.5, label="")
            plot!(p, s.x, per(s.hi); color=:gray70, lw=1.5, label="")
            plot!(p, s.x, per(s.lo); color=:gray70, lw=1.5, label="")
            plot!(p, s.x, per(s.d.rho[i, :]); color=:black, lw=3, label=n == 1 ? "time stepping      " : "")
            plot!(p, s.x, per(W[n][k]); color=RGB(0xe8/255, 0x31/255, 0x2a/255), ls=:dash, lw=3, label=n == 1 ? "weakly nonlinear" : "",
                  xlims=(0, s.d.Lx), ylims=yr, xlabel=L"x", ylabel=L"\rho", framestyle=:box, grid=false,
                  legend=n == 1 ? :top : false, legend_column=2, legendfontsize=14, foreground_color_legend=nothing,
                  title=latexstring(@sprintf("\\mathrm{%s}:\\quad \\phi = %.2f,\\quad \\phi_a = %.4f,\\quad \\mathrm{Pe} = %.0f,\\quad t = %.0f",
                                             s.d.sol_type, s.d.ϕ, s.d.ϕa, s.d.Pe, τ)))
            p
        end
        plot(ps...; layout=grid(2, 1), size=size, dpi=VIDEO_DPI, fontfamily="sans-serif", titlefontsize=17, guidefontsize=18,
             tickfontsize=12, left_margin=6Plots.mm, right_margin=4Plots.mm, bottom_margin=5Plots.mm, top_margin=3Plots.mm)
    end
    return save_video(anim, name; fps=fps)
end

# The three fields of the MPS over its last nper periods, one frame every dt_frame; thin grey: the profiles at the largest
# and smallest |ρ̂₁| of the cycle; dotted: the means
function mps_video(d; nper=2, dt_frame=20.0, name="06 Modulated Pattern (MPS).mp4")
    t = d.t; P = d.period
    F = (Float64.(d.rho), Float64.(d.rhoa), Float64.(d.m))
    shown = findall(>=(t[end] - nper*P - 1e-9), t)  # rows of the last nper periods
    frames = shown[1:max(1, round(Int, dt_frame/(t[2] - t[1]))):end]  # rows of the frames
    a1 = mode1_amplitude(F[1])  # |ρ̂₁| of each row
    jmax, jmin = shown[argmax(a1[shown])], shown[argmin(a1[shown])]  # rows of the largest and smallest |ρ̂₁| (grey profiles)
    x = range(0, d.Lx; length=size(F[1], 2) + 1)
    per(v) = vcat(v, v[1:1])
    means = (d.ϕ, d.ϕa, 0.0); labels = (L"\rho", L"\rho_a", L"m")  # means (dotted) and axis labels of ρ, ρa, m
    # y range of each field over the shown rows
    ylims_ = [let (lo, hi) = extrema(f[shown, :]); (lo - 0.1(hi - lo), hi + 0.1(hi - lo)) end for f in F]
    ttl = latexstring(@sprintf("\\mathrm{MPS}:\\quad \\phi = %.5f,\\quad \\phi_a = %.6f,\\quad \\mathrm{Pe} = %.5f", d.ϕ, d.ϕa, d.Pe))
    anim = @animate for j in frames
        ps = map(1:3) do k
            f = F[k]
            p = hline([means[k]]; color=:gray60, ls=:dot, lw=1.5, label="")
            plot!(p, x, per(f[jmax, :]); color=:gray70, lw=1.5, label="")
            plot!(p, x, per(f[jmin, :]); color=:gray70, lw=1.5, label="")
            plot!(p, x, per(f[j, :]); color=:black, lw=3, ylims=ylims_[k], xlims=(0, d.Lx), ylabel=labels[k], label="",
                  xlabel=k == 3 ? L"x" : "", xformatter=k == 3 ? :auto : _ -> "", framestyle=:box, grid=false, title=k == 1 ? ttl : "")
        end
        plot(ps...; layout=grid(3, 1), size=(1000, 1100), dpi=VIDEO_DPI, fontfamily="sans-serif", titlefontsize=17,
             guidefontsize=18, tickfontsize=12, left_margin=8Plots.mm, right_margin=4Plots.mm, bottom_margin=3Plots.mm, top_margin=3Plots.mm)
    end
    return save_video(anim, name; fps=25)
end
