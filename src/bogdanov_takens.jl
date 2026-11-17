# Bogdanov–Takens notebook (Lx = 10): coefficients of the amplitude equation at the BT point for all φ > 1/2, its
# weakly nonlinear bifurcation diagram at φ = 0.88998, and the homoclinic orbit where the modulated PS (MPS) glues into
# a standing wave (CP) in the PDE.
include(joinpath(@__DIR__, "common.jl"))
load_src("timestepping.jl"); load_src("bt_amplitude.jl")

const LX_BT = 10.0
const BT_PHI = 0.88998        # φ of the MPS: middle of the interval where K2 + K3 > 0 and Γ < 0
const BT_DELTA = 0.01         # δ of the MPS runs: φa = φa* + δ²φa¹, Pe = Pe* + δ²μ¹
const BT_SRANGE = (-6.0, 6.0) # range of the path parameter s = φa¹
const MPS_NX = 100            # spatial discretisation
const MPS_DT = 0.01           # time step
const FS_BT = 18; const GFS_BT = 24

# (φa¹, μ¹) at K1 = 4.9, η¹ = 0.35; it fixes |μ¹| of the two paths: φa¹ = s at μ¹ = ±μ¹
mps_point(c) = Tuple([c.k1_ϕa c.k1_μ; c.Π c.Ω] \ [4.9, 0.35])
# the two paths: φa¹ = s runs, μ¹ fixed at ± the μ¹ of mps_point
bt_paths(c) = (μ1 = mps_point(c)[2]; ["μ¹ > 0" => (ϕa0=0.0, vϕ=1.0, μ0=μ1, vμ=0.0), "μ¹ < 0" => (ϕa0=0.0, vϕ=1.0, μ0=-μ1, vμ=0.0)])

# ==================================================================================================================
# Data
# ==================================================================================================================

# Γ (A), K2 - K3 (D) and K2 + K3 (M) at the BT point for φ in (1/2, 1), and their sign changes (by bisection)
function coefficients_vs_phi()
    ϕs = collect(range(0.5005, 0.9995; length=4000))  # grid in φ over (1/2, 1)
    keep = [bt_point(ϕ, 2π/LX_BT)[2] <= ϕ for ϕ in ϕs]          # leave out φa* > φ 
    ϕs = ϕs[keep]
    cs = [bt_coefficients(ϕ; Lx=LX_BT) for ϕ in ϕs]  # coefficients at the BT point of each φ
    # A = Γ, D = K2 - K3, M = K2 + K3 on the grid
    vals = Dict("A" => [c.Γ for c in cs], "D" => [c.K2 - c.K3 for c in cs], "M" => [c.K2 + c.K3 for c in cs])
    # coefficient `name` at any φ (for the bisection)
    coef(ϕ, name) = (c = bt_coefficients(ϕ; Lx=LX_BT); name == "A" ? c.Γ : name == "D" ? c.K2 - c.K3 : c.K2 + c.K3)
    # zeros: bisection between neighbouring grid points where the sign changes
    zeros_ = Dict(name => [find_zero(ϕ -> coef(ϕ, name), (ϕs[j], ϕs[j+1]), Bisection()) for j in 1:length(ϕs)-1
                           if sign(v[j]) != sign(v[j+1])] for (name, v) in vals)
    # phia_star, Pe_star: the BT point (φa*, Pe*) of each φ
    return (phi=ϕs, A=vals["A"], D=vals["D"], M=vals["M"], zeros=zeros_, phia_star=[c.ϕas for c in cs], Pe_star=[c.Pe for c in cs])
end

# Weakly nonlinear bifurcation diagram at BT_PHI along both paths: the distinguished values of κ = K1|Γ|/((K2+K3)|η¹|),
# the bifurcation points, and the branches H, PS, TW and CP, MPS on a grid refined near the events
function wnl_diagram()
    c = bt_coefficients(BT_PHI; Lx=LX_BT); δ = BT_DELTA; smin, smax = BT_SRANGE
    kc = kappa_constants(c)  # κ of the rays: Hopf of PS (-1), homoclinic (-4/5), fold of CP, drift of PS
    paths = Dict{String,Any}()
    for (name, P) in bt_paths(c)
        ev = path_events(c, P, smin, smax, δ, δ, kc)  # bifurcation points of the path in [smin, smax]
        # s grid: uniform, ±1e-9 about each event, and finer over 1.45 ≤ |s| ≤ 1.95 (near the events of both paths)
        ss = sort(unique(vcat(collect(range(smin, smax; length=6001)), [e.s + d for e in ev for d in (-1e-9, 0.0, 1e-9)],
                              collect(range(1.45, 1.95; length=2001)), collect(range(-1.95, -1.45; length=2001)))))
        st = steady_branches(c, P, filter(s -> smin <= s <= smax, ss), δ)  # H, PS and TW at each s
        # L = 0 periodic orbits traced in m: SW (CP) for η¹ < 0 and η¹ > 0, MPS for η¹ > 0 (kind stored as a string)
        L0 = Dict("$(fam)_$(σ)" => columns([merge(o, (kind=string(o.kind),)) for o in lzero_branch(c, P, fam, σ, smin, smax; n=6000)])
                  for (fam, σ) in ((:SW, -1), (:SW, 1), (:MPS, 1)))
        # events and branches as Dicts of columns (string keys, for saving)
        paths[name] = Dict("events" => [Dict(string(k) => v for (k, v) in pairs(e)) for e in ev], "H" => columns(st.H),
                           "PS" => columns(st.PS), "TW" => columns(st.TW), "L0" => L0)
    end
    return (kappa=Dict(string(k) => v for (k, v) in pairs(kc)), paths=paths)
end

# PDE run at BT_PHI (Pe = Pe* + δ²μ¹, φa = φa* + δ²φa¹): the MPS straight from small noise, on the path μ¹ < 0 at
# φa¹ = s, where the weakly nonlinear MPS is stable and so is the PDE one. 
const MPS_RAND = (s=1.84, T=600000.0, rint=10.0, pert_amp=1e-4, seed=1)

function mps_rand_run(r=MPS_RAND)
    δ = BT_DELTA; c = bt_coefficients(BT_PHI); μ1 = -mps_point(c)[2]  # μ¹ < 0 path
    ϕa = c.ϕas + δ^2*r.s; Pe = c.Pe + δ^2*μ1  # φa = φa* + δ²s, Pe = Pe* + δ²μ¹
    res = run_spectral(BT_PHI, ϕa, Pe, LX_BT, MPS_NX; pert_type="rand", pert_amp=r.pert_amp, seed=r.seed, T=r.T, dt=MPS_DT,
                       classify=false, save_interval=100.0, record_interval=r.rint)
    M(v) = Float32.(permutedims(reduce(hcat, v)))  # records (one vector per time) -> Float32 matrix, one row per time
    # K1t, ηt: K1 and η¹ at (s, μ¹); rho_end, rhoa_end, m_end: the final state (Float64), continued by other runs
    return (t=res.rec.t, rho=M(res.rec.rho), rhoa=M(res.rec.rhoa), m=M(res.rec.m), K1t=c.k1_ϕa*r.s + c.k1_μ*μ1,
            ηt=c.Π*r.s + c.Ω*μ1, s=r.s, μ1=μ1, δ=δ, ϕ=BT_PHI, ϕa=ϕa, Pe=Pe, Nx=MPS_NX, dt=MPS_DT, scheme="sbdf2",
            pert_amp=r.pert_amp, rho_end=res.rho, rhoa_end=res.rhoa, m_end=res.m)
end

# Homoclinic orbit on the path μ¹ < 0, where the MPS changes into a CP as φa¹ grows past s_ref from the MPS rand_s1.84, 
# φa¹ is raised slowly to s_app and s_near; at s_glue (just above the change) the run is recorded until the signed 
# amplitude changes sign, and from s_app, φa¹ = s_ref + offsets are held T_conv each (for the video)
const HOM = (s0=1.84, s_app=1.8817, ds_app=0.001, T_app=5000.0, s_near=1.89095, ds_near=0.0005, T_near=10000.0,
             s_glue=1.8918, T_chunk=20000.0, T_max=600000.0, rint=10.0, s_ref=1.8917,
             offsets=[-0.010, -0.005, -0.0025, -0.001, -0.0003, 0.0003, 0.001, 0.003, 0.008], T_conv=50000.0)
# a:step:b stops at the last step not past b, which is b only if ds divides b - a
ramp(a, b, ds) = a:(b >= a ? abs(ds) : -abs(ds)):b                  # φa¹ from a towards b in steps of ds

function homoclinic_setup()
    c = bt_coefficients(BT_PHI)
    μ1 = -mps_point(c)[2]  # μ¹ < 0 path
    f = loaddata("bogdanov_takens/mps_runs/rand_s1.84.jld2")  # the MPS from noise (mps_rand_run)
    # first Fourier mode ρ̂₁ of ρ (coefficient of e^{iqx})
    mode1(ρ) = sum(ρ .* exp.(-2π*im .* (0:length(ρ)-1) ./ length(ρ)))/length(ρ)
    θ0 = angle(mode1(f.rho_end))                              # phase of the pattern (the MPS and the CP do not drift)
    X(ρ) = 2real(mode1(ρ)*exp(-im*θ0))                        # signed amplitude of the first mode
    function hold(state, s, T; rint=HOM.rint)                # run at φa¹ = s from `state` for T (ρa rescaled)
        rho, rhoa, m = copy.(state); ϕa = c.ϕas + BT_DELTA^2*s  # φa at φa¹ = s
        # continue with ρa rescaled to the new mean φa
        rhoa .*= ϕa/mean(rhoa)
        # same Pe (μ¹ fixed on the path)
        res = run_spectral(BT_PHI, ϕa, c.Pe + BT_DELTA^2*μ1, LX_BT, MPS_NX; init_fields=(rho, rhoa, m), T=T, dt=MPS_DT,
                           classify=false, save_interval=1000.0, record_interval=rint)
        return (res.rho, res.rhoa, res.m), res.rec
    end
    # all the records of a sequence of holds: times (continuous), φa¹, and the three fields (Float32)
    function hold_log!(L, state, s, T)
        state, rec = hold(state, s, T); n = length(rec.t); k0 = isempty(L.t) ? 1 : 2      # t = 0 repeats the previous end
        # times continue from the end of the previous hold
        append!(L.t, L.tnow[] .+ rec.t[k0:n]); append!(L.s, fill(s, n - k0 + 1))
        append!(L.rho, rec.rho[k0:n]); append!(L.rhoa, rec.rhoa[k0:n]); append!(L.m, rec.m[k0:n])
        L.tnow[] += T  # clock at the end of this hold
        return state
    end
    # empty log; tnow: time at the end of its last hold
    newlog() = (t=Float64[], s=Float64[], rho=Vector{Float32}[], rhoa=Vector{Float32}[], m=Vector{Float32}[], tnow=Ref(0.0))
    return (c=c, μ1=μ1, f=f, θ0=θ0, X=X, hold=hold, hold_log! = hold_log!, newlog=newlog)
end

M32(v) = Float32.(permutedims(reduce(hcat, v)))  # vector of records -> Float32 matrix, one row per time
# a log as saved: times, φa¹ and the fields as matrices
loggedfields(L) = (t=L.t, s=L.s, rho=M32(L.rho), rhoa=M32(L.rhoa), m=M32(L.m))

# From the final state of rand_s1.84: φa¹ raised by ds_app (T_app each) towards s_app, then by ds_near (T_near each)
# towards s_near; the whole log, the states at the end of both ramps and the time at the end of the first
function homoclinic_approach()
    H = homoclinic_setup()
    state = (copy(H.f.rho_end), copy(H.f.rhoa_end), copy(H.f.m_end))
    L = H.newlog()
    for s in ramp(HOM.s0, HOM.s_app, HOM.ds_app); state = H.hold_log!(L, state, s, HOM.T_app); end  # first ramp
    state_app = state; t_app = L.tnow[]  # state and time at the end of the first ramp
    # second, finer ramp
    for s in ramp(HOM.s_app, HOM.s_near, HOM.ds_near); state = H.hold_log!(L, state, s, HOM.T_near); end
    return merge(loggedfields(L), (state_app=state_app, state_near=state, t_app=t_app))
end

# Run at φa¹ = s_glue from the end of the approach, in chunks of T_chunk, until the signed amplitude has turned negative
# (the MPS glued into a CP) at least 5000 time units before the end; period: the last MPS cycle before the gluing
function homoclinic_cycle(approach)
    H = homoclinic_setup(); c = H.c
    # times and fields of all the chunks
    state = approach.state_near; tnow = 0.0; T_ = Float64[]; R = Vector{Float64}[]; RA = Vector{Float64}[]; MM = Vector{Float64}[]
    while tnow < HOM.T_max
        state, rec = H.hold(state, HOM.s_glue, HOM.T_chunk)
        n = length(rec.t)
        # each chunk without its t = 0 (repeats the previous end)
        append!(T_, tnow .+ rec.t[2:n]); append!(R, rec.rho[2:n]); append!(RA, rec.rhoa[2:n]); append!(MM, rec.m[2:n])
        tnow += HOM.T_chunk
        i0 = findfirst(<(0), H.X.(R))  # first negative signed amplitude: the gluing
        i0 !== nothing && T_[i0] < tnow - 5000 && break       # glued, with at least 5000 time units after it
    end
    x = H.X.(R); i0 = findfirst(<(0), x)  # signed amplitude of every record; i0: the gluing
    i0 === nothing && error("no gluing within T_max")
    # maxima of the signed amplitude before the gluing (above half its largest value)
    imax = [i for i in 2:i0-1 if x[i] > x[i-1] && x[i] >= x[i+1] && x[i] > 0.5maximum(x)]
    m1, m2 = imax[end-1], imax[end]                          # the last complete cycle
    P = T_[m2] - T_[m1]  # period of the last MPS cycle before the gluing
    # t_glue: time of the gluing; t_cycle: the maxima at both ends of the last cycle; θ0: phase of the pattern
    return (t=T_, rho=M32(R), rhoa=M32(RA), m=M32(MM), period=P, ϕ=BT_PHI, ϕa=c.ϕas + BT_DELTA^2*HOM.s_glue,
            Pe=c.Pe + BT_DELTA^2*H.μ1, ηt=c.Π*HOM.s_glue + c.Ω*H.μ1, δ=BT_DELTA, s=HOM.s_glue, μ1=H.μ1, t_glue=T_[i0],
            t_cycle=(T_[m1], T_[m2]), θ0=H.θ0, Γ=c.Γ)
end

# Steps φa¹ = s_ref + offsets across the gluing, each held T_conv and continued from the previous one; reached from
# the state at the end of the first ramp of the approach
function homoclinic_steps(approach)
    H = homoclinic_setup(); c = H.c
    ss = HOM.s_ref .+ HOM.offsets  # φa¹ of the steps
    state = approach.state_app
    P0 = H.newlog()                                            # from s_app to the first step (kept as pre_*)
    for s in ramp(HOM.s_app, ss[1], HOM.ds_near); state = H.hold_log!(P0, state, s, HOM.T_near); end
    T_all = Float64[]; S_all = Float64[]; R_all = Vector{Float32}[]; RA_all = Vector{Float32}[]; M_all = Vector{Float32}[]
    # starts: time at which each step begins
    starts = Float64[]; tnow = 0.0
    for s in ss
        state, rec = H.hold(state, s, HOM.T_conv)
        n = length(rec.t); push!(starts, tnow)
        # each step without its t = 0 (repeats the previous end)
        append!(T_all, tnow .+ rec.t[2:n]); append!(S_all, fill(s, n - 1))
        append!(R_all, rec.rho[2:n]); append!(RA_all, rec.rhoa[2:n]); append!(M_all, rec.m[2:n])
        tnow += HOM.T_conv
        @printf("φa¹ = %.5f done\n", s)
    end
    pre = loggedfields(P0)
    # steps: φa¹ of each step; ϕas, Pes: the BT point; pre_*: the log from s_app to the first step
    return (t=T_all, s=S_all, rho=M32(R_all), rhoa=M32(RA_all), m=M32(M_all), steps=ss, starts=starts, μ1=H.μ1, ϕ=BT_PHI,
            δ=BT_DELTA, ϕas=c.ϕas, Pes=c.Pe, θ0=H.θ0, pre_t=pre.t, pre_s=pre.s, pre_rho=pre.rho, pre_rhoa=pre.rhoa, pre_m=pre.m)
end

# ==================================================================================================================
# Figures
# ==================================================================================================================

# (K2-K3)/(K2+K3) = D/M against φ: Γ = 0 lines (the sign of Γ written between them), values of D/M as dotted levels
# where they apply (red: D, M > 0; blue: D, M < 0), and the regions on the φ axis, labelled on the axis when they fit
# and otherwise in a strip below (two rows, with thin lines to their region). The levels of D/M (levels_A_neg for Γ < 0,
# levels_A_pos for Γ > 0) and the region names are the boundaries and cases of the classification in Dangelmayr &
# Knobloch, "The Takens–Bogdanov bifurcation with O(2)-symmetry" (Phil. Trans. R. Soc. Lond. A 322, 1987)
const RATIO_SETTINGS = (size=(1600, 800), xlims=(0.5, 1.0), ylims=(-1.6, 1.1), levels_A_neg=[1.0, 4/5, 3/4, 0.74, 0.7, 3/5, 1/2],
    levels_A_pos=[1/2, 1/5], level_lw=3.9, A0_lw=4.9, lw=6.3, guidefontsize=34, tickfontsize=26, phi_label_y=0.6,
    gamma_label_y=(0.85, -0.5, 0.85), gamma_label_dx=(0.0, 0.0, 0.0), gamma_fontsize=30,
    region_names=["II+", "I+", "IV-", "III-", "IV-", "V-", "VI-", "VII-", "VIII-", "IX-", "I+", "I+", "II+"],
    region_label_y=0.11, region_tick_up=0.07, region_tick_down=0.08, xticks=0.5:0.1:1.0, strip_ticks_y=0.86, strip_bend_y=0.72,
    label_scale=1.0, region_gap=0.006, strip_gap=0.02, line_clearance=0.005, rows=2, strip_height=0.26,
    strip_rows_y=(0.45, 0.17), strip_halfheight=0.1, label_dx=Dict{Int,Float64}())

function ratio_figure(C; S=RATIO_SETTINGS)
    ϕ = C.phi; z = C.zeros
    sgn(v, x) = sign(v[argmin(abs.(ϕ .- x))])  # sign of v at the grid point nearest to x
    cuts = sort(vcat(S.xlims[1], S.xlims[2], z["A"], z["D"], z["M"]))  # edges of the φ-intervals: all zeros of A, D and M
    # sign of A, D and M on each interval (at its midpoint)
    pieces = [(a=cuts[k], b=cuts[k+1], A=sgn(C.A, (cuts[k]+cuts[k+1])/2), D=sgn(C.D, (cuts[k]+cuts[k+1])/2), M=sgn(C.M, (cuts[k]+cuts[k+1])/2))
              for k in 1:length(cuts)-1]
    # dotted levels: each value of D/M for its sign of Γ, over the intervals with that sign of Γ and D/M > 0
    segs = [(a=pc.a, b=pc.b, value=ℓ, col=(pc.D > 0 ? :red : :blue)) for (levels, sA) in ((S.levels_A_neg, -1), (S.levels_A_pos, +1))
            for ℓ in levels for pc in pieces if pc.A == sA && pc.D == pc.M]
    p = plot(; xlims=S.xlims, ylims=S.ylims, ylabel=L"\dfrac{\mathcal{K}_2-\mathcal{K}_3}{\mathcal{K}_2+\mathcal{K}_3}", legend=false,
             guidefontsize=S.guidefontsize, tickfontsize=S.tickfontsize, yticks=(-1.5:0.5:1.0, [latexstring(@sprintf("%.1f", v)) for v in -1.5:0.5:1.0]),
             xticks=(S.xticks, fill("", length(S.xticks))), bottom_margin=-6Plots.mm)
    for x in z["A"]; vline!(p, [x]; color=:black, ls=:dot, lw=S.A0_lw, label=""); end  # Γ = 0 lines
    edges = vcat(S.xlims[1], z["A"], S.xlims[2])  # intervals of constant sign of Γ, each labelled with it
    for k in 1:length(edges)-1
        sA = sgn(C.A, (edges[k] + edges[k+1])/2)
        annotate!(p, (edges[k] + edges[k+1])/2 + S.gamma_label_dx[k], S.gamma_label_y[k],
                  txt(sA > 0 ? L"\Gamma>0" : L"\Gamma<0", S.gamma_fontsize, :black, :center))
    end
    for s in segs; plot!(p, [s.a, s.b], [s.value, s.value]; color=s.col, ls=:dot, lw=S.level_lw, label=""); end  # the levels
    yv = copy(C.D ./ C.M); for x in z["M"]; yv[argmin(abs.(ϕ .- x))] = NaN; end        # break at the poles (M = 0)
    plot!(p, ϕ, yv; color=:purple, lw=S.lw, label="")  # the curve D/M
    # regions on the φ axis: edges at the zeros of A, D, M and the crossings of the curve with the blue levels
    edges_r = sort(vcat(z["A"], z["D"], z["M"]))
    for s in segs
        s.col == :blue || continue
        # grid points inside the level's interval: crossings of the curve with the level, by linear interpolation
        for i in findall(x -> s.a < x < s.b, ϕ)[1:end-1]
            f1, f2 = yv[i] - s.value, yv[i+1] - s.value
            f1*f2 < 0 && push!(edges_r, ϕ[i] - f1*(ϕ[i+1] - ϕ[i])/(f2 - f1))
        end
    end
    edges_r = vcat(S.xlims[1], sort(edges_r), S.xlims[2])
    nreg = length(edges_r) - 1
    nreg == length(S.region_names) || error("there are $nreg regions but $(length(S.region_names)) names")
    names = S.region_names
    mids = [(edges_r[k] + edges_r[k+1])/2 for k in 1:nreg]  # midpoints of the regions
    pxw = Dict('I' => 12, 'V' => 22, 'X' => 22, '+' => 35, '-' => 35)       # character widths in px (font size 30)
    # label widths in units of φ
    wlab = [S.label_scale*sum(pxw[ch] for ch in n)*(S.gamma_fontsize/30)*(S.xlims[2] - S.xlims[1])/(0.86*S.size[1]) for n in names]
    ybot = S.ylims[1]
    label(k) = txt(latexstring("\\mathrm{$(names[k][1:end-1])}{$(names[k][end])}"), S.gamma_fontsize, :black, :center)
    # region edges as ticks on the φ axis
    for x in edges_r[2:end-1]; plot!(p, [x, x], [ybot, ybot + S.region_tick_up]; color=:black, lw=2.8, label=""); end
    # labels that fit inside their region (placed on the axis)
    fits = [edges_r[k+1] - edges_r[k] ≥ wlab[k] + S.region_gap for k in 1:nreg]
    for k in findall(fits); annotate!(p, mids[k] + get(S.label_dx, k, 0.0), ybot + S.region_label_y, label(k)); end
    # the other labels in the strip, alternating between rows in order of φ
    ks = findall(.!fits); rowof(i) = mod1(i, S.rows)
    # offsets of the strip labels: each at least its distance to the earlier ones (same row: no overlap; other row:
    # clear of its line)
    offs = zeros(length(ks))
    for j in 2:length(ks), i in 1:j-1
        wi, wj = wlab[ks[i]], wlab[ks[j]]
        need = rowof(i) == rowof(j) ? (wi + wj)/2 + S.strip_gap : (rowof(i) < rowof(j) ? wi : wj)/2 + S.line_clearance
        offs[j] = max(offs[j], offs[i] + need)
    end
    # the strip labels centred about the mean of their regions,
    labx = isempty(ks) ? Float64[] : offs .- sum(offs)/length(ks) .+ sum(mids[ks])/length(ks)
    # then shifted (if needed) to stay inside xlims
    isempty(ks) || (labx .+= clamp(0.0, S.xlims[1] + wlab[ks[1]]/2 - labx[1], S.xlims[2] - wlab[ks[end]]/2 - labx[end]))
    # strip below the panel: φ ticks, φ label, edges and the labels that did not fit
    strip = plot(; xlims=S.xlims, ylims=(0, 1), showaxis=false, grid=false, ticks=false, legend=false, top_margin=0Plots.mm)
    for x in S.xticks; annotate!(strip, x, S.strip_ticks_y, txt(@sprintf("%.1f", x), S.tickfontsize, :black, :center)); end
    annotate!(strip, sum(S.xlims)/2, S.phi_label_y, txt(L"\phi", S.guidefontsize, :black, :center))
    # region edges at the top of the strip
    for x in edges_r[2:end-1]; plot!(strip, [x, x], [1.0, 1.0 - S.region_tick_down]; color=:black, lw=2.8, label=""); end
    for (i, k) in enumerate(ks)
        x = labx[i] + get(S.label_dx, k, 0.0); y = S.strip_rows_y[rowof(i)]
        # grey line from the region down to its label
        plot!(strip, [mids[k], mids[k], x, x], [1.0, S.strip_bend_y, S.strip_rows_y[1] + S.strip_halfheight + 0.05, y + S.strip_halfheight];
              color=:gray, lw=1.4, label="")
        annotate!(strip, x, y, label(k))
    end
    hs = S.strip_height
    return plot(p, strip; layout=grid(2, 1, heights=[1 - hs, hs]), size=(S.size[1], round(Int, S.size[2]/(1 - hs))),
                left_margin=14Plots.mm, right_margin=12Plots.mm, top_margin=6Plots.mm)
end

# Bifurcation rays: (angle in the plane (K1, (K2+K3)η¹/|Γ|), label, colour, line style)
function wnl_rays(kc)
    θκ(κ) = mod(atan(1.0, κ), 2π)                        # rays with η¹ > 0
    # η¹ = 0 (K1 > 0 and K1 < 0), K1 = 0 with η¹ < 0, then the κ rays with η¹ > 0
    return [(0.0, "pitchfork of H", :black, :solid), (π, nothing, :black, :solid), (3π/2, "Hopf of H", :black, :dash),
            (θκ(kc["drift_PS"]), "drift of PS", :purple, :solid), (θκ(kc["fold_SW"]), "fold of CP", :forestgreen, :solid),
            (θκ(kc["hom"]), "homoclinic", :darkorange, :solid), (θκ(kc["hopf_PS"]), "Hopf of PS", :royalblue, :solid)]
end
const PATH_COLORS = [:gray35, :gray70]
# Pe of a path (fixed μ¹)
path_label(c, P) = latexstring(@sprintf("\\mathrm{Pe} = %.5f", c.Pe + BT_DELTA^2*P.μ0))

# Two-parameter diagram in the (K1, η¹) plane: to scale (scale) and schematic (schem, Dangelmayr–Knobloch style: the
# angles between the rays stretched, arcs for the range of angles where each solution exists, continuous if stable)
const TWO_PARAM_SETTINGS = (lw=3.1, ms=3.3,
    scale=(xlims=(-6.0, 6.0), ylims=(-0.4, 0.4), size=(900, 650), legend_height=0.26, columns=2,
           labels=[("H", -3.5, -0.25), ("PS", -5.3, 0.035), ("none", 3.0, 0.25), ("none", 3.0, -0.25), ("none", -1.5, 0.3)]),
    schem=(xlims=(-2.55, 2.0), ylims=(-1.75, 1.6), size=(1000, 800), legend_height=0.08,
           disp=(drift=35.0, fold=100.0, hom=118.0, hopf=145.0), ray_r=1.5, r0=4.0,
           path_r=[(1.02, 1.22), (1.26, 1.46)], radii=(H=0.18, PS=0.33, TW=0.48, SWout=0.63, SWin=0.78, MPS=0.92),
           arc_label_deg=Dict(:H => 225.0, :PS => 70.0, :TW => 320.0, :SWout => 200.0, :SWin => 109.0, :MPS => 131.0),
           arc_label_dr=Dict(:H => 0.085, :PS => 0.085, :TW => 0.13, :SWout => 0.13, :SWin => 0.085, :MPS => 0.13),
           arrow=0.045, path_head=0.1, path_label=[(x=0.93, y=-0.78, ha=:left), (x=-1.18, y=-0.78, ha=:right)], hopfH_y=-1.38))

function two_parameter_figure(W; S=TWO_PARAM_SETTINGS)
    c = bt_coefficients(BT_PHI; Lx=LX_BT); T = S.scale; M = c.K2 + c.K3  # M = K2 + K3 (> 0 at BT_PHI)
    pa = plot(; xlims=T.xlims, ylims=T.ylims, xlabel=L"\mathcal{K}_1", ylabel=L"\eta^1", legend=false, guidefontsize=GFS_BT,
              tickfontsize=FS_BT)
    R = 100*max(abs.(T.xlims)..., abs.(T.ylims)...)  # ray length well beyond the axes
    for (θ, lab, col, ls) in wnl_rays(W.kappa)
        # direction (K1, η¹): η¹ = (|Γ|/M) × the second coordinate of the ray
        d = [cos(θ), sin(θ)*abs(c.Γ)/M]; d ./= hypot(d...)
        plot!(pa, [0, R*d[1]], [0, R*d[2]]; color=col, ls=ls, lw=S.lw, label=lab === nothing ? "" : lab)
    end
    # the paths (s over BT_SRANGE), their bifurcation points and an arrowhead at the end (increasing s)
    for (k, (name, P)) in enumerate(bt_paths(c))
        ss = range(BT_SRANGE...; length=400)
        plot!(pa, [K1s(c, P, s) for s in ss], [ηs(c, P, s) for s in ss]; color=PATH_COLORS[k], lw=3S.lw, alpha=0.7, label=path_label(c, P))
        es = W.paths[name]["events"]
        scatter!(pa, [e["K1"] for e in es], [e["η"] for e in es]; markercolor=:black, markerstrokecolor=:black, ms=S.ms, label="")
        s2 = BT_SRANGE[2]; dx = (T.xlims[2] - T.xlims[1])/30; dy = (T.ylims[2] - T.ylims[1])/30      # direction of increasing s
        # tip and direction of the path at s = s2, in units of dx, dy
        tip = [K1s(c, P, s2), ηs(c, P, s2)]; d = (tip .- [K1s(c, P, s2 - 1), ηs(c, P, s2 - 1)]) ./ [dx, dy]
        t = d ./ hypot(d...); n = [-t[2], t[1]]; b = -t
        plot!(pa, Shape(tip[1] .+ dx .* [0, b[1] + 0.6n[1], b[1] - 0.6n[1]], tip[2] .+ dy .* [0, b[2] + 0.6n[2], b[2] - 0.6n[2]]);
              fillcolor=PATH_COLORS[k], linecolor=PATH_COLORS[k], label="")
    end
    scatter!(pa, [0.0], [0.0]; markercolor=:black, markerstrokecolor=:black, ms=S.ms, label="")  # the BT point
    for (t, x, y) in T.labels; annotate!(pa, x, y, txt(t, FS_BT, :black, :center)); end  # region labels
    return with_legend(pa, legend_entries(pa); size=T.size, legend_height=T.legend_height, columns=T.columns, fontsize=FS_BT,
                       sample=0.06, margin=8Plots.mm)
end

function schematic_figure(W; S=TWO_PARAM_SETTINGS)
    c = bt_coefficients(BT_PHI; Lx=LX_BT); T = S.schem; kc = W.kappa; M = c.K2 + c.K3
    ang(k, y) = mod(atan(y, k), 2π)  # angle in [0, 2π)
    θκ(κ) = ang(κ, 1.0)  # angle of the ray K1 = κ(K2+K3)η¹/|Γ| with η¹ > 0
    # true angles of the rays: drift of PS, fold of CP, homoclinic, Hopf of PS
    θd, θf, θh, θH = θκ(kc["drift_PS"]), θκ(kc["fold_SW"]), θκ(kc["hom"]), θκ(kc["hopf_PS"])
    # true angle θ of (K1, (K2+K3)η¹/|Γ|) mapped piecewise linearly so that the rays sit at the angles T.disp
    knots_t = [0.0, θd, θf, θh, θH, π, 3π/2, 2π]
    knots_d = deg2rad.([0.0, T.disp.drift, T.disp.fold, T.disp.hom, T.disp.hopf, 180.0, 270.0, 360.0])
    function dmap(θ)
        θ = mod(θ, 2π); j = clamp(searchsortedlast(knots_t, θ), 1, length(knots_t) - 1)
        return knots_d[j] + (θ - knots_t[j])*(knots_d[j+1] - knots_d[j])/(knots_t[j+1] - knots_t[j])
    end
    pb = plot(; xlims=T.xlims, ylims=T.ylims, aspect_ratio=1, framestyle=:none, grid=false, ticks=false, legend=false)
    # filled triangle with its tip at `tip`, pointing away from `base`
    tri!(tip, base, sz, color) = (t = (tip .- base) ./ hypot((tip .- base)...); n = [-t[2], t[1]]; b = tip .- sz .* t;
        plot!(pb, Shape([tip[1], b[1] + 0.6sz*n[1], b[1] - 0.6sz*n[1]], [tip[2], b[2] + 0.6sz*n[2], b[2] - 0.6sz*n[2]]);
              fillcolor=color, linecolor=color, label=""))
    Rr = T.ray_r; off = 0.05  # ray length; offset of the labels
    for (θ, lab, col, ls) in wnl_rays(kc)
        φ = dmap(θ)
        plot!(pb, [0, Rr*cos(φ)], [0, Rr*sin(φ)]; color=col, ls=ls, lw=S.lw, label="")
        lab === nothing && continue
        if θ == 0.0  # the pitchfork (both η¹ = 0 rays): label at the left end
            annotate!(pb, -Rr - off, 0.0, txt(lab, FS_BT, :black, :right, :vcenter))
        elseif θ == 3π/2  # Hopf of H: label below the ray, at hopfH_y
            annotate!(pb, off, T.hopfH_y, txt(lab, FS_BT, :black, :left, :vcenter))
        else  # the other rays: label beyond the end, aligned by its direction
            annotate!(pb, (Rr + off)*cos(φ), (Rr + off)*sin(φ), txt(lab, FS_BT, col, cos(φ) > 0.3 ? :left : (cos(φ) < -0.3 ? :right : :center),
                      sin(φ) > 0.3 ? :bottom : (sin(φ) < -0.3 ? :top : :vcenter)))
        end
    end
    # axis names at the ends of the rays
    annotate!(pb, Rr + off, 0.0, txt(L"\mathcal{K}_1", GFS_BT, :black, :left, :vcenter))
    annotate!(pb, 0.0, -Rr - off, txt(L"\eta^1", GFS_BT, :black, :center, :top))
    # paths: true angle stretched like the rays, each in its own band of radii r = a + (b - a)(1 - exp(-r/r0))
    for (k, (name, P)) in enumerate(bt_paths(c))
        (ra, rb) = T.path_r[k]
        xy(s) = (k1 = K1s(c, P, s); y = M*ηs(c, P, s)/abs(c.Γ); ρd = ra + (rb - ra)*(1 - exp(-hypot(k1, y)/T.r0));
                 φ = dmap(ang(k1, y)); [ρd*cos(φ), ρd*sin(φ)])
        pts = xy.(range(BT_SRANGE...; length=2000))  # the path (s over BT_SRANGE) mapped to the schematic plane
        # last point at least path_head from the end: the line stops there and the arrowhead fills the rest
        j = findlast(q -> hypot((q .- pts[end])...) >= T.path_head, pts)
        plot!(pb, first.(pts[1:j]), last.(pts[1:j]); color=PATH_COLORS[k], lw=3S.lw, alpha=0.8, label="")
        tri!(pts[end], pts[j], T.path_head, PATH_COLORS[k])
        # bifurcation points of the path
        e_xy = [xy(e["s"]) for e in W.paths[name]["events"]]
        scatter!(pb, first.(e_xy), last.(e_xy); markercolor=:black, markerstrokecolor=:black, ms=S.ms, label="")
        L = T.path_label[k]
        annotate!(pb, L.x, L.y, txt(path_label(c, P), FS_BT, PATH_COLORS[k], L.ha, :vcenter))
    end
    # existence arcs: (label, radius key, [(θ from, θ to, stable)], arrowheads at both ends?)
    function arrowhead!(φ, r, dir)
        t = dir .* [-sin(φ), cos(φ)]; n = [cos(φ), sin(φ)]; tip = r .* n; h = T.arrow; w = 0.6T.arrow; b = tip .- h .* t
        plot!(pb, Shape([tip[1], b[1] + w*n[1], b[1] - w*n[1]], [tip[2], b[2] + w*n[2], b[2] - w*n[2]]); fillcolor=:black,
              linecolor=:black, label="")
    end
    # set by hand from the existence and stability conditions of each branch (H, PS, TW, CP, MPS)
    arcs = [("H", :H, [(0.0, π, false), (π, 3π/2, true), (3π/2, 2π, false)], false), ("PS", :PS, [(0.0, θH, false), (θH, π, true)], true),
            ("TW", :TW, [(3π/2, 2π + θd, false)], true), ("CP", :SWout, [(θf, 3π/2, false)], true),
            ("CP", :SWin, [(θf, θh, true)], true), ("MPS", :MPS, [(θh, θH, true)], true)]
    for (lab, key, segs, heads) in arcs
        r = T.radii[key]
        for (a, b, st) in segs
            # displayed angles (past 2π for an arc that crosses the positive K1 axis)
            φs = range(dmap(a), a < 2π <= b ? 2π + dmap(b - 2π) : (b == 2π ? 2π : dmap(b)); length=200)
            plot!(pb, r .* cos.(φs), r .* sin.(φs); color=:black, ls=st ? :solid : :dash, lw=S.lw, label=st ? "stable" : "unstable")
        end
        # displayed angles of both ends
        φa_ = dmap(segs[1][1]); b = segs[end][2]; φb_ = b > 2π ? 2π + dmap(b - 2π) : (b == 2π ? 2π : dmap(b))
        heads && (arrowhead!(φa_, r, -1); arrowhead!(φb_, r, 1))
        φm = deg2rad(T.arc_label_deg[key])  # label of the arc
        annotate!(pb, (r + T.arc_label_dr[key])*cos(φm), (r + T.arc_label_dr[key])*sin(φm), txt(lab, FS_BT, :black, :center))
    end
    scatter!(pb, [0.0], [0.0]; markercolor=:black, markerstrokecolor=:black, ms=S.ms, label="")
    ents = sort(legend_entries(pb); by=e -> e[1] != "stable")  # legend: stable first
    return with_legend(pb, ents; size=T.size, legend_height=T.legend_height, columns=2, fontsize=FS_BT, margin=2Plots.mm)
end

# One-parameter diagrams along the two paths: every branch as Δρ = 4δ|A| against φa = φa* + δ²s (SW and MPS: maximum of
# the signed amplitude over the period); stable continuous, unstable dashed; markers at the bifurcation points
const PATHS_SETTINGS = (lw=3.1, ms=6, size=(1800, 560), legend_height=0.25, columns=4,
    panels=["μ¹ < 0" => (tag="(a)", xlims=(1.45, 1.92), ylims=(0.0, 0.47), xtick=1e-5,
                         labels=[("H", 1.47, 0.02), ("CP", 1.6, 0.44), ("PS", 1.62, 0.2), ("MPS", 1.838, 0.25), ("CP", 1.897, 0.31), ("PS", 1.905, 0.272)]),
            "μ¹ > 0" => (tag="(b)", xlims=(-4.0, -1.0), ylims=(0.0, 0.32), xtick=5e-5,
                         labels=[("H", -3.8, 0.02), ("CP", -3.55, 0.17), ("TW", -2.3, 0.157), ("PS", -1.25, 0.28)])])
const EVENT_MARKERS = [("primary Hopf", :circle, "Hopf bifurcation"), ("stationary", :hexagon, "pitchfork bifurcation"),
                       ("secondary Hopf", :rect, "Hopf bifurcation of PS"), ("homoclinic", :star5, "homoclinic bifurcation"),
                       ("fold", :utriangle, "fold of CP"), ("drift", :diamond, "drift of PS")]

function paths_figure(W; S=PATHS_SETTINGS)
    c = bt_coefficients(BT_PHI; Lx=LX_BT); δ = BT_DELTA
    X(s) = c.ϕas .+ δ^2 .* s; Y(a) = 4δ .* a  # X: φa from s; Y: Δρ = 4δ|A| from the amplitude
    function pieces(x, y, st)                  # pieces of constant stability, each sharing its end with the next
        out = Tuple{Vector{Float64},Vector{Float64},Bool}[]
        isempty(x) && return out
        i0 = 1
        for i in 2:length(x)+1
            if i > length(x) || st[i] != st[i0]
                j = min(i, length(x)); push!(out, (x[i0:j], y[i0:j], st[i0])); i0 = i
            end
        end
        return out
    end
    panels = map(S.panels) do (name, P)
        d = W.paths[name]
        # φa ticks every xtick
        xt = collect(ceil(X(P.xlims[1])/P.xtick)*P.xtick:P.xtick:X(P.xlims[2]))
        yl = Y.(P.ylims)
        p = plot(; xlims=X.(P.xlims), ylims=yl, xlabel=L"\phi_a", ylabel=L"\Delta\rho", legend=false,
                 xticks=(xt, [@sprintf("%.5f", v) for v in xt]), guidefontsize=GFS_BT, tickfontsize=FS_BT)
        # a branch as pieces, solid where stable, dashed where unstable
        line!(x, y, st) = for (xs, ys, s) in pieces(x, y, st)
            plot!(p, X(xs), Y(ys); color=:black, ls=s ? :solid : :dash, lw=S.lw, label=s ? "stable" : "unstable")
        end
        line!(d["H"]["s"], zeros(length(d["H"]["s"])), d["H"]["stable"])  # H (zero amplitude)
        isempty(d["PS"]) || line!(d["PS"]["s"], d["PS"]["r"], d["PS"]["stable"])  # PS (amplitude r)
        isempty(d["TW"]) || line!(d["TW"]["s"], d["TW"]["r"], d["TW"]["stable"])  # TW (amplitude r)
        # L = 0 orbits (CP, MPS): largest signed amplitude
        for (_, b) in d["L0"]; isempty(b) || line!(b["s"], b["xmax"], b["stable"]); end
        for e in d["events"]  # bifurcation points inside the panel, marker by type
            P.xlims[1] <= e["s"] <= P.xlims[2] || continue
            m = EVENT_MARKERS[findfirst(m -> occursin(m[1], e["type"]), EVENT_MARKERS)]
            scatter!(p, X([e["s"]]), Y([e["amp"]]); markershape=m[2], markercolor=:white, markerstrokecolor=:black,
                     markerstrokewidth=2.1, ms=S.ms, label=m[3])
        end
        # panel tag
        annotate!(p, X(P.xlims[1]), yl[2] + 0.04(yl[2] - yl[1]), txt(latexstring("\\mathrm{$(P.tag)}"), GFS_BT, :black, :left, :bottom))
        for (t, x, y) in P.labels; annotate!(p, X(x), Y(y), txt(t, FS_BT, :black, :center)); end  # branch labels
        p
    end
    order = ["stable", "unstable", (m[3] for m in EVENT_MARKERS)...]  # legend order: stable, unstable, then the markers
    ents = sort(legend_entries(panels...); by=e -> something(findfirst(==(e[1]), order), length(order) + 1))
    return with_legend(plot(panels...; layout=(1, length(panels))), ents; size=S.size, legend_height=S.legend_height,
                       columns=S.columns, fontsize=FS_BT, legend_bottom=8Plots.mm)
end

# Phase plane (a, da/dT) near the homoclinic orbit, with a = Re(ρ̂₁e^{-iθ₀})/δ: grey, the MPS (last period of the steps
# below the gluing and the last cycle before it, with their mirror images) and the CP of sw_steps; black dashed, the
# leading-order homoclinic orbit a = A sech(√η¹ T) at the value of φa¹ selected by the O(δ) condition
# (bt_homoclinic_wnl), and its mirror image; dots, the PS of the PDE (Newton) and H; an arrow for
# increasing φa (on this path the MPS grows towards H and glues into the CP)
const HOM_PHASE_SETTINGS = (size=(780, 560), legend_height=0.13, xlims=(-0.36, 0.36), ylims=(-0.032, 0.032), lw_ts=1.7, lw_ae=2.8,
                            ts_color=:gray55, sw_steps=(1.8947, 1.8997), arrow_from=(0.07, 0.0053), arrow_to=(0.046, 0.013),
                            arrow_label_offset=(0.006, -0.0022), mps_label=(0.13, -0.0063), cp_label=(0.0, 0.0078), ps_label_dy=0.0033,
                            H_label=(0.0, -0.0062))

function homoclinic_phase_figure(h, st; S=HOM_PHASE_SETTINGS)
    function amp(t, ρ, θ, δ)  # a = Re(ρ̂₁e^{-iθ₀})/δ and da/dT = (da/dt)/δ (central differences)
        # ρ̂₁ of each row
        nx = size(ρ, 2); r1 = [sum(ρ[i, :] .* exp.(-2π*im .* (0:nx-1) ./ nx))/nx for i in axes(ρ, 1)]
        a = real.(r1 .* exp(-im*θ)) ./ δ
        # central differences (T = δt)
        da = similar(a); da[2:end-1] .= (a[3:end] .- a[1:end-2]) ./ (t[3:end] .- t[1:end-2]) ./ δ
        da[1] = da[2]; da[end] = da[end-1]
        return a, da
    end
    δ = h.δ; θ0 = h.θ0
    mps = Tuple{Vector{Float64},Vector{Float64}}[]; sw = Tuple{Vector{Float64},Vector{Float64}}[]
    for sv in st.steps              # last period (maximum to maximum) of each step that has not glued, and of sw_steps
        k = findall(==(sv), st.s)  # rows of this step
        a, da = amp(st.t[k], Float64.(st.rho[k, :]), θ0, δ)
        g = minimum(a) < 0  # glued: a changes sign (CP)
        # a glued step is kept only if it is one of sw_steps
        (g && !any(s -> isapprox(sv, s; atol=1e-7), S.sw_steps)) && continue
        # maxima of a (above half the largest)
        imax = [i for i in 2:length(a)-1 if a[i] > a[i-1] && a[i] >= a[i+1] && a[i] > 0.5maximum(a)]
        r = vcat(imax[end-1]:imax[end], imax[end-1])  # last period, closed
        push!(g ? sw : mps, (a[r], da[r]))  # CP -> sw, MPS -> mps
    end
    # last MPS cycle before the gluing (homoclinic_cycle), plus 2% of a period to close it
    k = findall(τ -> h.t_cycle[1] <= τ <= h.t_cycle[2] + 0.02h.period, h.t)
    push!(mps, amp(h.t[k], Float64.(h.rho[k, :]), θ0, δ))
    c = bt_coefficients(BT_PHI; Lx=LX_BT)
    # weakly nonlinear homoclinic orbit on the path μ¹ < 0 (at its own φa¹)
    hw = bt_homoclinic_wnl(c, (ϕa0=0.0, vϕ=1.0, μ0=h.μ1, vμ=0.0))
    hw === nothing && error("no weakly nonlinear homoclinic orbit on this path")
    ps = ps_breathing(c, h.s, h.μ1, δ; Lx=LX_BT, N=32)  # PS of the PDE (Newton) at φa¹ = s_glue
    # its a = |ρ̂₁|/δ
    N = length(ps.u) ÷ 3; aps = abs(sum(ps.u[1:N] .* exp.(-2π*im .* (0:N-1) ./ N))/N)/δ
    p = plot(; xlabel=L"a", ylabel=L"\mathrm{d}a/\mathrm{d}T", legend=false, xlims=S.xlims, ylims=S.ylims, guidefontsize=GFS_BT,
             tickfontsize=FS_BT)
    # the MPS and their mirror images (a -> -a)
    for (a, da) in mps, sg in (1, -1); plot!(p, sg .* a, sg .* da; color=S.ts_color, lw=S.lw_ts, label=""); end
    for (a, da) in sw; plot!(p, a, da; color=S.ts_color, lw=S.lw_ts, label=""); end  # the CP (already symmetric)
    for sg in (1, -1)
        plot!(p, sg .* hw.a, sg .* hw.da; color=:black, ls=:dash, lw=S.lw_ae,
              label=sg == 1 ? "homoclinic orbit (weakly nonlinear estimation)" : "")
    end
    scatter!(p, [-aps, 0, aps], [0, 0, 0]; color=:black, ms=4.5, markerstrokewidth=0, label="")  # PS (±) and H (origin)
    arrow!(p, S.arrow_from..., S.arrow_to..., S.xlims, S.ylims, (640, 400); lw=2.5, hl=16, hw=6)
    annotate!(p, S.arrow_from[1] + S.arrow_label_offset[1], S.arrow_from[2] + S.arrow_label_offset[2], txt(L"\phi_a", FS_BT, :black, :left))
    for sg in (1, -1)
        annotate!(p, sg*S.mps_label[1], sg*S.mps_label[2], txt("MPS", FS_BT, :black, :center))
        annotate!(p, sg*aps, S.ps_label_dy, txt("PS", FS_BT, :black, :center))
    end
    annotate!(p, S.cp_label..., txt("CP", FS_BT, :black, :center))
    annotate!(p, S.H_label..., txt("H", FS_BT, :black, :center))
    return with_legend(p, legend_entries(p); size=S.size, legend_height=S.legend_height, columns=1, fontsize=FS_BT, margin=8Plots.mm)
end

# ==================================================================================================================
# Video: the steps of φa¹ across the homoclinic orbit, each over its last nper periods (for the first step above the
# homoclinic value, the gluing itself): ρ(x), the phase plane (a, da/dT) (grey: previous φa) and a(t)
# ==================================================================================================================

function homoclinic_video(d; dframe=100.0, fps=20, nper=2, name="07 Homoclinic Bifurcation - MPS Gluing into CP.mp4")
    t = d.t; s = d.s; rho = Float64.(d.rho); nx = size(rho, 2); δ = d.δ
    x = range(0, LX_BT; length=nx + 1)
    r1 = [sum(rho[i, :] .* exp.(-2π*im .* (0:nx-1) ./ nx))/nx for i in axes(rho, 1)]  # ρ̂₁ of each row
    a = real.(r1 .* exp(-im*d.θ0)) ./ δ  # signed amplitude a = Re(ρ̂₁e^{-iθ₀})/δ
    # da/dT = (da/dt)/δ by central differences
    da = similar(a); da[2:end-1] .= (a[3:end] .- a[1:end-2]) ./ (t[3:end] .- t[1:end-2]) ./ δ; da[1] = da[2]; da[end] = da[end-1]
    Pe = d.Pes + δ^2*d.μ1  # Pe on the path (fixed)
    function period(k)                                         # mean period from the maxima of |a| (SW: two per period)
        # times of the maxima of |a| (above 60% of the largest)
        tm = [t[i] for i in k[2:end-1] if abs(a[i]) > abs(a[i-1]) && abs(a[i]) >= abs(a[i+1]) && abs(a[i]) > 0.6maximum(abs.(a[k]))]
        sw = minimum(a[k]) < 0 < maximum(a[k])  # SW (CP): a changes sign
        tm = tm[max(1, end-4):end]  # the last 5 maxima
        return (sw ? 2 : 1)*(tm[end] - tm[1])/(length(tm) - 1)
    end
    windows = Vector{Int}[]; first_glued = true  # windows: the rows shown for each step
    for (n, sv) in enumerate(d.steps)
        k = findall(==(sv), s)  # rows of this step
        if minimum(a[k]) < 0 < maximum(a[k]) && first_glued   # the gluing itself
            first_glued = false
            i0 = k[findfirst(<(0), a[k] .* sign(a[k[1]]))]  # first sign change of a
            Pm = period(windows[end])  # MPS period of the previous step
            # rows of the next step (its period sets the time shown after the gluing)
            kn = findall(==(d.steps[min(n + 1, end)]), s)
            # from one MPS period before the gluing to nper periods after it
            w = findall(τ -> t[i0] - Pm <= τ <= t[i0] + nper*period(kn[end÷2:end]), t)
        else
            P = period(k[end÷2:end])
            w = findall(τ -> t[k[end]] - nper*P <= τ <= t[k[end]], t)  # the last nper periods of the step
        end
        push!(windows, w)
    end
    al = maximum(abs.(a))*1.08; dal = maximum(abs.(da))*1.08  # half-ranges of the phase-plane axes
    yr = extrema(rho) .+ (-1, 1) .* 0.05 .* (maximum(rho) - minimum(rho))  # ρ range of the whole run
    font = (guidefontsize=16, tickfontsize=11, framestyle=:box, grid=false)
    step = max(1, round(Int, dframe/(t[2] - t[1])))  # rows between frames
    anim = Animation()
    for (n, w) in enumerate(windows), j in w[1:step:end]
        p1 = hline([d.ϕ]; color=:gray60, ls=:dot, lw=1.5, label="")
        plot!(p1, x, vcat(rho[j, :], rho[j, 1]); color=:black, lw=3, ylims=yr, xlims=(0, 10), xlabel=L"x", ylabel=L"\rho", label="", font...)
        p2 = plot(; xlims=(-al, al), ylims=(-dal, dal), xlabel=L"a", ylabel=L"\mathrm{d}a/\mathrm{d}T", font...)
        # earlier windows in grey
        for m in 1:n-1; plot!(p2, a[windows[m]], da[windows[m]]; color=:gray75, lw=1, label=""); end
        k = w[1]:j  # from the start of the window to the current frame
        plot!(p2, a[k], da[k]; color=:black, lw=1.8, label="")
        scatter!(p2, [a[j]], [da[j]]; color=:crimson, ms=6, markerstrokewidth=0, label="")
        scatter!(p2, [0.0], [0.0]; marker=:xcross, color=:gray40, ms=7, label="")
        p3 = plot(t[w], a[w]; color=:gray80, lw=1.5, xlabel=L"t", ylabel=L"a", label="", xlims=(t[w[1]], t[w[end]]), ylims=(-al, al), font...)
        plot!(p3, t[k], a[k]; color=:black, lw=1.8, label="")
        vline!(p3, [t[j]]; color=:crimson, lw=1.5, label="")
        hline!(p3, [0.0]; color=:gray50, ls=:dot, lw=1, label="")
        ttl = latexstring(@sprintf("\\phi = %.5f,\\quad \\phi_a = %.9f,\\quad \\mathrm{Pe} = %.5f", d.ϕ, d.ϕas + δ^2*s[j], Pe))
        p0 = plot(; framestyle=:none, grid=false, ticks=false, xlims=(0, 1), ylims=(0, 1))
        annotate!(p0, 0.5, 0.5, Plots.text(ttl, 16, :black, :center))
        frame(anim, plot(p0, p1, p2, p3; layout=@layout([t{0.06h}; [a b]; c{0.33h}]), size=(1300, 900), dpi=VIDEO_DPI,
                         fontfamily="sans-serif", left_margin=6Plots.mm, bottom_margin=5Plots.mm, top_margin=2Plots.mm))
    end
    return save_video(anim, name; fps=fps)
end
