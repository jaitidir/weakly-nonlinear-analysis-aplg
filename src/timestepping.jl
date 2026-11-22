# Pseudospectral time stepping of the APLG hydrodynamic equations
#   ∂t ρ  = ∂x²ρ - Pe ∂x[(1-ρ)m]
#   ∂t ρa = ∂x[(1-ρ)∂x ρa + ρa ∂x ρ] - Pe ∂x[(1-ρ)m]
#   ∂t m  = ∂x[(1-ρ)∂x m + m ∂x ρ] - 2m - Pe ∂x[(1-ρ)ρa]
# on [0, Lx) periodic: FFT derivatives (2/3-rule dealiasing: the fields are masked before every product and the whole
# explicit right-hand side after) and the second-order IMEX scheme SBDF2 (the constant-coefficient
# diffusion and -2m implicit, the rest explicit; the first step is IMEX Euler). The masses of ρ and ρa are conserved
# exactly. A run can classify its final state with the recurrence classifier below
# (H, PS, T or CP) and stop as soon as it has converged.
using FFTW

struct SpectralParams
    Nx::Int
    Lx::Float64
    Pe::Float64
    phi::Float64
    phia::Float64
    D0::Float64                # 1 - φ: mean diffusivity of ρa and m (treated implicitly)
    k::Vector{Float64}
    k2::Vector{Float64}
    dealias::Vector{Float64}   # 2/3-rule mask
end

function SpectralParams(Nx::Int, Lx::Real, Pe::Real, phi::Real, phia::Real)
    @assert iseven(Nx) "Nx must be even"
    # FFT ordering of the wavenumbers: 0, 1, ..., Nx/2-1, -Nx/2 (Nyquist), ..., -1 times 2π/Lx
    k = (2π/Float64(Lx)) .* vcat(0:(Nx÷2 - 1), -(Nx÷2):-1)
    dealias = Float64.(abs.(k) .< (2/3) * maximum(abs.(k)))     # keeps |n| < Nx/3 (Nyquist mode removed too)
    return SpectralParams(Nx, Float64(Lx), Float64(Pe), Float64(phi), Float64(phia), 1.0 - Float64(phi), k, k .^ 2, dealias)
end

# Explicit part of the right-hand side (nonlinear terms and the fluctuation of the diffusivity 1-ρ about 1-φ), returned
# in Fourier space and masked: every term is a product of two masked fields (|n| < Nx/3), so its aliased modes fall in
# |n| ≥ Nx/3 and the mask removes them (2/3 rule)
function nonlinear_terms(rho::Vector{Float64}, rhoa::Vector{Float64}, m::Vector{Float64}, p::SpectralParams)
    k, k2, mask = p.k, p.k2, p.dealias
    # dealiased fields (suffix _d) and their second derivatives -k² û
    rho_hat = fft(rho) .* mask; rhoa_hat = fft(rhoa) .* mask; m_hat = fft(m) .* mask
    rho_d = real(ifft(rho_hat)); rhoa_d = real(ifft(rhoa_hat)); m_d = real(ifft(m_hat))
    rho_xx = real(ifft(@. -k2 * rho_hat)); rhoa_xx = real(ifft(@. -k2 * rhoa_hat)); m_xx = real(ifft(@. -k2 * m_hat))
    F1 = @. (1.0 - rho_d) * m_d          # flux of ρ and ρa
    F2 = @. (1.0 - rho_d) * rhoa_d       # flux of m
    F1_x = real(ifft(@. im * k * $(fft(F1))))     # ∂x F1
    F2_x = real(ifft(@. im * k * $(fft(F2))))
    # ∂x[(1-ρ)∂x u + u ∂x ρ] = (1-ρ)u_xx + u ρ_xx (the ρ_x u_x terms cancel), minus the implicit part (1-φ)u_xx
    diffuse_rhoa = @. (1.0 - rho_d - p.D0) * rhoa_xx + rhoa_d * rho_xx
    diffuse_m = @. (1.0 - rho_d - p.D0) * m_xx + m_d * rho_xx
    # (N̂ρ, N̂ρa, N̂m), masked; ρ has no explicit diffusion part (its ∂x²ρ is fully implicit)
    return fft(@. -p.Pe * F1_x) .* mask, fft(@. diffuse_rhoa - p.Pe * F1_x) .* mask, fft(@. diffuse_m - p.Pe * F2_x) .* mask
end

# Clip v into [lo, hi] and give it mean `target`, spreading the correction over the points with room to move
function clip_to_mean!(v, lo, hi, target)
    v = clamp.(v, lo, hi)
    for _ in 1:20                                        # (one pass is exact up to rounding; the rest mop up roundoff)
        d = target - mean(v)
        abs(d) <= 1e-15 && break
        room = d > 0 ? (hi .- v) : (v .- lo)             # how far each point can move towards the needed side
        total = sum(room)
        total >= abs(d) * length(v) || error("clip_to_mean!: mean $target not reachable within the bounds")
        # move each point in proportion to its room: the sum changes by exactly d·Nx and no bound is crossed
        v = v .+ sign(d) .* room .* (abs(d) * length(v) / total)
    end
    return v
end

# Homogeneous state plus a perturbation of norm pert_amp:
#   "rand": uniform random noise from seeded_rng(seed) (common.jl);
#   "non-symmetric", "double", "biased": Re[(A⁺vp + A⁻vm) e^{i qc x}] with the linear eigenvectors vp, vm of the two
#   travelling modes and (A⁺, A⁻) = (1, 0), (1, 1) (reflection symmetric) and (pert_Ap, pert_Am)
# then clipped strictly inside the physical bounds, at least `margin` from each of them, with the masses kept:
#   2ε ≤ ρ ≤ 1 - ε,  ε ≤ ρa ≤ ρ - ε,  |m| ≤ ρa - ε     (ε = margin; no point starts on a bound, e.g. ρa = ρ)
const IC_MARGIN = 1e-3
function initial_condition(p::SpectralParams; pert_type::String="rand", pert_amp::Float64=0.1, seed::Int=1,
                           pert_qc=nothing, pert_vp=nothing, pert_vm=nothing, pert_Ap::Float64=1.0, pert_Am::Float64=1.0,
                           margin::Float64=IC_MARGIN)
    Nx = p.Nx
    if pert_type == "rand"
        rng = seeded_rng(seed)
        pr = rand(rng, Nx) .- 0.5; pa = rand(rng, Nx) .- 0.5; pm = rand(rng, Nx) .- 0.5
    elseif pert_type in ("non-symmetric", "double", "biased")
        mode = cis.(pert_qc .* ((0:Nx-1) .* (p.Lx / Nx)))      # e^{i qc x} on the grid x_j = (j-1) Lx/Nx
        Ap, Am = pert_type == "non-symmetric" ? (1.0, 0.0) : pert_type == "double" ? (1.0, 1.0) : (pert_Ap, pert_Am)
        vm = pert_vm === nothing ? (0.0+0.0im, 0.0+0.0im, 0.0+0.0im) : pert_vm
        # (ρ, ρa, m) components of A⁺vp + A⁻vm
        combined = (Ap*pert_vp[1] + Am*vm[1], Ap*pert_vp[2] + Am*vm[2], Ap*pert_vp[3] + Am*vm[3])
        pr = real.(combined[1] .* mode); pa = real.(combined[2] .* mode); pm = real.(combined[3] .* mode)
    else
        error("pert_type must be \"rand\", \"non-symmetric\", \"double\" or \"biased\"")
    end
    pr .-= mean(pr); pa .-= mean(pa); pm .-= mean(pm)                 # zero-mean perturbation: masses unchanged
    c = sqrt(sum(pr.^2) + sum(pa.^2) + sum(pm.^2)) / sqrt(Nx)        # joint RMS of the three components
    pr .*= pert_amp / c; pa .*= pert_amp / c; pm .*= pert_amp / c   # rescaled to joint RMS pert_amp
    ε = margin
    rho = clip_to_mean!(fill(p.phi, Nx) .+ pr, 2ε, 1.0 - ε, p.phi)       # 2ε: leaves room for ε ≤ ρa ≤ ρ - ε
    rhoa = clip_to_mean!(fill(p.phia, Nx) .+ pa, ε, rho .- ε, p.phia)
    m = clamp.(zeros(Nx) .+ pm, .-(rhoa .- ε), rhoa .- ε)  # |m| ≤ ρa - ε; m is not conserved, so its mean is not restored
    return rho, rhoa, m
end

# Eigenvectors (ρ, ρa, m) of the two travelling modes at onset, for the "non-symmetric"/"double"/"biased" ICs
# Normalised to ρ̂ = 1 for the modes e^{i qc x + σ t} of the equations linearised about (φ, ϕac, 0), σ = +i wc (vp) and
# -i wc (vm): m̂ from the ρ equation σ = -qc² - i qc Pe (1-φ) m̂, ρ̂a from the ρa equation
function travelling_eigenvectors(ϕ, ϕac, qc, wc, Pe)
    i = im
    vp = (ComplexF64(1.0), (qc^2*(1-ϕac)+i*wc)/(qc^2*(1-ϕ)+i*wc), (i*qc^2 - wc) / (qc * Pe * (1-ϕ)))
    vm = (ComplexF64(1.0), (qc^2*(1-ϕac)-i*wc)/(qc^2*(1-ϕ)-i*wc), (i*qc^2 + wc) / (qc * Pe * (1-ϕ)))
    return vp, vm
end

"""
    run_spectral(phi, phia, Pe, Lx, Nx; T, dt, pert_type, pert_amp, seed, init_fields, classify, check_after,
                 save_interval, record_interval, ...)

Time steps from the perturbed homogeneous state (or from `init_fields = (ρ, ρa, m)`) up to T. With `classify` the
recurrence classifier is applied every 100 time units after `check_after` to each of ρ, ρa and m separately (from
their snapshots every 2 time units; it is called at the first multiple of `save_interval` after each 100), and the run
stops only when all three have converged at the same check (a field that converged earlier keeps being checked). Δρ,
Δρa and Δm are the largest peak-to-peak values of each field over the snapshots on which its classification was made
(the last one for H, the last 100 time units for PS and T, the last period for CP and MPS). The type of the run is that
of ρ; the types of all three are returned in `sol_types`. `record_interval` records all the fields (Float32) in `rec`.
If the explicit step blows up, dt is halved and the run restarted. The physical bounds 0 ≤ ρ ≤ 1, 0 ≤ ρa ≤ ρ, |m| ≤ ρa are checked on the initial and on the final state only:
the discrete scheme does not preserve them exactly, and a small transient excursion (e.g. in the first steps from
grid-scale noise clipped to the bounds) is accepted as long as the final state satisfies them. The scheme is SBDF2
(second order). Returns the final fields and (sol_type, sol_types, converged, speed, period, drho_eff, drhoa_eff, dm_eff, detail).
"""
function run_spectral(phi::Real, phia::Real, Pe::Real, Lx::Real, Nx::Int; dt::Float64=0.01, dt_min::Float64=1e-5,
                      max_dt_halvings::Int=6, kwargs...)
    dt_try = dt
    for attempt in 0:max_dt_halvings
        try
            return run_spectral_once(phi, phia, Pe, Lx, Nx; dt=dt_try, kwargs...)
        catch e
            # only a blow-up is retried (from t = 0 with dt/2); any other error, including a final state outside the
            # physical bounds, is passed on. (The returned tuple does not say which dt was finally used.)
            retry = e isa ErrorException && occursin("blew up", e.msg)
            (retry && dt_try/2 >= dt_min) || rethrow(e)
            @warn "$(e.msg): restarting with dt = $(dt_try/2)"
            dt_try /= 2
        end
    end
    error("run_spectral: still blowing up after $max_dt_halvings halvings of dt")
end

# Physical bounds of the lattice gas: 0 ≤ ρ ≤ 1, 0 ≤ ρa ≤ ρ, |m| ≤ ρa. bound_violation is the largest amount by which
# any of them is broken (≤ 0 when all hold); BOUND_TOL only absorbs rounding (fields at a bound after clipping)
const BOUND_TOL = 1e-10
function bound_violation(ρ, ρa, m)
    v = -Inf
    @inbounds for i in eachindex(ρ)
        v = max(v, -ρ[i], ρ[i] - 1, -ρa[i], ρa[i] - ρ[i], abs(m[i]) - ρa[i])
    end
    return v
end
check_bounds(ρ, ρa, m, t, dt) = (v = bound_violation(ρ, ρa, m); v > BOUND_TOL &&
    error("run_spectral: final state outside the physical bounds at t=$t by $v (dt=$dt)"))

function run_spectral_once(phi::Real, phia::Real, Pe::Real, Lx::Real, Nx::Int; T::Float64=1000.0, dt::Float64=0.01,
                           pert_type::String="rand", pert_amp::Float64=0.1, seed::Int=1,
                           pert_qc=nothing, pert_vp=nothing, pert_vm=nothing, pert_Ap::Float64=1.0, pert_Am::Float64=1.0,
                           save_interval::Float64=10.0, check_after::Float64=0.0, classify::Bool=true,
                           record_interval::Union{Nothing,Float64}=nothing, rc_interval::Float64=2.0,
                           rc_check_every::Float64=100.0, init_fields=nothing, verbose::Bool=false)
    p = SpectralParams(Nx, Lx, Pe, phi, phia)
    rho, rhoa, m = init_fields === nothing ?
        initial_condition(p; pert_type=pert_type, pert_amp=pert_amp, seed=seed, pert_qc=pert_qc, pert_vp=pert_vp,
                          pert_vm=pert_vm, pert_Ap=pert_Ap, pert_Am=pert_Am) : copy.(init_fields)
    # a start outside the bounds is not retried (dt cannot fix it)
    v0 = bound_violation(rho, rhoa, m)
    v0 > BOUND_TOL && error("run_spectral: initial state outside the physical bounds (by $v0)")
    # implicit operators L: -k², -(1-φ)k², -(1-φ)k² - 2. SBDF2: (3u' - 4u + u_prev)/(2dt) = L u' + 2N - N_prev, after a
    # first IMEX Euler step (u' - u)/dt = L u' + N
    denom_rho = @. 1.0 - dt * (-p.k2)                    # IMEX Euler: 1 - dt L
    denom_rhoa = @. 1.0 - dt * (-p.D0 * p.k2)
    denom_m = @. 1.0 - dt * (-p.D0 * p.k2 - 2.0)
    # SBDF2: 3 - 2dt L
    denom2_rho = @. 3.0 + 2dt * p.k2; denom2_rhoa = @. 3.0 + 2dt * p.D0 * p.k2; denom2_m = @. 3.0 + 2dt * (p.D0 * p.k2 + 2.0)
    prev = nothing                                       # (ρ̂, ρ̂a, m̂, N̂ρ, N̂ρa, N̂m) of the last step
    rho_hat = fft(rho); rhoa_hat = fft(rhoa); m_hat = fft(m)
    nsteps = round(Int, T / dt)
    save_every = max(1, round(Int, save_interval / dt))
    # snapshots of ρ, ρa and m for the classifier every rc_interval, keeping the last 8000 time units
    rc_every = max(1, round(Int, rc_interval / dt)); rc_keep = round(Int, 8000.0 / rc_interval)
    rc_t = Float64[]; rc_f = (Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]); rc_next = 0.0
    record_every = record_interval === nothing ? 0 : max(1, round(Int, record_interval / dt))
    rec_t = Float64[]; rec_rho = Vector{Float32}[]; rec_rhoa = Vector{Float32}[]; rec_m = Vector{Float32}[]
    if record_every > 0
        push!(rec_t, 0.0); push!(rec_rho, Float32.(rho)); push!(rec_rhoa, Float32.(rhoa)); push!(rec_m, Float32.(m))
    end
    result = (type=:unknown, types=(:unknown, :unknown, :unknown), converged=false, speed=nothing, period=nothing,
              drho_eff=nothing, drhoa_eff=nothing, dm_eff=nothing, detail="")
    t = 0.0
    for step in 1:nsteps
        rho_cur = real(ifft(rho_hat)); rhoa_cur = real(ifft(rhoa_hat)); m_cur = real(ifft(m_hat))
        Nh = nonlinear_terms(rho_cur, rhoa_cur, m_cur, p); cur = (rho_hat, rhoa_hat, m_hat)
        if prev === nothing                              # first step: IMEX Euler
            rho_hat = @. (rho_hat + dt * Nh[1]) / denom_rho
            rhoa_hat = @. (rhoa_hat + dt * Nh[2]) / denom_rhoa
            m_hat = @. (m_hat + dt * Nh[3]) / denom_m
        else                                             # SBDF2: û' = (4û - û_prev + 2dt(2N̂ - N̂_prev)) / (3 - 2dt L)
            rho_hat = @. (4cur[1] - prev[1] + 2dt * (2Nh[1] - prev[4])) / denom2_rho
            rhoa_hat = @. (4cur[2] - prev[2] + 2dt * (2Nh[2] - prev[5])) / denom2_rhoa
            m_hat = @. (4cur[3] - prev[3] + 2dt * (2Nh[3] - prev[6])) / denom2_m
        end
        prev = (cur..., Nh...)
        t += dt
        if classify && step % rc_every == 0
            push!(rc_t, t)
            for (v, h) in zip(rc_f, (rho_hat, rhoa_hat, m_hat)); push!(v, real(ifft(h))); end
            length(rc_t) > rc_keep && (popfirst!(rc_t); foreach(popfirst!, rc_f))     # sliding window of rc_keep snapshots
        end
        if record_every > 0 && step % record_every == 0
            push!(rec_t, t); push!(rec_rho, Float32.(real(ifft(rho_hat))))
            push!(rec_rhoa, Float32.(real(ifft(rhoa_hat)))); push!(rec_m, Float32.(real(ifft(m_hat))))
        end
        if step % save_every == 0 || step == nsteps
            if classify && t >= check_after && t + 1e-9 >= rc_next && length(rc_t) > 3
                rc_next = t + rc_check_every
                # each field classified on its own (snapshots as rows, time × x); converged only if all three are. The
                # result is that of the last check, so for an unconverged run it can be up to rc_check_every older than
                # the final fields
                rs = [classify_recurrence(rc_t, permutedims(reduce(hcat, v)); Lx=Float64(Lx)) for v in rc_f]
                detail = join(("$(n): $(r.detail)" for (n, r) in zip(("ρ", "ρa", "m"), rs) if !r.converged), "; ")
                result = (type=rs[1].type, types=Tuple(r.type for r in rs), converged=all(r -> r.converged, rs),
                          speed=rs[1].speed, period=rs[1].period, drho_eff=rs[1].drho_eff, drhoa_eff=rs[2].drho_eff,
                          dm_eff=rs[3].drho_eff, detail=detail)
                verbose && @printf("  t = %.0f  %s  converged = %s  %s\n", t, result.type, result.converged, result.detail)
            end
            classify && result.converged && break
        end
        # (ρa is not checked: a blow-up shows in ρ or m as well)
        (!all(isfinite, rho_hat) || !all(isfinite, m_hat)) && error("run_spectral: blew up at t=$t (dt=$dt)")
    end
    check_bounds(real(ifft(rho_hat)), real(ifft(rhoa_hat)), real(ifft(m_hat)), t, dt)   # the final state (not retried)
    return (t=t, rho=real(ifft(rho_hat)), rhoa=real(ifft(rhoa_hat)), m=real(ifft(m_hat)),
            rec=(t=rec_t, rho=rec_rho, rhoa=rec_rhoa, m=rec_m), sol_type=result.type, sol_types=result.types,
            converged=result.converged,
            speed=result.speed, period=result.period, drho_eff=result.drho_eff, drhoa_eff=result.drhoa_eff,
            dm_eff=result.dm_eff, detail=result.detail)
end

# Time stepping points of the bifurcation diagrams are run with record_interval = HISTORY and their whole history saved
const HISTORY = 5.0

# Save a run as data/<file>: times t, fields rho, rhoa, m at those times (one row per time, Float32), the result of the
# run (stop time, type of ρ and types of ρ, ρa, m, converged, Δρ, Δρa, Δm) and its settings
function save_history(file, r; settings...)
    rows(v) = permutedims(reduce(hcat, v))       # vector of snapshots -> matrix with one row per time
    savedata(file, (t=r.rec.t, rho=rows(r.rec.rho), rhoa=rows(r.rec.rhoa), m=rows(r.rec.m), t_stop=r.t, type=r.sol_type,
                    types=r.sol_types, converged=r.converged, drho=r.drho_eff, drhoa=r.drhoa_eff, dm=r.dm_eff, scheme="sbdf2", settings...))
end

# ==================================================================================================================
# Recurrence classifier: a converged solution repeats itself. With S_s a shift by s in x,
#   steady in a moving frame U(t) = S_{cτ} U(t-τ)  ->  PS (c = 0, and |c| still decreasing) or T (c ≠ 0);
#   periodic U(t) = U(t-P) (compared without shift)  ->  CP if the profile half a period earlier matches after a
#   translation (≥ 1e-3) and a quarter period earlier matches no translate; MPS if half a period earlier it matches
#   best without translation and the profile changes over half a period.
# Anything else (still changing, a modulated travelling wave, a quasi-periodic state) is :unknown and the run goes on.
# Comparisons use the relative error ||U_a - S_s U_b|| / ||U_a - mean(U_a)|| of one field (minimised over the shift s for
# the steady test, without shift for the periodic one); a run classifies ρ, ρa and m separately.
# ==================================================================================================================

const _KCACHE = Dict{Int,Vector{Float64}}()
wavenumbers(N) = get!(_KCACHE, N) do; vcat(0:(N÷2-1), -(N÷2):-1) .* (2π / N) end   # per grid point
shift_hat(uh, s) = uh .* cis.(-wavenumbers(length(uh)) .* s)                      # shift by s grid points

# Best shift s (grid points, continuous) and relative error between a and S_s b
function best_shift(a::AbstractVector, b::AbstractVector)
    N = length(a); ah = fft(a .- mean(a)); bh = fft(b .- mean(b))
    cc = real(ifft(ah .* conj(bh)))                  # circular cross-correlation: cc[s+1] = Σₙ a[n] b[n-s]
    j = argmax(cc); jm = mod1(j - 1, N); jp = mod1(j + 1, N)
    den = cc[jm] - 2cc[j] + cc[jp]
    frac = den == 0 ? 0.0 : 0.5 * (cc[jm] - cc[jp]) / den      # parabolic fit through the peak and its neighbours
    s = (j - 1) + frac; s = s > N / 2 ? s - N : s    # shift in grid points, mapped to (-N/2, N/2]
    k = wavenumbers(N)
    for _ in 1:3                                     # Newton refinement on the continuous objective
        # f(s) = Re Σ conj(â) b̂ e^{-iks} (the correlation): g = f'(s), h = -f''(s) > 0 at a maximum
        g = sum(real(conj(ah) .* (im .* k) .* bh .* cis.(-k .* s)) .* -1)
        h = sum(real(conj(ah) .* (k .^ 2) .* bh .* cis.(-k .* s)))
        h > 0 || break
        s += g / h
    end
    return s, norm(ah .- shift_hat(bh, s)) / max(norm(ah), 1e-30)
end

# Band-limited (Lanczos, a = 4) interpolation of the snapshots F (rows at t0, t0 + τ, ...) at time tq
lanczos(x, a=4) = x == 0 ? 1.0 : (abs(x) < a ? a * sin(π * x) * sin(π * x / a) / (π^2 * x^2) : 0.0)
function interp_field(F::AbstractMatrix, t0, τ, tq; a=4)
    u = (tq - t0) / τ + 1                            # fractional row index (row 1 at t0)
    j0 = floor(Int, u)
    out = zeros(size(F, 2)); wsum = 0.0
    for j in (j0 - a + 1):(j0 + a)                   # the 2a nearest rows
        (1 <= j <= size(F, 1)) || return nothing     # nothing: tq too close to the ends of the stored history
        w = lanczos(u - j, a); wsum += w
        out .+= w .* @view F[j, :]
    end
    return out ./ wsum
end

# Relative mismatch between two profiles without any shift: ||a - b|| / ||a|| (means removed)
rel_mismatch(a, b) = (am = a .- mean(a); norm(am .- (b .- mean(b))) / max(norm(am), 1e-30))
# Classification at the last of the uniformly spaced times t (snapshots: rows of rho, which can be any of the three
# fields). Returns (type, converged, period, speed, drho_eff, detail); drho_eff: the largest peak-to-peak value over the
# snapshots the type was judged on (the last one for H and unknown, the last steady_window for PS and T, the last period
# for CP and MPS).
function classify_recurrence(t::AbstractVector, rho::AbstractMatrix; window=1500.0, pwindow=3000.0, steady_window=100.0,
                             h_tol=1e-4, steady_tol=1e-3, steady_lag=20.0, rec_tol=1e-2, long_tol=1e-2, c_tol=1e-4,
                             shape_tol=0.5, Lx=10.0, convergence=true)
    τ = t[2] - t[1]; N = size(rho, 2); dx = Lx / N     # τ: time between snapshots
    amp = [maximum(r) - minimum(r) for r in eachrow(rho)]      # Δρ of each snapshot
    n = length(t)
    unknown(detail; period=nothing, speed=nothing) = (type=:unknown, converged=false, period=period, speed=speed,
                                                      drho_eff=amp[end], detail=detail)
    # H and steady states are judged over steady_window, periodic ones over window
    ns = round(Int, steady_window / τ)
    n < 2ns && return unknown("not enough history")
    sidx = (n - ns + 1):n
    # H: Δρ below h_tol over the last steady_window AND still decreasing (its largest value there below the largest over
    # the steady_window before), since a slowly growing perturbation can also be small; at rounding level (< 1e-10) it
    # cannot decrease any further and is accepted as it is
    if maximum(amp[sidx]) < h_tol
        a_now = maximum(amp[sidx]); a_before = maximum(amp[(n - 2ns + 1):(n - ns)])
        (a_now < 1e-10 || a_now < a_before) || return unknown("small amplitude but not decaying")
        return (type=:H, converged=true, period=nothing, speed=0.0, drho_eff=amp[end], detail="")
    end

    # 1. steady in a moving frame: every snapshot equals the one steady_lag earlier, shifted
    ml = max(1, round(Int, steady_lag / τ))          # lag in rows
    n < ns + ml && return unknown("not enough history")
    sh = Float64[]; er = Float64[]
    for k in sidx[2:end]
        # sh: shift per snapshot interval (grid points / τ); er: mismatch after the best shift
        s, e = best_shift(rho[k, :], rho[k-ml, :]); push!(sh, s / ml); push!(er, e)
    end
    if maximum(er) < steady_tol
        A = amp[sidx]
        nst = ns                                     # length of the steady stretch (rows)
        # extend the stretch backwards, snapshot by snapshot, while each still matches its shifted lagged copy
        while nst < min(n - ml - 1, round(Int, window / τ))
            best_shift(rho[n-nst, :], rho[n-nst-ml, :])[2] < steady_tol || break
            nst += 1
        end
        c = mean(sh) * dx / τ                        # drift speed (length/time; c > 0: towards +x)
        convergence || return (type=(abs(c) < c_tol ? :PS : :T), converged=true, period=nothing, speed=c,
                               drho_eff=maximum(A), detail="")
        # no slow change of shape: newest vs oldest snapshot of the stretch, the change rescaled to a whole window
        # (the rate of change observed over the stretch, applied over `window` time units)
        _, elong = best_shift(rho[n, :], rho[n-nst, :])
        elong * (window / (nst * τ)) > long_tol && return unknown("steady shape but still changing slowly"; speed=c)
        # the drift speed must have stopped changing too
        nst < 2ns && return unknown("steady; waiting to check that the speed is constant"; speed=c)
        # c1: speed over the oldest ns snapshots of the stretch; dc: its change rescaled to a whole window
        k1 = (n - nst + 1):(n - nst + ns); c1 = mean(best_shift(rho[k, :], rho[k-ml, :])[1] / ml for k in k1) * dx / τ
        dc = abs(c - c1) * (window / ((nst - ns) * τ))
        ((abs(c) >= c_tol && dc > 0.05abs(c)) || (abs(c) < c_tol && dc > c_tol / 2)) &&
            return unknown("steady shape but drift speed still changing"; speed=c)
        # PS: as for H, the drift speed must also be still decreasing (below its value at the start of the stretch), since
        # near a drift instability a PS drifts away only slowly, from a very small speed; at rounding level (< 1e-10) it
        # cannot decrease any further and is accepted as it is
        abs(c) < c_tol && !(abs(c) < 1e-10 || abs(c) < abs(c1)) && return unknown("stationary but drift speed growing"; speed=c)
        return (type=(abs(c) < c_tol ? :PS : :T), converged=true, period=nothing, speed=c, drho_eff=maximum(A), detail="")
    end

    # 2. periodic: candidate periods P where the field matches its own past at t - P WITHOUT any shift (so half a
    # period of a standing wave, which returns shifted, and drifting patterns do not count), over pwindow; trial lags
    # from 4 snapshot intervals to pwindow/4
    nw = round(Int, pwindow / τ)
    n < nw && return unknown("not steady; not enough history yet for the period test")
    idx = (n - nw + 1):n
    F = rho; t0 = t[1]
    ks = idx[end-round(Int, pwindow / (3τ)):4:end]                      # comparison times: last third of the window
    # E(P): mean over kset of the mismatch (no shift) between the field and its (interpolated) value P earlier
    function E(P, kset)
        v = Float64[]
        for k in kset
            past = interp_field(F, t0, τ, t[k] - P); past === nothing && return Inf
            push!(v, rel_mismatch(F[k, :], past))
        end
        return mean(v)
    end
    Ps = (4τ):(τ / 2):(pwindow / 4)                  # trial periods
    Es = [E(P, ks) for P in Ps]
    # candidates: every local minimum of E(P) on the coarse grid of lags. The minimum at a true period is very sharp
    # (the mismatch grows by ~ω·|lag - P|), so a grid lag next to it can be far above rec_tol: candidates are only
    # tested against rec_tol after being refined to non-integer lags below (E < 0.5 only drops lags with no
    # resemblance at all)
    cand = [j for j in 2:length(Es)-1 if Es[j] <= Es[j-1] && Es[j] <= Es[j+1] && Es[j] < 0.5]
    isempty(cand) && return unknown("no recurrence found")
    last_detail = ""
    for j in cand                                                       # shortest period that really works
        a, b = Ps[j] - τ / 2, Ps[j] + τ / 2                             # golden-section refinement
        for _ in 1:20
            c1 = b - (b - a) / 1.618; c2 = a + (b - a) / 1.618
            E(c1, ks) < E(c2, ks) ? (b = c2) : (a = c1)
        end
        P = (a + b) / 2
        # 3. the same period over the whole window ...
        # (every other row whose value P earlier lies inside the window, with room for the interpolation stencil)
        kall = (idx[1] + ceil(Int, P / τ) + 4):2:n
        length(kall) < 8 && continue
        e_first = E(P, kall[1:end÷2]); e_last = E(P, kall[end÷2+1:end])      # older and newer half of the window
        if e_last > rec_tol
            last_detail = "P=$(round(P, sigdigits=4)) recurs only to $(round(e_last, sigdigits=2))"; continue
        end
        # 4. ... not still converging: the mismatch is not still improving, and the field matches itself nper periods
        # back (at the best lag near nper·P, averaged over a period of comparison times)
        convergence && (e_first > 2e_last && e_first > rec_tol / 2) && return unknown("recurrence still improving"; period=P)
        nper = floor(Int, (pwindow - 1.5P - 8τ) / P)    # most whole periods that fit in the window
        klast = (n - ceil(Int, P / τ)):2:n              # comparison times: the last period
        function elag(Λ)                                # mean mismatch with the field Λ earlier
            v = 0.0
            for k in klast
                pl = interp_field(F, t0, τ, t[k] - Λ); pl === nothing && return Inf
                v += rel_mismatch(F[k, :], pl)
            end
            return v / length(klast)
        end
        a2, b2 = nper * P - P / 4, nper * P + P / 4
        for _ in 1:(convergence ? 30 : 0)
            c1 = b2 - (b2 - a2) / 1.618; c2 = a2 + (b2 - a2) / 1.618
            elag(c1) < elag(c2) ? (b2 = c2) : (a2 = c1)
        end
        Λ = (a2 + b2) / 2; elong = convergence ? elag(Λ) : Inf
        if isfinite(elong)                              # (Inf: Λ reaches before the stored history; P kept)
            elong > long_tol && return unknown("periodic but still changing slowly"; period=P)
            P = Λ / nper                                # period refined from the long lag
        end
        # 5. CP or MPS: translation s that best matches the profile half a period earlier with the current one
        # (the smallest such translation: for a pattern with several wavelengths, a whole-wavelength translation gives
        # the same match as none). s ≥ 1e-3 and, after the translation, mismatch ≤ rec_tol: CP (peaks and troughs
        # exchange), provided the profile changes shape in between (below). s < 1e-3: MPS (stays in place, its
        # amplitude changes), unless the profile hardly changes over half a period (mismatch < 1e-4), which could be a
        # PS: then unknown. Anything else moves: unknown
        past(k) = interp_field(F, t0, τ, t[k] - P / 2)
        e0 = Float64[]; eb = Float64[]; sk = Float64[]
        for k in klast
            pk = past(k); sb, e_b = best_shift(F[k, :], pk); ez = rel_mismatch(F[k, :], pk)
            push!(e0, ez); push!(eb, e_b); push!(sk, ez - e_b <= 1e-6 * max(ez, 1e-12) ? 0.0 : abs(sb) * dx)
        end
        s_half = mean(sk)                               # translation (length)
        if s_half >= 1e-3
            mean(eb) > rec_tol && return unknown("periodic but moving"; period=P)
            # A T is periodic without shift too (P = Lx/|c|) and is also a translate of itself half a period earlier; a
            # T still carrying a weak counter-propagating wave (relative size r) fails the steady test above but passes
            # every test up to here. A CP changes shape: a quarter period earlier no translation matches it (mismatch
            # after the best translation ≥ 1.2 in all our CPs, about 1.4 r, i.e. ~0.01, for such a T)
            eq = mean(best_shift(F[k, :], interp_field(F, t0, τ, t[k] - P / 4))[2] for k in klast)
            eq < shape_tol && return unknown("a translate of itself a quarter period earlier (possibly a T)"; period=P)
            return (type=:CP, converged=true, period=P, speed=0.0, drho_eff=maximum(amp[klast[1]:n]), detail="")
        end
        mean(e0) < 1e-4 && return unknown("no change over half a period (possibly a PS)"; period=P)
        return (type=:MPS, converged=true, period=P, speed=0.0, drho_eff=maximum(amp[klast[1]:n]), detail="")
    end
    return unknown(last_detail)
end

# Classification with a longer period-search window pwindow, for the slow modulated PS (MPS) near the Bogdanov–Takens
# point: its period (~4000 time units) is beyond the default pwindow = 3000 (trial periods up to pwindow/4, and the
# long-lag check needs pwindow ≳ 2.5P); classify_recurrence tells MPS and CP apart itself
classify_modulated_ps(t::AbstractVector, rho::AbstractMatrix; pwindow=40000.0, Lx=10.0) =
    classify_recurrence(t, rho; pwindow=pwindow, Lx=Lx)
