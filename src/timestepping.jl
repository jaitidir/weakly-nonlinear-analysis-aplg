# Pseudospectral time stepping of the APLG hydrodynamic equations
#   ∂t ρ  = ∂x²ρ - Pe ∂x[(1-ρ)m]
#   ∂t ρa = ∂x[(1-ρ)∂x ρa + ρa ∂x ρ] - Pe ∂x[(1-ρ)m]
#   ∂t m  = ∂x[(1-ρ)∂x m + m ∂x ρ] - 2m - Pe ∂x[(1-ρ)ρa]
# on [0, Lx) periodic: FFT derivatives (2/3 dealiasing) and the second-order IMEX scheme SBDF2 (the constant-coefficient
# diffusion and -2m implicit, the rest explicit; the first step is IMEX Euler). The masses of ρ and ρa are conserved
# exactly. A run can classify its final state with the recurrence classifier below
# (H, PS, T or CP) and stop as soon as it has converged.
using FFTW, Random

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

# Explicit part of the right-hand side (nonlinear terms and the fluctuation of the diffusivity 1-ρ about 1-φ)
function nonlinear_terms(rho::Vector{Float64}, rhoa::Vector{Float64}, m::Vector{Float64}, p::SpectralParams)
    k, k2, mask = p.k, p.k2, p.dealias
    # dealiased fields (suffix _d) and their second derivatives -k² û
    rho_hat = fft(rho) .* mask; rhoa_hat = fft(rhoa) .* mask; m_hat = fft(m) .* mask
    rho_d = real(ifft(rho_hat)); rhoa_d = real(ifft(rhoa_hat)); m_d = real(ifft(m_hat))
    rho_xx = real(ifft(@. -k2 * rho_hat)); rhoa_xx = real(ifft(@. -k2 * rhoa_hat)); m_xx = real(ifft(@. -k2 * m_hat))
    F1 = @. (1.0 - rho_d) * m_d          # flux of ρ and ρa
    F2 = @. (1.0 - rho_d) * rhoa_d       # flux of m
    F1_x = real(ifft(@. im * k * $(fft(F1) .* mask)))     # ∂x F1, the product masked again before differentiating
    F2_x = real(ifft(@. im * k * $(fft(F2) .* mask)))
    # ∂x[(1-ρ)∂x u + u ∂x ρ] = (1-ρ)u_xx + u ρ_xx (the ρ_x u_x terms cancel), minus the implicit part (1-φ)u_xx
    diffuse_rhoa = @. (1.0 - rho_d - p.D0) * rhoa_xx + rhoa_d * rho_xx
    diffuse_m = @. (1.0 - rho_d - p.D0) * m_xx + m_d * rho_xx
    # (Nρ, Nρa, Nm) in real space; ρ has no explicit diffusion part (its ∂x²ρ is fully implicit)
    return (@. -p.Pe * F1_x), (@. diffuse_rhoa - p.Pe * F1_x), (@. diffuse_m - p.Pe * F2_x)
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
#   "rand": random noise (seeded);
#   "non-symmetric", "double", "biased": Re[(A⁺vp + A⁻vm) e^{i qc x}] with the linear eigenvectors vp, vm of the two
#   travelling modes and (A⁺, A⁻) = (1, 0), (1, 1) (reflection symmetric) and (pert_Ap, pert_Am)
# then clipped to 0 ≤ ρ ≤ 1, 0 ≤ ρa ≤ ρ, |m| ≤ ρa with the masses kept
function initial_condition(p::SpectralParams; pert_type::String="rand", pert_amp::Float64=0.1, seed::Union{Nothing,Int}=1,
                           pert_qc=nothing, pert_vp=nothing, pert_vm=nothing, pert_Ap::Float64=1.0, pert_Am::Float64=1.0)
    seed === nothing || Random.seed!(seed)
    Nx = p.Nx
    if pert_type == "rand"
        pr = rand(Nx) .- 0.5; pa = rand(Nx) .- 0.5; pm = rand(Nx) .- 0.5
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
    rho = clip_to_mean!(fill(p.phi, Nx) .+ pr, 0.0, 1.0, p.phi)
    rhoa = clip_to_mean!(fill(p.phia, Nx) .+ pa, 0.0, rho, p.phia)
    m = clamp.(zeros(Nx) .+ pm, .-rhoa, rhoa)          # |m| ≤ ρa; m is not conserved, so its mean is not restored
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
recurrence classifier is applied every 100 time units after `check_after` (from ρ stored every 2 time units) and the
run stops once it has converged. `save_interval` is the window over which the largest Δρa, Δm are tracked;
`record_interval` records all the fields (Float32) in `rec`. If the explicit step blows up, or a step leaves the
physical bounds 0 ≤ ρ ≤ 1, 0 ≤ ρa ≤ ρ, |m| ≤ ρa (checked after every step), dt is halved and the run restarted. The scheme is SBDF2 (second order): the first-order IMEX Euler used before damped slow growth rates by
O(dt), which mattered for the slow modes near the Bogdanov–Takens point. Returns the final fields and
(sol_type, converged, speed, period, drho_eff, drhoa_eff, dm_eff, detail).
"""
function run_spectral(phi::Real, phia::Real, Pe::Real, Lx::Real, Nx::Int; dt::Float64=0.01, dt_min::Float64=1e-5,
                      max_dt_halvings::Int=6, kwargs...)
    dt_try = dt
    for attempt in 0:max_dt_halvings
        try
            return run_spectral_once(phi, phia, Pe, Lx, Nx; dt=dt_try, kwargs...)
        catch e
            # only a blow-up or a step out of the physical bounds is retried (from t = 0 with dt/2); any other error is
            # passed on. (The returned tuple does not say which dt was finally used.)
            retry = e isa ErrorException && (occursin("blew up", e.msg) || occursin("left the physical bounds", e.msg))
            (retry && dt_try/2 >= dt_min) || rethrow(e)
            @warn "$(e.msg): restarting with dt = $(dt_try/2)"
            dt_try /= 2
        end
    end
    error("run_spectral: still blowing up or leaving the physical bounds after $max_dt_halvings halvings of dt")
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
    error("run_spectral: left the physical bounds at t=$t by $v (dt=$dt)"))

function run_spectral_once(phi::Real, phia::Real, Pe::Real, Lx::Real, Nx::Int; T::Float64=1000.0, dt::Float64=0.01,
                           pert_type::String="rand", pert_amp::Float64=0.1, seed::Union{Nothing,Int}=1,
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
    # ρ snapshots for the classifier every rc_interval, keeping the last 8000 time units
    rc_every = max(1, round(Int, rc_interval / dt)); rc_keep = round(Int, 8000.0 / rc_interval)
    rc_t = Float64[]; rc_rho = Vector{Float64}[]; rc_next = 0.0
    record_every = record_interval === nothing ? 0 : max(1, round(Int, record_interval / dt))
    rec_t = Float64[]; rec_rho = Vector{Float32}[]; rec_rhoa = Vector{Float32}[]; rec_m = Vector{Float32}[]
    if record_every > 0
        push!(rec_t, 0.0); push!(rec_rho, Float32.(rho)); push!(rec_rhoa, Float32.(rhoa)); push!(rec_m, Float32.(m))
    end
    drhoa_series = Float64[]; dm_series = Float64[]
    result = (type=:unknown, converged=false, speed=nothing, period=nothing, drho_eff=nothing, drhoa_eff=nothing,
              dm_eff=nothing, detail="")
    # largest Δρa and Δm since the last save (checked every step)
    drhoa_win_max = 0.0; dm_win_max = 0.0
    t = 0.0
    for step in 1:nsteps
        rho_cur = real(ifft(rho_hat)); rhoa_cur = real(ifft(rhoa_hat)); m_cur = real(ifft(m_hat))
        step > 1 && check_bounds(rho_cur, rhoa_cur, m_cur, t, dt)   # the state reached by the last step
        drhoa_win_max = max(drhoa_win_max, maximum(rhoa_cur) - minimum(rhoa_cur))
        dm_win_max = max(dm_win_max, maximum(m_cur) - minimum(m_cur))
        Nrho, Nrhoa, Nm = nonlinear_terms(rho_cur, rhoa_cur, m_cur, p)
        Nh = (fft(Nrho), fft(Nrhoa), fft(Nm)); cur = (rho_hat, rhoa_hat, m_hat)
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
            push!(rc_t, t); push!(rc_rho, real(ifft(rho_hat)))
            length(rc_t) > rc_keep && (popfirst!(rc_t); popfirst!(rc_rho))     # sliding window of rc_keep snapshots
        end
        if record_every > 0 && step % record_every == 0
            push!(rec_t, t); push!(rec_rho, Float32.(real(ifft(rho_hat))))
            push!(rec_rhoa, Float32.(real(ifft(rhoa_hat)))); push!(rec_m, Float32.(real(ifft(m_hat))))
        end
        if step % save_every == 0 || step == nsteps
            rhoa_now = real(ifft(rhoa_hat)); m_now = real(ifft(m_hat))
            # largest peak-to-peak Δρa, Δm over this save window
            push!(drhoa_series, max(drhoa_win_max, maximum(rhoa_now) - minimum(rhoa_now)))
            push!(dm_series, max(dm_win_max, maximum(m_now) - minimum(m_now)))
            drhoa_win_max = 0.0; dm_win_max = 0.0
            if classify && t >= check_after && t + 1e-9 >= rc_next && length(rc_t) > 3
                rc_next = t + rc_check_every
                # snapshots as rows (time × x) for the classifier
                r = classify_recurrence(rc_t, permutedims(reduce(hcat, rc_rho)); Lx=Float64(Lx))
                # Δρa, Δm: largest over the last 75 save windows (75·save_interval time units). The result is that of
                # the last check, so for an unconverged run it can be up to rc_check_every older than the final fields
                result = (type=r.type, converged=r.converged, speed=r.speed, period=r.period, drho_eff=r.drho_eff,
                          drhoa_eff=maximum(drhoa_series[max(1, end-74):end]), dm_eff=maximum(dm_series[max(1, end-74):end]),
                          detail=r.detail)
                verbose && @printf("  t = %.0f  %s  converged = %s  %s\n", t, result.type, result.converged, result.detail)
            end
            classify && result.converged && break
        end
        # (ρa is not checked: a blow-up shows in ρ or m as well)
        (!all(isfinite, rho_hat) || !all(isfinite, m_hat)) && error("run_spectral: blew up at t=$t (dt=$dt)")
    end
    check_bounds(real(ifft(rho_hat)), real(ifft(rhoa_hat)), real(ifft(m_hat)), t, dt)   # the final state as well
    return (t=t, rho=real(ifft(rho_hat)), rhoa=real(ifft(rhoa_hat)), m=real(ifft(m_hat)),
            rec=(t=rec_t, rho=rec_rho, rhoa=rec_rhoa, m=rec_m), sol_type=result.type, converged=result.converged,
            speed=result.speed, period=result.period, drho_eff=result.drho_eff, drhoa_eff=result.drhoa_eff,
            dm_eff=result.dm_eff, detail=result.detail)
end

# Time stepping points of the bifurcation diagrams are run with record_interval = HISTORY and their whole history saved
const HISTORY = 5.0

# Save a run as data/<file>: times t, fields rho, rhoa, m at those times (one row per time, Float32), the result of the
# run (stop time, type, converged, Δρ, Δρa, Δm) and its settings
function save_history(file, r; settings...)
    rows(v) = permutedims(reduce(hcat, v))       # vector of snapshots -> matrix with one row per time
    savedata(file, (t=r.rec.t, rho=rows(r.rec.rho), rhoa=rows(r.rec.rhoa), m=rows(r.rec.m), t_stop=r.t, type=r.sol_type,
                    converged=r.converged, drho=r.drho_eff, drhoa=r.drhoa_eff, dm=r.dm_eff, scheme="sbdf2", settings...))
end

# ==================================================================================================================
# Recurrence classifier: a converged solution repeats itself. With S_s a shift by s in x,
#   steady in a moving frame U(t) = S_{cτ} U(t-τ)  ->  PS (c = 0) or T (c ≠ 0);
#   periodic U(t) = S_s U(t-P), standing (s = 0)  ->  CP.
# Anything else (still changing, a modulated travelling wave, a quasi-periodic state) is :unknown and the run goes on.
# Comparisons use the relative error ||U_a - S_s U_b|| / ||U_a - mean(U_a)|| of ρ minimised over the shift s.
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

# Balance of the forward- and backward-rotating parts of the first Fourier mode over the period P ending at row n:
# 1 for a standing pattern
function standing_balance(F, t, n, P)
    # z(t): complex amplitude of the first Fourier mode (one wavelength in the box) over the last period
    kk = findall((t .> t[n] - P) .& (t .<= t[n])); z = [fft(F[k, :])[2] for k in kk]; tk = t[kk]
    # bal(w): (min/max of the e^{+iwt} and e^{-iwt} parts of z, their total weight)
    bal(w) = (cp = abs(mean(z .* cis.(-w .* tk))); cm = abs(mean(z .* cis.(w .* tk))); (min(cp, cm) / max(cp, cm, 1e-30), cp + cm))
    # z may oscillate at the frequency 2π/P or at twice it: use whichever carries more weight
    b1, s1 = bal(2π / P); b2, s2 = bal(4π / P)
    return s1 >= s2 ? b1 : b2
end

# Classification at the last of the uniformly spaced times t (snapshots: rows of rho). Returns (type, converged,
# period, speed, drho_eff, detail).
function classify_recurrence(t::AbstractVector, rho::AbstractMatrix; window=1500.0, steady_window=100.0, h_tol=2e-3,
                             steady_tol=1e-3, steady_lag=20.0, rec_tol=1e-2, long_tol=1e-2, c_tol=1e-4, Lx=10.0)
    τ = t[2] - t[1]; N = size(rho, 2); dx = Lx / N     # τ: time between snapshots
    amp = [maximum(r) - minimum(r) for r in eachrow(rho)]      # Δρ of each snapshot
    n = length(t)
    unknown(detail; period=nothing, speed=nothing) = (type=:unknown, converged=false, period=period, speed=speed,
                                                      drho_eff=amp[end], detail=detail)
    # H and steady states are judged over steady_window, periodic ones over window
    ns = round(Int, steady_window / τ)
    n < ns && return unknown("not enough history")
    sidx = (n - ns + 1):n
    maximum(amp[sidx]) < h_tol && return (type=:H, converged=true, period=nothing, speed=0.0, drho_eff=amp[end], detail="")

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
        nst = ns                                     # length of the steady stretch
        # extend the stretch backwards, snapshot by snapshot, while each still matches its shifted lagged copy
        while nst < min(n - ml - 1, round(Int, window / τ))
            best_shift(rho[n-nst, :], rho[n-nst-ml, :])[2] < steady_tol || break
            nst += 1
        end
        # no slow drift of the shape: newest vs oldest snapshot of the stretch, extrapolated to a window
        _, elong = best_shift(rho[n, :], rho[n-nst, :])
        elong * (window / (nst * τ)) > long_tol && return unknown("steady shape but still changing slowly")
        c = mean(sh) * dx / τ                        # drift speed (length/time; c > 0: towards +x)
        # the drift speed must have stopped changing too
        nst < 2ns && return unknown("steady; waiting to check that the speed is constant"; speed=c)
        # c1: speed over the oldest ns snapshots of the stretch; dc: its change extrapolated to a whole window
        k1 = (n - nst + 1):(n - nst + ns); c1 = mean(best_shift(rho[k, :], rho[k-ml, :])[1] / ml for k in k1) * dx / τ
        dc = abs(c - c1) * (window / ((nst - ns) * τ))
        ((abs(c) >= c_tol && dc > 0.05abs(c)) || (abs(c) < c_tol && dc > c_tol / 2)) &&
            return unknown("steady shape but drift speed still changing"; speed=c)
        return (type=(abs(c) < c_tol ? :PS : :T), converged=true, period=nothing, speed=c, drho_eff=maximum(A), detail="")
    end

    # 2. periodic: candidate periods P where the field matches its own past at t - P
    nw = round(Int, window / τ)
    n < nw && return unknown("not steady; not enough history yet for the period test")
    idx = (n - nw + 1):n
    A = amp[idx]
    F = rho; t0 = t[1]
    ks = idx[end-round(Int, window / (3τ)):4:end]                       # comparison times: last third of the window
    # E(P): median over kset of the mismatch between the field and its (interpolated) value P earlier, best shift
    function E(P, kset)
        v = Float64[]
        for k in kset
            past = interp_field(F, t0, τ, t[k] - P); past === nothing && return Inf
            push!(v, best_shift(F[k, :], past)[2])
        end
        return median(v)
    end
    Ps = (4τ):(τ / 2):(window / 4)                   # trial periods
    Es = [E(P, ks) for P in Ps]
    # candidates: local minima of E(P) below 5 rec_tol
    cand = [j for j in 2:length(Es)-1 if Es[j] <= Es[j-1] && Es[j] <= Es[j+1] && Es[j] < 5rec_tol]
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
        (e_first > 2e_last && e_first > rec_tol / 2) && return unknown("recurrence still improving"; period=P)
        nper = floor(Int, (window - 1.5P - 8τ) / P)     # most whole periods that fit in the window
        klast = (n - ceil(Int, P / τ)):2:n              # comparison times: the last period
        function elag(Λ)                                # mean mismatch with the field Λ earlier
            v = 0.0
            for k in klast
                pl = interp_field(F, t0, τ, t[k] - Λ); pl === nothing && return Inf
                v += best_shift(F[k, :], pl)[2]
            end
            return v / length(klast)
        end
        a2, b2 = nper * P - P / 4, nper * P + P / 4
        for _ in 1:30
            c1 = b2 - (b2 - a2) / 1.618; c2 = a2 + (b2 - a2) / 1.618
            elag(c1) < elag(c2) ? (b2 = c2) : (a2 = c1)
        end
        Λ = (a2 + b2) / 2; elong = elag(Λ)
        if isfinite(elong)                              # (Inf: Λ reaches before the stored history; P kept)
            elong > long_tol && return unknown("periodic but still changing slowly"; period=P)
            P = Λ / nper                                # period refined from the long lag
        end
        # standing or drifting: shift over one period (a half-box shift means a standing pattern of period 2P)
        s, _ = best_shift(F[n, :], interp_field(F, t0, τ, t[n] - P))
        Pout = P
        if abs(abs(s) - N / 2) < 0.05N                  # shift within 5% of half the box
            Pout = 2P; s = 0.0
        end
        c = s * dx / Pout                               # drift speed (length/time)
        # standing (CP): no drift and equal forward- and backward-rotating parts; otherwise a modulated travelling
        # wave, a transient
        (abs(c) < c_tol && standing_balance(F, t, n, Pout) > 0.8) || return unknown("modulated travelling wave";
                                                                                    period=Pout, speed=c)
        return (type=:CP, converged=true, period=Pout, speed=0.0, drho_eff=maximum(A), detail="")
    end
    return unknown(last_detail)
end

# ==================================================================================================================
# Modulated (breathing) PS: passes the CP test, but |ρ̂₁| stays away from zero over the last period
# (min/max > ps_frac) and the half-width of its oscillation does not drift over the last env_periods periods (less
# than env_tol per period). Returns (type ∈ :MPS, :CP, :PS, :T, :H, :unknown; converged; period; min_over_max).
# ==================================================================================================================

# |ρ̂₁| of each snapshot (rows of rho)
mode1_amplitude(rho::AbstractMatrix) = [abs(fft(r)[2]) / length(r) for r in eachrow(rho)]

function classify_modulated_ps(t::AbstractVector, rho::AbstractMatrix; window=6000.0, Lx=10.0, ps_frac=0.1,
                               env_periods=20, env_tol=1e-3)
    r = classify_recurrence(t, rho; window=window, Lx=Lx)
    (r.converged && r.type == :CP) || return (type=r.type, converged=r.converged, period=r.period, min_over_max=nothing)
    a1 = mode1_amplitude(rho)
    P = r.period
    lastp = findall(x -> x > t[end] - P, t)                       # snapshots of the last period
    ratio = minimum(a1[lastp]) / maximum(a1[lastp])
    ratio < ps_frac && return (type=:CP, converged=true, period=P, min_over_max=ratio)
    floor(Int, (t[end] - t[1]) / P) - 1 < env_periods && return (type=:unknown, converged=false, period=P, min_over_max=ratio)
    # tc: middle of the k-th last period; hw: half-width of the oscillation of |ρ̂₁| over it
    tc = Float64[]; hw = Float64[]
    for k in 1:env_periods
        idx = findall(x -> t[end] - k * P < x <= t[end] - (k - 1) * P, t)
        push!(tc, t[end] - (k - 0.5) * P); push!(hw, (maximum(a1[idx]) - minimum(a1[idx])) / 2)
    end
    rate = ([ones(env_periods) tc] \ log.(hw))[2]                # least-squares slope of log(hw) vs t
    abs(rate) * P > env_tol && return (type=:unknown, converged=false, period=P, min_over_max=ratio)
    return (type=:MPS, converged=true, period=P, min_over_max=ratio)
end
