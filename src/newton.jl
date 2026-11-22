# Newton solvers for steady patterns (PS) and travelling waves (T) of the APLG hydrodynamic equations
#   ∂t ρ = ∂x²ρ - Pe ∂x[(1-ρ)m],  ∂t ρa = ∂x[(1-ρ)∂x ρa + ρa ∂x ρ] - Pe ∂x[(1-ρ)m],
#   ∂t m = ∂x[(1-ρ)∂x m + m ∂x ρ] - 2m - Pe ∂x[(1-ρ)ρa]
# on a periodic grid of Nx points with second-order finite differences, and their continuation in φa.
using SparseArrays, KLU

struct Params
    Nx::Int
    Lx::Float64
    dx::Float64
    Pe::Float64
    phi::Float64
    phia::Float64
    pin_idx::Int                 # grid point of the travelling-wave phase pin ρ[pin_idx] = φ (steady pin: always m[1])
    pin_value::Float64           # (not read by any solver: the travelling-wave pin uses p.phi)
    D1::SparseMatrixCSC{Float64,Int}
    D2::SparseMatrixCSC{Float64,Int}
end

function Params(Nx::Int, Lx::Real, Pe::Real, phi::Real, phia::Real; pin_idx::Int=1, pin_value::Real=phi)
    dx = Float64(Lx) / Nx
    return Params(Nx, Float64(Lx), dx, Float64(Pe), Float64(phi), Float64(phia), pin_idx, Float64(pin_value),
                  periodic_D1(Nx, dx), periodic_D2(Nx, dx))
end

# Periodic centred first and second derivatives: (u[i+1] - u[i-1])/(2dx) and (u[i+1] - 2u[i] + u[i-1])/dx², indices mod N
function periodic_D1(N::Int, dx::Real)
    I = Int[]; J = Int[]; V = Float64[]
    c = 1.0 / (2.0 * dx)
    for i in 1:N
        ip = i == N ? 1 : i + 1; im = i == 1 ? N : i - 1
        push!(I, i); push!(J, ip); push!(V, c)
        push!(I, i); push!(J, im); push!(V, -c)
    end
    return sparse(I, J, V, N, N)
end

function periodic_D2(N::Int, dx::Real)
    I = Int[]; J = Int[]; V = Float64[]
    c = 1.0 / dx^2
    for i in 1:N
        ip = i == N ? 1 : i + 1; im = i == 1 ? N : i - 1
        push!(I, i); push!(J, im); push!(V, c)
        push!(I, i); push!(J, i); push!(V, -2.0 * c)
        push!(I, i); push!(J, ip); push!(V, c)
    end
    return sparse(I, J, V, N, N)
end

# U = [ρ; ρa; m]
pack_state(rho, rhoa, m) = vcat(rho, rhoa, m)
@views unpack_state(U, p::Params) = (U[1:p.Nx], U[p.Nx+1:2p.Nx], U[2p.Nx+1:3p.Nx])
unpack_state(U, Nx::Int) = (U[1:Nx], U[Nx+1:2Nx], U[2Nx+1:3Nx])

# Shift ρ and ρa to the masses φ and φa (adds a constant: only the mean changes, used on initial guesses)
function enforce_masses!(rho, rhoa, phi, phia)
    rho .-= mean(rho) - phi
    rhoa .-= mean(rhoa) - phia
    return nothing
end

# Peak-to-peak amplitudes (Δρ, Δρa, Δm) of a solution
amplitudes(s) = (maximum(s.rho) - minimum(s.rho), maximum(s.rhoa) - minimum(s.rhoa), maximum(s.m) - minimum(s.m))

# ==================================================================================================================
# Steady states: Z = [U; λρ; λρa; λm] with Lagrange multipliers for the two masses and the pin m[1] = mfix,
# deflated against the homogeneous state. Doing the fix in m is valid as long-times solutions preserve that mean(m) = 0.
# ==================================================================================================================

# Right-hand sides (R1, R2, R3) of the three equations; ∂x[(1-ρ)∂x u + u ∂x ρ] is written as (1-ρ)u_xx + u ρ_xx (the
# ρ_x u_x terms cancel). D2 is symmetric, so Σ R2 = 0 exactly on the grid, as for the conservative form
function raw_blocks(U, p)
    D1, D2, Pe = p.D1, p.D2, p.Pe
    rho, rhoa, m = unpack_state(U, p)
    f = (1 .- rho) .* m                  # flux of ρ and ρa
    h = (1 .- rho) .* rhoa               # flux of m
    R1 = D2 * rho .- Pe .* (D1 * f)
    R2 = (1 .- rho) .* (D2 * rhoa) .+ rhoa .* (D2 * rho) .- Pe .* (D1 * f)
    R3 = (1 .- rho) .* (D2 * m) .+ m .* (D2 * rho) .- Pe .* (D1 * h) .- 2.0 .* m
    return R1, R2, R3
end

# Exact Jacobian of raw_blocks: Jij = ∂Ri/∂(ρ, ρa, m)j, e.g. ∂/∂ρ of (1-ρ)D2ρa = -diag(D2ρa)
function jacobian_raw(U, p)
    N, D1, D2, Pe = p.Nx, p.D1, p.D2, p.Pe
    rho, rhoa, m = unpack_state(U, p)
    A = 1 .- rho
    Isp = spdiagm(0 => ones(N))
    Mrhoa = spdiagm(0 => rhoa); Mm = spdiagm(0 => m); MA = spdiagm(0 => A)
    J11 = D2 + Pe .* (D1 * Mm)
    J12 = spzeros(N, N)
    J13 = -Pe .* (D1 * MA)
    J21 = -spdiagm(0 => D2 * rhoa) + Mrhoa * D2 + Pe .* (D1 * Mm)
    J22 = MA * D2 + spdiagm(0 => D2 * rho)
    J23 = -Pe .* (D1 * MA)
    J31 = -spdiagm(0 => D2 * m) + Mm * D2 + Pe .* (D1 * Mrhoa)
    J32 = -Pe .* (D1 * MA)
    J33 = MA * D2 + spdiagm(0 => D2 * rho) - 2.0 .* Isp
    return sparse([J11 J12 J13; J21 J22 J23; J31 J32 J33])
end

pack_aug(U, lamrho, lamrhoa, lamm) = vcat(U, [lamrho, lamrhoa, lamm])

# Z = [U (3Nx); λρ; λρa; λm] -> (U, λρ, λρa, λm)
function unpack_aug(Z, p)
    M = 3 * p.Nx
    return Z[1:M], Z[M+1], Z[M+2], Z[M+3]
end

# 3Nx + 3 equations for 3Nx + 3 unknowns. λρ, λρa (added to every row of R1, R2) vanish at a solution, since
# Σ R1 = Σ R2 = 0 identically; λm (added to R3[1] only) balances the extra phase pin and is ≈ 0 when a steady state
# with m[1] = mfix exists. Rows 3Nx+1..3Nx+3: mean(ρ) = φ, mean(ρa) = φa and the phase pin m[1] = mfix
function residual_aug(Z, p, mfix)
    N = p.Nx
    U, lamrho, lamrhoa, lamm = unpack_aug(Z, p)
    rho, rhoa, m = unpack_state(U, p)
    R1, R2, R3 = raw_blocks(U, p)
    e1 = zeros(N); e1[1] = 1.0                       # (pin at index 1, whatever p.pin_idx is)
    return vcat(R1 .+ lamrho .* ones(N), R2 .+ lamrhoa .* ones(N), R3 .+ lamm .* e1,
                [mean(rho) - p.phi, mean(rhoa) - p.phia, m[1] - mfix])
end

sparse_col(v::AbstractVector) = (idx = findall(!iszero, v); sparse(idx, ones(Int, length(idx)), v[idx], length(v), 1))
sparse_row(v::AbstractVector) = (idx = findall(!iszero, v); sparse(ones(Int, length(idx)), idx, v[idx], 1, length(v)))

function jacobian_aug(Z, p, mfix)
    N = p.Nx
    U, _, _, _ = unpack_aug(Z, p)
    J = jacobian_raw(U, p)
    e1 = zeros(N); e1[1] = 1.0
    # bordered matrix [J  columns of λρ, λρa, λm; gradient rows of mean(ρ), mean(ρa), m[1]  0]
    A = [J sparse_col(vcat(ones(N), zeros(N), zeros(N))) sparse_col(vcat(zeros(N), ones(N), zeros(N))) sparse_col(vcat(zeros(N), zeros(N), e1));
         sparse_row(vcat(fill(1.0 / N, N), zeros(2N))) spzeros(1, 1) spzeros(1, 1) spzeros(1, 1);
         sparse_row(vcat(zeros(N), fill(1.0 / N, N), zeros(N))) spzeros(1, 1) spzeros(1, 1) spzeros(1, 1);
         sparse_row(vcat(zeros(2N), e1)) spzeros(1, 1) spzeros(1, 1) spzeros(1, 1)]
    A = sparse(A)
    dropzeros!(A)
    return A
end

# Deflation R/d^(1/N) (N = 1 here: R/d) with d the Euclidean distance of the whole Z (multipliers included) to the
# homogeneous state
homogeneous_state_aug(p::Params) = pack_aug(pack_state(fill(p.phi, p.Nx), fill(p.phia, p.Nx), zeros(p.Nx)), 0.0, 0.0, 0.0)

function residual_deflated(Z, p::Params, mfix::Float64; N::Float64=1, eps::Float64=1e-14)
    d = max(norm(Z .- homogeneous_state_aug(p)), eps)
    return residual_aug(Z, p, mfix) ./ d^(1/N)
end

# d/dZ [R d^(-1/N)] = J d^(-1/N) - (1/N) R vᵀ d^(-1/N-2), v = Z - Z_hom (dense rank-one term)
function jacobian_deflated(Z, p::Params, mfix::Float64; N::Float64=1, eps::Float64=1e-14)
    R = residual_aug(Z, p, mfix)
    J = jacobian_aug(Z, p, mfix)
    v = Z .- homogeneous_state_aug(p)
    d = max(norm(v), eps)
    return sparse(J ./ d^(1/N) .- sparse(R * transpose(v)) .* (1/N) ./ d^(1/N + 2))
end

# Newton step of the deflated system (N = 1): the solution of jacobian_deflated(Z) dZ = -G, i.e. (A + u vᵀ) dZ = -G with
# A = J/d sparse and the dense rank-one term u vᵀ = -R vᵀ/d³, by one sparse LU of A and the Sherman–Morrison formula
# (the same step as a factorisation of the whole deflated Jacobian, whose rank-one term fills it completely: much
# faster on fine grids)
function deflated_newton_step(Z, p::Params, mfix::Float64, G; eps::Float64=1e-14)
    R = residual_aug(Z, p, mfix)
    v = Z .- homogeneous_state_aug(p)
    d = max(norm(v), eps)
    F = lu(jacobian_aug(Z, p, mfix) ./ d)
    y = F \ (-G); w = F \ (-R ./ d^3)
    return y .- w .* (dot(v, y) / (1 + dot(v, w)))
end

# Damped Newton for the deflated steady system, from U0 with the pin m[1] = mfix: returns (Z, converged, iterations,
# residual)
function newton_aug(U0, p; tol=1e-10, maxiter=35, mfix=nothing)
    mfix === nothing && (mfix = unpack_state(U0, p)[3][1])      # default pin: the guess's own m[1]
    Z = pack_aug(copy(U0), 0.0, 0.0, 0.0)                       # multipliers start at 0
    G = residual_deflated(Z, p, mfix, N=1.0)
    for k in 1:maxiter
        dZ = deflated_newton_step(Z, p, mfix, G)
        alpha = 1.0
        Gnorm = norm(G)
        accepted = false
        for _ in 1:20                                            # backtracking: halve the step until G decreases
            Ztrial = Z .+ alpha .* dZ
            Gtrial = residual_deflated(Ztrial, p, mfix, N=1.0)
            if norm(Gtrial) < (1 - 1e-4 * alpha) * Gnorm         # sufficient-decrease (Armijo) test
                Z, G, accepted = Ztrial, Gtrial, true
                break
            end
            alpha *= 0.5
        end
        accepted || return Z, false, k, norm(G)
        # tol applies to the deflated residual R/d: the raw residual is d times it (d = O(1) on the branches)
        norm(G) < tol && return Z, true, k, norm(G)
    end
    return Z, false, maxiter, norm(G)
end

# A steady solution as a named tuple (rho, rhoa, m are views of U). it and resn keep their defaults 0 when not passed
# (bifurcation_scan), so they are not diagnostics there
function steady_result(Z, p; ok=true, it=0, resn=0.0)
    U, _, _, _ = unpack_aug(Z, p)
    rho, rhoa, m = unpack_state(U, p)
    return (Z=Z, U=U, rho=rho, rhoa=rhoa, m=m, ok=ok, it=it, resn=resn)
end

function converge_guess(U0, p::Params; tol=1e-10, maxiter=100, mfix=nothing)
    Z, ok, it, resn = newton_aug(U0, p; tol=tol, maxiter=maxiter, mfix=mfix)
    return steady_result(Z, p; ok=ok, it=it, resn=resn)
end

# ==================================================================================================================
# Travelling waves U(x + c t): Z = [U; λρ; λρa; λm; T] with c = 2π/(qc T): Lagrange multipliers for the three
# populations (as for the steady states) with mean(ρ) = φ, mean(ρa) = φa, mean(m) = 0, and the phase pin ρ[pin_idx] = φ;
# deflated by η = shift + d^(-power) (d: RMS distance to the homogeneous state). Σ R1 = Σ R2 = 0 and Σ R3 = -2 Σ m
# identically on the grid, so at a solution λρ = λρa = λm = 0.
# ==================================================================================================================

homogeneous_U(p::Params) = vcat(fill(p.phi, p.Nx), fill(p.phia, p.Nx), zeros(p.Nx))

function deflation_factor_and_grad(Z::Vector{Float64}, p::Params; power::Float64=1.0, shift::Float64=0.0,
                                   dmin::Float64=1e-14)
    Nx = p.Nx
    dU = Z[1:3Nx] .- homogeneous_U(p)                # only U enters d, not the multipliers or T
    nU = length(dU)
    d = sqrt(dot(dU, dU) / nU)
    d <= dmin && error("Deflation evaluated too close to the homogeneous branch.")
    gradη = zeros(length(Z))
    gradη[1:3Nx] .= (-power / nU) .* d^(-power-2) .* dU     # ∇η = -power d^(-power-1) ∇d, ∇d = dU/(nU d)
    return shift + d^(-power), gradη, d              # (η, ∇η, d)
end

# Wavenumber of the wave (only converts the period T into c): qc of the box onset unless given
wave_number(p, q_override) = q_override === nothing ? find_qc_phiac_wc(p.phi, p.Pe, p.Lx)[1] : q_override

# Residual of the travelling-wave system (3Nx + 4 rows): in the frame ξ = x + ct, 0 = RHS(U) - c U' + multipliers
function residual_tw_raw(Z::Vector{Float64}, p::Params; q_override::Union{Nothing,Float64}=nothing)
    Nx = p.Nx
    λρ, λρa, λm = Z[3Nx+1], Z[3Nx+2], Z[3Nx+3]
    c = 2*pi / wave_number(p, q_override) / Z[3Nx+4]  # c = 2π/(q T), T = Z[end]
    rho, rhoa, m = unpack_state(Z[1:3Nx], Nx)
    A = 1 .- rho
    f = A .* m
    h = A .* rhoa
    R1 = p.D2*rho .- p.Pe.*(p.D1*f) .- c.*(p.D1*rho) .+ λρ
    R2 = A.*(p.D2*rhoa) .+ rhoa.*(p.D2*rho) .- p.Pe.*(p.D1*f) .- c.*(p.D1*rhoa) .+ λρa
    R3 = A.*(p.D2*m) .+ m.*(p.D2*rho) .- p.Pe.*(p.D1*h) .- 2.0.*m .- c.*(p.D1*m) .+ λm
    # constraints: the three means and the phase pin
    return vcat(R1, R2, R3, mean(rho) - p.phi, mean(rhoa) - p.phia, mean(m), rho[p.pin_idx] - p.phi)
end

function jacobian_tw_raw(Z::Vector{Float64}, p::Params; q_override::Union{Nothing,Float64}=nothing)
    Nx, pidx, D1, D2, Pe = p.Nx, p.pin_idx, p.D1, p.D2, p.Pe
    T = Z[3Nx+4]
    qc = wave_number(p, q_override)
    c = 2*pi / qc / T
    rho, rhoa, m = unpack_state(Z[1:3Nx], Nx)
    A = 1 .- rho
    Isp = sparse(I, Nx, Nx)
    MA = spdiagm(0 => A); Mm = spdiagm(0 => m); Mrhoa = spdiagm(0 => rhoa)
    J11 = D2 .+ Pe.*(D1*Mm) .- c.*D1
    J12 = spzeros(Nx, Nx)
    J13 = -Pe.*(D1*MA)
    J21 = -spdiagm(0 => D2*rhoa) .+ Mrhoa*D2 .+ Pe.*(D1*Mm)
    J22 = MA*D2 .+ spdiagm(0 => D2*rho) .- c.*D1
    J23 = -Pe.*(D1*MA)
    J31 = -spdiagm(0 => D2*m) .+ Mm*D2 .+ Pe.*(D1*Mrhoa)
    J32 = -Pe.*(D1*MA)
    J33 = MA*D2 .+ spdiagm(0 => D2*rho) .- 2.0.*Isp .- c.*D1
    fac = 2*pi / qc / T^2                              # d/dT of -c D1 u
    dRdT_rho = fac .* (D1*rho); dRdT_rhoa = fac .* (D1*rhoa); dRdT_m = fac .* (D1*m)
    # rows: (R1, R2, R3, mean ρ, mean ρa, mean m, pin); columns: (ρ, ρa, m, λρ, λρa, λm, T)
    J = spzeros(3Nx + 4, 3Nx + 4)
    J[1:Nx, 1:Nx] = J11;         J[1:Nx, Nx+1:2Nx] = J12;         J[1:Nx, 2Nx+1:3Nx] = J13
    J[Nx+1:2Nx, 1:Nx] = J21;     J[Nx+1:2Nx, Nx+1:2Nx] = J22;     J[Nx+1:2Nx, 2Nx+1:3Nx] = J23
    J[2Nx+1:3Nx, 1:Nx] = J31;    J[2Nx+1:3Nx, Nx+1:2Nx] = J32;    J[2Nx+1:3Nx, 2Nx+1:3Nx] = J33
    J[1:Nx, 3Nx+1] = ones(Nx); J[Nx+1:2Nx, 3Nx+2] = ones(Nx); J[2Nx+1:3Nx, 3Nx+3] = ones(Nx)   # λρ, λρa, λm
    J[1:Nx, 3Nx+4] = dRdT_rho; J[Nx+1:2Nx, 3Nx+4] = dRdT_rhoa; J[2Nx+1:3Nx, 3Nx+4] = dRdT_m
    J[3Nx+1, 1:Nx] = fill(1.0/Nx, Nx)                   # mean(ρ) - φ
    J[3Nx+2, Nx+1:2Nx] = fill(1.0/Nx, Nx)               # mean(ρa) - φa
    J[3Nx+3, 2Nx+1:3Nx] = fill(1.0/Nx, Nx)              # mean(m)
    J[3Nx+4, pidx] = 1.0                                # pin ρ[pin_idx] - φ
    return J
end

# Exact deflated Newton step from the raw sparse Jacobian (Sherman–Morrison): δ = -η/(η + ∇η·J⁻¹F) J⁻¹F
function deflated_newton_step(Z::Vector{Float64}, p::Params; power::Float64=2.0, shift::Float64=1.0,
                              dmin::Float64=1e-14, denom_tol::Float64=1e-12, q_override::Union{Nothing,Float64}=nothing)
    Rraw = residual_tw_raw(Z, p; q_override=q_override)
    η, gradη, _ = deflation_factor_and_grad(Z, p; power=power, shift=shift, dmin=dmin)
    a = klu(jacobian_tw_raw(Z, p; q_override=q_override)) \ Rraw     # a = J⁻¹F, the undeflated Newton step is -a
    denom = η + dot(gradη, a)
    # (if the denominator degenerates, fall back to the plain Newton step -a)
    scale = (!isfinite(denom) || abs(denom) < denom_tol) ? -1.0 : -η / denom
    return scale .* a
end

# (‖F‖, ‖ηF‖, η, RMS distance to the homogeneous state)
function residual_norms(Z::Vector{Float64}, p::Params; power::Float64=2.0, shift::Float64=1.0, dmin::Float64=1e-14,
                        q_override::Union{Nothing,Float64}=nothing)
    raw_norm = norm(residual_tw_raw(Z, p; q_override=q_override))
    η, _, d_hom = deflation_factor_and_grad(Z, p; power=power, shift=shift, dmin=dmin)
    return raw_norm, η * raw_norm, η, d_hom
end

# Initial guess [U; 0; 0; 0; T0] from the weakly nonlinear solution at t = 0
function build_initial_guess_tw(p::Params, Pe, qc, ϕ, ϕac, ϕa, delta, ϕ1, w, T0::Float64; mode=:plus)
    z = [weaklynonlinear_osc(xi, 0.0, Pe, qc, ϕ, ϕac, ϕa, delta, ϕ1, w, mode) for xi in range(0.0, p.Lx - p.dx; length=p.Nx)]
    rho0 = Float64[zi[1] for zi in z]; rhoa0 = Float64[zi[2] for zi in z]; m0 = Float64[zi[3] for zi in z]
    enforce_masses!(rho0, rhoa0, ϕ, ϕa)
    return vcat(pack_state(rho0, rhoa0, m0), 0.0, 0.0, 0.0, T0)      # λρ = λρa = λm = 0, period guess T0
end

# Deflated Newton with backtracking for a travelling wave; converged when the raw residual is below tol away from
# the homogeneous state
function solve_tw(Z0::Vector{Float64}, p::Params; tol=1e-10, maxiter=50, defl_power=1.0, defl_shift=0.0,
                  defl_dmin=1e-14, min_accept_dist=1e-5, max_backtracks=20, q_override::Union{Nothing,Float64}=nothing)
    Z = copy(Z0)
    conv = false
    qc = wave_number(p, q_override)
    kw = (power=defl_power, shift=defl_shift, dmin=defl_dmin, q_override=q_override)
    for iter in 0:maxiter
        raw_norm, defl_norm, η, d_hom = residual_norms(Z, p; kw...)
        # converged: raw residual small, away from the homogeneous state, and period T = Z[end] > 0
        if raw_norm < tol && d_hom > min_accept_dist && Z[end] > 0
            conv = true
            break
        end
        iter == maxiter && break
        δ = deflated_newton_step(Z, p; kw...)
        α = 1.0
        accepted = false
        for bt in 1:max_backtracks                      # halve α until the deflated residual decreases (T stays > 0)
            Ztrial = Z .+ α .* δ
            if all(isfinite, Ztrial) && Ztrial[end] > 0
                try
                    defl_trial = residual_norms(Ztrial, p; kw...)[2]
                    if isfinite(defl_trial) && defl_trial < defl_norm
                        Z = Ztrial
                        accepted = true
                        break
                    end
                catch                                   # too close to the homogeneous state
                end                                     # (any other error also just rejects the trial step)
            end
            α *= 0.5
        end
        accepted || break                               # no decrease: stop, conv stays false
    end
    Nx = p.Nx
    rho, rhoa, m = unpack_state(Z[1:3Nx], Nx)
    T = Z[end]
    raw_norm, defl_norm, η, d_hom = residual_norms(Z, p; kw...)
    return (Z=Z, rho=rho, rhoa=rhoa, m=m, T=T, c=2*pi / qc / T, conv=conv, raw_norm=raw_norm, defl_norm=defl_norm,
            eta=η, dist_hom=d_hom)
end

# ==================================================================================================================
# Continuation
# ==================================================================================================================

# Natural continuation in a parameter s from s_start towards s_target, in steps of `step`; after the first failure it
# bisects between the last success and the failure. solve(s, last) returns the solution at s (from the last accepted
# one) or nothing. It stops at s_target, or once the bracket is below min_step (scan_rule: once the bisection step
# itself is below min_step, the rule of bifurcation_scan). Returns the accepted (s, solution) pairs.
function bisect_continuation(solve, s_start, sol_start, s_target; step, min_step, scan_rule=false)
    dir = sign(s_target - s_start)
    out = Tuple{Float64,Any}[]
    dir == 0 && return out
    # hi: last accepted s (prev its solution); lo: first failed s (s_target until a failure); bracket: a failure seen
    hi, prev, lo, bracket = s_start, sol_start, s_target, false
    while abs(hi - s_target) > min_step
        if bracket                                       # bisection between hi and lo
            !scan_rule && abs(hi - lo) <= min_step && break
            trial = 0.5 * (hi + lo)
            scan_rule && abs(trial - hi) <= min_step && break
        else                                             # full step, not past s_target
            trial = dir > 0 ? min(hi + dir * step, s_target) : max(hi + dir * step, s_target)
            abs(trial - hi) <= min_step && break
        end
        sol = solve(trial, prev)
        if sol === nothing                               # failure: it becomes the far end of the bracket
            lo, bracket = trial, true
        else
            push!(out, (trial, sol)); hi, prev = trial, sol
        end
    end
    return out
end

# Steady branch in φa from res_start (at p_start.phia) towards phia_target
function continue_in_phia(res_start, p_start::Params, ϕ; phia_target, step0=0.01, min_step=1e-4, tol=1e-10, maxiter=50)
    function solve(ϕa, prev)
        p = Params(p_start.Nx, p_start.Lx, p_start.Pe, ϕ, ϕa; pin_idx=p_start.pin_idx)
        Z, ok, it, resn = newton_aug(prev.U, p; tol=tol, maxiter=maxiter, mfix=prev.m[1])
        return ok ? steady_result(Z, p; ok=ok, it=it, resn=resn) : nothing
    end
    steps = bisect_continuation(solve, p_start.phia, res_start, phia_target; step=step0, min_step=min_step)
    return Any[(phia=p_start.phia, res=res_start); [(phia=s, res=r) for (s, r) in steps]]
end

# Travelling-wave branch in δ (φa = ϕac + ϕ1 δ²), from res_start at delta_start down to delta_target
function continue_in_delta(res_start, p_start::Params, qc, ϕ, ϕac, ϕ1; delta_start=0.2, delta_target=0.01, step0=0.01,
                           min_step=1e-4, tol=1e-10, maxiter=50, defl_power=1.0, defl_shift=1.0)
    function solve(δ, prev)
        p = Params(p_start.Nx, p_start.Lx, p_start.Pe, ϕ, ϕac + ϕ1 * δ^2; pin_idx=p_start.pin_idx)
        r = try
            solve_tw(copy(prev.Z), p; tol=tol, maxiter=maxiter, defl_power=defl_power, defl_shift=defl_shift, q_override=qc)
        catch                                            # any error counts as a failed step (bisection follows)
            nothing
        end
        return (r !== nothing && r.conv) ? r : nothing
    end
    steps = bisect_continuation(solve, delta_start, res_start, delta_target; step=step0, min_step=min_step)
    return Any[(delta=delta_start, phia=p_start.phia, res=res_start); [(delta=δ, phia=ϕac + ϕ1 * δ^2, res=r) for (δ, r) in steps]]
end

# Steady branch of the stationary bifurcation with wavenumber qc (at ϕac), continued from ε = eps_max down to eps_min
# (φa = ϕac + ε ϕ1), with the weakly nonlinear amplitudes on a finer grid of ε
function bifurcation_scan(ϕ, Pe, Lx, Nx, qc, ϕac, ϕ1; eps_min=0.0, eps_max=0.035, n_eps=100, n_eps_WNL=10000,
                          tol=1e-10, maxiter=50, min_step=1e-10)
    # grid x_j = (j-1) Lx/Nx (a float range: its length is Nx for Lx = 10 and the Nx used, not for every Lx/Nx)
    x = 0:Lx/Nx:(Lx - Lx/Nx)
    phia_WNL = [ϕac + eps * ϕ1 for eps in range(0.0, eps_max; length=n_eps_WNL)]
    # weakly nonlinear (Δρ, Δρa, Δm) with δ = √ε, so that φa = ϕac + δ² ϕ1
    wnl = [wnl_stat_amplitudes(x, Pe, qc, ϕ, ϕac, ϕac + eps * ϕ1, eps^0.5, ϕ1) for eps in range(0.0, eps_max; length=n_eps_WNL)]
    # first solution at eps_max, from the weakly nonlinear one
    z = [weaklynonlinear_stat(xi, Pe, qc, ϕ, ϕac, ϕac + eps_max * ϕ1, eps_max^0.5, ϕ1) for xi in x]
    rho0 = real([zi[1] for zi in z]); rhoa0 = real([zi[2] for zi in z]); m0 = real([zi[3] for zi in z])
    enforce_masses!(rho0, rhoa0, ϕ, ϕac + eps_max * ϕ1)
    p0 = Params(Nx, Lx, Pe, ϕ, ϕac + eps_max * ϕ1; pin_idx=1)
    Z, ok, _, _ = newton_aug(pack_state(rho0, rhoa0, m0), p0; tol=tol, maxiter=maxiter, mfix=m0[1])
    @assert ok "initial Newton solve failed at eps = $eps_max"
    start = steady_result(Z, p0)
    function solve(eps, prev)
        p = Params(Nx, Lx, Pe, ϕ, ϕac + eps * ϕ1; pin_idx=1)
        Z, ok, _, _ = newton_aug(prev.U, p; tol=tol, maxiter=maxiter, mfix=prev.U[2*Nx + 1])   # pin m[1] of prev
        return ok ? steady_result(Z, p) : nothing
    end
    steps = bisect_continuation(solve, eps_max, start, eps_min; step=(eps_max - eps_min) / n_eps, min_step=min_step,
                                scan_rule=true)
    # all accepted solutions, ordered by increasing ε
    sols = reverse([(phia=ϕac + eps_max * ϕ1, res=start); [(phia=ϕac + e * ϕ1, res=r) for (e, r) in steps]])
    amp = [amplitudes(s.res) for s in sols]
    return (phia=[s.phia for s in sols], drho=getindex.(amp, 1), drhoa=getindex.(amp, 2), dm=getindex.(amp, 3),
            phia_wnl=phia_WNL, drho_wnl=getindex.(wnl, 1), drhoa_wnl=getindex.(wnl, 2), dm_wnl=getindex.(wnl, 3))
end

# Pseudo-arclength continuation in (Z, φa) from the last two solutions sol0, sol1 (named tuples (phia, res)), which
# goes around folds. residual(Z, φa) and jacobian(Z, φa) give F and dF/dZ, φa enters row dphia_row as -1, and
# result(Z, φa, iterations, residual) builds the stored solution. skip: an entry of Z left out of the tangent and
# extrapolated by the secant instead (the period T of a travelling wave, which diverges where c -> 0); pinv_fallback:
# use the minimum-norm step where the bordered Jacobian is singular. sparse_solve: factorise the bordered Jacobian as a
# sparse matrix (the same linear system; much faster on fine grids) instead of a dense one.
function continue_arclength_generic(sol0, sol1, residual, jacobian, result; dphia_row, ds=0.01, n_steps=100, tol=1e-10,
                                    maxiter=30, skip=nothing, pinv_fallback=false, sparse_solve=false)
    sols = Any[(phia=sol0.phia, res=sol0.res), (phia=sol1.phia, res=sol1.res)]
    Zprev, ϕaprev = sol0.res.Z, sol0.phia
    Zcur, ϕacur = sol1.res.Z, sol1.phia
    M = length(Zcur)
    for step in 1:n_steps
        # secant tangent (tZ, tϕa) through the last two solutions, unit norm
        dZ = Zcur .- Zprev
        dϕa = ϕacur - ϕaprev
        dZ_dir = copy(dZ); skip === nothing || (dZ_dir[skip] = 0.0)
        nrm = sqrt(norm(dZ_dir)^2 + dϕa^2)
        tZ = dZ_dir ./ nrm
        tϕa = dϕa / nrm
        # predictor: a step ds along the tangent (the skipped entry: one more previous increment)
        Z = Zcur .+ ds .* tZ
        skip === nothing || (Z[skip] = Zcur[skip] + dZ[skip])
        ϕa = ϕacur + ds * tϕa
        ok = false; Fnorm = Inf; it_used = 0
        for k in 1:maxiter                               # corrector: Newton on (F = 0, arclength condition)
            F = residual(Z, ϕa)
            arc = dot(tZ, Z .- Zcur) + tϕa * (ϕa - ϕacur) - ds     # projection of the step on the tangent = ds
            Fnorm = norm(vcat(F, arc))
            if Fnorm < tol
                ok = true; it_used = k - 1
                break
            end
            dFdphia = zeros(M); dFdphia[dphia_row] = -1.0
            if sparse_solve  # bordered Jacobian [dF/dZ  dF/dφa; tZᵀ  tϕa] (sparse)
                As = [sparse(jacobian(Z, ϕa)) sparse_col(dFdphia); sparse_row(tZ) sparse([tϕa;;])]
                delta = lu(As) \ (-vcat(F, arc))
                Z .+= delta[1:M]; ϕa += delta[M+1]; it_used = k
                continue
            end
            # bordered Jacobian [dF/dZ  dF/dφa; tZᵀ  tϕa] (dense)
            A = zeros(M + 1, M + 1)
            A[1:M, 1:M] = Matrix(jacobian(Z, ϕa))
            A[1:M, M+1] = dFdphia
            A[M+1, 1:M] .= tZ
            A[M+1, M+1] = tϕa
            local delta
            try
                delta = A \ (-vcat(F, arc))
            catch e
                # (dense LU throws only for an exactly zero pivot; a nearly singular A gives no exception)
                (pinv_fallback && e isa LinearAlgebra.SingularException) || rethrow(e)
                try
                    delta = pinv(A) * (-vcat(F, arc))
                    any(!isfinite, delta) && break
                catch
                    break                                # (ends the corrector: ok = false, the branch stops)
                end
            end
            Z .+= delta[1:M]
            ϕa += delta[M+1]
            it_used = k
        end
        ok || break                                      # no convergence: stop the branch here
        Zprev, ϕaprev = Zcur, ϕacur
        Zcur, ϕacur = Z, ϕa
        push!(sols, (phia=ϕacur, res=result(Zcur, ϕacur, it_used, Fnorm)))
    end
    return sols
end

# Steady branch (around a fold), with the pin m[1] = mfix
function continue_arclength(sol0, sol1, p::Params, ϕ, mfix; ds=0.01, n_steps=100, tol=1e-10, maxiter=30)
    P(ϕa) = Params(p.Nx, p.Lx, p.Pe, ϕ, ϕa; pin_idx=p.pin_idx)
    # undeflated system; φa enters only the row mean(ρa) - φa (row 3Nx + 2)
    return continue_arclength_generic(sol0, sol1, (Z, ϕa) -> residual_aug(Z, P(ϕa), mfix),
                                      (Z, ϕa) -> jacobian_aug(Z, P(ϕa), mfix),
                                      (Z, ϕa, it, resn) -> steady_result(Z, P(ϕa); it=it, resn=resn);
                                      dphia_row=3p.Nx + 2, ds=ds, n_steps=n_steps, tol=tol, maxiter=maxiter, sparse_solve=true)
end

# Travelling-wave branch (around folds and through c = 0)
function continue_arclength_tw(sol0, sol1, p::Params, ϕ; qc, ds=0.01, n_steps=100, tol=1e-10, maxiter=30)
    N = p.Nx
    P(ϕa) = Params(N, p.Lx, p.Pe, ϕ, ϕa; pin_idx=p.pin_idx)
    # conv = true because only converged points are stored; raw_norm, defl_norm, eta, dist_hom are 0.0 placeholders
    # here, not computed values
    function result(Z, ϕa, it, resn)
        rho, rhoa, m = unpack_state(Z[1:3N], N)
        return (Z=Z, rho=rho, rhoa=rhoa, m=m, T=Z[end], c=2*pi / qc / Z[end], conv=true, raw_norm=0.0, defl_norm=0.0,
                eta=0.0, dist_hom=0.0)
    end
    # φa enters row 3N + 2 (mean(ρa) - φa); the period T = Z[3N + 4] is the skipped entry
    return continue_arclength_generic(sol0, sol1, (Z, ϕa) -> residual_tw_raw(Z, P(ϕa); q_override=qc),
                                      (Z, ϕa) -> jacobian_tw_raw(Z, P(ϕa); q_override=qc), result;
                                      dphia_row=3N + 2, ds=ds, n_steps=n_steps, tol=tol, maxiter=maxiter, skip=3N + 4,
                                      pinv_fallback=true)
end
