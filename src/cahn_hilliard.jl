# Cahn–Hilliard notebook: Newton steady profiles against the weakly nonlinear steady state (φ0 = 0.1 and 0.5), bifurcation
# diagrams of the supercritical (φ0 = 0.1) and subcritical (φ0 = 0.5) cases, and the fold near the tricritical point
# (φ0 ≈ 0.446).
include(joinpath(@__DIR__, "common.jl"))
using SparseArrays

# ==================================================================================================================
# Weakly nonlinear estimates of the steady states
# ==================================================================================================================

# Cubic amplitude equation, gamma = gammaC + δ^2 gamma1:
#   phi* = phi0 - 2δ sqrt(gamma1 kC^2 / (3kC^2 - 6phi0^2)) cos(kC x),
# or the homogeneous state phi0 where that amplitude is not real.
function wnl_profile(x, phi0, gamma1, δ, kC)
    A2 = gamma1 * kC^2 / (3 * kC^2 - 6 * phi0^2)  # |A|²
    return A2 >= 0 ? phi0 .- 2 * δ * sqrt(A2) .* cos.(kC .* x) : fill(phi0, length(x))
end

# Its branch (gamma, Δphi/phi0) over gammaVec, from (gammaC, 0) to where it exists. It is stable (supercritical) if
# 3kC^2 > 6phi0^2 and unstable (subcritical) otherwise.
function wnl_branch(gammaVec, x, phi0, δ, kC)
    gammaC = 3 * phi0^2 + kC^2  # linear onset
    pts = [(gammaC, 0.0)]
    for gamma in gammaVec
        gamma1 = (gamma - gammaC) / δ^2  # gamma = gammaC + δ² gamma1
        if gamma1 * kC^2 / (3 * kC^2 - 6 * phi0^2) > 0  # |A|² > 0: amplitude real
            phi = wnl_profile(x, phi0, gamma1, δ, kC)
            push!(pts, (gamma, (maximum(phi) - minimum(phi)) / phi0))
        end
    end
    sort!(pts)
    return first.(pts), last.(pts), 3 * kC^2 > 6 * phi0^2  # supercritical if the cubic coefficient is positive
end

# Quintic amplitude equation near phi0^c, gamma = gammaC + δ^4 gamma1:
#   15 X^2 - 2 mu1 X - 2 gamma1 kC^2 = 0,  X = |A|^2,  Δphi/phi0 = 4δ sqrt(X)/phi0.
# Returns the upper (stable) root and the lower (unstable) one, which only exists for gamma1 <= 0.
function wnl_fold(gamma1, mu1, δ, kC, phi0)
    sq = sqrt(max(mu1^2 + 30 * gamma1 * kC^2, 0))  # (clamped at 0 below the fold)
    upper = 4 * δ * sqrt((mu1 + sq) / 15) / phi0
    lower = gamma1 <= 0 ? 4 * δ * sqrt(max((mu1 - sq) / 15, 0)) / phi0 : NaN
    return upper, lower
end

# ==================================================================================================================
# Steady states (Newton with deflation, gamma sweeps)
# ==================================================================================================================

# Second-order finite-difference Laplacian with periodic boundary conditions
function laplacian(N, dx)
    A = spdiagm(-1 => ones(N-1), 0 => -2 .* ones(N), 1 => ones(N-1))
    A[1, end] = 1.0  # periodic wrap
    A[end, 1] = 1.0
    return A / dx^2
end

# Steady state at gamma = gammaC + gamma1 δ^2 by Newton (the numerical continuation's solver), to compare with the
# weakly nonlinear profile: from the weakly nonlinear profile if supercritical (the small branch), from a
# large-amplitude cosine if subcritical (the large stable branch; the weakly nonlinear one does not exist above gammaC)
function steady_profile(phi0; L=10.0, N=256, δ=0.1, gamma1=1.0)
    dx = L / N
    x  = ((0:N-1) .+ 0.5) .* dx  # cell centres
    Δ  = laplacian(N, dx)
    kC = 2π / L
    gamma = 3 * phi0^2 + kC^2 + gamma1 * δ^2
    guess = 3kC^2 > 6phi0^2 ? wnl_profile(x, phi0, gamma1, δ, kC) : phi0 .+ (100 * phi0^2) .* cos.(kC .* x)
    guess = guess .+ (phi0 - mean(guess))  # exact mean phi0
    z, ok = newton(initial_z(guess, Δ, gamma), Δ, gamma, phi0; maxiter=100)
    ok || error("Newton did not converge for phi0 = $phi0")
    R, _ = residual_and_jacobian(z, Δ, gamma, phi0)
    return (; x=collect(x), phi=z[1:N], mu=z[end], phi0, gamma, gamma1, δ, L, N, converged=ok, residual=norm(R) / sqrt(N + 1))
end

# Steady states phi^3 - gamma phi - Δ phi = mu with mean(phi) = phi0, unknowns z = [phi; mu]
function residual_and_jacobian(z, Δ, gamma, phi0)
    N = length(z) - 1
    phi = z[1:N]
    # residual: the equation (with the multiplier mu) and the mass constraint
    R = vcat(phi .^ 3 .- gamma .* phi .- Δ * phi .- z[N+1], mean(phi) - phi0)
    J11 = spdiagm(0 => 3 .* phi .^ 2 .- gamma) - Δ
    J = [J11 -ones(N); (1/N) .* ones(1, N) 0.0]  # bordered Jacobian
    return R, J
end

# Distance between phi and phi_ref, minimised over the translations of phi_ref (and the difference vector)
function min_shift_distance(phi, phi_ref)
    dmin, dvmin = Inf, phi .- phi_ref
    for k in 0:length(phi)-1  # all grid translations
        dv = phi .- circshift(phi_ref, k)
        norm(dv) < dmin && ((dmin, dvmin) = (norm(dv), dv))
    end
    return dmin, dvmin
end

# Deflation factor D = prod_j (1/d_j^p + alpha) over the known solutions and its gradient in phi
function deflation_factor(phi, sols, p, alpha)
    d, dv = zeros(length(sols)), zeros(length(phi), length(sols))
    for (j, s) in enumerate(sols)
        dj, dv[:, j] = min_shift_distance(phi, s.phi)  # distance to the nearest translate
        d[j] = max(dj, 1e-14)  # avoid division by zero
    end
    g = 1 ./ d .^ p .+ alpha  # factor of each known solution
    D = prod(g)
    # dD = Σ_j (D/g_j) dg_j, dg_j = -p d_j^(-p-2) (phi - translated s_j)
    dD = sum((D / g[j]) .* (-p .* dv[:, j] ./ d[j]^(p+2)) for j in eachindex(sols); init=zeros(length(phi)))
    return D, dD
end

# Deflated residual: the PDE equations are multiplied by D, the mean constraint is not
function deflated_residual(z, Δ, gamma, phi0, sols, p, alpha)
    R, J = residual_and_jacobian(z, Δ, gamma, phi0)
    isempty(sols) && return R, R, J
    N = length(z) - 1
    D, dD = deflation_factor(z[1:N], sols, p, alpha)
    Jdef = Matrix(J)
    Jdef[1:N, :] .*= D  # PDE rows scaled by D
    Jdef[1:N, :] .+= R[1:N] * transpose(vcat(dD, 0.0))   # d(D R1)/dz = D dR1/dz + R1 (dD/dz)'
    return R, vcat(D .* R[1:N], R[end]), Jdef
end

# Newton with backtracking line search; with known solutions `sols` the residual is deflated so the iteration cannot
# converge to them again. Returns (z, converged).
function newton(z, Δ, gamma, phi0; sols=[], p=1, alpha=1.0, tol=1e-10, maxiter=25)
    for _ in 1:maxiter
        R, Rdef, J = deflated_residual(z, Δ, gamma, phi0, sols, p, alpha)
        rms(v) = norm(v) / sqrt(length(v))  # RMS norm
        rms(R) < tol && rms(Rdef) < tol && return z, true      # a true solution, not a known one

        dz = try -(J \ Rdef) catch; return z, false end  # Newton step on the deflated system
        all(isfinite, dz) || return z, false

        # Armijo backtracking on the deflated residual; if it fails, take the best decreasing trial
        merit0 = norm(Rdef)
        best_z, best_merit, accepted = nothing, merit0, false  # best trial so far, its merit, accepted?
        α = 1.0
        for _ in 1:10
            ztrial = z .+ α .* dz
            merit = norm(deflated_residual(ztrial, Δ, gamma, phi0, sols, p, alpha)[2])
            if isfinite(merit)
                merit < best_merit && ((best_z, best_merit) = (ztrial, merit))
                if merit <= (1 - 1e-4 * α) * merit0  # Armijo sufficient decrease
                    z, accepted = ztrial, true
                    break
                end
            end
            α *= 0.5
        end
        accepted || (best_z === nothing ? (return z, false) : (z = best_z))  # else the best decreasing trial; none: give up
    end
    return z, false  # not converged within maxiter
end

# Starting point z = [phi; mu] for Newton from a guess of phi
initial_z(phi, Δ, gamma) = vcat(phi, mean(phi .^ 3 .- gamma .* phi .- Δ * phi))

# Supercritical sweep: one Newton solve per gamma, from phi0 - 2δ cos(kC x) below gammaC and from a large-amplitude
# guess above it. Returns every steady state.
function newton_sweep(phi0, gammaVec; L=10.0, N=256, δ=0.1)
    dx = L / N
    x  = ((0:N-1) .+ 0.5) .* dx
    Δ  = laplacian(N, dx)
    gammaC = 3 * phi0^2 + (2π / L)^2
    phi_low = phi0 .- 2δ .* cos.(2π .* x ./ L)  # small-amplitude guess
    phi_low .-= mean(phi_low) .- phi0

    phi_all = zeros(N, length(gammaVec))
    for (j, gamma) in enumerate(gammaVec)
        guess = gamma > gammaC ? phi0 .- 100 .* cos.(2π .* x ./ L) : phi_low  # amplitude 100 above gammaC
        # tol = 1e-18 is below round-off: newton returns after maxiter (or when no step decreases R); the flag is ignored
        z, _ = newton(initial_z(guess, Δ, gamma), Δ, gamma, phi0; tol=1e-18, maxiter=100)
        phi_all[:, j] = z[1:N]
    end
    amp = [(maximum(c) - minimum(c)) / phi0 for c in eachcol(phi_all)]  # Δphi/phi0
    return (; gammaVec=collect(gammaVec), amp, phi_all, x=collect(x), phi0, gammaC, δ, L, N)
end

# Subcritical sweep with deflation: from the largest gamma down, every gamma is solved from numGuesses guesses,
# deflating the solutions already found. The guesses are the middle (unstable) solution of the previous gamma (or a
# small cosine), a large-amplitude cosine once one solution is known, and phi0. Returns (gamma, Δphi/phi0) of every
# distinct steady state.
function deflation_sweep(phi0, gammaVec; L=10.0, N=256, δ=0.05, numGuesses=10)
    dx = L / N
    x  = (0:N-1) .* dx
    Δ  = laplacian(N, dx)
    gammaC = 3 * phi0^2 + (2π / L)^2
    amp_of(phi) = (maximum(phi) - minimum(phi)) / phi0

    gamma_all, amp_all = Float64[], Float64[]
    solWNL = phi0 .+ 0.1 .* cos.(2π .* x ./ L)  # small cosine: the first guess at the largest gamma
    for gamma in reverse(gammaVec)  # from the largest gamma down
        sols = []  # solutions found at this gamma
        for g in 1:numGuesses
            # last guess phi0; with one solution known a large cosine (amplitude 100 phi0²); otherwise solWNL
            phi_guess = g == numGuesses ? phi0 .* ones(N) :
                        length(sols) == 1 ? phi0 .+ (100 * phi0^2) .* cos.(2π .* x ./ L) : solWNL
            phi_guess .+= phi0 - mean(phi_guess)  # exact mean (in place: shifts solWNL itself when it is the guess)

            z, ok = newton(initial_z(phi_guess, Δ, gamma), Δ, gamma, phi0; sols=sols)  # deflated against the solutions found
            ok || continue
            # keep it only if it is not a translation of a known solution
            all(s -> min_shift_distance(z[1:N], s.phi)[1] >= 1e-4, sols) && push!(sols, (phi=z[1:N], mu=z[end]))
        end
        append!(gamma_all, fill(gamma, length(sols)))  # one (gamma, amplitude) per solution
        append!(amp_all, amp_of.(s.phi for s in sols))

        # first guess of the next gamma: the middle (unstable) solution if three were found, else the small cosine
        solWNL = length(sols) == 3 ? sols[sortperm(amp_of.(s.phi for s in sols))[2]].phi :
                                     phi0 .+ 0.1 .* cos.(2π .* x ./ L)
    end
    return (; gamma_all, amp_all, gammaVec=collect(gammaVec), x=collect(x), phi0, gammaC, δ, L, N)
end

# ==================================================================================================================
# Fold of the subcritical branch near the tricritical phi0^c = sqrt(3kC^2/6): weakly nonlinear seed, natural
# continuation in gamma down to the fold and pseudo-arclength continuation around it
# ==================================================================================================================

# Parameters. phi0^2 = (3kC^2 + mu1 δ^2)/6 is O(δ^2) above phi0^c and gamma = gammaC + δ^4 gamma1. The derivatives are
# spectral: a finite-difference Laplacian shifts kC^2 by more than the whole subcritical window.
function fold_setup(; L=10.0, N=256, δ=0.1, mu1=1.0)
    x  = collect((0:N-1) .* (L / N))  # x_j = (j-1) L/N
    kC = 2π / L
    phi0   = sqrt((3 * kC^2 + mu1 * δ^2) / 6)  # O(δ²) above the tricritical phi0^c
    gammaC = 3 * phi0^2 + kC^2
    gamma1_fold = -mu1^2 / (30 * kC^2)  # weakly nonlinear fold: -mu1²/(30 kC²)

    kvec = 2π / L .* collect(fftfreq(N, N))  # wavenumbers in FFT order
    k1vec = copy(kvec); k1vec[N ÷ 2 + 1] = 0.0          # no Nyquist mode in the first derivative
    Id = Matrix{Float64}(I, N, N)
    D2 = real.(ifft(-(kvec .^ 2) .* fft(Id, 1), 1))  # spectral second derivative
    D1 = real.(ifft((im .* k1vec) .* fft(Id, 1), 1))  # spectral first derivative

    # plot window in gamma: 1.5 times the weakly nonlinear fold below gammaC, 0.15 δ⁴ above it
    return (; L, N, x, kC, δ, mu1, phi0, gammaC, gamma1_fold,
              gamma_min = gammaC + 1.5 * gamma1_fold * δ^4, gamma_max = gammaC + 0.15 * δ^4,
              D1, D2, cosC = cos.(kC .* x), sinC = sin.(kC .* x))
end

# Steady states phi^3 - gamma phi - D2 phi = mu + nu sin(kC x), mean(phi) = phi0, <phi, sin(kC x)> = 0 (the phase
# condition removes the translation mode; nu is zero at a solution). z = [phi; mu; nu].
function residual_fold(P, z, gamma)
    N = P.N; phi = z[1:N]
    return vcat(phi .^ 3 .- gamma .* phi .- P.D2 * phi .- z[N+1] .- z[N+2] .* P.sinC,
                mean(phi) - P.phi0,  # mass
                (2 / N) * dot(phi, P.sinC))  # phase condition
end

function jacobian_fold(P, z, gamma)
    N = P.N; phi = z[1:N]
    J = zeros(N + 2, N + 2)
    J[1:N, 1:N] = Diagonal(3 .* phi .^ 2 .- gamma) - P.D2  # dF/dphi
    J[1:N, N+1] .= -1.0  # dF/dmu
    J[1:N, N+2]  = -P.sinC  # dF/dnu
    J[N+1, 1:N] .= 1 / N  # mass row
    J[N+2, 1:N]  = (2 / N) .* P.sinC  # phase row
    return J
end

# plain Newton on the bordered system; converged when the largest residual is below tol
function newton_fold(P, z, gamma; tol=1e-11, maxiter=30)
    for _ in 1:maxiter
        F = residual_fold(P, z, gamma)
        norm(F, Inf) < tol && return z, true
        z = z - jacobian_fold(P, z, gamma) \ F
    end
    return z, norm(residual_fold(P, z, gamma), Inf) < tol
end

# Δphi/phi0 of a solution z
amp_of(P, z) = (maximum(z[1:P.N]) - minimum(z[1:P.N])) / P.phi0

# Weakly nonlinear seed on the lower (unstable) branch: phi0 + a cos(kC x) - (phi0 a^2/(2kC^2)) cos(2kC x),
# a = 2δ sqrt(X), X the lower root of the quintic amplitude equation
function wnl_seed(P, gamma1)
    (; x, kC, δ, mu1, phi0) = P
    X = (mu1 - sqrt(mu1^2 + 30 * gamma1 * kC^2)) / 15  # lower root
    a = 2 * δ * sqrt(X)
    phi = phi0 .+ a .* P.cosC .- (phi0 * a^2 / (2 * kC^2)) .* cos.(2kC .* x)
    gamma = P.gammaC + gamma1 * δ^4
    mu = mean(phi .^ 3 .- gamma .* phi .- P.D2 * phi)  # mu from the mean of the equation
    return vcat(phi, mu, 0.0), gamma  # nu = 0
end

# Natural continuation from (z0, gamma0) towards gamma_target, bisecting the step when Newton fails (or falls back onto
# phi0). There is no solution beyond the fold, so it stops there.
function continue_in_gamma(P, z0, gamma0; gamma_target, step0, min_step)
    dir  = sign(gamma_target - gamma0)  # towards the target
    sols = Any[(gamma=gamma0, z=z0)]
    g_hi, z_hi = gamma0, z0  # last accepted point
    g_lo, have_bracket = gamma_target, false  # failure bracket (none yet)
    while abs(g_hi - gamma_target) > min_step
        if have_bracket
            abs(g_hi - g_lo) <= min_step && break  # bracket below min_step: stop (the fold)
            g_trial = 0.5 * (g_hi + g_lo)  # bisect between success and failure
        else
            g_trial = dir > 0 ? min(g_hi + step0, gamma_target) : max(g_hi - step0, gamma_target)
        end
        z, ok = newton_fold(P, z_hi, g_trial)
        ok && amp_of(P, z) < 0.5 * amp_of(P, z_hi) && (ok = false)  # fell onto phi0: a failure
        if ok
            push!(sols, (gamma=g_trial, z=z))
            g_hi, z_hi = g_trial, z
        else
            g_lo, have_bracket = g_trial, true
        end
    end
    return sols
end

# Pseudo-arclength continuation from two points: secant predictor and Newton corrector on the bordered system
# [J dF/dgamma1; t_z'/N t_g1]. Arclength is measured in (z, gamma1) with an RMS norm on z, since gamma itself only
# moves by ~1e-5.
function continue_arclength(P, sol0, sol1; ds, n_steps, gamma_stop=Inf, tol=1e-11, maxiter=30)
    (; N, gammaC, δ) = P
    wdot(az, ag, bz, bg) = dot(az, bz) / N + ag * bg  # RMS on z plus gamma1
    sols = Any[sol0, sol1]
    zprev, gprev = sol0.z, (sol0.gamma - gammaC) / δ^4  # g = gamma1 = (gamma - gammaC)/δ⁴
    zcur,  gcur  = sol1.z, (sol1.gamma - gammaC) / δ^4

    for step in 1:n_steps
        # unit secant tangent in the weighted norm
        dz, dg = zcur .- zprev, gcur - gprev
        nrm = sqrt(wdot(dz, dg, dz, dg))
        tz, tg = dz ./ nrm, dg / nrm

        z, g = zcur .+ ds .* tz, gcur + ds * tg  # predictor
        ok = false
        for _ in 1:maxiter
            gamma = gammaC + g * δ^4
            F   = residual_fold(P, z, gamma)
            arc = wdot(tz, tg, z .- zcur, g - gcur) - ds  # arclength condition
            if norm(vcat(F, arc), Inf) < tol
                ok = true; break
            end
            M = N + 2
            A = zeros(M + 1, M + 1)
            A[1:M, 1:M] = jacobian_fold(P, z, gamma)
            A[1:M, M+1] = vcat(-z[1:N], 0.0, 0.0) .* δ^4      # dF/dgamma1
            A[M+1, 1:M] = tz ./ N
            A[M+1, M+1] = tg
            d = A \ (-vcat(F, arc))  # Newton correction of (z, gamma1)
            z = z .+ d[1:M]
            g += d[M+1]
        end
        if !ok
            @printf("arclength step %d failed to converge near gamma1 = %.5f\n", step, g)
            break
        end
        zprev, gprev, zcur, gcur = zcur, gcur, z, g
        push!(sols, (gamma=gammaC + g * δ^4, z=z))
        gammaC + g * δ^4 > gamma_stop && break  # past gamma_stop: done
    end
    return sols
end

# Linear stability: smallest eigenvalue of diag(3phi^2 - gamma) - D2 on mean-zero perturbations orthogonal to the
# translation mode phi_x (stable if positive)
function min_eig_stability(P, phi, gamma)
    Lin  = Diagonal(3 .* phi .^ 2 .- gamma) - P.D2  # Hessian of the free energy
    dphi = P.D1 * phi  # translation mode
    C = norm(dphi) > 1e-10 ? hcat(ones(P.N), dphi) : ones(P.N, 1)  # constraints: mean zero, ⊥ phi_x
    Q = nullspace(Matrix(C'))  # basis of the allowed perturbations
    return eigmin(Symmetric(Q' * Lin * Q))
end

# The whole branch: WNL seed at gamma1 = -0.002, continuation in gamma down to the fold, arclength around it and up the
# large-amplitude branch. Returns gamma, Δphi/phi0, stability and profile of every point.
function fold_branch(P)
    (; N, gammaC, δ) = P
    z0, gamma0 = wnl_seed(P, -0.002)  # gamma1 = -0.002: near onset
    z_seed, ok = newton_fold(P, z0, gamma0)
    ok || error("WNL seed did not converge")

    # lower (unstable) branch: natural continuation down in gamma to the fold
    lower = continue_in_gamma(P, z_seed, gamma0; gamma_target=P.gamma_min, step0=0.002 * δ^4, min_step=1e-6 * δ^4)
    # arclength from its last two points around the fold and up the large-amplitude branch to gamma_max
    arc = continue_arclength(P, lower[end-1], lower[end]; ds=0.002, n_steps=5000, gamma_stop=P.gamma_max)
    branch = vcat(lower, arc[3:end])  # arc[1:2] are lower[end-1:end]

    gamma = [s.gamma for s in branch]
    amp   = [amp_of(P, s.z) for s in branch]
    stable = [min_eig_stability(P, s.z[1:N], s.gamma) > 0 for s in branch]
    i = argmin(gamma)  # fold: smallest gamma
    @printf("numerical fold: gamma1 = %.5f, amp = %.4f   (WNL: gamma1 = %.5f)\n",
            (gamma[i] - gammaC) / δ^4, amp[i], P.gamma1_fold)
    return (; gamma, amp, stable, phi_all = reduce(hcat, [s.z[1:N] for s in branch]), x = P.x,
              P.phi0, gammaC, δ, P.mu1, P.L, N)
end

# ==================================================================================================================
# Figures: one framed legend above the panels and (a), (b) in the upper left corner of each panel
# ==================================================================================================================

const FS = 18          # ticks, legend
const GFS = 24         # axis labels, (a), (b) and annotations
const LW = 4           # line width

const NUM_STABLE_LABEL   = L"\mathrm{numerical\ continuation\ (stable)}"
const NUM_UNSTABLE_LABEL = L"\mathrm{numerical\ continuation\ (unstable)}"
const WNL_STABLE_LABEL   = L"\mathrm{weakly\ nonlinear\ estimation\ (stable)}"
const WNL_UNSTABLE_LABEL = L"\mathrm{weakly\ nonlinear\ estimation\ (unstable)}"
num_style(stable; lw=LW) = (linewidth=lw, linestyle=(stable ? :solid : :dash), color=:blue)
wnl_style(stable; lw=LW) = (linewidth=lw, linestyle=(stable ? :dashdot : :dash), color=:black)

# Empty panel
paper_axes(tag=""; kw...) = plot(; title=tag, titlelocation=:left, legend=false, guidefontsize=GFS, titlefontsize=GFS,
                                 tickfontsize=FS, kw...)

# Framed legend: a centred grid filled row by row with `ncol` entries per row, each column as wide as its longest label
# (estimated from the label length). entries = [(label, line style), ...]
function legend_panel(entries, figwidth; ncol, fs)
    leg = plot(xlims=(0, 1), ylims=(0, 1), framestyle=:none, legend=false, grid=false)
    # label length without its LaTeX markup
    visible(lab) = length(replace(lab, "\\phi" => "p", "\\mathrm{" => "", "\\ " => " ", "\$" => "", "{" => "", "}" => ""))
    u = 1 / (0.9 * figwidth)              # one pixel, in panel units (the panel spans ~90% of the figure)
    cw = 0.65 * fs * u                    # average character width
    sample = 80u; gap = 10u; spacing = 45u; pad = 25u
    nrow   = cld(length(entries), ncol)
    widths = [sample + gap + cw * visible(lab) for (lab, _) in entries]
    colw   = [maximum(widths[c:ncol:end]) for c in 1:ncol]
    left   = (1 - sum(colw) - spacing * (ncol - 1)) / 2
    for (i, (lab, st)) in enumerate(entries)
        r, c = divrem(i - 1, ncol)  # row and column of entry i
        xi = left + sum(colw[1:c]; init=0.0) + spacing * c
        y  = 0.88 - (r + 0.5) * 0.76 / nrow
        plot!(leg, [xi, xi + sample], [y, y]; st...)
        annotate!(leg, xi + sample + gap, y, txt(lab, fs, :black, :left))
    end
    x0, x1 = left - pad, 1 - left + pad
    plot!(leg, [x0, x1, x1, x0, x0], [0.05, 0.05, 0.95, 0.95, 0.05]; color=:black, linewidth=1)
    return leg, nrow
end

# The panels in one row with the legend above them; `size` is the size of the panels row
function legend_on_top(entries, panels...; size, ncol=length(entries), fs=FS)
    leg, nrow = legend_panel(entries, size[1]; ncol, fs)
    leg_px = 30 + 40 * nrow  # legend height in px
    h = leg_px / (size[2] + leg_px)
    lay = grid(2, 1, heights=[h, 1 - h])  # legend row above the panels row
    lay[2, 1] = grid(1, length(panels))
    fig = plot(leg, panels...; layout=lay, size=(size[1], size[2] + leg_px), margin=8Plots.mm)
    plot!(fig[1]; top_margin=2Plots.mm, bottom_margin=0Plots.mm, left_margin=0Plots.mm, right_margin=0Plots.mm)
    return fig
end

# Homogeneous state phi = phi0: stable below gammaC, unstable above
function plot_homogeneous!(p, gmin, gmax, gammaC)
    plot!(p, [gmin, gammaC], [0.0, 0.0]; num_style(true)...)
    plot!(p, [gammaC, gmax], [0.0, 0.0]; num_style(false)...)
end

# Steady profiles of the numerical continuation against the weakly nonlinear steady state, one panel per phi0. Where |A|² < 0
# (subcritical case above gammaC) the weakly nonlinear analysis predicts only the homogeneous state phi0, which is
# unstable there: wnl_profile then returns that flat profile, drawn in the unstable style.
function profiles_figure(runs)
    panels = map(zip(runs, (L"\mathrm{(a)}", L"\mathrm{(b)}"))) do (r, tag)
        kC = 2π / r.L
        wnl = wnl_profile(r.x, r.phi0, r.gamma1, r.δ, kC)  # weakly nonlinear steady state (or the flat phi0)
        p = paper_axes(tag; xlabel=L"x", ylabel=L"\rho", xlims=(0, r.L))
        plot!(p, r.x, r.phi; num_style(true)...)  # steady state of the numerical continuation (Newton)
        plot!(p, r.x, wnl; wnl_style(3kC^2 > 6r.phi0^2)...)  # stable style if supercritical
    end
    entries = [(NUM_STABLE_LABEL, num_style(true)), (WNL_STABLE_LABEL, wnl_style(true)), (WNL_UNSTABLE_LABEL, wnl_style(false))]
    return legend_on_top(entries, panels...; size=(1400, 520), ncol=2)
end

# Bifurcation diagram of a gamma sweep (gamma, Δphi/phi0 of every steady state found): homogeneous state, patterned
# branch from (gammaC, 0) ordered by amplitude, unstable up to the fold (the last solution at the smallest gamma) and
# stable after it, and the weakly nonlinear branch
function bifurcation_panel(gammas, amps, x, phi0, δ, L, tag)
    kC = 2π / L
    gammaC = 3 * phi0^2 + kC^2
    gmin, gmax = extrema(gammas)
    p = paper_axes(tag; xlabel=L"\gamma", ylabel=L"\Delta\rho/\phi", xlims=(gmin, gmax))
    plot_homogeneous!(p, gmin, gmax, gammaC)

    pat = amps .> 1e-5  # patterned states only
    i = sortperm(amps[pat])  # by amplitude along the branch
    g, a = vcat(gammaC, gammas[pat][i]), vcat(0.0, amps[pat][i])  # from the bifurcation point
    k = findlast(==(minimum(g[2:end])), g)  # the fold: last point at the smallest gamma
    k > 2 && plot!(p, g[1:k], a[1:k]; num_style(false)...)  # unstable up to the fold
    plot!(p, g[(k > 2 ? k : 1):end], a[(k > 2 ? k : 1):end]; num_style(true)...)  # stable after it (all if no fold)

    gw, aw, stable = wnl_branch(gammas, x, phi0, δ, kC)
    plot!(p, gw, aw; wnl_style(stable)...)
    return p
end

function bifurcation_figure(sup, sub)
    p1 = bifurcation_panel(sup.gammaVec, sup.amp, sup.x, sup.phi0, sup.δ, sup.L, L"\mathrm{(a)}")
    p2 = bifurcation_panel(sub.gamma_all, sub.amp_all, sub.x, sub.phi0, sub.δ, sub.L, L"\mathrm{(b)}")
    entries = [(NUM_STABLE_LABEL, num_style(true)), (WNL_STABLE_LABEL, wnl_style(true)),
               (NUM_UNSTABLE_LABEL, num_style(false)), (WNL_UNSTABLE_LABEL, wnl_style(false))]
    return legend_on_top(entries, p1, p2; size=(1400, 560), ncol=2)
end

# Fold diagram: continued branch (stability from the eigenvalues), homogeneous state, weakly nonlinear branches
# (thinner, so the continuation underneath stays visible) and the weakly nonlinear fold gamma^F
function fold_figure(br, P)
    (; gammaC, δ, kC, mu1, phi0, gamma_min, gamma_max, gamma1_fold) = P
    # gamma only moves by O(1e-5): a grid line every 5e-6, a label every 1e-5, no offset
    xt = collect(ceil(gamma_min / 5e-6) * 5e-6 : 5e-6 : gamma_max)
    xl = [iseven(round(Int, t / 5e-6)) ? latexstring(@sprintf("%.5f", t)) : "" for t in xt]
    p = paper_axes(; xlabel=L"\gamma", ylabel=L"\Delta\rho/\phi", xlims=(gamma_min, gamma_max), xticks=(xt, xl))

    # branch from the bifurcation point (gammaC, 0), taken as unstable
    gb, ab, sb = vcat(gammaC, br.gamma), vcat(0.0, br.amp), vcat(false, br.stable)
    plot!(p, gb, ifelse.(sb, NaN, ab); num_style(false)...)  # unstable points
    plot!(p, gb, ifelse.(sb, ab, NaN); num_style(true)...)  # stable points
    plot_homogeneous!(p, gamma_min, gamma_max, gammaC)

    g1 = range(gamma1_fold, 0.15, length=400)  # from the fold to gamma1 = 0.15
    w = [wnl_fold(g, mu1, δ, kC, phi0) for g in g1]
    plot!(p, gammaC .+ g1 .* δ^4, first.(w); wnl_style(true; lw=LW - 1)...)
    plot!(p, gammaC .+ g1 .* δ^4, last.(w); wnl_style(false; lw=LW - 1)...)

    vline!(p, [gammaC + gamma1_fold * δ^4]; color=:black, linestyle=:dot, linewidth=2)
    annotate!(p, gammaC + (gamma1_fold - 0.012) * δ^4, 0.18, txt(L"\gamma^F", GFS, :black))

    entries = [(NUM_STABLE_LABEL, num_style(true)), (WNL_STABLE_LABEL, wnl_style(true; lw=LW - 1)),
               (NUM_UNSTABLE_LABEL, num_style(false)), (WNL_UNSTABLE_LABEL, wnl_style(false; lw=LW - 1))]
    return legend_on_top(entries, p; size=(1150, 750), ncol=2, fs=16)
end
