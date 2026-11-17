# Binodal of the active-passive lattice gas (coexisting densities of the phase-separated state), for the spinodal
# figure of the stationary notebook. With γ = (1 - φa)/(1 - φ), the coexisting densities ϕ1 < ϕ2 have equal g0 and
# equal h0 = g0 R - Φ∘R; they are found by shrinking brackets around both for γ from 1 up to the critical point.
# Obtained from the work: Dynamical patterns and nonreciprocal effective interactions in an active-passive mixture 
# through exact hydrodynamic analysis (Mason et al., 2025, Nat. Commun., 16). The code is adapted from the 
# original code by the authors for the mean field assumption ds(ρ)=1-ρ                                                                           .
module Binodal
using QuadGK, Roots
export binodal_curve

self_diff(ρ) = 1 - (ρ ≤ 0.0 ? 1e-10 : min(ρ, 1.0))     # 1 - ρ, ρ kept in (0, 1]
# passive density ϕp = ϕ - ϕa on the spinodal (q -> 0) at density x; same as phia_stationary(0, x, Pe)
pspin(x; Pe=10.0) = (-2 - (1-x)*(self_diff(x) - x)*Pe^2)/(self_diff(x) + (1-x))/Pe^2
R_prime(ρ) = 1/self_diff(ρ)/(1-ρ)^3     # = 1/(1-ρ)⁴ = dR/dρ
R(ρ) = 1/3/(1-ρ)^3
# g0 and its derivative; dg0 = 0 reproduces the spinodal pspin with γ = 1 + ϕp/(1 - ρ)
g0(ρ; Pe=10, γ=1) = Pe*(1-γ*(1 - ρ))*self_diff(ρ) - 2*log(1 - ρ)/Pe
dg0(ρ; Pe=10, γ=1) = Pe*γ*self_diff(ρ) - Pe*(1-γ*(1 - ρ)) + 2/(1 - ρ)/Pe

# Φ∘R(x) = ∫₀ˣ g0 R' dy + const: quadrature of g0 R' - c log(1-y)/(1-y)⁴ plus the exact primitive
# c(1+3log(1-x))/(9(1-x)³) of the subtracted term (note: g0 R' already contains -c log(1-y)/(1-y)⁴, so this doubles
# rather than removes that term in the quadrature; the result is still exact). The lower limit used for the primitive
# is 0.95 instead of 0: a constant offset (depends on Pe only) that cancels in the equal-h0 condition
function ΦoR(x; Pe=10, γ=3.4, tol=1e-14)
    c = 2/Pe
    f(y) = g0(y; Pe=Pe, γ=γ)*R_prime(y) - c*log(1 - y)/(1-y)^4
    return quadgk(f, 0.0, x, rtol=1e-14, atol=tol)[1] + c*(1+3*log(1-x))/9/(1-x)^3 - c*(1+3*log(1-0.95))/9/(1-0.95)^3
end
h0(ρ; Pe=10, γ=1, atol=1e-12) = g0(ρ; Pe=Pe, γ=γ)*R(ρ) - ΦoR(ρ; Pe=Pe, γ=γ, tol=atol)

# Density in `limits` with the same value of f as ϕ1 (`fail` if there is none)
function same_value(f, ϕ1, limits, fail)
    try
        return find_zero(x -> f(x) - f(ϕ1), limits)
    catch
        return fail
    end
end

# Turning points of g0 (where dg0 = 0), returned as (upper root = local minimum of g0, lower root = local maximum);
# prints and returns nothing if g0 is monotonic (no phase separation at this γ)
function g_turning_points(; Pe=5, γ=3.52, initial_Δ=1e-4)
    x = 0.0:initial_Δ:(1-initial_Δ)
    DG = dg0.(x; Pe=Pe, γ=γ)
    minimum(DG) > 0 && return println("no turn in g0")
    x0 = x[argmin(DG)]
    f(x) = dg0(x; Pe=Pe, γ=γ)
    return find_zero(f, (x0, 1.0)), find_zero(f, (0.0, x0))
end

# Brackets (lower, upper) of the two coexisting densities, shrunk alternately with h0 and g0
function colapse_sol_interval(; Pe=5, γ=1.0, rho_max=(1-10e-16), initial_Δ=1e-4, max_iter=20, tol=1e-8, atol=1e-12)
    G(x) = g0(x; Pe=Pe, γ=γ)
    H(x) = h0(x; Pe=Pe, γ=γ, atol=atol)
    gmin, gmax = g_turning_points(; Pe=Pe, γ=γ, initial_Δ=initial_Δ)
    # initial brackets: ϕ1 between the density where g0 = g0(gmin) on the left branch and the lower spinodal;
    # ϕ2 between the upper spinodal and the density where g0 = g0(gmax) on the right branch
    lower = (same_value(G, gmin, (0.0, gmax), 0.0), gmax)
    upper = (gmin, same_value(G, gmax, (gmin, rho_max), rho_max))
    bracket_width(lo, up) = abs(lo[2]-lo[1]) + abs(up[2]-up[1])
    precision = bracket_width(lower, upper)
    i = 0
    while (i < max_iter) & (precision > tol)
        # map the lower bracket into the upper one by equal h0, then back by equal g0
        u1 = same_value(H, lower[1], upper, upper[1]); u2 = same_value(H, lower[2], upper, upper[2])
        upper = (min(u1, u2), max(u1, u2))
        l1 = same_value(G, upper[1], lower, lower[1]); l2 = same_value(G, upper[2], lower, lower[2])
        lower = (min(l1, l2), max(l1, l2))
        precision = bracket_width(lower, upper)
        i += 1
    end
    return precision ≤ tol, lower, upper  # (converged, bracket of ϕ1, bracket of ϕ2)
end

# Binodal at Pe: γ, the coexisting densities ϕ1, ϕ2 and their passive densities ϕp1, ϕp2 (ϕa = 1 - γ(1 - ϕ))
function binodal_curve(Pe)
    # critical point: extremum of γ = 1 + ϕp/(1-ϕ) along the spinodal (forward difference, step 1e-10; analytically
    # ϕc = 1 - 4/Pe², which must lie in the bracket (0.6, 0.99)); γ_max = γ there
    ϕc = find_zero(x -> (pspin.(x; Pe=Pe)/(1-x) - pspin.(x+1e-10; Pe=Pe)/(1-x-1e-10))/1e-10, (0.6, 0.99))
    γ_max = (1-ϕc+pspin.(ϕc; Pe=Pe))/(1-ϕc)
    γs = Float64[]; ϕ1s = Float64[]; ϕ2s = Float64[]
    # finer steps of γ towards the critical point
    for (γrange, Δ, iters) in ((1:0.001:(γ_max-0.001), 1e-6, 40), ((γ_max-0.001):0.0001:(γ_max-0.0001), 1e-8, 200),
                               ((γ_max-0.0001):0.00001:(γ_max-0.00001), 1e-8, 200), ((γ_max-0.00001):0.000001:γ_max, 1e-8, 200))
        for γ in γrange
            try
                ok, lower, upper = colapse_sol_interval(; Pe=Pe, γ=γ, initial_Δ=Δ, max_iter=iters, tol=1e-8, atol=1e-12)
                # only converged points are kept (outer ends of the brackets)
                ok && (push!(ϕ1s, lower[1]); push!(ϕ2s, upper[2]); push!(γs, γ))
            catch
                break  # first failure ends this γ range (the next, finer range is still tried)
            end
        end
    end
    ϕp(γ, ϕ) = ϕ - (1 - γ*(1 - ϕ))
    return (γ=γs, ϕ1=ϕ1s, ϕ2=ϕ2s, ϕp1=ϕp.(γs, ϕ1s), ϕp2=ϕp.(γs, ϕ2s))
end
end

using .Binodal
