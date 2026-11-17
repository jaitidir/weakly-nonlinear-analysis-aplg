# Amplitude equations near the codimension-two (Bogdanov–Takens) point where the stationary and the
# oscillatory instabilities meet (ϕa_st = ϕa_osc = ϕac at q = q_c, i.e. a double zero eigenvalue):
#
#   A'' = η¹ A + Γ|A|²A + δ (K1 A' + K2 |A|²A' + K3 A² conj(A')),     η¹ = ϕa¹ Π + μ¹ Ω,
#
# with ϕa = ϕa* + δ² ϕa¹ and Pe = Pe* + δ² μ¹. Written as A = a e^{iθ}, J = a²θ' it is the real system
#   a'' + ∂V/∂a = δ (K1 + (K2+K3) a²) a',   J' = δ (K1 + (K2-K3) a²) J,   V = [2J²/a² - 2η¹a² - Γa⁴]/4.
# The solution types (H, PS, TW, SW, MPS), their existence and stability conditions follow Dangelmayr & Knobloch,
# "The Takens–Bogdanov bifurcation with O(2)-symmetry" (Phil. Trans. R. Soc. Lond. A 322, 1987).
using QuadGK, Roots, ForwardDiff
# (eigvals, eigen, norm, dot come from LinearAlgebra, loaded by common.jl before this file)

# BT point at wavenumber q: (Pe*, ϕa*) where phia_stationary(q) = phia_oscillatory(q), i.e. L(q) has a double zero
# eigenvalue (a Jordan block); exists for ϕ > 1/2
function bt_point(ϕ, q)
    Pe2 = (2(1-ϕ)^2*q^2 + 3(1-ϕ)*q^2 + 4(1-ϕ) + 2) / ((1-ϕ)*(2ϕ-1))
    ϕas = 2(1 + (1-ϕ)*q^2) / (Pe2*(1-ϕ)*(2ϕ-1))
    return sqrt(Pe2), ϕas
end


# All coefficients of the amplitude equation at the BT point for density ϕ (> 1/2), box length Lx (q_c = 2π/Lx).
# K1 is linear in (ϕa¹, μ¹): K1 = k1_ϕa ϕa¹ + k1_μ μ¹
function bt_coefficients(ϕ; Lx=10.0)
    ϕ > 0.5 || error("the BT point exists only for ϕ > 1/2")
    i = im; q = 2π / Lx
    Pe, ϕas = bt_point(ϕ, q)
    α = (1 - ϕas) / (1 - ϕ)  # null vector v0 = (1, α, iβ) of L(q) at the BT point (ρ component 1)
    β = q / (Pe*(1 - ϕ))
    Λ = β/q - Pe*(ϕas - ϕ)/(q^2*(1 - ϕ)^2)  # solvability normalisation (makes the coefficient of A'' equal to 1)
    # η¹ = ϕa¹Π + μ¹Ω: the product of the two small eigenvalues of L(q) is -δ²η¹ (their sum is δ²K1)
    Π = 2Pe*q^2/Λ
    Ω = 2q^2*(2ϕas - 1)/Λ
    Kp = q*(Pe^2*(1 - 2ϕas) + 3q^2); Km = q*(Pe^2*(1 - 2ϕas) - 3q^2)
    # C's (stationary second-order harmonics), as in coefs_stationary: the e^{2iqx} part of (ρ, ρa, m) is 2qPeβ C A²
    den = Pe^2*(1 - ϕ)*(1 - ϕas) - 2q^2*(1 - ϕ) - 1
    C1 = -inv(4q^2 + 2q^2*Pe^2*(1 - ϕ)*(4ϕas - 3)/den) + 0im
    C2 = (1 - ϕas)/(1 - ϕ)*C1
    C3 = -(i*q*Pe*(4ϕas - 3))/den*C1
    Cρ, Cρa, Cm = C1, C2, C3
    Γ = (-2q*Pe*β/Λ)*(i*Kp*C3 + Km*β*C1 + 2Pe*q^2*C2 - 4Pe*q^2*α*C1)  # same Γ as coefs_stationary at the BT point
    # D, D*, W: harmonics needed at O(δ) of the solvability condition (terms in |A|²A' and A²conj(A'))
    Dρ = -(((Pe^2*(1 - ϕ) + 4q^2*(1 - ϕ) + 2)*Cρ - Pe^2*(1 - ϕ)*Cρa - 2i*q*Pe*(1 - ϕ)*Cm) / (12q^4*(1 - ϕ)))
    Dρa = α*Dρ + (Cρ - Cρa)/(4q^2*(1 - ϕ))
    Dm = i*(Cρ + 4q^2*Dρ)/(2q*Pe*(1 - ϕ))
    Dsρ = -(4q^2*(1 - ϕ) + 2 + 2Pe^2*(ϕ - ϕas))/(12q^4*(1 - ϕ))
    Dsρa = α*Dsρ
    Dsm = i*(1 + 4q^2*Dsρ)/(2q*Pe*(1 - ϕ))
    Wρ, Wρa, Wm = (2β*q*Pe .* (Dρ, Dρa, Dm)) .+ ((Dsρ, Dsρa, Dsm) ./ (1 - ϕ))
    P = 6Pe*q^2*(α*Wρ - Wρa) - 2Pe^2*q*(1 - 2ϕas)*(i*Wm + β*Wρ) - 6i*q^3*(Wm + i*β*Wρ) + 2q^2*Pe*(Wρa + α*Wρ)
    Q = 6β*q*Pe^2*(ϕas - ϕ)/(1 - ϕ)^2*Cρ - 2β*q*Pe^2*(1 - 2ϕas)/(1 - ϕ)*Cρ +
        (6β*q^3/(1 - ϕ) + 2β*Pe^2*q*(ϕas - ϕ)/(1 - ϕ)^2)*Cρ
    K2 = (12q*Pe^2*β/(1 - ϕ)*(Cρa - α*Cρ) + 4i*q^2*Pe*β/(1 - ϕ)*(Cm - i*β*Cρ) -
          2Γ*Pe*(ϕas - ϕ)/(q^4*(1 - ϕ)^3) + P)/Λ    # = +2ΓPe*(ϕ-ϕa*)/(q⁴(1-ϕ)³)
    K3 = (6q*Pe^2*β/(1 - ϕ)*(Cρa - α*Cρ) + 2i*q^2*Pe*β/(1 - ϕ)*(Cm - i*β*Cρ) -
          Γ*Pe*(ϕas - ϕ)/(q^4*(1 - ϕ)^3) + Q)/Λ
    # Γ, K2, K3 are real (imaginary parts returned in imag_parts as a check)
    # K1 = [μ¹q²/Pe*² - μ¹(1-2ϕ)(1-ϕa*) - Pe* ϕa¹ + η¹ Pe*(ϕ-ϕa*)/(q⁴(1-ϕ)²)] / (Λ(1-ϕ))
    c = Pe*(ϕ - ϕas)/(q^4*(1 - ϕ)^2)
    k1_ϕa = (-Pe + Π*c)/(Λ*(1 - ϕ))
    k1_μ  = (q^2/Pe^2 - (1 - 2ϕ)*(1 - ϕas) + Ω*c)/(Λ*(1 - ϕ))
    return (ϕ=ϕ, q=q, Pe=Pe, ϕas=ϕas, α=α, β=β, Λ=Λ, Π=Π, Ω=Ω, Γ=real(Γ), K2=real(K2), K3=real(K3),
            k1_ϕa=k1_ϕa, k1_μ=k1_μ, C=(Cρ, Cρa, Cm), D=(Dρ, Dρa, Dm), Ds=(Dsρ, Dsρa, Dsm), W=(Wρ, Wρa, Wm),
            P=P, Q=Q, imag_parts=(Γ=imag(Γ), K2=imag(K2), K3=imag(K3)))
end

# η¹ and K1 at (ϕa¹, μ¹)
η(c, ϕa1, μ1)  = ϕa1*c.Π + μ1*c.Ω

K1(c, ϕa1, μ1) = ϕa1*c.k1_ϕa + μ1*c.k1_μ

const _GL = QuadGK.gauss(96)  # 96-point Gauss–Legendre nodes and weights on [-1, 1] (orbit averages for m < 1/2)

# ====================================================================================================
# Averaging: the L = 0 families SW (standing wave, CP) and MPS, and the steady branches H, PS, TW along a path
# ====================================================================================================

# Complete elliptic integrals K(m), E(m) (parameter m = k²) by the arithmetic–geometric mean:
# E = K (1 - Σ 2^{n-1} c_n²), c_0² = m
function ellipKE(m)
    0 <= m < 1 || error("ellipKE needs 0 ≤ m < 1 (m = $m)")
    a, b = 1.0, sqrt(1 - m); s = m/2; p = 1.0
    for _ in 1:60
        abs(a - b) <= 4eps(a) && break
        a, b, c = (a + b)/2, sqrt(a*b), (a - b)/2
        p *= 2; s += p*c^2/2
    end
    K = π/(2a)
    return K, K*(1 - s)
end

# Periodic orbit of the O(1) system (δ = 0) in z = a² = |A|², which obeys z'² = 2Γ(z - z1)(z - z2)(z - z3):
# roots sum -2η¹/Γ, pair products 4E/Γ, product 2L²/Γ (E energy, L = J angular momentum). Written for Γ < 0,
# z3 ≤ z1 ≤ z ≤ z2, z = z3 + (z2 - z3) dn²(ωT, m). Returns roots, E, L², m, K(m), E(m), ω, the period Tz of z and
# the averages <z>, <z²>, W = <|A'|²>, P = <a²a'²>
function orbit_from_roots(c, ηv, z1, z2; z3=-2ηv/c.Γ - z1 - z2)
    Γ = c.Γ
    E = Γ/4*(z1*z2 + z1*z3 + z2*z3)
    L2 = max(Γ/2*z1*z2*z3, 0.0)  # round-off below 0 set to 0 (L = 0 families)
    m = (z2 - z1)/(z2 - z3)
    Kk, Ek = ellipKE(m)
    ω = sqrt(abs(Γ)*(z2 - z3)/2)
    zav = z3 + (z2 - z3)*Ek/Kk  # <dn²> = E/K
    # <a a''> = -<a'²> and conservation of E give W and <z²>; <(a³a')'> = 0 gives P
    z2av = -4(ηv*zav + E)/(3Γ)
    if m >= 0.5
        W = (4E + ηv*zav)/3                # <|A'|²>
        P = (ηv*z2av + 4E*zav - 3L2)/5     # <r² v²>
    else
        # near the Hopf ends W and P are small differences of O(1) terms: average them directly instead, with
        # z = zc + h sin ϑ, dT ∝ dϑ/√(z - z3) (smooth for m < 1/2) and P in factored form
        zc = (z1 + z2)/2; h = (z2 - z1)/2; sw = 0.0; sP = 0.0; sQ = 0.0
        for (x, w) in zip(_GL...)
            ϑ = π/2*x; z = zc + h*sin(ϑ); wt = w/sqrt(z - z3)
            Pz = abs(Γ)/2*h^2*cos(ϑ)^2*(z - z3)
            # W integrand |A'|² = z'²/(4z) + L²/z, with (z - z1) cancelled by hand when z1 = 0
            sw += wt; sP += wt*Pz; sQ += wt*(z1 == 0 ? abs(Γ)/2*(z2 - z)*(z - z3) : Pz/z + L2/z)
        end
        W = sQ/sw; P = sP/sw
    end
    return (z1=z1, z2=z2, z3=z3, E=E, L2=L2, m=m, K=Kk, Ek=Ek, ω=ω, Tz=2Kk/ω, zav=zav, z2av=z2av, W=W, P=P)
end

# O(δ) balances averaged over the orbit: dE/dT = δ(K1|A'|² + (K2+K3)a²a'² + (K2-K3)J²), dJ/dT = δ(K1 + (K2-K3)a²)J
Fbal(c, K1v, o) = K1v*o.W + (c.K2 + c.K3)*o.P + (c.K2 - c.K3)*o.L2      # <dE/dT>/δ

Gbal(c, K1v, o) = K1v + (c.K2 - c.K3)*o.zav                              # <dL/dT>/(δL)


# ---------------------------------------------------------------------------------------------------------
# L = 0 families, in the units z = (|η¹|/|Γ|) ζ. Each family is parametrised by m = k² ∈ (0, 1):
#   SW,  η¹ < 0 : ζ1 = 0, ζ2 = 2m/(1-2m)         (m ∈ (0, 1/2); m → 0 Hopf of H, m → 1/2 large amplitude)
#   SW,  η¹ > 0 : ζ1 = 0, ζ2 = 2m/(2m-1)         (m ∈ (1/2, 1); m → 1 homoclinic, m → 1/2 large amplitude)
#   MPS, η¹ > 0 : ζ2 = 2/(2-m), ζ1 = 2(1-m)/(2-m) (m → 0 Hopf of PS, m → 1 homoclinic)
# Persistence F = 0 gives K1 = -M<P>/<|A'|²> = κ M|η¹|/|Γ|: κ(m) depends only on the family.
# ---------------------------------------------------------------------------------------------------------
# (ζ1, ζ2) of the family (σ = sign η¹); ζ3 follows from the sum of the roots
function lzero_roots(fam, σ, m)
    fam == :MPS && return (2(1 - m)/(2 - m), 2/(2 - m))
    return (0.0, σ < 0 ? 2m/(1 - 2m) : 2m/(2m - 1))
end

# Orbit of family fam at η¹ = ηv and parameter m, in physical units z = s ζ
function lzero_orbit(c, ηv, fam, m)
    s = abs(ηv)/abs(c.Γ)
    ζ1, ζ2 = lzero_roots(fam, sign(ηv), m)
    # MPS: z3 = 0 exactly (L = 0), not the rounded -2η¹/Γ - z1 - z2
    return fam == :MPS ? orbit_from_roots(c, ηv, s*ζ1, s*ζ2; z3=0.0) : orbit_from_roots(c, ηv, s*ζ1, s*ζ2)
end

# κ(m) = -P|Γ|/(W|η¹|), evaluated at |η¹| = 1 (it does not depend on |η¹|)
function kappa_lzero(c, σ, fam, m)
    o = lzero_orbit(c, σ*1.0, fam, m)
    return -o.P/o.W/(1/abs(c.Γ))
end

# Interval of m of the family
mrange(fam, σ) = fam == :MPS ? (0.0, 1.0) : (σ < 0 ? (0.0, 0.5) : (0.5, 1.0))


"""
Grid in m, refined near both ends: down to 1e-11 from the homoclinic and large-amplitude ends, and to 1e-8 from
a Hopf end (m = 0), below which the finite differences for λ_E lose their accuracy.
"""
function mgrid(fam, σ; n=4000)
    a, b = mrange(fam, σ)
    # logarithmic near both ends, uniform in between
    lo = a .+ (b - a) .* exp10.(range(a == 0 ? -8 : -11, -1; length=n ÷ 2)) ./ 2
    hi = b .- (b - a) .* exp10.(range(-1, -11; length=n ÷ 2)) ./ 2
    return vcat(lo, range(lo[end], hi[1]; length=n)[2:end-1], hi)
end


"""
All L = 0 periodic orbits (SW and MPS) at (η¹, K1): for each, the orbit, the energy exponent λ_E/δ = dF/dE and the
angular-momentum exponent λ_L/δ = G = K1 + D<x²>.
"""
function lzero_solutions(c, ηv, K1v; n=4000)
    out = NamedTuple[]
    ηv == 0 && return out
    σ = sign(ηv); κ = K1v*abs(c.Γ)/((c.K2 + c.K3)*abs(ηv))
    # families that exist for this sign of η¹ (the SW roots are written for Γ < 0; Γ > 0 with η¹ < 0 throws in ellipKE)
    fams = σ < 0 ? (:SW,) : (c.Γ < 0 ? (:MPS, :SW) : ())
    for fam in fams
        ms = mgrid(fam, σ; n=n)
        κs = [kappa_lzero(c, σ, fam, m) for m in ms]
        # every sign change of κ(m) - κ on the grid is an orbit with F = 0, refined by bisection
        for j in 1:length(ms)-1
            (κs[j] - κ)*(κs[j+1] - κ) < 0 || continue
            m = find_zero(x -> kappa_lzero(c, σ, fam, x) - κ, (ms[j], ms[j+1]), Bisection())
            push!(out, lzero_point(c, ηv, K1v, fam, m))
        end
    end
    return out
end


"""
dE/dm along an L = 0 family at fixed η¹ (s = |η¹|/|Γ|). SW: E = -η¹z2/2 - Γz2²/4 with z2 = s ζ2(m);
MPS: E = Γ z1 z2/4 = Γ s² (1-m)/(2-m)².
"""
function dEdm(c, ηv, fam, m)
    s = abs(ηv)/abs(c.Γ)
    fam == :MPS && return abs(c.Γ)*s^2*m/(2 - m)^3
    ζ2 = lzero_roots(fam, sign(ηv), m)[2]
    dζ2 = ηv < 0 ? 2/(1 - 2m)^2 : -2/(2m - 1)^2
    return (-ηv/2 - c.Γ*s*ζ2/2)*s*dζ2
end

# One L = 0 orbit with its exponents λ_E, λ_L and its stability
function lzero_point(c, ηv, K1v, fam, m)
    o = lzero_orbit(c, ηv, fam, m)
    # λ_E/δ = dF/dE along the family at fixed (η¹, K1): dF/dm by central differences, dE/dm exactly (near the Hopf
    # of PS, E - E_min ∝ m², too small to difference)
    a, b = mrange(fam, sign(ηv)); h = 1e-3*min(m - a, b - m)
    op = lzero_orbit(c, ηv, fam, m + h); om = lzero_orbit(c, ηv, fam, m - h)
    λE = (Fbal(c, K1v, op) - Fbal(c, K1v, om))/(2h*dEdm(c, ηv, fam, m))
    λL = Gbal(c, K1v, o)
    xmin = fam == :MPS ? sqrt(o.z1) : -sqrt(o.z2)  # range [xmin, xmax] of the signed amplitude a
    period = fam == :SW ? 2o.Tz : o.Tz           # period of the signed amplitude x(T)
    return (kind=fam, m=m, E=o.E, xmin=xmin, xmax=sqrt(o.z2), xav2=o.zav, W=o.W, period=period,
            λE=λE, λL=λL, F=Fbal(c, K1v, o), stable=(λE < 0 && λL < 0))
end

# Eigenvalues (in T) of the homogeneous state H, A = 0: λ² - δK1λ - η¹ = 0
H_eigs(ηv, K1v, δ) = (d = complex(δ^2*K1v^2 + 4ηv); ((δ*K1v + sqrt(d))/2, (δ*K1v - sqrt(d))/2))

# Stationary pattern PS, A = r real: r² = -η¹/Γ. λr: amplitude pair λ² - δfλ + 2η¹ = 0, f = K1 + (K2+K3)r²;
# λphase = δg, g = K1 + (K2-K3)r² (drift, J' = δgJ); translation (θ) is neutral
function PS_solution(c, ηv, K1v, δ)
    r2 = -ηv/c.Γ
    r2 > 0 || return nothing
    f = K1v + (c.K2 + c.K3)*r2; g = K1v + (c.K2 - c.K3)*r2
    d = complex(δ^2*f^2 - 8ηv)
    λr = ((δ*f + sqrt(d))/2, (δ*f - sqrt(d))/2)
    return (r=sqrt(r2), f=f, g=g, λr=λr, λphase=δ*g, stable=(ηv > 0 && f < 0 && g < 0))
end

# Travelling wave TW, A = r e^{iΩT}: imaginary part K1 + (K2-K3)r² = 0, real part Ω² = -η¹ - Γr²
function TW_solution(c, ηv, K1v, δ)
    D = c.K2 - c.K3
    r2 = -K1v/D; r2 > 0 || return nothing
    Ω2 = -ηv - c.Γ*r2; Ω2 > 0 || return nothing
    r = sqrt(r2); Ωt = sqrt(Ω2)
    # Jacobian of (a, a', J) at the TW (J = r²Ω; the phase θ is neutral and left out)
    J = [0 1 0; 2c.Γ*r2-4Ω2 2δ*c.K3*r2 2Ωt/r; 2δ*D*r^3*Ωt 0 0]
    ev = eigvals(J)
    # characteristic polynomial λ³ + a2λ² + a1λ + a0; Routh–Hurwitz Hopf condition a2a1 - a0 = 0
    a2 = -2δ*c.K3*r2; a1 = 4Ω2 - 2c.Γ*r2; a0 = -4δ*D*r2*Ω2
    return (r=r, Ω=Ωt, ev=ev, a=(a2, a1, a0), hopf=a2*a1 - a0, stable=all(real.(ev) .< 0))
end


# =========================================================================================================
# Straight paths (ϕa¹, μ¹) = (ϕa¹₀ + v_ϕ s, μ¹₀ + v_μ s): branches, stability and bifurcation points
# =========================================================================================================
pathpoint(P, s) = (P.ϕa0 + P.vϕ*s, P.μ0 + P.vμ*s)  # (ϕa¹, μ¹) at s

ηs(c, P, s) = η(c, pathpoint(P, s)...)

K1s(c, P, s) = K1(c, pathpoint(P, s)...)

κs(c, P, s) = K1s(c, P, s)*abs(c.Γ)/((c.K2 + c.K3)*abs(ηs(c, P, s)))


"""
Value of s where K1 = κ (K2+K3) σ η¹/|Γ| with sign(η¹) = σ (both K1 and η¹ are linear in s), or nothing.
This is where the path crosses the ray of the (ϕa¹, μ¹) plane on which κ takes the given value.
"""
function s_at_kappa(c, P, κ, σ)
    a = κ*σ*(c.K2 + c.K3)/abs(c.Γ)
    # K1(s) - a η¹(s) = (k0 + k1 s) - a (e0 + e1 s)
    k0 = K1(c, P.ϕa0, P.μ0); k1 = c.k1_ϕa*P.vϕ + c.k1_μ*P.vμ
    e0 = η(c, P.ϕa0, P.μ0);  e1 = c.Π*P.vϕ + c.Ω*P.vμ
    abs(k1 - a*e1) < 1e-300 && return nothing
    s = -(k0 - a*e0)/(k1 - a*e1)
    return sign(e0 + e1*s) == σ ? s : nothing
end

s_zero(f0, f1) = -f0/f1  # zero of f0 + f1 s

s_eta0(c, P)  = s_zero(η(c, P.ϕa0, P.μ0), c.Π*P.vϕ + c.Ω*P.vμ)  # s where η¹ = 0

s_K10(c, P)   = s_zero(K1(c, P.ϕa0, P.μ0), c.k1_ϕa*P.vϕ + c.k1_μ*P.vμ)  # s where K1 = 0

# Distinguished values of κ (analytic except the fold): Hopf of PS f = 0 -> κ = -1; homoclinic (Melnikov, sech orbit)
# -> κ = -4/5; fold of SW = maximum of κ(m) on the η¹ > 0 SW family (golden section); PS drift g = 0 -> κ = -(K2-K3)/(K2+K3)
function kappa_constants(c)
    f(m) = -kappa_lzero(c, 1, :SW, m)
    r = optimize_1d(f, 0.5 + 1e-6, 1 - 1e-9)
    return (hopf_PS=-1.0, hom=-4/5, fold_SW=-r.fmin, m_fold=r.xmin, drift_PS=-(c.K2 - c.K3)/(c.K2 + c.K3))
end

# Golden-section minimisation of f on [a, b] (f unimodal)
function optimize_1d(f, a, b; tol=1e-13)
    g = (sqrt(5) - 1)/2
    x1 = b - g*(b - a); x2 = a + g*(b - a); f1 = f(x1); f2 = f(x2)
    while b - a > tol
        if f1 < f2
            b, x2, f2 = x2, x1, f1; x1 = b - g*(b - a); f1 = f(x1)
        else
            a, x1, f1 = x1, x2, f2; x2 = a + g*(b - a); f2 = f(x2)
        end
    end
    x = (a + b)/2
    return (xmin=x, fmin=f(x))
end


"""
L = 0 branch of family `fam` (:SW or :MPS) for sign σ of η¹, traced by the elliptic parameter m: each orbit
shape fixes κ(m), hence the point s(m) of the path (K1 and η¹ are linear in s). Returns the points inside
[smin, smax], ordered in m.
"""
function lzero_branch(c, P, fam, σ, smin, smax; n=3000)
    pts = NamedTuple[]
    for m in mgrid(fam, σ; n=n)
        κ = kappa_lzero(c, σ, fam, m)
        abs(κ) < 1e7 || continue            # closer to η¹ = 0 than round-off allows (the amplitude stays finite there)
        s = s_at_kappa(c, P, κ, σ)
        (s === nothing || !(smin <= s <= smax)) && continue
        o = lzero_point(c, ηs(c, P, s), K1s(c, P, s), fam, m)
        push!(pts, merge((s=s,), o))
    end
    return pts
end

# H, PS and TW (with their eigenvalues and stability) at each s of ss; PS and TW only where they exist
function steady_branches(c, P, ss, δ)
    H = [(s=s, stable=(ηs(c, P, s) < 0 && K1s(c, P, s) < 0), eig=H_eigs(ηs(c, P, s), K1s(c, P, s), δ)) for s in ss]
    PS = [merge((s=s,), o) for s in ss for o in (PS_solution(c, ηs(c, P, s), K1s(c, P, s), δ),) if o !== nothing]
    TW = [merge((s=s,), o) for s in ss for o in (TW_solution(c, ηs(c, P, s), K1s(c, P, s), δ),) if o !== nothing]
    return (H=H, PS=PS, TW=TW)
end


"""
Bifurcation points of the path in [smin, smax], all from the analytic conditions (no plot reading):
primary Hopf of H (K1 = 0, η¹ < 0), stationary bifurcation (η¹ = 0), and the rays κ = -1 (Hopf of PS → MPS),
κ = -4/5 (homoclinic: MPS ↔ SW), κ = κ_fold (fold of SW), κ = -R (PS drift instability, where TW ends on PS).
Each has the stability on both sides (s ∓ ε), computed from the branches themselves.
"""
function path_events(c, P, smin, smax, δ, δphys, kc)
    ev = NamedTuple[]
    M = c.K2 + c.K3; D = c.K2 - c.K3
    # the event in physical parameters: ϕa = ϕa* + δphys²ϕa¹, Pe = Pe* + δphys²μ¹
    phys(s) = (ϕa1 = pathpoint(P, s)[1]; μ1 = pathpoint(P, s)[2];
               (ϕa1=ϕa1, μ1=μ1, η=ηs(c, P, s), K1=K1s(c, P, s), ϕa=c.ϕas + δphys^2*ϕa1, Pe=c.Pe + δphys^2*μ1))
    eps_s = 1e-7*(smax - smin)
    # stability of each branch at s ("none" where it does not exist); side(f, s) = "f(s - ε) → f(s + ε)"
    Hst(s) = (ηs(c, P, s) < 0 && K1s(c, P, s) < 0) ? "stable" : "unstable"
    PSst(s) = (o = PS_solution(c, ηs(c, P, s), K1s(c, P, s), δ); o === nothing ? "none" : (o.stable ? "stable" : "unstable"))
    TWst(s) = (o = TW_solution(c, ηs(c, P, s), K1s(c, P, s), δ); o === nothing ? "none" : (o.stable ? "stable" : "unstable"))
    L0st(s, kind) = (os = filter(o -> o.kind == kind, lzero_solutions(c, ηs(c, P, s), K1s(c, P, s)));
                     isempty(os) ? "none" : join([o.stable ? "stable" : "unstable" for o in sort(os, by=o -> o.xmax)], " + "))
    side(f, s) = "$(f(s - eps_s)) → $(f(s + eps_s))"
    # primary Hopf of H
    sh = s_K10(c, P)
    if smin <= sh <= smax && ηs(c, P, sh) < 0
        ω = sqrt(-ηs(c, P, sh))
        push!(ev, merge((s=sh, type="primary Hopf of H (K1 = 0, η¹ < 0)", branches="H, TW (±), SW",
                         amp=0.0, freq="Ω_H = √(-η¹) = $(round(ω, sigdigits=5)) (period $(round(2π/ω, sigdigits=5)))",
                         change="H: $(side(Hst, sh)); TW: $(side(TWst, sh)); SW: $(side(s -> L0st(s, :SW), sh))"), phys(sh)))
    end
    # stationary bifurcation of H
    s0 = s_eta0(c, P)
    if smin <= s0 <= smax
        push!(ev, merge((s=s0, type="stationary bifurcation of H (η¹ = 0)", branches="H, PS",
                         amp=0.0, freq="0 (real eigenvalue)",
                         change="H: $(side(Hst, s0)) (real eigenvalue through 0); PS: $(side(PSst, s0))"), phys(s0)))
    end
    rS(s) = sqrt(abs(ηs(c, P, s)/c.Γ))  # PS amplitude r
    # secondary Hopf of PS → MPS
    s1 = s_at_kappa(c, P, kc.hopf_PS, 1)
    if s1 !== nothing && smin <= s1 <= smax && c.Γ < 0
        ω = sqrt(2ηs(c, P, s1))
        push!(ev, merge((s=s1, type="secondary Hopf of PS (K1 + (K2+K3) r_PS² = 0)", branches="PS, MPS",
                         amp=rS(s1), freq="√(2η¹) = $(round(ω, sigdigits=5)) (period $(round(2π/ω, sigdigits=5)))",
                         change="PS: $(side(PSst, s1)); MPS: $(side(s -> L0st(s, :MPS), s1))"), phys(s1)))
    end
    # homoclinic (infinite period): MPS ↔ SW
    s2 = s_at_kappa(c, P, kc.hom, 1)
    if s2 !== nothing && smin <= s2 <= smax && c.Γ < 0
        push!(ev, merge((s=s2, type="homoclinic orbit of H (κ = -4/5): infinite period", branches="MPS, SW",
                         amp=sqrt(2ηs(c, P, s2)/abs(c.Γ)), freq="period → ∞",
                         change="MPS: $(side(s -> L0st(s, :MPS), s2)); SW: $(side(s -> L0st(s, :SW), s2))"), phys(s2)))
    end
    # fold of SW
    s3 = s_at_kappa(c, P, kc.fold_SW, 1)
    if s3 !== nothing && smin <= s3 <= smax && c.Γ < 0
        o = lzero_point(c, ηs(c, P, s3), K1s(c, P, s3), :SW, kc.m_fold)
        push!(ev, merge((s=s3, type="fold of SW periodic orbits (λ_E = 0)", branches="SW (stable), SW (unstable)",
                         amp=o.xmax, freq="period $(round(o.period, sigdigits=5))",
                         change="SW: $(side(s -> L0st(s, :SW), s3))"), phys(s3)))
    end
    # PS drift instability: TW ends on PS
    s4 = s_at_kappa(c, P, kc.drift_PS, 1)
    if s4 !== nothing && smin <= s4 <= smax && -ηs(c, P, s4)/c.Γ > 0
        push!(ev, merge((s=s4, type="drift (phase) instability of PS: TW ends on PS (K1 + (K2-K3) r_PS² = 0)",
                         branches="PS, TW (±)", amp=rS(s4), freq="Ω_TW → 0",
                         change="PS: $(side(PSst, s4)) (phase eigenvalue through 0); TW: $(side(TWst, s4))"), phys(s4)))
    end
    return sort(ev, by=e -> e.s)
end

# Vector of named tuples -> Dict of columns (for saving)
columns(v) = isempty(v) ? Dict{String,Any}() : Dict(string(k) => [getfield(x, k) for x in v] for k in keys(v[1]))


# ====================================================================================================
# Exact PS of the PDE near the BT point (Fourier collocation, Newton) and its breathing eigenvalue
# ====================================================================================================

# Fourier spectral differentiation matrices (first and second derivative) on N (even) equispaced points of [0, L)
function fourier_matrices(N, L)
    h = 2π/N; D1 = zeros(N, N); D2 = zeros(N, N)
    for i in 1:N, j in 1:N
        i == j && continue
        d = (i - j)*h
        D1[i, j] = 0.5*(-1)^(i - j)*cot(d/2)
        D2[i, j] = -0.5*(-1)^(i - j)/sin(d/2)^2
    end
    for i in 1:N; D2[i, i] = -π^2/(3h^2) - 1/6; end
    s = 2π/L
    return D1*s, D2*s^2
end

# Right-hand side of the PDE (ρ, ρa, m stacked); ∂x[(1-ρ)∂xf + f∂xρ] = (1-ρ)∂x²f + f∂x²ρ
function aplg_rhs(u, Pe, D1, D2)
    N = length(u) ÷ 3
    ρ = u[1:N]; ρa = u[N+1:2N]; m = u[2N+1:3N]
    F1 = (1 .- ρ) .* m; F2 = (1 .- ρ) .* ρa
    return vcat(D2*ρ - Pe*(D1*F1),
                (1 .- ρ) .* (D2*ρa) + ρa .* (D2*ρ) - Pe*(D1*F1),
                (1 .- ρ) .* (D2*m) + m .* (D2*ρ) - 2m - Pe*(D1*F2))
end

# Steady state of the PDE by Newton (least squares: 3N equations + the two means + the phase condition ∫ρ sin(qx) = 0
# that fixes the translation). Returns u, the residual and the eigenvalues of the Jacobian at u
function ps_newton(ϕ, ϕa, Pe, Lx, N, u0; tol=1e-13)
    D1, D2 = fourier_matrices(N, Lx)
    x = (0:N-1) .* (Lx/N); sinq = sin.(2π/Lx .* x) ./ N
    G(u) = vcat(aplg_rhs(u, Pe, D1, D2), sum(u[1:N])/N - ϕ, sum(u[N+1:2N])/N - ϕa, dot(sinq, u[1:N]))
    u = copy(u0)
    for _ in 1:60
        du = ForwardDiff.jacobian(G, u) \ G(u); u .-= du
        norm(du) < tol && break
    end
    J = ForwardDiff.jacobian(v -> aplg_rhs(v, Pe, D1, D2), u)
    return u, norm(aplg_rhs(u, Pe, D1, D2)), eigvals(J)
end


"""
PS of the PDE at φa = φa* + δ²φa¹, Pe = Pe* + δ²μ¹ (one wavelength in Lx) and its breathing eigenvalue.
Returns the half-amplitude of ρ, the breathing pair σ, f_PDE = 2 Re σ/δ² and the state u = (ρ, ρ_a, m) (nothing if
no PS is found).
"""
function ps_breathing(c, ϕa1, μ1, δ; Lx=10.0, N=32)
    ϕ = c.ϕ; q = 2π/Lx; ϕa = c.ϕas + δ^2*ϕa1; Pe = c.Pe + δ^2*μ1; ηv = η(c, ϕa1, μ1)
    ηv > 0 || return nothing
    # initial guess: critical linear eigenvector at k = q, with the amplitude of the amplitude equation's PS
    Lk = ComplexF64[-q^2 0 -im*q*Pe*(1-ϕ); -ϕa*q^2 -(1-ϕ)*q^2 -im*q*Pe*(1-ϕ); im*q*Pe*ϕa -im*q*Pe*(1-ϕ) -(1-ϕ)*q^2-2]
    ev = eigen(Lk); v = ev.vectors[:, argmin(abs.(ev.values))]; v ./= v[1]
    x = (0:N-1) .* (Lx/N); e = exp.(im*q .* x)
    for a in (1.0, 0.5, 2.0, 4.0) .* (2δ*sqrt(ηv/abs(c.Γ)))
        u0 = vcat(ϕ .+ a .* real.(v[1] .* e), ϕa .+ a .* real.(v[2] .* e), a .* real.(v[3] .* e))
        u, res, λ = ps_newton(ϕ, ϕa, Pe, Lx, N, u0)
        amp = (maximum(u[1:N]) - minimum(u[1:N]))/2
        (res < 1e-10 && amp > 1e-6) || continue
        # breathing pair: in t, δλr with λr ≈ ±i√(2η¹) (complex while δ²f² < 8η¹; a real pair would not be picked)
        cand = filter(z -> imag(z) > 0, λ)
        σ = cand[argmin(abs.(cand .- im*δ*sqrt(2ηv)))]
        return (amp=amp, σ=σ, f=2real(σ)/δ^2, ω=imag(σ)/δ, u=u)
    end
    return nothing
end



# ====================================================================================================
# Weakly nonlinear MPS as fields: the L = 0 orbit A = a(T) e^{iθ} (T = δt) of a'' = η¹a + Γa³, to O(δ²),
#   (ρ, ρa, m) = (φ, φa, 0) + δ(a v₀ e^{iqx} + c.c.) + δ²(a' v₁ e^{iqx} + a² C e^{2iqx} + c.c.)
# with v₀ the null vector of the linear operator at the BT point (ρ component 1), v₁ its generalised eigenvector
# (L v₁ = v₀, ρ component 0) and C = -L(2q)⁻¹N₂(v₀, v₀) the second harmonic of the quadratic terms
# ====================================================================================================

function bt_mps_wnl(c, ϕa1, μ1, δ; dT=1e-3)
    ϕ = c.ϕ; q = c.q; ϕas = c.ϕas; Pe = c.Pe
    L(k) = ComplexF64[-k^2 0 -im*k*Pe*(1-ϕ); -ϕas*k^2 -(1-ϕ)*k^2 -im*k*Pe*(1-ϕ); im*k*Pe*ϕas -im*k*Pe*(1-ϕ) -(1-ϕ)*k^2-2]
    L1 = L(q); v0 = vcat(1.0, -(L1[2:3, 2:3] \ L1[2:3, 1])); v1 = vcat(0.0, L1[:, 2:3] \ v0)
    # N₂(v₀, v₀) at e^{2iqx}: from Pe∂x(ρm) in the ρ and ρa equations and Pe∂x(ρρa) in the m equation (the
    # diffusive quadratic terms cancel); C = 2qPeβ c.C
    C = -(L(2q) \ (2im*q*Pe .* [v0[1]*v0[3], v0[1]*v0[3], v0[1]*v0[2]]))
    ηv = η(c, ϕa1, μ1)
    o = filter(o -> o.kind == :MPS && o.stable, lzero_solutions(c, ηv, K1(c, ϕa1, μ1)))
    isempty(o) && return nothing
    o = o[1]
    # one period of a(T) from its maximum (RK4)
    f(y) = (y[2], ηv*y[1] + c.Γ*y[1]^3)
    n = ceil(Int, o.period/dT); h = o.period/n
    a = zeros(n + 1); da = zeros(n + 1); a[1] = o.xmax
    for i in 1:n
        y = (a[i], da[i]); k1 = f(y); k2 = f(y .+ h/2 .* k1); k3 = f(y .+ h/2 .* k2); k4 = f(y .+ h .* k3)
        a[i+1], da[i+1] = y .+ h/6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end
    P = o.period/δ                                          # period in t
    # fields at (x, t), with a(T), a'(T) linearly interpolated on the RK4 grid
    function field(x, t)
        θ = mod(t, P)/P*n; i = min(floor(Int, θ), n - 1); w = θ - i
        ai = (1 - w)*a[i+1] + w*a[i+2]; dai = (1 - w)*da[i+1] + w*da[i+2]
        e1 = exp(im*q*x)
        u = 2 .* real.(δ*ai .* v0 .* e1 .+ δ^2*dai .* v1 .* e1 .+ δ^2*ai^2 .* C .* e1^2)
        return [ϕ + u[1], ϕas + δ^2*ϕa1 + u[2], u[3]]
    end
    # closure: distance between the start and end of the integrated period (check of the period)
    return (field=field, period=P, orbit=o, a=a, da=da, v0=v0, v1=v1, C=C, closure=hypot(a[end] - a[1], da[end] - da[1]))
end


"""
    bt_homoclinic_wnl(c, P; n=4001, τmax=25.0)

Weakly nonlinear homoclinic orbit on the path P. For real A = a (J = 0) the amplitude equation is
a'' = η¹a + Γa³ + δ(K1 + M a²)a', M = K2 + K3. At O(1) it conserves E = a'²/2 - η¹a²/2 - Γa⁴/4, and the orbit with E = 0
is a = A sech τ, τ = √η¹ T, A = √(2η¹/|Γ|), for every η¹ > 0. The O(δ) solvability (Melnikov) condition,
K1∫a'² dT + M∫a²a'² dT = 0, i.e. K1|Γ|/(M η¹) = -4/5, selects the point s of the path; the orbit is the O(1) one there.
Returns s and a, da/dT along τ ∈ [-τmax, τmax].
"""
function bt_homoclinic_wnl(c, P; n=4001, τmax=25.0)
    s = s_at_kappa(c, P, kappa_constants(c).hom, 1)
    s === nothing && return nothing
    ηv = ηs(c, P, s); k = sqrt(ηv); A = sqrt(2ηv/abs(c.Γ))  # τ = kT
    τ = range(-τmax, τmax; length=n)
    return (s=s, a=A .* sech.(τ), da=-k*A .* sech.(τ) .* tanh.(τ))
end
