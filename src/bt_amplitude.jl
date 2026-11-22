# Amplitude equations near the codimension-two (Bogdanov–Takens) point where the stationary and the
# oscillatory instabilities meet (ϕa_st = ϕa_osc = ϕac at q = q_c, i.e. a double zero eigenvalue):
#
#   A'' = η¹ A + Γ|A|²A + δ (K1 A' + K2 |A|²A' + K3 A² conj(A')),     η¹ = ϕa¹ Π + μ¹ Ω,
#
# with ϕa = ϕa* + δ² ϕa¹ and Pe = Pe* + δ² μ¹. Written as A = a e^{iθ}, J = a²θ' it is the real system
#   a'' + ∂V/∂a = δ (K1 + (K2+K3) a²) a',   J' = δ (K1 + (K2-K3) a²) J,   V = [2J²/a² - 2η¹a² - Γa⁴]/4.
# The solution types (H, PS, TW, CP, MPS; CP and MPS are SW2 and SW3 in the paper below), their existence and stability and the
# bifurcation conditions are those of Dangelmayr & Knobloch (D&K), "The Takens–Bogdanov bifurcation with O(2)-symmetry"
# (Phil. Trans. R. Soc. Lond. A 322, 1987), here with A = Γ, M = K2 + K3 and D = K2 - K3.
using Roots, ForwardDiff
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
    Λ = β/q - Pe*(ϕas - ϕ)/(q^2*(1 - ϕ)^2)  # solvability normalisation 
    # η¹ = ϕa¹Π + μ¹Ω
    Π = 2Pe*q^2/Λ
    Ω = 2q^2*(2ϕas - 1)/Λ
    Kp = q*(Pe^2*(1 - 2ϕas) + 3q^2); Km = q*(Pe^2*(1 - 2ϕas) - 3q^2)
    # C's (stationary second-order harmonics), as in coefs_stationary: the e^{2iqx} part of (ρ, ρa, m) is 2qPeβ C A²
    den = Pe^2*(1 - ϕ)*(1 - ϕas) - 2q^2*(1 - ϕ) - 1
    C1 = -inv(4q^2 + 2q^2*Pe^2*(1 - ϕ)*(4ϕas - 3)/den) + 0im
    C2 = (1 - ϕas)/(1 - ϕ)*C1
    C3 = -(i*q*Pe*(4ϕas - 3))/den*C1
    Cρ, Cρa, Cm = C1, C2, C3
    Γ = (-2q*Pe*β/Λ)*(i*Kp*C3 + Km*β*C1 + 2Pe*q^2*C2 - 4Pe*q^2*α*C1)  
    # D, D*, W: needed at O(δ) of the solvability condition (terms in |A|²A' and A²conj(A'))
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

# ====================================================================================================
# The periodic orbits with J = 0 of D&K §5 (Γ < 0): CP (their SW2) and MPS (their SW3), and the steady branches
# H, PS, TW along a path. D&K's (μ, ν, A) are our (η¹, K1, Γ); each orbit is labelled by the elliptic modulus k, here
# through m = k² (k'² = 1 - m)
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

# Jacobi elliptic functions sn, cn, dn of (u | m) by the same AGM (descending Landen transformation, Abramowitz & Stegun
# 16.4): a_n, b_n, c_n as above until c_N ≈ 0, then φ_N = 2^N a_N u, φ_{n-1} = [φ_n + asin(c_n sin φ_n / a_n)]/2 and
# sn = sin φ_0, cn = cos φ_0, dn = (cn² + k'² sn²)^{1/2} (= (1 - k² sn²)^{1/2} without the cancellation; dn > 0)
function ellipj(u, m)
    0 <= m < 1 || error("ellipj needs 0 ≤ m < 1 (m = $m)")
    a = [1.0]; c = [sqrt(m)]; b = sqrt(1 - m)
    while abs(c[end]) > eps(a[end])
        push!(c, (a[end] - b)/2); an = (a[end] + b)/2; b = sqrt(a[end]*b); push!(a, an)
    end
    N = length(a) - 1
    φ = 2.0^N*a[end]*u
    for n in N:-1:1
        φ = (φ + asin(c[n+1]/a[n+1]*sin(φ)))/2
    end
    sn, cn = sin(φ), cos(φ)
    return sn, cn, sqrt(cn^2 + (1 - m)*sn^2)
end

"""
Orbit of family `fam` (:CP or :MPS) with k = √m at η¹ = ηv, in r² = |A|² (D&K's roots s1, s2, s3):
- CP (SW2, (5.12)): As1 = -2μk²/(k²-k'²), As3 = 2μk'²/(k²-k'²), s2 = 0; r² = s1 cn²θ, θ = [A(s3-s1)/2]^{1/2} T.
  μ(k²-k'²) ≥ 0: m < 1/2 for η¹ < 0 (from the Hopf of H at m → 0) and m > 1/2 for η¹ > 0 (homoclinic at m → 1)
- MPS (SW3, (5.20)): As1 = -2μ/(1+k'²), As2 = -2μk'²/(1+k'²), s3 = 0; r² = s1 dn²θ, θ = (-As1/2)^{1/2} T.
  η¹ > 0, from the Hopf of PS at m → 0 to the homoclinic orbit at m → 1
Returns the roots, the range [xmin, xmax] of the signed amplitude a (xmax = √s1; CP: a changes sign) and its period
(cn: 4K in θ, dn: 2K in θ).
"""
function sw_orbit(c, fam, ηv, m)
    A = c.Γ; μ = ηv; kp2 = 1 - m; Kk = ellipKE(m)[1]
    if fam == :CP
        d = m - kp2  # k² - k'²
        s1 = -2μ*m/(A*d); s3 = 2μ*kp2/(A*d)
        return (s1=s1, s2=0.0, s3=s3, xmin=-sqrt(s1), xmax=sqrt(s1), period=4Kk/sqrt(A*(s3 - s1)/2))
    end
    q = 1 + kp2; s1 = -2μ/(A*q); s2 = -2μ*kp2/(A*q)
    return (s1=s1, s2=s2, s3=0.0, xmin=sqrt(s2), xmax=sqrt(s1), period=2Kk/sqrt(-A*s1/2))
end

"""
Persistence condition of the orbit with k = √m, written like the rays of kappa_constants as the line ΓK1 = c(k)Mη¹:
- CP:  5ΓK1 = -2Mη¹R3(k) (5.13), R3 = [k'²(1+k'²) - 2(k'²+k⁴)Φ] / ((k²-k'²)[k'² + (k²-k'²)Φ]) (5.14)
- MPS: 5ΓK1 = 2Mη¹R5(k) (5.21), R5 = [k'²(1+k'²) - 2(k'²+k⁴)Φ] / ((1+k'²)[2k'² - (1+k'²)Φ]) (5.22)
with Φ = ℰ(k)/𝒦(k) (4.12). For k → 0 both fractions are 0/0 and lose all their digits, so below m = 1e-4 D&K's
expansions R3 ~ 5k²/4 and R5 ~ 5/2 - 5k⁴/64 are used instead.
"""
function sw_line(fam, m)
    kp2 = 1 - m                                    # k'² (m = k²)
    if m < 1e-4                                    # small k: D&K's expansions
        R3 = 5/4*m                                 # R3 ~ 5k²/4
        R5 = 5/2 - 5/64*m^2                        # R5 ~ 5/2 - 5k⁴/64
    else
        Kk, Ek = ellipKE(m); Φ = Ek/Kk             # Φ = ℰ/𝒦 (4.12)
        num = kp2*(1 + kp2) - 2(kp2 + m^2)*Φ       # numerator of both R3 and R5
        R3 = num/((m - kp2)*(kp2 + (m - kp2)*Φ))   # (5.14)
        R5 = num/((1 + kp2)*(2kp2 - (1 + kp2)*Φ))  # (5.22)
    end
    if fam == :CP
        return -2/5*R3                             # (5.13): 5ΓK1 = -2Mη¹R3
    else
        return 2/5*R5                              # (5.21): 5ΓK1 = 2Mη¹R5
    end
end

# The same line as a value of κ = K1|Γ|/(M|η¹|): ΓK1 = cMη¹ is κ = -σc for Γ < 0 (σ = sign η¹)
kappa_sw(fam, σ, m) = -σ*sw_line(fam, m)

# Interval of m of the family
mrange(fam, σ) = fam == :MPS ? (0.0, 1.0) : (σ < 0 ? (0.0, 0.5) : (0.5, 1.0))

# Grid in m, logarithmic down to 1e-11 from both ends (Hopf, homoclinic, large amplitude), uniform in between
function mgrid(fam, σ; n=4000)
    a, b = mrange(fam, σ)
    lo = a .+ (b - a) .* exp10.(range(-11, -1; length=n ÷ 2)) ./ 2
    hi = b .- (b - a) .* exp10.(range(-1, -11; length=n ÷ 2)) ./ 2
    return vcat(lo, range(lo[end], hi[1]; length=n)[2:end-1], hi)
end

# The MPS at (η¹, K1): the k whose persistence line passes through the point (bisection in m), or nothing outside the
# region between the Hopf of PS and the homoclinic line where it exists (MPS stable there, see lzero_branch)
function mps_orbit(c, ηv, K1v)
    (c.Γ < 0 && ηv > 0) || return nothing
    κ = K1v*abs(c.Γ)/((c.K2 + c.K3)*ηv)
    f(m) = kappa_sw(:MPS, 1, m) - κ
    a, b = 1e-11, 1 - 1e-11
    f(a)*f(b) < 0 || return nothing
    m = find_zero(f, (a, b), Bisection())
    return merge((kind=:MPS, k=sqrt(m)), sw_orbit(c, :MPS, ηv, m), (stable=true,))
end

# Stationary pattern PS, A = r real: r² = -η¹/Γ (D&K (3.13)). Stable (Γ < 0, η¹ > 0) where f = K1 + Mr² < 0 and
# g = K1 + Dr² < 0, i.e. ΓK1 > Mη¹ and ΓK1 > Dη¹: beyond the Hopf of PS (3.17) and the drift line (3.16)
function PS_solution(c, ηv, K1v)
    r2 = -ηv/c.Γ
    r2 > 0 || return nothing
    f = K1v + (c.K2 + c.K3)*r2; g = K1v + (c.K2 - c.K3)*r2
    return (r=sqrt(r2), stable=(ηv > 0 && f < 0 && g < 0))
end

# Travelling wave TW, A = r e^{iΩT}: r² = -K1/D, D = K2 - K3 (D&K (3.18)), and Ω² = -η¹ - Γr² (Ω → 0 on the drift line
# (3.16), where the TW ends on the PS). Unstable wherever it exists (D&K)
function TW_solution(c, ηv, K1v)
    D = c.K2 - c.K3
    r2 = -K1v/D; r2 > 0 || return nothing
    Ω2 = -ηv - c.Γ*r2; Ω2 > 0 || return nothing
    return (r=sqrt(r2), Ω=sqrt(Ω2), stable=false)
end


# =========================================================================================================
# Straight paths (ϕa¹, μ¹) = (ϕa¹₀ + v_ϕ s, μ¹₀ + v_μ s): branches, stability and bifurcation points
# =========================================================================================================
pathpoint(P, s) = (P.ϕa0 + P.vϕ*s, P.μ0 + P.vμ*s)  # (ϕa¹, μ¹) at s

ηs(c, P, s) = η(c, pathpoint(P, s)...)

K1s(c, P, s) = K1(c, pathpoint(P, s)...)


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

# Bifurcation rays of D&K with η¹ > 0, as κ = K1|Γ|/(M|η¹|) (for Γ < 0, ΓK1 = c Mη¹ is the ray κ = -c); the other two,
# the pitchfork η¹ = 0 (3.11) and the Hopf K1 = 0, η¹ < 0 (3.12) of H, are fixed lines (wnl_rays, path_events):
#   Hopf of PS     ΓK1 = Mη¹      (3.17)  ->  κ = -1
#   drift of PS    ΓK1 = Dη¹      (3.16)  ->  κ = -D/M
#   homoclinic     5ΓK1 = 4Mη¹    (5.19)  ->  κ = -4/5
#   fold of CP     ΓK1 ≈ 0.74Mη¹  (5.18)  ->  κ = -0.74
kappa_constants(c) = (hopf_PS=-1.0, drift_PS=-(c.K2 - c.K3)/(c.K2 + c.K3), hom=-4/5, fold_CP=-0.74)


"""
Branch of family `fam` (:CP or :MPS) for sign σ of η¹ on the path P, traced in m = k²: the path crosses the persistence
line of each orbit (sw_line) at one s (K1 and η¹ are linear in s), where the orbit is sw_orbit. Returns the points
inside [smin, smax], ordered in m. Stability from the diagram (the arcs of the schematic, from D&K): MPS stable, CP
unstable; wnl_diagram then marks the CP between the homoclinic and fold points of the path as stable.
"""
function lzero_branch(c, P, fam, σ, smin, smax; n=3000)
    c.Γ < 0 || return NamedTuple[]
    pts = NamedTuple[]
    for m in mgrid(fam, σ; n=n)
        κ = kappa_sw(fam, σ, m)
        abs(κ) < 1e7 || continue            # closer to η¹ = 0 than round-off allows 
        s = s_at_kappa(c, P, κ, σ)
        (s === nothing || !(smin <= s <= smax)) && continue
        push!(pts, merge((s=s, kind=fam, k=sqrt(m)), sw_orbit(c, fam, ηs(c, P, s), m), (stable=fam == :MPS,)))
    end
    return pts
end

# H, PS and TW (with their stability) at each s of ss; PS and TW only where they exist. H is stable for η¹ < 0 and
# K1 < 0: below the pitchfork (3.11) and the Hopf bifurcation (3.12)
function steady_branches(c, P, ss)
    H = [(s=s, stable=(ηs(c, P, s) < 0 && K1s(c, P, s) < 0)) for s in ss]
    PS = [merge((s=s,), o) for s in ss for o in (PS_solution(c, ηs(c, P, s), K1s(c, P, s)),) if o !== nothing]
    TW = [merge((s=s,), o) for s in ss for o in (TW_solution(c, ηs(c, P, s), K1s(c, P, s)),) if o !== nothing]
    return (H=H, PS=PS, TW=TW)
end


"""
Bifurcation points of the path in [smin, smax]: where it crosses the bifurcation sets of D&K, the pitchfork (η¹ = 0,
(3.11)) and the Hopf bifurcation (K1 = 0, η¹ < 0, (3.12)) of H and the rays of `kc` (kappa_constants; all with η¹ > 0,
so only for Γ < 0). `ends` is the branch that starts or ends there: wnl_diagram puts the event at its amplitude.
"""
function path_events(c, P, smin, smax, δ, kc)
    # the event in physical parameters: ϕa = ϕa* + δ²ϕa¹, Pe = Pe* + δ²μ¹
    phys(s) = (ϕa1 = pathpoint(P, s)[1]; μ1 = pathpoint(P, s)[2];
               (ϕa1=ϕa1, μ1=μ1, η=ηs(c, P, s), K1=K1s(c, P, s), ϕa=c.ϕas + δ^2*ϕa1, Pe=c.Pe + δ^2*μ1))
    sh = s_K10(c, P)
    # (s, type, branches that meet there, branch that starts or ends there)
    cands = Tuple{Union{Float64,Nothing},String,String,String}[(ηs(c, P, sh) < 0 ? sh : nothing, "primary Hopf of H (D&K (3.12): K1 = 0, η¹ < 0)", "H, TW (±), CP", "H"),
             (s_eta0(c, P), "stationary bifurcation of H (D&K (3.11): η¹ = 0)", "H, PS", "H")]
    c.Γ < 0 && append!(cands, [
             (s_at_kappa(c, P, kc.hopf_PS, 1), "secondary Hopf of PS (D&K (3.17): ΓK1 = Mη¹)", "PS, MPS", "PS"),
             (s_at_kappa(c, P, kc.drift_PS, 1), "drift (phase) instability of PS: TW ends on PS (D&K (3.16): ΓK1 = Dη¹)",
              "PS, TW (±)", "TW"),
             (s_at_kappa(c, P, kc.hom, 1), "homoclinic orbit of H (D&K (5.19): 5ΓK1 = 4Mη¹)", "MPS, CP", "MPS_1"),
             (s_at_kappa(c, P, kc.fold_CP, 1), "fold of CP periodic orbits (D&K (5.18))", "CP (stable), CP (unstable)", "CP_1")])
    ev = [merge((s=s, type=t, branches=b, ends=e), phys(s)) for (s, t, b, e) in cands if s !== nothing && smin <= s <= smax]
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
# Weakly nonlinear MPS as fields: the L = 0 orbit A = a(T) e^{iθ} (T = δt) of a'' = η¹a + Γa³, in closed form
# a = √s1 dn(ωT, k), ω = (-Γs1/2)^{1/2} (D&K: r² = s1 dn²θ, (5.20)), to O(δ²),
#   (ρ, ρa, m) = (φ, φa, 0) + δ(a v₀ e^{iqx} + c.c.) + δ²(a' v₁ e^{iqx} + a² C e^{2iqx} + c.c.)
# with v₀ the null vector of the linear operator at the BT point (ρ component 1), v₁ its generalised eigenvector
# (L v₁ = v₀, ρ component 0) and C = -L(2q)⁻¹N₂
# ====================================================================================================

function bt_mps_wnl(c, ϕa1, μ1, δ)
    ϕ = c.ϕ; q = c.q; ϕas = c.ϕas; Pe = c.Pe
    L(k) = ComplexF64[-k^2 0 -im*k*Pe*(1-ϕ); -ϕas*k^2 -(1-ϕ)*k^2 -im*k*Pe*(1-ϕ); im*k*Pe*ϕas -im*k*Pe*(1-ϕ) -(1-ϕ)*k^2-2]
    L1 = L(q); v0 = vcat(1.0, -(L1[2:3, 2:3] \ L1[2:3, 1])); v1 = vcat(0.0, L1[:, 2:3] \ v0)
    # N₂ at e^{2iqx}: from Pe∂x(ρm) in the ρ and ρa equations and Pe∂x(ρρa) in the m equation (the
    # diffusive quadratic terms cancel); C = 2qPeβ c.C
    C = -(L(2q) \ (2im*q*Pe .* [v0[1]*v0[3], v0[1]*v0[3], v0[1]*v0[2]]))
    ηv = η(c, ϕa1, μ1)
    o = mps_orbit(c, ηv, K1(c, ϕa1, μ1))
    o === nothing && return nothing
    # a(T) = √s1 dn(ωT, k) from its maximum at T = 0, and a'(T) = -√s1 ω k² sn cn (d dn/du = -k² sn cn)
    m = o.k^2; ω = sqrt(-c.Γ*o.s1/2); r1 = sqrt(o.s1)
    P = o.period/δ                                          # period in t (2𝒦 in ωT)
    function field(x, t)
        sn, cn, dn = ellipj(ω*δ*mod(t, P), m)
        a = r1*dn; da = -r1*ω*m*sn*cn
        e1 = exp(im*q*x)
        u = 2 .* real.(δ*a .* v0 .* e1 .+ δ^2*da .* v1 .* e1 .+ δ^2*a^2 .* C .* e1^2)
        return [ϕ + u[1], ϕas + δ^2*ϕa1 + u[2], u[3]]
    end
    return (field=field, period=P, orbit=o, v0=v0, v1=v1, C=C)
end


"""
    bt_homoclinic_wnl(c, P; n=4001, τmax=25.0)

Weakly nonlinear homoclinic orbit on the path P. For real A = a (J = 0) the amplitude equation is
a'' = η¹a + Γa³ + δ(K1 + M a²)a', M = K2 + K3. At O(1) the orbit homoclinic to H (energy E = 0) is a = A sech τ,
τ = √η¹ T, A = √(2η¹/|Γ|), for every η¹ > 0; the point s of the path is where it crosses the homoclinic ray of D&K,
5ΓK1 = 4Mη¹ (5.19). Returns s and a, da/dT along τ ∈ [-τmax, τmax].
"""
function bt_homoclinic_wnl(c, P; n=4001, τmax=25.0)
    s = s_at_kappa(c, P, kappa_constants(c).hom, 1)
    s === nothing && return nothing
    ηv = ηs(c, P, s); k = sqrt(ηv); A = sqrt(2ηv/abs(c.Γ))  # τ = kT
    τ = range(-τmax, τmax; length=n)
    return (s=s, a=A .* sech.(τ), da=-k*A .* sech.(τ) .* tanh.(τ))
end
