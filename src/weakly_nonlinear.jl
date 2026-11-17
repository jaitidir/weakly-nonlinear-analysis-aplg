# Linear stability of the homogeneous state and weakly nonlinear (amplitude-equation) solutions, stationary and
# oscillatory. Fields are (ρ, ρ_a, m); φ = mean ρ, φa = mean ρ_a; ϕac is the onset, qc the critical wavenumber, wc
# the frequency at onset, and φa = ϕac + δ²ϕ1.

# Thresholds of the stationary and oscillatory instabilities at wavenumber q. Linear operator at e^{iqx}, about
# (ρ, ρ_a, m) = (ϕ, ϕa, 0):  L(q) = [-q² 0 -iqPe(1-ϕ); -ϕa q² -(1-ϕ)q² -iqPe(1-ϕ); iqPeϕa -iqPe(1-ϕ) -(1-ϕ)q²-2].
# phia_stationary: det L(q) = 0 (a real eigenvalue crosses 0; ϕa above it is unstable).
# phia_oscillatory: Routh–Hurwitz/Hopf condition (an imaginary pair ±iω crosses); it is only meaningful where
# ω² = F_omega > 0 (below, find_qc_phiac_wc), otherwise it is a spurious root.
phia_stationary(q, ϕ, Pe) = (2 + (1 - ϕ) * Pe^2 + (1 - ϕ) * q^2) / (2 * Pe^2 * (1 - ϕ))
phia_oscillatory(q, ϕ, Pe) = 2 * (1 + (1 - ϕ) * q^2) * (2 + 2 * (1 - ϕ) + (1 - ϕ)^2 * Pe^2 + (2 - ϕ)^2 * q^2) /
                             ((1 - ϕ) * Pe^2 * (2 + q^2))

# Onset over the wavenumbers of the box, q = 2πn/Lx for n = 1, ..., up to q ≈ 200 (stationary diagrams): returns (qc, ϕac)
function find_qc_and_phiac(ϕ::Float64, Pe::Float64, Lx::Float64; n_min::Int=1, n_max::Int=ceil(Int, 200Lx/(2π)))
    q_vals = 2π .* collect(n_min:n_max) ./ Lx
    phi_osc_vals = phia_oscillatory.(q_vals, ϕ, Pe)
    idx = argmin(phi_osc_vals)
    # phia_stationary grows with q, so its lowest box mode is n = 1. qc is the minimiser of the oscillatory threshold
    # even when the stationary one is lower (in practice both are n = 1, q = 2π/Lx)
    return q_vals[idx], min(phia_stationary(2π/Lx, ϕ, Pe), phi_osc_vals[idx])
end

# Onset over the wavenumbers of the box, q = 2πn/Lx (oscillatory diagrams): returns (qc, ϕac, wc, :oscillatory or
# :stationary)
function find_qc_phiac_wc(ϕ::Float64, Pe::Float64, Lx::Float64; n_min::Int=1, n_max::Int=1000)
    # ω² of the imaginary pair ±iω of L(q) on the oscillatory threshold
    F_omega(q, ϕa) = q^2 * (2 + 2 * (1 - ϕ) * (1 + q^2) + (1 - ϕ)^2 * (Pe^2 + q^2) - ϕa * (1 - ϕ) * Pe^2)
    q_vals = 2π .* collect(n_min:n_max) ./ Lx
    phi_osc_vals = phia_oscillatory.(q_vals, ϕ, Pe)
    idx = argmin(phi_osc_vals)
    qc = q_vals[idx]
    phi_osc_inf = phi_osc_vals[idx]
    ϕac = min(phia_stationary(2π/Lx, ϕ, Pe), phi_osc_inf)
    inst_type = abs(ϕac - phi_osc_inf) < 1e-10 ? :oscillatory : :stationary
    # wc is the onset frequency only for :oscillatory (for :stationary it is F_omega at the stationary threshold)
    return qc, ϕac, sqrt(F_omega(qc, ϕac)), inst_type
end

# ------------------------------------------------------------------------------------------------------------------
# Stationary instability
# ------------------------------------------------------------------------------------------------------------------

# Coefficients of the stationary amplitude equation dA/dT = ϕ1 Π A + Γ |A|²A (T = δ²t), for the mode
# A v e^{iq_c x} with critical eigenvector v = (1, α, iβ) (ρ component 1)
function coefs_stationary(params; tol=1e-14, check_denoms=true)
    Pe, q_c, phi, phi_a_star = params.Pe, params.q_c, params.phi, params.phi_a_star
    i = im
    α = (1 - phi_a_star) / (1 - phi)  # ρ_a component of v
    β = q_c / (Pe * (1 - phi))         # m component of v is iβ
    Λ = 2 * Pe * phi_a_star - Pe * α + β * q_c  # solvability normalisation: Π = 2Pe q_c²/Λ = dσ/dϕa at onset
    Kp = -2 * Pe^2 * q_c * phi_a_star + 3 * q_c^3 + q_c * Pe^2  # = q_c (Pe²(1 - 2ϕa*) + 3q_c²)
    Km = -2 * Pe^2 * q_c * phi_a_star - 3 * q_c^3 + q_c * Pe^2  # = q_c (Pe²(1 - 2ϕa*) - 3q_c²)
    if check_denoms
        abs(1 - phi) < tol && error("1 - phi is ~0 -> singular scaling")
        abs(Pe) < tol && error("Pe is ~0 -> β singular")
        abs(Λ) < tol && error("Lambda Λ is ~0 (|Λ|=$(abs(Λ))) -> Pi/Gamma singular")
    end
    denom_beta = Pe^2 * (1 - phi) * (1 - phi_a_star) - 2 * q_c^2 * (1 - phi) - 1
    check_denoms && abs(denom_beta) < tol && error("C1/C3 denominator is ~0")
    # Second harmonic (O(δ²), e^{2iq_c x}) of (ρ, ρ_a, m): it is 2q_c Pe β (C1, C2, C3) A² (C's scaled by 1/(2q_c Pe β))
    C1 = -inv(4 * q_c^2 + 2 * q_c^2 * Pe^2 * (1 - phi) * (4 * phi_a_star - 3) / denom_beta)
    C2 = α * C1
    C3 = -(i * q_c * Pe * (4 * phi_a_star - 3)) * (C1 / denom_beta)
    Π = (2 * Pe * q_c^2) / Λ
    # O(δ³) solvability condition: projection of the resonant cubic terms (through the second harmonic) onto e^{iq_c x}
    bracket = i * Kp * C3 + Km * β * C1 + 2 * Pe * q_c^2 * C2 - 4 * Pe * q_c^2 * (conj(α) * C1)
    Γ = (-2 * q_c * Pe * β / Λ) * bracket
    return (alpha=α, beta=β, Lambda=Λ, Kplus=Kp, Kminus=Km, C1=C1, C2=C2, C3=C3, bracket=bracket, Gamma=Γ, Pi=Π)
end

# true if the stationary bifurcation is subcritical (Π > 0, so the branch |A|² = -ϕ1Π/Γ lies at ϕ1 < 0 iff Π/Γ ≥ 0).
# Any error (a singular coefficient) is also reported as subcritical
function stationary_subcritical(ϕ, Pe, qc, ϕac)
    try
        out = coefs_stationary((Pe=Pe, q_c=qc, phi=ϕ, phi_a_star=ϕac))
        return real(out.Pi / out.Gamma) >= 0
    catch
        return true
    end
end

# Weakly nonlinear stationary solution [ρ, ρ_a, m] at x, leading order: (ϕ, ϕa, 0) + 2δ|A| Re(v e^{iqc x}),
# |A|² = -Π ϕ1/Γ; ϕa (= phi_a_star + δ²ϕ1 by the caller) is only the mean of ρ_a
function weaklynonlinear_stat(x, Pe, qc, ϕ, phi_a_star, ϕa, delta, ϕ1)
    out = coefs_stationary((Pe=Pe, q_c=qc, phi=ϕ, phi_a_star=phi_a_star))
    Γ, Π = out.Gamma, out.Pi
    i = im
    real(Π * ϕ1 / Γ) >= 0 && error("No real solution")
    A1eq = sqrt(-real(Π * ϕ1 / Γ))
    v = ComplexF64[1, (1 - phi_a_star) / (1 - ϕ), i * qc / Pe / (1 - ϕ)]  # null vector of L(qc) at phi_a_star
    return [ϕ, ϕa, 0] .+ 2 .* delta .* A1eq .* real(v .* exp(i * qc * x))
end

# +1 if the stationary branch is real for ϕa = ϕac + ε ϕ1 (supercritical), -1 if only for ϕac - ε ϕ1 (subcritical)
function branch_sign(ϕ, Pe, qc, ϕac, ϕ1)
    out = coefs_stationary((Pe=Pe, q_c=qc, phi=ϕ, phi_a_star=ϕac))
    return real(out.Pi * ϕ1 / out.Gamma) < 0 ? 1.0 : -1.0
end

# Peak-to-peak amplitudes (Δρ, Δρ_a, Δm) of the weakly nonlinear stationary solution on the grid x
function wnl_stat_amplitudes(x, Pe, qc, ϕ, ϕac, ϕa, δ, ϕ1)
    z = [weaklynonlinear_stat(xi, Pe, qc, ϕ, ϕac, ϕa, δ, ϕ1) for xi in x]
    return Tuple(maximum(real(zi[k]) for zi in z) - minimum(real(zi[k]) for zi in z) for k in 1:3)
end

# ------------------------------------------------------------------------------------------------------------------
# Oscillatory instability: travelling modes A⁺ e^{i(qc x + wc t)} (:plus), A⁻ e^{i(qc x - wc t)} (:minus) and their
# coexistence (:coexistence, the counter-propagating pattern CP)
# ------------------------------------------------------------------------------------------------------------------

# Second-order harmonics (O(δ²), e^{2iq_c x}) of the oscillatory expansion, components (C1, C2, C3) ~ (ρ, ρ_a, m), with
# μ = C1/C3. compute_plus_minus(±w): harmonics of (A±)² at frequency ±2w; compute_zero(±w): the two pieces of the
# zero-frequency harmonic A⁺A⁻ e^{2iq_c x}, combined in D1–D3 of coefs_osc (cross-coupling Σ±)
function compute_Cs(Pe, q_c, phi, phi_a_star, w)
    i = im
    function compute_plus_minus(ω)
        num = (i*q_c^2 - ω) * (q_c^2*(1-phi) + i*ω) * (2*q_c^2*(1-phi) + 1 + i*ω) -
              i*(Pe*q_c)^2 * (1-phi)^2 * (q_c^2*(1-phi_a_star) + i*ω)
        den = Pe*q_c*(1-phi) * (q_c^2*(1-phi_a_star) + i*ω) * (2*q_c^2 + i*ω) -
              Pe*q_c * (q_c^2 + i*ω) * (q_c^2*(1-phi) + i*ω) *
              (phi_a_star - (1-phi) * (2*q_c^2*(1-phi_a_star) + i*ω) / (2*q_c^2*(1-phi) + i*ω))
        μ  = num / den
        C1 = -inv(4*q_c^2 + 2*i*ω + 2*i*q_c*Pe*(1-phi)/μ)
        C2 = ((2*q_c^2*(1-phi_a_star) + i*ω) / (2*q_c^2*(1-phi) + i*ω)) * C1
        C3 = (1/μ) * C1
        return C1, C2, C3, μ
    end
    function compute_zero(ω)
        num = (i*q_c^2 - ω) * (q_c^2*(1-phi) + i*ω) * (2*q_c^2*(1-phi) + 1) -
              i*(Pe*q_c)^2 * (1-phi)^2 * (q_c^2*(1-phi_a_star) + i*ω)
        den = Pe*q_c*(1-phi) * (q_c^2*(1-phi_a_star) + i*ω) * 2*q_c^2 -
              Pe*q_c * (q_c^2 + i*ω) * (q_c^2*(1-phi) + i*ω) * (2*phi_a_star - 1)
        μ0 = num / den
        C1 = -inv(4*q_c^2 + 2*i*q_c*Pe*(1-phi)/μ0)
        C2 = ((1-phi_a_star) / (1-phi)) * C1
        C3 = (1/μ0) * C1
        return C1, C2, C3, μ0
    end
    C1p, C2p, C3p, μp = compute_plus_minus(w)
    C1m, C2m, C3m, μm = compute_plus_minus(-w)
    C1z_p, C2z_p, C3z_p, μz_p = compute_zero(w)
    C1z_m, C2z_m, C3z_m, μz_m = compute_zero(-w)
    return (C1p=C1p, C2p=C2p, C3p=C3p, mup=μp, C1m=C1m, C2m=C2m, C3m=C3m, mum=μm,
            C1zp=C1z_p, C2zp=C2z_p, C3zp=C3z_p, muzp=μz_p, C1zm=C1z_m, C2zm=C2z_m, C3zm=C3z_m, muzm=μz_m)
end

# Coefficients of the coupled amplitude equations dA±/dT = ϕ1 Π± A± + Γ± |A±|²A± + Σ± |A∓|²A± (T = δ²t) for the modes
# A± v± e^{i(q_c x ± w t)}, v± = (1, α±, β±) the eigenvectors of L(q_c) for ±iw. Π± = dλ±/dϕa at onset; Γ⁻ = conj(Γ⁺),
# Σ⁻ = conj(Σ⁺), Π⁻ = conj(Π⁺)
function coefs_osc(params, C; tol=1e-14, check_denoms=true)
    Pe, q_c, phi, phi_a_star, w = params.Pe, params.q_c, params.phi, params.phi_a_star, params.w
    i = im
    # sgn = +1 for A⁺ (eigenvalue +iw), -1 for A⁻
    function pm_coefs(sgn)
        σ = q_c^2 * (1 - phi) + sgn * i * w
        β = (i * q_c^2 - sgn * w) / (q_c * Pe * (1 - phi))  # m component of v±
        κ = (2 * q_c^2 * (1 - phi) + sgn * i * w) / (q_c^2 + sgn * i * w)
        α = (q_c^2 * (1 - phi_a_star) + sgn * i * w) / σ  # ρ_a component of v±
        a = Pe^2 * q_c^2 * ((1 - phi) - phi_a_star * κ)
        Λ = Pe * phi_a_star * κ / (1 - phi) - Pe * α - i * σ * β / (q_c * (1 - phi))  # solvability normalisation
        return (σ=σ, β=β, κ=κ, α=α, a=a, Λ=Λ)
    end
    cp = pm_coefs(+1)
    cm = pm_coefs(-1)
    if check_denoms
        for (name, val) in [("σ⁺", cp.σ), ("σ⁻", cm.σ), ("q_c²+iw", q_c^2 + i*w), ("q_c²-iw", q_c^2 - i*w),
                            ("q_c·Pe·(1-phi)", q_c * Pe * (1 - phi)), ("Λ⁺", cp.Λ), ("Λ⁻", cm.Λ)]
            abs(val) < tol && error("$name ≈ 0 -> singular")
        end
    end
    C1p, C2p, C3p = C.C1p, C.C2p, C.C3p
    C1m, C2m, C3m = C.C1m, C.C2m, C.C3m
    # zero-frequency harmonic of A⁺A⁻ (enters only the cross-coupling Σ±)
    D1 = cp.β * C.C1zp + cm.β * C.C1zm
    D2 = cp.β * C.C2zp + cm.β * C.C2zm
    D3 = cp.β * C.C3zp + cm.β * C.C3zm
    Πp = (2 * Pe * q_c^2 + i * Pe * w / (1 - phi)) / cp.Λ
    Πm = (2 * Pe * q_c^2 - i * Pe * w / (1 - phi)) / cm.Λ
    # O(δ³) solvability: Γ± from the (A±)² harmonics (α_cross: ρ_a component of the other eigenvector, conj(v±) = v∓
    # for that component), Σ± from the A⁺A⁻ harmonic
    pref_Γ(β, Λ) = -2 * Pe * β / ((1 - phi) * Λ)
    pref_Σ(Λ) = -2 * Pe / ((1 - phi) * Λ)
    bracket_Γ(σ, a, β, C1, C2, C3, α_cross) =
        a * (conj(β) * C1 + C3) - 3 * i * q_c^3 * Pe * (1 - phi) * (C2 - α_cross * C1) +
        3 * q_c^2 * σ * (C3 - conj(β) * C1) + i * σ * q_c * Pe * (α_cross * C1 + C2)
    Γp = pref_Γ(cp.β, cp.Λ) * bracket_Γ(cp.σ, cp.a, cp.β, C1p, C2p, C3p, cm.α)
    Γm = pref_Γ(cm.β, cm.Λ) * bracket_Γ(cm.σ, cm.a, cm.β, C1m, C2m, C3m, cp.α)
    bracket_Σ(σ, a, β_cross, α_same) =
        a * (conj(β_cross) * D1 + D3) - 3 * i * q_c^3 * Pe * (1 - phi) * (D2 - α_same * D1) +
        3 * q_c^2 * σ * (D3 - conj(β_cross) * D1) + i * σ * q_c * Pe * (α_same * D1 + D2)
    Σp = pref_Σ(cp.Λ) * bracket_Σ(cp.σ, cp.a, cm.β, cp.α)
    Σm = pref_Σ(cm.Λ) * bracket_Σ(cm.σ, cm.a, cp.β, cm.α)
    return (Pip=Πp, Pim=Πm, Gammap=Γp, Gammam=Γm, Sigmap=Σp, Sigmam=Σm)
end

# (Pip, Pim, Gammap, Gammam, Sigmap, Sigmam) at the onset (qc, ϕac, w)
osc_coefficients(Pe, qc, ϕ, ϕac, w) = coefs_osc((Pe=Pe, q_c=qc, phi=ϕ, phi_a_star=ϕac, w=w),
                                                compute_Cs(Pe, qc, ϕ, ϕac, w); tol=1e-14)

# Stability of a fixed point of the amplitude equations from its two eigenvalues
classify_eigenvalues(λ1, λ2) = (λ1 > 0 && λ2 > 0) ? :unstable : (λ1 < 0 && λ2 < 0) ? :stable : :saddle

# Existence and stability of T (one travelling mode) and CP (both modes) with ϕ1 = ±1 on the side given; used along
# the spinodal. Returns (has_T, has_CP, stab_T, stab_CP); :none where the branch does not exist.
# Only real parts matter: with A± = a± e^{iθ±}, (a±²)' = 2a±²(ϕ1Πr± + Γr± a±² + Σr± a∓²)
function osc_classes(Pe, qc, ϕ, ϕac, wc; ϕ1=1.0)
    out = osc_coefficients(Pe, qc, ϕ, ϕac, wc)
    Πrp, Γrp, Σrp = real(out.Pip), real(out.Gammap), real(out.Sigmap)
    Πrm, Γrm, Σrm = real(out.Pim), real(out.Gammam), real(out.Sigmam)
    lm = (Γrm != 0) && (-ϕ1 * Πrm / Γrm > 0)  # T, here as the A⁻ wave (same as A⁺ by symmetry): a₋² = -ϕ1Πr⁻/Γr⁻ > 0
    cp = false; slm = :none; scp = :none
    # CP: a₊², a₋² from the 2×2 linear system Γr± a±² + Σr± a∓² = -ϕ1 Πr± (Cramer's rule)
    denom1 = Γrm * Γrp - Σrm * Σrp
    denom2 = Σrm * Σrp - Γrm * Γrp
    if abs(denom1) > 1e-14
        a_plus  = ϕ1 * (Πrm * Σrp - Γrm * Πrp) / denom1
        a_minus = ϕ1 * (Πrm * Γrp - Σrm * Πrp) / denom2
        cp = (a_plus > 0) && (a_minus > 0)
        if cp
            # eigenvalues tr ± sd of the Jacobian of (a₊², a₋²)' at CP, 2[Γr⁺a₊² Σr⁺a₊²; Σr⁻a₋² Γr⁻a₋²]
            tr_val = Γrp * a_plus + Γrm * a_minus
            sd = sqrt(complex((Γrp * a_plus - Γrm * a_minus)^2 + 4 * Σrp * Σrm * a_plus * a_minus))
            scp = classify_eigenvalues(real(tr_val + sd), real(tr_val - sd))
        end
    end
    if lm
        # T eigenvalues: growth of the other wave ϕ1Πr⁺ + Σr⁺a₋², and its own amplitude -2ϕ1Πr⁻
        a_lm = -ϕ1 * Πrm / Γrm
        slm = classify_eigenvalues(ϕ1 * Πrp + Σrp * a_lm, -2 * ϕ1 * Πrm)
    end
    return lm, cp, slm, scp
end

# Stability (:stable, :unstable, :saddle) of the T (mode=:plus) or CP (mode=:coexistence) branch for ϕ1 = ±1
function classify_branch_stability(Pe, qc, ϕ, ϕac, ϕ1, w, mode)
    out = osc_coefficients(Pe, qc, ϕ, ϕac, w)
    Πrp, Γrp, Σrp = real(out.Pip), real(out.Gammap), real(out.Sigmap)
    Πrm, Γrm, Σrm = real(out.Pim), real(out.Gammam), real(out.Sigmam)
    if mode == :plus
        a_rm = -ϕ1 * Πrp / Γrp  # a₊² of T (named a_rm, but it is the A⁺ amplitude squared)
        return classify_eigenvalues(-2 * ϕ1 * Πrp, ϕ1 * Πrm + Σrm * a_rm)
    elseif mode == :coexistence
        # as in osc_classes (no check of denom1 ≈ 0 here)
        denom1 = Γrm * Γrp - Σrm * Σrp
        a_plus  = ϕ1 * (Πrm * Σrp - Γrm * Πrp) / denom1
        a_minus = ϕ1 * (Πrm * Γrp - Σrm * Πrp) / (-denom1)
        tr_val = Γrp * a_plus + Γrm * a_minus
        sqrt_disc = sqrt(complex((Γrp * a_plus - Γrm * a_minus)^2 + 4 * Σrp * Σrm * a_plus * a_minus))
        return classify_eigenvalues(real(tr_val + sqrt_disc), real(tr_val - sqrt_disc))
    else
        error("classify_branch_stability is implemented for :plus and :coexistence only")
    end
end

# +1 if the T branch (mode :plus or :minus) is real for ϕa = ϕac + ε ϕ1, -1 if only for ϕac - ε ϕ1
function branch_sign_osc(Pe, qc, ϕ, ϕac, ϕ1, w; mode=:plus)
    out = osc_coefficients(Pe, qc, ϕ, ϕac, w)
    Πr, Γr = mode == :plus ? (real(out.Pip), real(out.Gammap)) :
             mode == :minus ? (real(out.Pim), real(out.Gammam)) : error("mode must be :plus or :minus")
    return ϕ1 * Πr / Γr < 0 ? 1.0 : -1.0
end

# The same for the CP branch: both squared amplitudes must be non-negative
function branch_sign_osc_cp(Pe, qc, ϕ, ϕac, ϕ1, w)
    out = osc_coefficients(Pe, qc, ϕ, ϕac, w)
    Γrp, Γrm = real(out.Gammap), real(out.Gammam)
    Πrp, Πrm = real(out.Pip), real(out.Pim)
    Σrp, Σrm = real(out.Sigmap), real(out.Sigmam)
    denom = Γrm * Γrp - Σrm * Σrp
    ap2 = ϕ1 * (Πrm * Σrp - Γrm * Πrp) / denom
    am2 = ϕ1 * (Πrm * Γrp - Σrm * Πrp) / (-denom)
    return (ap2 >= 0 && am2 >= 0) ? 1.0 : -1.0
end

# Everything of the oscillatory weakly nonlinear solution that does not depend on (x, t): the mode amplitudes
# A1p, A1m, their frequency corrections θp, θm and the eigenvectors vp, vm.
# The steady states A± = a± e^{iθ± T} of the amplitude equations: real parts give a±², imaginary parts θ±.
# sqrt(abs(...)) gives an amplitude even if a±² < 0 (no branch on that side of ϕ1): the caller must pick ϕ1 accordingly
function wnl_osc_setup(Pe, qc, ϕ, phi_a_star, ϕa, delta, ϕ1, w, mode)
    out = osc_coefficients(Pe, qc, ϕ, phi_a_star, w)
    i = im
    Πrp, Πip = real(out.Pip), imag(out.Pip)
    Πrm, Πim = real(out.Pim), imag(out.Pim)
    Γrp, Γip = real(out.Gammap), imag(out.Gammap)
    Γrm, Γim = real(out.Gammam), imag(out.Gammam)
    Σrp, Σip = real(out.Sigmap), imag(out.Sigmap)
    Σrm, Σim = real(out.Sigmam), imag(out.Sigmam)
    βp = (i*qc^2 - w) / (qc * Pe * (1-ϕ))
    βm = (i*qc^2 + w) / (qc * Pe * (1-ϕ))
    vp = ComplexF64[1, (qc^2*(1-phi_a_star)+i*w)/(qc^2*(1-ϕ)+i*w), βp]
    vm = ComplexF64[1, (qc^2*(1-phi_a_star)-i*w)/(qc^2*(1-ϕ)-i*w), βm]
    if mode == :plus
        A1p = sqrt(abs(-ϕ1*Πrp/Γrp)); A1m = 0.0
        θp = ϕ1*(Πip-(Γip/Γrp)*Πrp); θm = 0.0  # θ = ϕ1Πi + Γi a² with a² = -ϕ1Πr/Γr
    elseif mode == :minus
        A1p = 0.0; A1m = sqrt(abs(-ϕ1*Πrm/Γrm))
        θp = 0.0; θm = ϕ1*(Πim-(Γim/Γrm)*Πrm)
    elseif mode == :coexistence
        denom = Γrm*Γrp - Σrm*Σrp
        abs(denom) < 1e-14 && error("Denominator ≈ 0")
        ap2 = ϕ1*(Πrm*Σrp-Γrm*Πrp)/denom  # a₊², a₋² as in osc_classes
        am2 = ϕ1*(Πrm*Γrp-Σrm*Πrp)/(-denom)
        A1p = sqrt(abs(ap2)); A1m = sqrt(abs(am2))
        θp = ϕ1*Πip+Γip*ap2+Σip*am2
        θm = ϕ1*Πim+Γim*am2+Σim*ap2
    else
        error("mode must be :plus, :minus, or :coexistence")
    end
    return (A1p=A1p, A1m=A1m, θp=θp, θm=θm, vp=vp, vm=vm)
end

# Weakly nonlinear oscillatory solution [ρ, ρ_a, m] at (x, t), from a wnl_osc_setup result:
# (ϕ, ϕa, 0) + 2δ Re(A⁺ v⁺ e^{i(qc x + w t)} + A⁻ v⁻ e^{i(qc x - w t)}), A± = a± e^{iθ± δ² t}
function wnl_osc_eval(x, t, qc, ϕ, ϕa, delta, w, setup)
    i = im; xs = x - π/(2*qc)  # shift by a quarter wavelength (a free translation)
    Ap = setup.A1p*exp(i*setup.θp*delta^2*t)*exp(i*(qc*xs+w*t))
    Am = setup.A1m*exp(i*setup.θm*delta^2*t)*exp(i*(qc*xs-w*t))
    z = [ϕ, ϕa, 0.0] .+ delta.*2 .*real(Ap.*setup.vp) .+ delta.*2 .*real(Am.*setup.vm)
    return real.(z)
end

weaklynonlinear_osc(x, t, Pe, qc, ϕ, phi_a_star, ϕa, delta, ϕ1, w, mode=:plus) =
    wnl_osc_eval(x, t, qc, ϕ, ϕa, delta, w, wnl_osc_setup(Pe, qc, ϕ, phi_a_star, ϕa, delta, ϕ1, w, mode))

# Frequency of the travelling mode with its nonlinear correction (:plus or :minus): ±w + θ±δ², with the same
# θ = ϕ1(Πi - (Γi/Γr)Πr) as wnl_osc_setup
function wnl_effective_frequency(Pe, qc, ϕ, ϕac, delta, ϕ1, w; mode=:plus)
    out = osc_coefficients(Pe, qc, ϕ, ϕac, w)
    Πrp, Πip = real(out.Pip), imag(out.Pip)
    Πrm, Πim = real(out.Pim), imag(out.Pim)
    Γrp, Γip = real(out.Gammap), imag(out.Gammap)
    Γrm, Γim = real(out.Gammam), imag(out.Gammam)
    if mode == :plus
        return w + ϕ1*(Πip - (Γip/Γrp)*Πrp)*delta^2
    elseif mode == :minus
        return -w + ϕ1*(Πim - (Γim/Γrm)*Πrm)*delta^2
    else
        error("wnl_effective_frequency is implemented for :plus and :minus only")
    end
end

# Peak-to-peak amplitudes (Δρ, Δρ_a, Δm) of the weakly nonlinear oscillatory solution
#   z_k = z0_k + 2δ Re(A₊ v₊_k e^{i(qx + ω₊t)} + A₋ v₋_k e^{i(qx - ω₋t)}),
# over all x and all times: both terms are cosines whose phases qx ± ωt run independently over [0, 2π) (one wave period
# for T; for CP the relative phase also covers [0, 2π) in each period), so the largest value is reached when they align:
#   Δz_k = 4δ (|A₊||v₊_k| + |A₋||v₋_k|)     (T: one of A₊, A₋ is zero)
function wnl_amplitude_envelope(Pe, qc, ϕ, ϕac, ϕa, δ, ϕ1, w, mode)
    s = wnl_osc_setup(Pe, qc, ϕ, ϕac, ϕa, δ, ϕ1, w, mode)
    return Tuple(4δ .* (s.A1p .* abs.(s.vp) .+ s.A1m .* abs.(s.vm)))
end
