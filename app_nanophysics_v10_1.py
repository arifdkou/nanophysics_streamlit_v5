# app_nanophysics_week10.py
# ============================================================
# Nanophysics – Week 10
# Optical Transitions in Quantum Wells and Nanostructures
# Optical Properties, Joint DOS, and Selection Rules
# ============================================================

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Streamlit page settings
# ------------------------------------------------------------
st.set_page_config(
    page_title="Nanophysics – Week 10: Optical Transitions in Quantum Wells",
    layout="wide"
)

# ------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------
h = 6.626_070_15e-34      # Planck constant (J·s)
hbar = h / (2 * np.pi)
eV = 1.602_176_634e-19    # J
m_e = 9.109_383_56e-31    # kg
c = 2.997_924_58e8        # m/s

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def E_infinite_well(n, m_eff_rel, L_nm):
    """
    Energy of n-th level in an infinite quantum well (1D confinement):

        E_n = (hbar^2 * pi^2 * n^2) / (2 m* L^2)

    Parameters
    ----------
    n         : integer or array of integers
    m_eff_rel : effective mass in units of m_e
    L_nm      : well width in nanometers

    Returns
    -------
    E_eV : energy in eV (relative to bottom of the well)
    """
    n = np.array(n, dtype=float)
    L = L_nm * 1e-9  # m
    m_eff = m_eff_rel * m_e
    E_J = (hbar ** 2) * (np.pi ** 2) * (n ** 2) / (2.0 * m_eff * L ** 2)
    E_eV = E_J / eV
    return E_eV


def lorentzian(x, x0, gamma):
    """
    Normalized Lorentzian line shape:

        L(x) = (1/pi) * (gamma / [ (x - x0)^2 + gamma^2 ])

    gamma is HWHM (half-width at half-maximum).
    """
    return (1.0 / np.pi) * (gamma / ((x - x0) ** 2 + gamma ** 2))


def qw_absorption_spectrum(E_photon_eV, Eg_eV, m_e_rel, m_h_rel, L_nm,
                           N_subbands=3, gamma_eV=0.02):
    """
    Very simple model of quantum well absorption spectrum:

    - Infinite well confinement for electrons and heavy holes.
    - Transition energy for n-th subband:

        E_trans(n) = Eg + E_e(n) + E_h(n)

    - Only transitions with Δn = 0 (n_c = n_v) are allowed.
    - Each allowed transition contributes a Lorentzian peak.

    Parameters
    ----------
    E_photon_eV : 1D array of photon energies (eV)
    Eg_eV       : bandgap of well material (eV)
    m_e_rel     : electron effective mass (m*/m_e)
    m_h_rel     : hole effective mass (m*/m_e)
    L_nm        : well width (nm)
    N_subbands  : maximum subband index to include (integer)
    gamma_eV    : broadening HWHM for Lorentzian (eV)

    Returns
    -------
    alpha_norm : normalized absorption coefficient (arbitrary units)
    E_trans    : list of transition energies (eV) for each subband
    """
    E_photon_eV = np.array(E_photon_eV, dtype=float)
    alpha = np.zeros_like(E_photon_eV)

    ns = np.arange(1, N_subbands + 1)
    E_e = E_infinite_well(ns, m_e_rel, L_nm)  # electron confinement energies
    E_h = E_infinite_well(ns, m_h_rel, L_nm)  # hole confinement energies

    E_trans = Eg_eV + E_e + E_h  # transition energies for n=1..N_subbands

    # Oscillator strength (simplified): stronger for lower n
    osc_strength = 1.0 / (ns ** 0.5)

    for n_idx, En in enumerate(E_trans):
        alpha += osc_strength[n_idx] * lorentzian(E_photon_eV, En, gamma_eV)

    alpha_norm = alpha / np.max(alpha) if np.max(alpha) > 0 else alpha
    return alpha_norm, E_trans


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.title("Nanophysics – Week 10")
st.subheader("Optical Transitions in Quantum Wells and Nanostructures")

st.markdown("---")

# ============================================================
# 1. LEARNING OBJECTIVES & CONTEXT
# ============================================================
st.header("1. Learning Objectives and Context")

st.markdown(
"""
In **Weeks 6–9** we built the foundations:

- **Week 6:** Quantization in finite and infinite quantum wells (confined states).  
- **Week 7:** Coupled wells and tunneling – mini-bands and level splitting.  
- **Week 8:** Periodic potentials and energy bands (Kronig–Penney model).  
- **Week 9:** Density of States (DOS) and Fermi–Dirac occupation in low dimensions.

In **Week 10** we go one crucial step further:

> We connect the **electronic structure** of nanostructures to their **optical properties**.

This week focuses on:

- Optical transitions in **quantum wells** and low-dimensional systems.  
- **Selection rules** (which transitions are allowed/forbidden).  
- Simple models of **absorption spectra** in quantum wells.  
- Relation to real devices: **lasers, LEDs, photodetectors**.

By the end of this week, you should be able to:

1. **Explain** interband optical transitions between valence and conduction subbands in a quantum well.
2. **Write** the basic expression for transition energy in an infinite quantum well.
3. **Describe** optical selection rules (Δk ≈ 0, Δn = 0 for infinite wells, TE/TM polarization).
4. **Simulate** a simple quantum well absorption spectrum as a function of well width, effective masses and broadening.
5. **Interpret** how quantum confinement shifts optical transition energies (blue shift with decreasing well width).
6. **Relate** these concepts to real materials and devices (e.g., GaAs/AlGaAs quantum wells, InGaN LEDs).
"""
)

st.markdown("---")

# ============================================================
# 2. THEORY – OPTICAL TRANSITIONS & SELECTION RULES
# ============================================================
st.header("2. Theory: Optical Transitions in Quantum Wells")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### 2.1 Interband Transitions in Bulk vs Quantum Wells")

    st.markdown(
    """
    In a **bulk semiconductor**, electrons can be optically excited from the
    **valence band** to the **conduction band** by absorbing a photon with
    energy \\(\\hbar \\omega\\).

    The basic interband condition is:

    """
    )
    st.latex(r"""
    \hbar \omega \approx E_c(\mathbf{k}) - E_v(\mathbf{k}),
    """)

    st.markdown(
    """
    where:

    - \\(E_c(\\mathbf{k})\\): conduction band energy  
    - \\(E_v(\\mathbf{k})\\): valence band energy  
    - Photons carry small momentum → **vertical transitions in k-space**.

    In a **quantum well**, carriers are confined in one direction (e.g. z-axis):

    - Motion in z is quantized (subbands).  
    - Motion in the in-plane directions (x, y) remains free (2D).  

    For each subband index \\(n\\), we have energies:

    """
    )

    st.latex(r"""
    E_{c,n} = E_{c0} + E_{e,n}, \quad
    E_{v,n} = E_{v0} - E_{h,n},
    """)

    st.markdown(
    """
    where:

    - \\(E_{c0}\\), \\(E_{v0}\\): band edges of the well material.  
    - \\(E_{e,n}\\): confinement energy of the n-th electron subband.  
    - \\(E_{h,n}\\): confinement energy of the n-th (heavy) hole subband.

    The **interband transition energy** between subband n in valence and
    conduction bands is:

    """
    )

    st.latex(r"""
    E_{\text{trans},n} = E_g + E_{e,n} + E_{h,n},
    """)

    st.markdown(
    """
    with \\(E_g = E_{c0} - E_{v0}\\) the bulk bandgap.

    Compared to the bulk case, the quantum well has **discrete transition
    energies** – this is the origin of sharp peaks in quantum well absorption
    and gain spectra.
    """
    )

with col2:
    st.markdown("### 2.2 Infinite Quantum Well Energies")

    st.markdown(
    """
    As a first approximation, we treat the well as an **infinite potential well**
    of width \\(L\\), with the Schrödinger equation:

    """
    )
    st.latex(r"""
    -\frac{\hbar^2}{2 m^*}
    \frac{d^2 \psi(z)}{dz^2}
    = E \psi(z),
    \qquad
    \psi(0) = \psi(L) = 0.
    """)

    st.markdown(
    """
    The solutions are:

    """
    )
    st.latex(r"""
    \psi_n(z) = \sqrt{\frac{2}{L}} \sin\left(\frac{n \pi z}{L}\right),
    \quad n = 1,2,3,\dots
    """)

    st.latex(r"""
    E_n = \frac{\hbar^2 \pi^2 n^2}{2 m^* L^2}.
    """)

    st.markdown(
    """
    Here \\(m^*\\) is the **effective mass** (electron or hole), and \\(L\\) the
    well width.

    - **Smaller L → larger E_n (stronger confinement, blue shift).**  
    - Different effective masses for electrons and holes give different sets
      of subband energies.

    In this app we use this infinite well formula to compute:

    - \\(E_{e,n}\\) for electrons (\\(m_e^*\\)),  
    - \\(E_{h,n}\\) for heavy holes (\\(m_h^*\\)).
    """
    )

st.markdown("### 2.3 Optical Selection Rules in Quantum Wells")

st.markdown(
r"""
Optical transitions are governed by the **dipole matrix element**:

\[
M_{cv}^{(n,m)} \propto \int \psi_{c,n}(z)\, z\, \psi_{v,m}(z)\, dz,
\]

or, more generally, by the overlap of valence and conduction subband
wavefunctions.

For an ideal symmetric infinite well with same envelope functions
for conduction and valence bands, the **envelope selection rule** is:

\[
\Delta n = n_c - n_v = 0
\quad \Rightarrow \quad
\text{allowed transitions: } n_c = n_v.
\]

Thus, the strongest interband transitions are:

- \(v_1 \to c_1\),
- \(v_2 \to c_2\),
- \(v_3 \to c_3\), ...

In addition:

- Photons carry negligible momentum → \(\Delta k_\parallel \approx 0\) (vertical transitions).  
- TE vs TM polarization leads to different selection rules (heavy-hole and
  light-hole mixing), but in our simple model we focus on envelope selection
  (\(\Delta n = 0\)).
"""
)

st.markdown("---")

# ============================================================
# 3. INTERACTIVE SIMULATIONS – QUANTUM WELL OPTICAL SPECTRA
# ============================================================
st.header("3. Interactive Simulations: Quantum Well Optical Transitions")

st.markdown(
"""
We now simulate a simple **quantum well absorption spectrum**:

1. Choose material (bandgap, typical effective masses).  
2. Choose well width \\(L\\).  
3. Compute subband energies \\(E_{e,n}\\), \\(E_{h,n}\\).  
4. Compute transition energies \\(E_{\\text{trans},n}\\) for \\(n = 1,2,...\\).  
5. Build a simple absorption spectrum as a sum of Lorentzian peaks.

This is not a full device model, but it illustrates:

- **Confinement-induced blue shift** (smaller \\(L\\) → higher transition energy).  
- **Discrete interband transition energies** (peaks in absorption).  
- Influence of **broadening** (interface roughness, phonons, etc.).
"""
)

# ---------------- Sidebar parameters ----------------
st.sidebar.title("Week 10 – Simulation Parameters")

material = st.sidebar.selectbox(
    "Reference material",
    ["Generic", "GaAs", "InGaAs", "GaN"]
)

# Default parameters
Eg_default = 1.40
m_e_default = 0.07
m_h_default = 0.45

if material == "GaAs":
    Eg_default = 1.42
    m_e_default = 0.067
    m_h_default = 0.45
elif material == "InGaAs":
    Eg_default = 0.75
    m_e_default = 0.05
    m_h_default = 0.45
elif material == "GaN":
    Eg_default = 3.40
    m_e_default = 0.20
    m_h_default = 1.0

Eg_eV = st.sidebar.slider(
    "Bandgap Eg (eV)",
    min_value=0.5,
    max_value=4.0,
    value=float(Eg_default),
    step=0.05
)

L_nm = st.sidebar.slider(
    "Well width L (nm)",
    min_value=2.0,
    max_value=20.0,
    value=8.0,
    step=0.5
)

m_e_rel = st.sidebar.slider(
    "Electron effective mass m*_e / m_e",
    min_value=0.02,
    max_value=1.0,
    value=float(m_e_default),
    step=0.01
)

m_h_rel = st.sidebar.slider(
    "Hole effective mass m*_h / m_e",
    min_value=0.05,
    max_value=2.0,
    value=float(m_h_default),
    step=0.05
)

N_subbands = st.sidebar.slider(
    "Number of subbands N",
    min_value=1,
    max_value=5,
    value=3,
    step=1
)

gamma_eV = st.sidebar.slider(
    "Broadening (HWHM) γ (eV)",
    min_value=0.005,
    max_value=0.1,
    value=0.02,
    step=0.005
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Observe how decreasing L (stronger confinement) shifts the peaks to "
    "higher energies (blue shift)."
)

# ------------------------------------------------------------
# Energy grid and calculations
# ------------------------------------------------------------
# Choose photon energy range around Eg with some margin
E_min = max(Eg_eV - 0.2, 0.0)
E_max = Eg_eV + 1.0
E_photon = np.linspace(E_min, E_max, 800)

alpha_norm, E_trans = qw_absorption_spectrum(
    E_photon,
    Eg_eV=Eg_eV,
    m_e_rel=m_e_rel,
    m_h_rel=m_h_rel,
    L_nm=L_nm,
    N_subbands=N_subbands,
    gamma_eV=gamma_eV
)

# ------------------------------------------------------------
# 3.1 Subband Energies (Figure)
# ------------------------------------------------------------
col_sub, col_abs = st.columns(2)

with col_sub:
    st.subheader("3.1 Subband Energies in the Quantum Well (Infinite Well Model)")

    ns = np.arange(1, N_subbands + 1)
    E_e = E_infinite_well(ns, m_e_rel, L_nm)
    E_h = E_infinite_well(ns, m_h_rel, L_nm)

    fig_sub, ax_sub = plt.subplots()
    ax_sub.plot(ns, E_e, marker="o", label="Electron subbands E_e,n")
    ax_sub.plot(ns, E_h, marker="s", label="Hole subbands E_h,n")
    ax_sub.set_xlabel("Subband index n")
    ax_sub.set_ylabel("Confinement energy (eV)")
    ax_sub.set_title("Electron and Hole Confinement Energies")
    ax_sub.grid(True)
    ax_sub.legend()
    st.pyplot(fig_sub)

    st.markdown(
    f"""
    **Parameters**

    - Material: **{material}**  
    - Bandgap: **Eg = {Eg_eV:.2f} eV**  
    - Well width: **L = {L_nm:.1f} nm**  
    - Electron mass: **m*_e = {m_e_rel:.3f} m_e**  
    - Hole mass: **m*_h = {m_h_rel:.3f} m_e**

    For smaller L, both electron and hole subband energies **increase**,
    leading to larger transition energies \\(E_{{\\text{{trans}},n}}\\).
    """
    )

# ------------------------------------------------------------
# 3.2 Absorption Spectrum (Figure)
# ------------------------------------------------------------
with col_abs:
    st.subheader("3.2 Quantum Well Absorption Spectrum (Simplified)")

    fig_abs, ax_abs = plt.subplots()
    ax_abs.plot(E_photon, alpha_norm, label="Normalized absorption")
    # Mark transition energies
    for idx, En in enumerate(E_trans):
        ax_abs.axvline(En, linestyle="--", alpha=0.5,
                       label=f"Transition n={idx+1}" if idx == 0 else None)

    ax_abs.set_xlabel("Photon energy (eV)")
    ax_abs.set_ylabel("Absorption (normalized)")
    ax_abs.set_title("Interband Transitions with Δn = 0")
    ax_abs.grid(True)
    ax_abs.legend()
    st.pyplot(fig_abs)

    # Convert fundamental transition to wavelength
    if len(E_trans) > 0:
        E1 = E_trans[0]
        wavelength_nm = 1240.0 / E1 if E1 > 0 else None
    else:
        wavelength_nm = None

    st.markdown(
    f"""
    **Interband transition energies (Δn = 0)**

    - E_trans,n = Eg + E_e,n + E_h,n

    For N = {N_subbands}, approximate transition energies (eV):

    """
    )

    for i, En in enumerate(E_trans, start=1):
        st.markdown(f"- n = {i}: E_trans,{i} ≈ **{En:.3f} eV**")

    if wavelength_nm is not None:
        st.markdown(
        f"""
        The **fundamental transition** (n = 1) corresponds to wavelength:

        - λ₁ ≈ **{wavelength_nm:.1f} nm**

        Decreasing L shifts this wavelength towards **shorter values** (blue shift),
        which is the basic principle behind **quantum well lasers** and wavelength-tunable LEDs.
        """
        )

st.markdown("---")

# ============================================================
# 4. WORKED EXAMPLES (3 Detailed Problems)
# ============================================================
st.header("4. Worked Examples (3 Detailed Problems)")

with st.expander("Example 1 – Confinement-Induced Blue Shift", expanded=True):
    st.markdown(
    """
    **Problem.**  
    Consider a GaAs quantum well with:

    - Bandgap: \\(E_g = 1.42\\,\\text{eV}\\),  
    - Electron mass: \\(m_e^* = 0.067 m_e\\),  
    - Heavy-hole mass: \\(m_h^* = 0.45 m_e\\).

    Compare the fundamental transition energy \\(E_{\\text{trans},1}\\)
    for well widths:

    - (a) \\(L = 10\\,\\text{nm}\\)  
    - (b) \\(L = 5\\,\\text{nm}\\).

    Use the infinite well approximation.

    ---
    **Solution (qualitative and semi-quantitative).**

    For each well width:

    - Compute electron confinement energy:

      \\[
      E_{e,1} = \\frac{\\hbar^2 \\pi^2}{2 m_e^* L^2}.
      \\]

    - Compute hole confinement energy:

      \\[
      E_{h,1} = \\frac{\\hbar^2 \\pi^2}{2 m_h^* L^2}.
      \\]

    - Then:

      \\[
      E_{\\text{trans},1} = E_g + E_{e,1} + E_{h,1}.
      \\]

    Because both \\(E_{e,1}\\) and \\(E_{h,1}\\) scale as \\(1/L^2\\):

    - Halving the well width (from 10 nm to 5 nm) **quadruples** the
      confinement energies.  
    - Therefore, \\(E_{\\text{trans},1}\\) is significantly **larger** for
      the 5 nm well (strong blue shift).

    In the simulation, decreasing L from 10 nm to 5 nm visibly shifts the
    fundamental absorption peak to higher photon energies.
    """
    )

with st.expander("Example 2 – Selection Rule Δn = 0 in an Infinite Well"):
    st.markdown(
    r"""
    **Problem.**  
    Show qualitatively why the envelope selection rule in a symmetric infinite
    quantum well leads to strongest transitions when \\(\Delta n = n_c - n_v = 0\\).

    ---
    **Solution.**

    The z-dependent parts of the conduction and valence subband wavefunctions
    (envelope functions) in an infinite well are:

    \[
    \psi_{c,n}(z) = \sqrt{\frac{2}{L}} \sin\left(\frac{n \pi z}{L}\right),
    \quad
    \psi_{v,m}(z) = \sqrt{\frac{2}{L}} \sin\left(\frac{m \pi z}{L}\right).
    \]

    The dipole matrix element for an interband transition involves an overlap
    integral like:

    \[
    M_{cv}^{(n,m)} \propto \int_0^L \psi_{c,n}(z)\, \psi_{v,m}(z)\, dz.
    \]

    Using orthogonality of sine functions:

    \[
    \int_0^L \sin\left(\frac{n \pi z}{L}\right)
             \sin\left(\frac{m \pi z}{L}\right)\, dz
    \propto \delta_{n,m},
    \]

    we see that:

    - If \(n \neq m\), the integral vanishes (in the ideal infinite well).  
    - If \(n = m\), the overlap is maximum.

    Thus, **envelope selection rule**:

    \[
    \Delta n = 0 \Rightarrow \text{strongest transitions}.
    \]

    In real heterostructures with finite barriers and band mixing, this rule
    is softened but still provides a good guideline.
    """
    )

with st.expander("Example 3 – Polarization and Optical Transitions"):
    st.markdown(
    r"""
    **Problem.**  
    In a quantum well grown along the z direction, discuss qualitatively how
    polarization (TE vs TM) interacts with selection rules for heavy-hole
    and light-hole transitions.

    ---
    **Solution (conceptual).**

    - The electric field of the light has components **parallel** (in-plane)
      or **perpendicular** (along z) to the quantum well plane.  
    - TE-polarized light: electric field lies **in the plane** of the well.  
    - TM-polarized light: electric field has a substantial **z-component**.

    In many III–V quantum wells (e.g. GaAs/AlGaAs):

    - **Heavy-hole (HH)** states couple strongly to **TE polarization**, because
      of their angular momentum orientation.  
    - **Light-hole (LH)** states can couple more strongly to **TM polarization**.

    As a result:

    - TE absorption/gain is dominated by **HH → conduction** transitions.  
    - TM absorption/gain has stronger contributions from **LH → conduction** transitions.

    Our simple envelope model in this app does not explicitly treat HH–LH
    mixing, but in real devices, polarization-resolved measurements and
    simulations are crucial for designing **laser diodes**, **modulators** and
    **polarization-sensitive detectors**.
    """
    )

st.markdown("---")

# ============================================================
# 5. SHORT QUIZ (3 Questions)
# ============================================================
st.header("5. Short Quiz (3 Questions)")

st.markdown("Answer the questions to check your understanding of Week 10 concepts.")

# Quiz 1
st.subheader("Quiz 1")
st.markdown(
"""
In a simple symmetric infinite quantum well, the **strongest** interband transitions
occur between which subbands (within the envelope function approximation)?

A. Any pair with n_c ≠ n_v (all equally strong)  
B. Only between n_c = 1 and n_v = 3  
C. Mostly between n_c = n_v (Δn = 0)  
D. Only between n_c = 1 and n_v = 1
"""
)
q1 = st.radio("Your answer for Quiz 1:", ["A", "B", "C", "D"], key="w10q1")

if q1:
    if q1 == "C":
        st.success("Correct! The envelope overlap is largest for n_c = n_v (Δn = 0).")
    else:
        st.error("Not correct. For an ideal infinite well, the overlap is maximal when n_c = n_v.")

# Quiz 2
st.subheader("Quiz 2")
st.markdown(
"""
What happens to the fundamental transition energy \\(E_{trans,1}\\) when the
quantum well width L is **decreased**, assuming all other parameters remain constant?

A. It stays the same.  
B. It decreases (red shift).  
C. It increases (blue shift).  
D. It oscillates randomly.
"""
)
q2 = st.radio("Your answer for Quiz 2:", ["A", "B", "C", "D"], key="w10q2")

if q2:
    if q2 == "C":
        st.success("Correct! Stronger confinement (smaller L) increases subband energies and thus the transition energy.")
    else:
        st.error("Not correct. Confinement energies scale like 1/L², so reducing L increases transition energy.")

# Quiz 3
st.subheader("Quiz 3")
st.markdown(
"""
In a quantum well, the photon momentum is very small. What does this imply
for the in-plane wavevector of electrons involved in optical transitions?

A. Δk_parallel is large and arbitrary.  
B. Δk_parallel ≈ 0 (vertical transitions in k-space).  
C. k_parallel must be exactly zero for all transitions.  
D. Only phonon-assisted transitions are possible.
"""
)
q3 = st.radio("Your answer for Quiz 3:", ["A", "B", "C", "D"], key="w10q3")

if q3:
    if q3 == "B":
        st.success("Correct! Photons have small momentum, so optical transitions are nearly vertical in k-space.")
    else:
        st.error("Not correct. Direct optical transitions are approximately vertical: Δk_parallel ≈ 0.")

st.markdown("---")

st.markdown(
"""
### Summary – Week 10

- We connected **quantum confinement** (from Weeks 6–9) to **optical transitions** in quantum wells.  
- We derived interband transition energies:

  \\[
  E_{\\text{trans},n} = E_g + E_{e,n} + E_{h,n},
  \\]

  and saw how they depend on **well width** and **effective masses**.  
- We discussed **selection rules** (Δn = 0) and the role of wavefunction overlap.  
- We simulated a simple **absorption spectrum** consisting of discrete peaks
  corresponding to quantum well interband transitions.  
- We highlighted how confinement leads to **blue shift** of optical transitions
  and how this principle is used in **lasers, LEDs, and photodetectors**.

You can now explore different materials (GaAs, InGaAs, GaN), change the well
width, and see how quantum engineering controls optical properties of
nanostructures.
"""
)
