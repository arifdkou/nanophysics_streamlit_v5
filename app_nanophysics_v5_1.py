# app_nanophysics_week5.py
# ============================================================
# Nanophysics – Week 5
# 1D Quantum Harmonic Oscillator
# Streamlit Interactive Lecture Note
# ============================================================

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from math import factorial

# ------------------------------------------------------------
# Streamlit page settings
# ------------------------------------------------------------
st.set_page_config(
    page_title="Nanophysics – Week 5: Quantum Harmonic Oscillator",
    layout="wide"
)

# ------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------
h = 6.626_070_15e-34      # Planck constant (J·s)
hbar = h / (2 * np.pi)
eV = 1.602_176_634e-19    # J
m_e = 9.109_383_56e-31    # kg (electron mass)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def hermite_phys(n, x):
    """
    Physicists' Hermite polynomials H_n(x), implemented by recurrence.
    x can be a scalar or NumPy array.
    """
    x = np.array(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x

    H_nm2 = np.ones_like(x)      # H_0
    H_nm1 = 2.0 * x              # H_1
    for k in range(2, n + 1):
        H_n = 2.0 * x * H_nm1 - 2.0 * (k - 1) * H_nm2
        H_nm2, H_nm1 = H_nm1, H_n
    return H_nm1


def ho_psi_n(n, x, m_eff, omega):
    """
    Harmonic oscillator eigenfunction ψ_n(x):

    ψ_n(x) = 1 / sqrt(2^n n!) * (1 / (π^1/4 * x0^1/2)) * H_n(x / x0) * exp(-x^2 / (2 x0^2))

    where x0 = sqrt(ħ / (m_eff * ω))
    """
    x = np.array(x, dtype=float)
    x0 = np.sqrt(hbar / (m_eff * omega))
    xi = x / x0

    Hn = hermite_phys(n, xi)

    norm_prefactor = 1.0 / np.sqrt((2.0 ** n) * factorial(n))
    norm_gauss = 1.0 / (np.pi ** 0.25 * np.sqrt(x0))
    psi = norm_prefactor * norm_gauss * Hn * np.exp(-xi ** 2 / 2.0)
    return psi, x0


def ho_energy_eV(n, hbar_omega_eV):
    """
    E_n = ħω (n + 1/2), with ħω given in eV.
    """
    return hbar_omega_eV * (n + 0.5)


def omega_from_hbaromega_eV(hbar_omega_eV):
    """
    Convert ħω in eV to ω in rad/s.
    ħω (J) = hbar_omega_eV * eV, then ω = (ħω_J) / ħ.
    """
    hbar_omega_J = hbar_omega_eV * eV
    return hbar_omega_J / hbar


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.title("Nanophysics – Week 5")
st.subheader("1D Quantum Harmonic Oscillator: Energy Levels & Wavefunctions")

st.markdown("---")

# ============================================================
# 1. LEARNING OBJECTIVES
# ============================================================
st.header("1. Learning Objectives (Haftanın Ana Amaçları)")

st.markdown(
"""
By the end of this week, you should be able to:

1. **Write** the 1D time–independent Schrödinger equation for the harmonic oscillator potential.
2. **Explain** the quadratic potential \\(V(x) = \\tfrac{1}{2} m \\omega^2 x^2\\) and its physical meaning.
3. **Derive / recall** the discrete energy spectrum: \\(E_n = \\hbar \\omega (n + 1/2)\\).
4. **Describe** the qualitative shapes of \\(\\psi_n(x)\\) and \\(|\\psi_n(x)|^2\\) for \\(n = 0,1,2,\\dots\\).
5. **Use** input parameters (\\(\\hbar\\omega\\), effective mass, quantum number n) to simulate:
   - energy levels
   - wavefunctions
   - probability densities.
6. **Solve** example problems about energy spacing and characteristic length \\(x_0\\).
"""
)

st.markdown("---")

# ============================================================
# 2. THEORY: 1D QUANTUM HARMONIC OSCILLATOR
# ============================================================
st.header("2. Theory: 1D Quantum Harmonic Oscillator")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### 2.1 Schrödinger Equation with Harmonic Potential")

    st.latex(r"""
    \hat{H}\,\psi(x) = E\,\psi(x)
    """)

    st.markdown("In 1D, the Hamiltonian is:")

    st.latex(r"""
    \hat{H} = -\frac{\hbar^2}{2m}\,\frac{d^2}{dx^2} + V(x).
    """)

    st.markdown("For the **harmonic oscillator**:")

    st.latex(r"""
    V(x) = \frac{1}{2} m \omega^2 x^2.
    """)

    st.markdown(
    """
    So the time–independent Schrödinger equation becomes:
    """
    )

    st.latex(r"""
    \left[
      -\frac{\hbar^2}{2m}\,\frac{d^2}{dx^2}
      + \frac{1}{2} m \omega^2 x^2
    \right] \psi(x)
    =
    E\,\psi(x).
    """)

with col2:
    st.markdown("### 2.2 Energy Spectrum & Eigenfunctions")

    st.markdown(
    """
    The harmonic oscillator is **exactly solvable**. Its energy levels are:
    """
    )

    st.latex(r"""
    E_n = \hbar \omega \left(n + \frac{1}{2}\right),
    \quad n = 0, 1, 2, \dots
    """)

    st.markdown(
    """
    - Levels are **equally spaced** by \\(\\hbar\\omega\\).  
    - There is a **zero-point energy**: ground state energy is \\(\\tfrac{1}{2} \\hbar\\omega\\), not zero.

    Define the characteristic length:
    """
    )

    st.latex(r"""
    x_0 = \sqrt{\frac{\hbar}{m \omega}}.
    """)

    st.markdown(
    """
    In terms of dimensionless coordinate \\(\\xi = x / x_0\\), the eigenfunctions are:
    """
    )

    st.latex(r"""
    \psi_n(x) =
      \frac{1}{\sqrt{2^n n!}}
      \frac{1}{(\pi)^{1/4} x_0^{1/2}}
      H_n\left(\frac{x}{x_0}\right)
      e^{-x^2 / (2 x_0^2)},
    """)

    st.markdown(
    """
    where \\(H_n\\) are the **physicists' Hermite polynomials**.

    - \\(n\\)-th state has **n nodes** (zeros) in \\(\\psi_n(x)\\).  
    - \\(|\\psi_n(x)|^2\\) gives the **probability density**.
    """
    )

st.markdown("---")

# ============================================================
# 3. INTERACTIVE SIMULATIONS
# ============================================================
st.header("3. Interactive Simulations")

st.markdown(
"""
We will:

1. Show the **potential** \\(V(x)\\) and **discrete energy levels** \\(E_n\\).  
2. Plot the **wavefunction** \\(\\psi_n(x)\\) and **probability density** \\(|\\psi_n(x)|^2\\)  
   for a chosen quantum number \\(n\\).  
3. Let you explore the effect of \\(\\hbar\\omega\\), effective mass \\(m^*\\) and \\(n\\).
"""
)

# Sidebar for input parameters
st.sidebar.title("Week 5 – Simulation Parameters")

hbar_omega_eV = st.sidebar.slider(
    "ħω (energy spacing) in eV",
    min_value=0.01,
    max_value=0.5,
    value=0.10,
    step=0.01
)

mass_option = st.sidebar.selectbox(
    "Effective mass m*",
    [
        "Free electron (m* = m_e)",
        "Light effective mass (m* = 0.2 m_e)",
        "Heavy effective mass (m* = 0.5 m_e)"
    ]
)

if mass_option.startswith("Free"):
    m_factor = 1.0
elif "0.2" in mass_option:
    m_factor = 0.2
else:
    m_factor = 0.5

n_max = st.sidebar.slider(
    "Max level to display (n_max)",
    min_value=0,
    max_value=6,
    value=3,
    step=1
)

n_selected = st.sidebar.slider(
    "Selected level n for ψₙ(x)",
    min_value=0,
    max_value=6,
    value=1,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.info("Değiştir: ħω, m* ve n ile enerji seviyelerini ve dalga fonksiyonlarını incele.")

# Compute parameters
m_eff = m_factor * m_e
omega = omega_from_hbaromega_eV(hbar_omega_eV)

# Characteristic length x0
x0 = np.sqrt(hbar / (m_eff * omega))           # in meters
x0_nm = x0 * 1e9                               # in nm

# x-grid around 0, say [-4 x0, 4 x0]
x_min = -4.0 * x0
x_max = 4.0 * x0
x = np.linspace(x_min, x_max, 600)

# Potential V(x) = 0.5 m ω² x², in Joules then eV
V_J = 0.5 * m_eff * omega ** 2 * x ** 2
V_eV = V_J / eV

# Wavefunction for selected n
psi_n, _ = ho_psi_n(n_selected, x, m_eff, omega)
prob_n = np.abs(psi_n) ** 2

# Energies
n_levels = np.arange(0, n_max + 1)
E_levels_eV = ho_energy_eV(n_levels, hbar_omega_eV)

# ------------------------------------------------------------
# 3.1 Potential + Energy Level Diagram
# ------------------------------------------------------------
col_pot, col_wave = st.columns(2)

with col_pot:
    st.subheader("3.1 Harmonic Potential and Discrete Energy Levels")

    fig1, ax1 = plt.subplots()
    ax1.plot(x / x0, V_eV, label="V(x)")
    for n_val, E_val in zip(n_levels, E_levels_eV):
        ax1.axhline(E_val, linestyle="--", linewidth=0.8)
        ax1.text(
            x_max / x0 * 0.8,
            E_val + 0.01 * hbar_omega_eV,
            f"n={n_val}",
            fontsize=8
        )

    ax1.set_xlabel(r"$x / x_0$")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title("Quadratic Potential & Energy Levels")
    ax1.set_ylim(bottom=0)
    ax1.grid(True)
    st.pyplot(fig1)

    st.markdown(
    f"""
    **Parameters:**

    - ħω = **{hbar_omega_eV:.3f} eV**  
    - Effective mass: **{mass_option}**  
    - Characteristic length: **x₀ ≈ {x0_nm:.2f} nm**  
    - Energy levels:  
      \\(E_n = \\hbar\\omega (n + 1/2)\\); for n = 0..{n_max}
    """
    )

with col_wave:
    st.subheader(f"3.2 Wavefunction ψₙ(x) and Probability Density |ψₙ(x)|² (n = {n_selected})")

    fig2, ax2 = plt.subplots()
    ax2.plot(x / x0, psi_n, label=r"$\psi_n(x)$")
    ax2.set_xlabel(r"$x / x_0$")
    ax2.set_ylabel(r"$\psi_n(x)$")
    ax2.set_title(fr"Wavefunction for n = {n_selected}")
    ax2.grid(True)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(x / x0, prob_n, label=r"$|\psi_n(x)|^2$")
    ax3.set_xlabel(r"$x / x_0$")
    ax3.set_ylabel(r"$|\psi_n(x)|^2$")
    ax3.set_title(fr"Probability Density for n = {n_selected}")
    ax3.grid(True)
    st.pyplot(fig3)

    st.markdown(
    """
    **Gözlemler:**

    - n = 0 (ground state) → **Gaussian** profile, nodes yok.  
    - n arttıkça **daha fazla düğüm** (node) ortaya çıkar.  
    - |ψₙ(x)|², parçacığın bulunma olasılığını gösterir, klasik enerji sınırlarına
      yakın bölgelerde maksimumlara yaklaşır.
    """
    )

st.markdown("---")

# ============================================================
# 4. WORKED EXAMPLES (3 Detailed Problems)
# ============================================================
st.header("4. Worked Examples (3 Detailed Problems)")

# Example 1
with st.expander("Example 1 – Ground State Energy for ħω = 0.1 eV", expanded=True):
    st.markdown(
    """
    **Problem.**  
    A 1D quantum harmonic oscillator has \\(\\hbar\\omega = 0.10\\,\\text{eV}\\).  

    1. Find the ground state energy \\(E_0\\).  
    2. Find the first excited state energy \\(E_1\\).  
    3. What is the energy spacing \\(\\Delta E = E_1 - E_0\\)?

    ---
    **Solution.**

    Energy levels:

    """
    )
    st.latex(r"""
    E_n = \hbar \omega (n + 1/2).
    """)

    hwo = 0.10
    E0 = ho_energy_eV(0, hwo)
    E1 = ho_energy_eV(1, hwo)
    dE = E1 - E0

    st.latex(r"""
    E_0 = \hbar \omega \left(0 + \frac{1}{2}\right) = \frac{1}{2} \hbar \omega.
    """)
    st.write(f"Numerically: **E₀ ≈ {E0:.3f} eV**")

    st.latex(r"""
    E_1 = \hbar \omega \left(1 + \frac{1}{2}\right) = \frac{3}{2} \hbar \omega.
    """)
    st.write(f"Numerically: **E₁ ≈ {E1:.3f} eV**")

    st.markdown("Energy spacing:")

    st.latex(r"""
    \Delta E = E_1 - E_0 = \hbar \omega.
    """)
    st.write(f"Numerically: **ΔE ≈ {dE:.3f} eV**, which equals ħω as expected.")

    st.markdown(
    """
    **Interpretation:**  
    - Energy levels are **equally spaced** by \\(\\hbar\\omega\\).  
    - Even at n = 0 there is **zero-point energy** \\(E_0 = \\tfrac{1}{2}\\hbar\\omega\\).
    """
    )

# Example 2
with st.expander("Example 2 – Characteristic Length x₀ and Confinement Scale"):
    st.markdown(
    """
    **Problem.**  
    Consider an electron in a harmonic potential with:

    - Effective mass: \\(m^* = m_e\\)  
    - \\(\\hbar\\omega = 0.05\\,\\text{eV}\\)

    1. Compute the oscillator angular frequency \\(\\omega\\) in rad/s.  
    2. Compute the characteristic length \\(x_0 = \\sqrt{\\hbar/(m\\omega)}\\) in nm.

    ---
    **Solution.**
    """
    )

    hwo2 = 0.05
    omega2 = omega_from_hbaromega_eV(hwo2)
    x0_2 = np.sqrt(hbar / (m_e * omega2))
    x0_2_nm = x0_2 * 1e9

    st.latex(r"""
    \omega = \frac{\hbar \omega}{\hbar}
    = \frac{(\hbar \omega)_{\text{(J)}}}{\hbar}.
    """)

    st.write(f"- ω ≈ {omega2:.3e} rad/s")

    st.latex(r"""
    x_0 = \sqrt{\frac{\hbar}{m \omega}}.
    """)

    st.write(f"- x₀ ≈ {x0_2_nm:.2f} nm")

    st.markdown(
    """
    **Interpretation:**  
    - x₀ sets the **spatial extent** of the ground state wavefunction.  
    - Smaller mass or larger ω → **smaller x₀** → stronger confinement.
    """
    )

# Example 3
with st.expander("Example 3 – Number of Nodes in ψₙ(x)"):
    st.markdown(
    """
    **Problem.**  
    For the quantum harmonic oscillator, how many **nodes** (zeros)
    does \\(\\psi_n(x)\\) have for a given quantum number \\(n\\)?

    1. Compute/observe the nodes for n = 0, 1, 2, 3 using the model.  
    2. Formulate the general rule.

    ---
    **Solution (qualitative + numerical check).**
    """
    )

    n_list = [0, 1, 2, 3]
    text_lines = []
    for n_test in n_list:
        psi_test, _ = ho_psi_n(n_test, x, m_eff, omega)
        # Say node where psi crosses 0: sign change
        sign_changes = np.where(np.diff(np.sign(psi_test)) != 0)[0]
        num_nodes = len(sign_changes)
        text_lines.append(f"n = {n_test}: approximately {num_nodes} nodes.")

    st.write("Approximate node counts (from the current parameter set):")
    for line in text_lines:
        st.write("- " + line)

    st.markdown(
    """
    By theory for the harmonic oscillator:

    - \\(n = 0\\) → 0 node  
    - \\(n = 1\\) → 1 node  
    - \\(n = 2\\) → 2 nodes  
    - …  

    **General rule:**  
    \\[
    \text{Number of nodes in } \psi_n(x) = n.
    \\]
    """
    )

st.markdown("---")

# ============================================================
# 5. QUIZ (3 QUESTIONS)
# ============================================================
st.header("5. Short Quiz (3 Questions)")

st.markdown("Answer the questions to check your understanding.")

# Quiz 1
st.subheader("Quiz 1")
st.markdown(
"""
For a 1D quantum harmonic oscillator, the energy levels are:

\\[
E_n = \hbar \omega \left(n + \frac{1}{2}\right).
\\]

What is the energy spacing between **adjacent levels** (\\(E_{n+1} - E_n\\))?

A. \\(\\hbar \\omega\\)  
B. \\(2 \\hbar \\omega\\)  
C. \\(\\hbar \\omega / 2\\)  
D. Depends on n
"""
)

q1 = st.radio("Your answer for Quiz 1:", ["A", "B", "C", "D"], key="w5q1")

if q1:
    if q1 == "A":
        st.success("Correct! E_{n+1} - E_n = ħω, independent of n.")
    else:
        st.error("Not correct. Energy levels are equally spaced by exactly ħω.")

# Quiz 2
st.subheader("Quiz 2")
st.markdown(
"""
The ground state energy of the harmonic oscillator is:

A. 0  
B. \\(\\tfrac{1}{2}\\hbar \\omega\\)  
C. \\(\\hbar \\omega\\)  
D. \\(\\tfrac{3}{2}\\hbar \\omega\\)
"""
)

q2 = st.radio("Your answer for Quiz 2:", ["A", "B", "C", "D"], key="w5q2")

if q2:
    if q2 == "B":
        st.success("Correct! The ground state has non-zero energy: E₀ = ½ ħω.")
    else:
        st.error("Not correct. Remember the zero-point energy: E₀ = ½ ħω.")

# Quiz 3
st.subheader("Quiz 3")
st.markdown(
"""
For the quantum harmonic oscillator, how many **nodes inside space**
does the wavefunction \\(\\psi_n(x)\\) have?

A. Always 0  
B. n/2  
C. n  
D. Depends on mass and ω
"""
)

q3 = st.radio("Your answer for Quiz 3:", ["A", "B", "C", "D"], key="w5q3")

if q3:
    if q3 == "C":
        st.success("Correct! The n-th eigenstate has exactly n nodes.")
    else:
        st.error("Not correct. For HO, node count is equal to n.")

st.markdown("---")

st.markdown(
"""
### Summary – Week 5

- We wrote the Schrödinger equation with **harmonic potential** \\(V(x)=\\tfrac{1}{2}m\\omega^2 x^2\\).  
- We discussed the **equally spaced energy levels** \\(E_n = \\hbar\\omega (n + 1/2)\\).  
- We introduced the **characteristic length** \\(x_0 = \\sqrt{\\hbar/(m\\omega)}\\).  
- We visualized \\(\\psi_n(x)\\) and \\(|\\psi_n(x)|^2\\) for different n.  
- We saw that the **n-th state has n nodes** and that the ground state has a **zero-point energy**.  
- We solved **3 worked examples** and checked the main ideas with **3 quiz questions**.

Şimdi kaydırıcılarla oynayarak farklı \\(\\hbar\\omega\\), \\(m^*\\) ve n değerlerinde
dalga fonksiyonlarının ve enerji seviyelerinin nasıl değiştiğini sezgisel olarak görebilirsin.
"""
)
