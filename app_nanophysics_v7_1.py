# app_nanophysics_week7.py
# ============================================================
# Nanophysics – Week 7
# 1D Time–Independent Schrödinger Equation & Infinite Quantum Well
# Streamlit Interactive Lecture Note
# ============================================================

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Streamlit page settings
# ------------------------------------------------------------
st.set_page_config(
    page_title="Nanophysics – Week 7: 1D Schrödinger Equation",
    layout="wide"
)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def psi_1d_infinite_well(n, L, x):
    """
    1D infinite potential well eigenfunction
    0 < x < L, V = 0; elsewhere V = ∞
    ψ_n(x) = sqrt(2/L) * sin(nπx/L)
    L in arbitrary units (here we use dimensionless x ∈ [0, L])
    """
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)


def energy_1d_dimensionless(n, L):
    """
    Energy in dimensionless units (ħ = 1, 2m = 1):
    H = -d²/dx² + V
    E_n = (n² π²) / L²
    """
    return (n ** 2) * (np.pi ** 2) / (L ** 2)


def energy_1d_electron_eV(n, L_nm, m_factor=1.0):
    """
    Physical energy in eV for a particle in 1D infinite well.

    E_n = (ħ² π² / 2 m*) * (n² / L²)

    - L_nm: well width in nanometers
    - m_factor: effective mass factor (m* = m_factor * m_e)
        m_factor = 1.0   → free electron mass
        m_factor = 0.2   → light effective mass (e.g., some semiconductors)
        m_factor = 0.5   → heavier effective mass
    """
    hbar = 1.054_571_817e-34      # J·s
    m_e = 9.109_383_56e-31        # kg
    eV = 1.602_176_634e-19        # J

    L = L_nm * 1e-9               # m
    m_eff = m_factor * m_e

    prefactor = (hbar ** 2 * np.pi ** 2) / (2.0 * m_eff)
    E_J = prefactor * (n ** 2 / L ** 2)
    E_eV = E_J / eV
    return E_eV


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.title("Nanophysics – Week 7")
st.subheader("1D Time–Independent Schrödinger Equation & Infinite Quantum Well")

st.markdown("---")

# ============================================================
# 1. LEARNING OBJECTIVES
# ============================================================
st.header("1. Learning Objectives (Haftanın Ana Amaçları)")

st.markdown(
"""
By the end of this week, you should be able to:

1. **Write** the 1D time–independent Schrödinger equation and identify the Hamiltonian.
2. **Explain** the boundary conditions for a particle in a 1D infinite potential well.
3. **Derive** the allowed energy eigenvalues \\(E_n\\) and eigenfunctions \\(\\psi_n(x)\\).
4. **Sketch** the wavefunctions and probability densities for low quantum numbers (\\(n = 1,2,3\\)).
5. **Use** input parameters (well width, quantum number, effective mass) to explore energy scaling.
6. **Solve** example problems related to quantum confinement in 1D nanoscale systems.
"""
)

st.markdown("---")

# ============================================================
# 2. THEORY: 1D TIME–INDEPENDENT SCHRÖDINGER EQUATION
# ============================================================
st.header("2. Theory: 1D Time–Independent Schrödinger Equation")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### 2.1 Hamiltonian and Schrödinger Equation")

    st.latex(r"""
    \hat{H}\,\psi(x) = E\,\psi(x)
    """)

    st.markdown(
    """
    - \\(\\hat{H}\\): Hamiltonian operator  
    - \\(\\psi(x)\\): stationary wavefunction  
    - \\(E\\): energy eigenvalue  

    In 1D, the Hamiltonian is
    """
    )

    st.latex(r"""
    \hat{H} = -\frac{\hbar^2}{2m}\,\frac{d^2}{dx^2} + V(x).
    """)

    st.markdown(
    """
    In **atomic-like units** (\\(\\hbar = 1\\), \\(2m = 1\\)), we use:

    """
    )

    st.latex(r"""
    \hat{H} = -\frac{d^2}{dx^2} + V(x),
    \quad
    \hat{H}\,\psi(x) = E\,\psi(x).
    """)

with col2:
    st.markdown("### 2.2 1D Infinite Potential Well (Infinite Quantum Box)")

    st.markdown(
    r"""
    We consider a **1D infinite potential well** (box) of width \\(L\\):

    - Region inside the well: \\(0 < x < L\\)  
    - Potential:
    """
    )

    st.latex(r"""
    V(x) =
    \begin{cases}
    0, & 0 < x < L, \\
    \infty, & \text{otherwise}.
    \end{cases}
    """)

    st.markdown(
    r"""
    Because \\(V(x) = \infty\\) outside the region, the wavefunction must vanish
    at the walls and outside:

    - \\(\psi(0) = 0\\)  
    - \\(\psi(L) = 0\\)

    Inside the well (\\(V=0\\)), the Schrödinger equation simplifies to:
    """
    )

    st.latex(r"""
    -\frac{\hbar^2}{2m}\,\frac{d^2 \psi}{dx^2} = E\,\psi
    \quad \Rightarrow \quad
    \frac{d^2 \psi}{dx^2} + k^2 \psi = 0,
    """)

    st.markdown(
    r"""
    where \\(k = \sqrt{2mE}/\hbar\\). The solutions are sinusoidal,
    and the boundary conditions select discrete \\(k\\) values → **quantization**.
    """
    )

st.markdown("### 2.3 Eigenfunctions and Energies")

st.latex(r"""
\psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n \pi x}{L}\right),
\quad n = 1,2,3,\dots
""")

st.latex(r"""
E_n = \frac{\hbar^2 \pi^2}{2m}\,\frac{n^2}{L^2}
\quad \text{or (in units } \hbar=1, 2m=1\text{):} \quad
E_n = \frac{n^2 \pi^2}{L^2}.
""")

st.markdown(
"""
- \\(n\\) is the **quantum number**: \\(n = 1,2,3,\\dots\\).  
- Higher \\(n\\) → more oscillations and higher energy.  
- The number of **nodes** (zeros inside the well) is \\(n-1\\).  
- Energy levels increase as \\(E_n \\propto n^2\\) and \\(E_n \\propto 1/L^2\\).
"""
)

st.markdown("---")

# ============================================================
# 3. INTERACTIVE SIMULATION
# ============================================================
st.header("3. Interactive Simulation: 1D Infinite Well")

st.markdown(
"""
Use the sidebar to choose:

- **Well width** (in nm)  
- **Quantum number** \\(n\\)  
- **Effective mass** (free electron or effective mass in semiconductors)  

The app will:

1. Plot the **wavefunction** \\(\\psi_n(x)\\)  
2. Plot the **probability density** \\(|\\psi_n(x)|^2\\)  
3. Compute the energy in **dimensionless units** and in **eV**
"""
)

# Sidebar for input parameters
st.sidebar.title("Week 7 – Simulation Parameters")

L_nm = st.sidebar.slider(
    "Well width L (nm)",
    min_value=1.0,
    max_value=20.0,
    value=10.0,
    step=0.5
)

n = st.sidebar.slider(
    "Quantum number n",
    min_value=1,
    max_value=6,
    value=1,
    step=1
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

grid_points = st.sidebar.slider(
    "Grid resolution (number of x points)",
    min_value=100,
    max_value=500,
    value=200,
    step=50
)

st.sidebar.markdown("---")
st.sidebar.info("Try changing L and n to see how the energy spectrum and wave shape change.")

# Prepare x-grid in dimensionless coordinates [0, L_dimless]
L_dimless = 1.0
x = np.linspace(0, L_dimless, grid_points)
x_nm = x * L_nm  # map to physical nm for plotting

psi = psi_1d_infinite_well(n, L_dimless, x)
prob_density = np.abs(psi) ** 2

E_dimless = energy_1d_dimensionless(n, L_dimless)
E_eV = energy_1d_electron_eV(n, L_nm, m_factor=m_factor)

# Plots
col_wave, col_prob = st.columns(2)

with col_wave:
    st.subheader("Wavefunction ψₙ(x)")
    fig1, ax1 = plt.subplots()
    ax1.plot(x_nm, psi)
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel(r"$\psi_n(x)$")
    ax1.set_title(fr"Wavefunction for n = {n}")
    ax1.axhline(0, color="black", linewidth=0.8)
    st.pyplot(fig1)

with col_prob:
    st.subheader("Probability Density |ψₙ(x)|²")
    fig2, ax2 = plt.subplots()
    ax2.plot(x_nm, prob_density)
    ax2.set_xlabel("x (nm)")
    ax2.set_ylabel(r"$|\psi_n(x)|^2$")
    ax2.set_title(fr"Probability Density for n = {n}")
    ax2.axhline(0, color="black", linewidth=0.8)
    st.pyplot(fig2)

st.markdown("### 3.1 Computed Energy")

st.latex(r"""
E_n = \frac{\hbar^2 \pi^2}{2m^*}\,\frac{n^2}{L^2}
\quad \text{and in our dimensionless units } E_n = \frac{n^2 \pi^2}{L^2}.
""")

st.write(
    f"- **Dimensionless energy** (ħ=1, 2m=1):  "
    f"\\(E_n = {E_dimless:.3f}\\)"
)
st.write(
    f"- **Physical energy** for {mass_option}, L = {L_nm:.1f} nm, n = {n}:  "
    f"\\(E_n \\approx {E_eV:.3f}\\ \\text{{eV}}\\)"
)

st.markdown("---")

# ============================================================
# 4. WORKED EXAMPLES (3 Detailed Problems)
# ============================================================
st.header("4. Worked Examples (3 Detailed Problems)")

# Example 1
with st.expander("Example 1 – Ground State Energy in a 10 nm Infinite Well", expanded=True):
    st.markdown(
    """
    **Problem.**  
    An electron is confined in a 1D infinite potential well of width  
    \\(L = 10\\,\\text{nm}\\). Find the **ground state energy** \\(E_1\\) in eV.

    ---
    **Solution.**

    For a 1D infinite well, the energy levels are:

    """
    )
    st.latex(r"""
    E_n = \frac{\hbar^2 \pi^2}{2m_e}\,\frac{n^2}{L^2}.
    """)

    st.markdown(
    r"""
    For the **ground state**, \\(n = 1\\), so:

    """
    )

    st.latex(r"""
    E_1 = \frac{\hbar^2 \pi^2}{2m_e}\,\frac{1^2}{L^2}.
    """)

    E1_10nm = energy_1d_electron_eV(1, 10.0, m_factor=1.0)
    st.write(f"Numerically, for L = 10 nm:  \n**E₁ ≈ {E1_10nm:.3f} eV.**")

    st.markdown(
    """
    **Interpretation:**  
    - The ground state energy is **non-zero** due to quantum confinement.  
    - If we make the well narrower (smaller L), the energy increases as \\(1/L^2\\).
    """
    )

# Example 2
with st.expander("Example 2 – Energy Ratio of First Two Levels"):
    st.markdown(
    """
    **Problem.**  
    In a 1D infinite well of width \\(L\\), compare the energies of
    the first two levels, \\(E_1\\) and \\(E_2\\).  
    Compute the ratio \\(E_2 / E_1\\).

    ---
    **Solution.**

    We have

    """
    )
    st.latex(r"""
    E_n = \frac{\hbar^2 \pi^2}{2m}\,\frac{n^2}{L^2}.
    """)

    st.markdown(
    r"""
    Therefore:

    """
    )

    st.latex(r"""
    E_1 = \frac{\hbar^2 \pi^2}{2m}\,\frac{1^2}{L^2}, \quad
    E_2 = \frac{\hbar^2 \pi^2}{2m}\,\frac{2^2}{L^2}
        = \frac{\hbar^2 \pi^2}{2m}\,\frac{4}{L^2}.
    """)

    st.latex(r"""
    \frac{E_2}{E_1} = \frac{4 / L^2}{1 / L^2} = 4.
    """)

    st.markdown(
    """
    **Result:**  
    - \\(E_2 = 4 E_1\\).  
    - In general, \\(E_n \\propto n^2\\).  
    - Thus, higher levels get **rapidly** more energetic (1, 4, 9, 16, ...).
    """
    )

# Example 3
with st.expander("Example 3 – Effect of Effective Mass on Energy (Semiconductor Well)"):
    st.markdown(
    """
    **Problem.**  
    Consider a 1D infinite well of width \\(L = 5\\,\\text{nm}\\) in a semiconductor
    where the electron has an effective mass \\(m^* = 0.2 m_e\\).  

    1. Compute the ground state energy \\(E_1\\).  
    2. Compare it with the free electron case \\(m^* = m_e\\).  

    ---
    **Solution.**

    We use

    """
    )

    st.latex(r"""
    E_1 = \frac{\hbar^2 \pi^2}{2m^*}\,\frac{1^2}{L^2}.
    """)

    E1_semic = energy_1d_electron_eV(1, 5.0, m_factor=0.2)
    E1_free  = energy_1d_electron_eV(1, 5.0, m_factor=1.0)

    st.write(f"- For m* = 0.2 m_e, L = 5 nm:  **E₁ ≈ {E1_semic:.3f} eV**")
    st.write(f"- For m* = 1.0 m_e, L = 5 nm:  **E₁ ≈ {E1_free:.3f} eV**")

    ratio = E1_semic / E1_free if E1_free != 0 else np.nan
    st.write(f"- Ratio E₁(m* = 0.2 m_e) / E₁(m* = m_e) ≈ {ratio:.2f}")

    st.markdown(
    """
    Since the energy is **inversely proportional** to the mass (\\(E_n \\propto 1/m^*\\)):

    - Smaller effective mass → **higher** energy levels.  
    - This is crucial in **nanostructures and quantum wells** in semiconductors,
      where band structure changes the effective mass.
    """
    )

st.markdown("---")

# ============================================================
# 5. QUIZ (3 QUESTIONS)
# ============================================================
st.header("5. Short Quiz (3 Questions)")

st.markdown("Answer the following questions to test your understanding.")

# Quiz 1
st.subheader("Quiz 1")
st.markdown(
"""
In a 1D infinite potential well, which statement about the
**ground state** is correct?

A. It has \\(n = 0\\) and zero energy.  
B. It has \\(n = 1\\) and non-zero energy.  
C. It has \\(n = 2\\) and is degenerate.  
D. It has negative energy.
"""
)

q1 = st.radio("Your answer for Quiz 1:", ["A", "B", "C", "D"], key="w7quiz1")

if q1:
    if q1 == "B":
        st.success("Correct! The ground state has n = 1 and a positive, non-zero energy.")
    else:
        st.error("Not correct. In an infinite well, quantum numbers start at n = 1, and E₁ is non-zero and positive.")

# Quiz 2
st.subheader("Quiz 2")
st.markdown(
"""
In the 1D infinite well, energy levels (for fixed mass) are:

\\[
E_n \propto \frac{n^2}{L^2}.
\\]

If the well width **L is doubled**, what happens to the ground state energy \\(E_1\\)?

A. It becomes 4 times larger.  
B. It becomes 2 times larger.  
C. It becomes 1/2 of the original value.  
D. It becomes 1/4 of the original value.
"""
)

q2 = st.radio("Your answer for Quiz 2:", ["A", "B", "C", "D"], key="w7quiz2")

if q2:
    # Energy ∝ 1/L². If L → 2L, then 1/L² → 1/(4L²): factor 1/4.
    if q2 == "D":
        st.success("Correct! Doubling L makes the energy 4 times smaller (E ∝ 1/L²).")
    else:
        st.error("Not correct. Remember that E ∝ 1/L²; increasing L lowers the energy.")

# Quiz 3
st.subheader("Quiz 3")
st.markdown(
"""
For the 1D infinite well, how many **nodes** (zeros inside the well,
excluding the boundaries) does the eigenfunction \\(\\psi_n(x)\\) have?

A. 0 for all n.  
B. n for all n.  
C. n − 1 nodes.  
D. It depends on L.
"""
)

q3 = st.radio("Your answer for Quiz 3:", ["A", "B", "C", "D"], key="w7quiz3")

if q3:
    if q3 == "C":
        st.success("Correct! The n-th eigenstate has n − 1 internal nodes.")
    else:
        st.error("Not correct. Check the plots: n=1 has no node, n=2 has 1 node, n=3 has 2 nodes, etc.")

st.markdown("---")

st.markdown(
"""
### Summary – Week 7

- We wrote and explained the **1D time–independent Schrödinger equation**.  
- We derived the **eigenfunctions** and **energies** for the 1D infinite potential well.  
- We visualized \\(\\psi_n(x)\\) and \\(|\\psi_n(x)|^2\\) and saw how they change with \\(n\\).  
- We explored how **well width** and **effective mass** affect energy levels.  
- We solved **3 worked examples** and used **3 quiz questions** to reinforce key ideas.

Use the sidebar to further experiment with different \\(L\\), \\(n\\), and \\(m^*\\) values and
deepen your intuition about quantum confinement in 1D.
"""
)
