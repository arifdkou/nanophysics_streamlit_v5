# app_nanophysics_week9.py
# ============================================================
# Nanophysics – Week 9
# Density of States in Low-Dimensional Systems
# (3D bulk, 2D quantum well, 1D quantum wire, 0D quantum dot)
# + Fermi–Dirac distribution
# with extended theory, figures, and material examples
# ============================================================

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Streamlit page settings
# ------------------------------------------------------------
st.set_page_config(
    page_title="Nanophysics – Week 9: Density of States",
    layout="wide"
)

# ------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------
h = 6.626_070_15e-34      # Planck constant (J·s)
hbar = h / (2 * np.pi)
eV = 1.602_176_634e-19    # J
m_e = 9.109_383_56e-31    # kg
k_B = 1.380_649e-23       # J/K
kB_eV = k_B / eV          # eV/K

# ------------------------------------------------------------
# Helper: DOS shapes (up to a multiplicative constant)
# ------------------------------------------------------------
def dos_shape(E_eV, m_factor=1.0, dim="3D", E_levels_eV=None):
    """
    Return normalized Density of States (DOS) g(E) shape for a given dimensionality.
    Only the shape is important here; overall constants are scaled out.

    Parameters
    ----------
    E_eV        : 1D array of energies in eV (>=0)
    m_factor    : effective mass in units of m_e (kept for completeness)
    dim         : "3D", "2D", "1D", or "0D"
    E_levels_eV : for 0D, list/array of discrete levels (eV)

    Returns
    -------
    g_norm : normalized DOS(E) (max = 1 if non-zero)
    """
    E_eV = np.array(E_eV, dtype=float)
    E_J = E_eV * eV
    _ = m_factor * m_e  # not explicitly used in shape, but kept for realism

    g = np.zeros_like(E_eV)

    if dim == "3D":
        # g_3D(E) ∝ sqrt(E)
        mask = E_J > 0
        g[mask] = np.sqrt(E_J[mask])

    elif dim == "2D":
        # g_2D(E) ∝ constant (for E > 0)
        mask = E_J > 0
        g[mask] = 1.0

    elif dim == "1D":
        # g_1D(E) ∝ 1 / sqrt(E)
        mask = E_J > 0
        g[mask] = 1.0 / np.sqrt(E_J[mask])

    elif dim == "0D":
        # 0D: sum of delta peaks at discrete levels.
        # Approximate with narrow Gaussians in energy.
        if E_levels_eV is None:
            E_levels_eV = np.array([0.05, 0.15, 0.30])
        else:
            E_levels_eV = np.array(E_levels_eV, dtype=float)

        if len(E_levels_eV) > 0:
            E_span = max(E_eV) - min(E_eV)
            sigma = max(E_span * 0.01, 1e-4)  # eV
            for E0 in E_levels_eV:
                g += np.exp(-(E_eV - E0) ** 2 / (2.0 * sigma ** 2))

    g_max = np.max(g)
    if g_max > 0:
        g_norm = g / g_max
    else:
        g_norm = g

    return g_norm


def fermi_dirac(E_eV, E_F_eV, T_K):
    """
    Fermi–Dirac distribution f(E):

        f(E) = 1 / (1 + exp((E - E_F) / (k_B T)))

    with energies in eV, temperature in K.
    """
    E_eV = np.array(E_eV, dtype=float)
    if T_K <= 0:
        f = np.zeros_like(E_eV)
        f[E_eV < E_F_eV] = 1.0
        return f

    beta = 1.0 / (kB_eV * T_K)  # 1/eV
    arg = (E_eV - E_F_eV) * beta
    arg = np.clip(arg, -700, 700)
    f = 1.0 / (1.0 + np.exp(arg))
    return f


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.title("Nanophysics – Week 9")
st.subheader("Density of States in Low-Dimensional Systems and Fermi–Dirac Occupation")

st.markdown("---")

# ============================================================
# 1. LEARNING OBJECTIVES & CONNECTION TO PREVIOUS WEEKS
# ============================================================
st.header("1. Learning Objectives and Connection to Weeks 6–8")

st.markdown(
"""
In **Weeks 6–8** we focused on *how energy levels form*:

- **Week 6:** Bound states in quantum wells and confined systems  
- **Week 7:** Coupled wells and tunneling (formation of mini-bands)  
- **Week 8:** Periodic potentials & band structure (Kronig–Penney model)

This week, we change viewpoint:

> Instead of tracking individual levels, we ask:  
> **How many states exist at each energy?**  

This is the role of the **Density of States (DOS)** \\(g(E)\\).

By the end of Week 9, you should be able to:

1. **Define** DOS \\(g(E)\\) and explain its physical meaning.
2. **Write and interpret** the energy dependence of DOS for:
   - 3D bulk (\\(g_{3D}(E) \\propto \\sqrt{E}\\)),
   - 2D quantum wells (constant DOS),
   - 1D quantum wires (\\(g_{1D}(E) \\propto E^{-1/2}\\)),
   - 0D quantum dots (discrete delta-like peaks).
3. **Explain** how moving from 3D → 2D → 1D → 0D changes DOS due to quantum confinement.
4. **Combine** DOS with the **Fermi–Dirac distribution** \\(f(E)\\) to obtain the occupied states \\(g(E) f(E)\\).
5. **Use simulations** to visualize DOS and occupation for different dimensions, effective masses, temperatures and Fermi levels.
6. **Relate** DOS to real **materials and devices** (GaAs, InP, Graphene, CNTs) and understand their scientific and technological potential.
"""
)

st.markdown("---")

# ============================================================
# 2. THEORY: DOS & FERMI–DIRAC (DETAILED)
# ============================================================
st.header("2. Theory: Density of States and Fermi–Dirac Statistics")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### 2.1 What is the Density of States?")

    st.markdown(
    """
    The **Density of States** \\(g(E)\\) tells us:

    > How many quantum states are available per unit energy at energy \\(E\\).

    Mathematically, if \\(E_n\\) are the energy eigenvalues, then:

    """
    )

    st.latex(r"""
    g(E) = \sum_{n} \delta(E - E_n).
    """)

    st.markdown(
    """
    In real materials, levels are dense and form **bands**, so \\(g(E)\\) becomes
    a smooth function in energy regions where bands exist, and it becomes **zero**
    inside band gaps.

    The DOS is crucial because the total number of electrons is roughly:

    """
    )

    st.latex(r"""
    n \sim \int g(E)\, f(E)\, dE,
    """)

    st.markdown(
    """
    where \\(f(E)\\) is the **Fermi–Dirac occupation**.
    This directly links quantum energy levels to **carrier densities** and
    thus to **conductivity** and **optical response**.
    """
    )

    st.markdown("### 2.2 DOS in Different Dimensions")

    st.markdown("Starting from the free-electron dispersion:")

    st.latex(r"""
    E = \frac{\hbar^2 k^2}{2m},
    """)

    st.markdown(
    """
    and counting states in **k-space**, one obtains the DOS in d dimensions.
    Up to constants, the energy dependence is:

    - **3D bulk:**
    """
    )
    st.latex(r"""
    g_{3D}(E) \propto \sqrt{E}.
    """)

    st.markdown("**2D quantum well:**")
    st.latex(r"""
    g_{2D}(E) = \text{constant} \quad (E > 0).
    """)

    st.markdown("**1D quantum wire:**")
    st.latex(r"""
    g_{1D}(E) \propto \frac{1}{\sqrt{E}} \quad (E > 0).
    """)

    st.markdown("**0D quantum dot:**")
    st.latex(r"""
    g_{0D}(E) = \sum_n \delta(E - E_n).
    """)

with col2:
    st.markdown("### 2.3 Fermi–Dirac Distribution and Occupied DOS")

    st.markdown(
    r"""
    Once we know how many states exist at energy \\(E\\), we ask:

    > What is the probability that a state at energy \\(E\\) is occupied?

    This is given by the **Fermi–Dirac distribution**:

    """
    )

    st.latex(r"""
    f(E) =
    \frac{1}{1 + \exp\left(\dfrac{E - E_F}{k_B T}\right)},
    """)

    st.markdown(
    """
    where:

    - \\(E_F\\) : Fermi level  
    - \\(T\\)   : temperature (K)  
    - \\(k_B\\) : Boltzmann constant  

    At **T = 0 K**, this becomes a sharp step:

    """
    )

    st.latex(r"""
    f(E) =
    \begin{cases}
      1, & E < E_F, \\
      0, & E > E_F.
    \end{cases}
    """)

    st.markdown(
    """
    At finite T, the step is smeared around \\(E_F\\), and some electrons are
    thermally excited to energies above \\(E_F\\).

    The **occupied DOS** is then:

    """
    )

    st.latex(r"""
    g_{\text{occ}}(E) = g(E)\, f(E),
    """)

    st.markdown(
    """
    which tells us, at each energy, how many of the available states are
    actually filled with electrons. This directly impacts:

    - **transport** (which states carry current),  
    - **optical absorption** (initial and final states),  
    - **emission spectra** (recombination from occupied to empty states).
    """
    )

st.markdown("---")

# ============================================================
# 3. INTERACTIVE SIMULATIONS: DOS & OCCUPATION
# ============================================================
st.header("3. Interactive Simulations: DOS and Occupation")

st.markdown(
"""
We will:

1. Select **dimensionality** (3D, 2D, 1D, 0D) and **effective mass**.  
2. Compute the **DOS** \\(g(E)\\) vs energy.  
3. Choose **temperature** and **Fermi level** and compute **Fermi–Dirac** \\(f(E)\\).  
4. Plot **g(E)**, **f(E)** and **g(E) f(E)**.  
5. Compare DOS shapes of all dimensions in a single figure.
"""
)

# ---------------- Sidebar parameters ----------------
st.sidebar.title("Week 9 – Simulation Parameters")

dim_option = st.sidebar.selectbox(
    "Dimensionality",
    ["3D bulk", "2D quantum well", "1D quantum wire", "0D quantum dot"]
)

if dim_option.startswith("3D"):
    dim = "3D"
elif dim_option.startswith("2D"):
    dim = "2D"
elif dim_option.startswith("1D"):
    dim = "1D"
else:
    dim = "0D"

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

E_max_eV = st.sidebar.slider(
    "Maximum energy E_max (eV)",
    min_value=0.2,
    max_value=2.0,
    value=1.0,
    step=0.1
)

n_E = st.sidebar.slider(
    "Number of energy points",
    min_value=200,
    max_value=2000,
    value=600,
    step=200
)

T_K = st.sidebar.slider(
    "Temperature T (K)",
    min_value=0.0,
    max_value=600.0,
    value=300.0,
    step=50.0
)

E_F_eV = st.sidebar.slider(
    "Fermi level E_F (eV)",
    min_value=0.0,
    max_value=2.0,
    value=0.4,
    step=0.05
)

if dim == "0D":
    st.sidebar.markdown("**0D: example discrete levels**")
    levels_choice = st.sidebar.selectbox(
        "Dot level set",
        ["Set A: 0.05, 0.15, 0.30 eV", "Set B: 0.20, 0.35, 0.50 eV"]
    )
    if levels_choice.startswith("Set A"):
        E_levels_eV = np.array([0.05, 0.15, 0.30])
    else:
        E_levels_eV = np.array([0.20, 0.35, 0.50])
else:
    E_levels_eV = None

st.sidebar.markdown("---")
st.sidebar.info(
    "Change dimensionality, effective mass, temperature and Fermi level to "
    "explore how DOS and occupation behave in different nanostructures."
)

# ------------------------------------------------------------
# Energy grid and main calculations
# ------------------------------------------------------------
E_eV = np.linspace(0.0, E_max_eV, n_E)
g_E = dos_shape(E_eV, m_factor=m_factor, dim=dim, E_levels_eV=E_levels_eV)
f_E = fermi_dirac(E_eV, E_F_eV=E_F_eV, T_K=T_K)
g_occ = g_E * f_E

# ------------------------------------------------------------
# 3.1 DOS vs Energy (selected dimension)
# ------------------------------------------------------------
col_dos, col_occ = st.columns(2)

with col_dos:
    st.subheader("3.1 Density of States g(E) vs Energy (Selected Dimension)")

    fig1, ax1 = plt.subplots()
    ax1.plot(E_eV, g_E)
    ax1.set_xlabel("Energy E (eV)")
    ax1.set_ylabel("DOS (normalized)")
    ax1.set_title(f"DOS for {dim_option}")
    ax1.grid(True)
    st.pyplot(fig1)

    if dim == "3D":
        shape_text = r"$g_{3D}(E) \propto \sqrt{E}$"
    elif dim == "2D":
        shape_text = r"$g_{2D}(E) = \text{constant}$"
    elif dim == "1D":
        shape_text = r"$g_{1D}(E) \propto 1/\sqrt{E}$"
    else:
        shape_text = r"$g_{0D}(E) = \sum_n \delta(E - E_n)$ (here shown as narrow peaks)"

    st.markdown(
    f"""
    **Selected parameters**

    - Dimensionality: **{dim_option}**  
    - Effective mass: **{mass_option}**  
    - Energy range: **0 – {E_max_eV:.2f} eV**

    **Characteristic DOS behavior**

    - {shape_text}
    """
    )

with col_occ:
    st.subheader("3.2 Fermi–Dirac f(E) and Occupied DOS g(E) f(E)")

    fig2, ax2 = plt.subplots()
    ax2.plot(E_eV, f_E, label="Fermi–Dirac f(E)")
    ax2.plot(E_eV, g_occ, label="Occupied DOS g(E) f(E)")
    ax2.axvline(E_F_eV, linestyle="--", label=f"E_F = {E_F_eV:.2f} eV")
    ax2.set_xlabel("Energy E (eV)")
    ax2.set_ylabel("Probability / normalized DOS")
    ax2.set_title(f"Occupation at T = {T_K:.0f} K")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    st.markdown(
    f"""
    **Fermi–Dirac parameters**

    - Temperature: **T = {T_K:.0f} K**  
    - Fermi level: **E_F = {E_F_eV:.2f} eV**

    **Observations**

    - At low T, f(E) is close to a **step** at E_F.  
    - At higher T, the step becomes **smeared** and electrons occupy states above E_F.  
    - g(E) f(E) shows where **electrons actually reside** in energy.
    """
    )

# ------------------------------------------------------------
# 3.3 Comparative DOS: 3D vs 2D vs 1D vs 0D
# ------------------------------------------------------------
st.subheader("3.3 Comparative DOS in 3D, 2D, 1D and 0D (Shape Only)")

E_cmp = np.linspace(0.0, E_max_eV, n_E)
g3 = dos_shape(E_cmp, dim="3D")
g2 = dos_shape(E_cmp, dim="2D")
g1 = dos_shape(E_cmp, dim="1D")
g0 = dos_shape(E_cmp, dim="0D", E_levels_eV=[0.25 * E_max_eV, 0.5 * E_max_eV, 0.75 * E_max_eV])

fig_cmp, ax_cmp = plt.subplots()
ax_cmp.plot(E_cmp, g3, label="3D")
ax_cmp.plot(E_cmp, g2, label="2D")
ax_cmp.plot(E_cmp, g1, label="1D")
ax_cmp.plot(E_cmp, g0, label="0D")
ax_cmp.set_xlabel("Energy E (eV)")
ax_cmp.set_ylabel("Normalized DOS (shape)")
ax_cmp.set_title("DOS Shape Comparison: 3D vs 2D vs 1D vs 0D")
ax_cmp.grid(True)
ax_cmp.legend()
st.pyplot(fig_cmp)

st.markdown(
"""
**Interpretation of the comparative figure**

- **3D bulk:** DOS grows smoothly from zero as \\(\\sqrt{E}\\).  
- **2D wells:** DOS is **flat** (constant) above the subband edge.  
- **1D wires:** DOS diverges as \\(1/\\sqrt{E}\\) at the band edge (van Hove singularity).  
- **0D dots:** DOS consists of **discrete peaks**, resembling δ-functions.
"""
)

st.markdown("---")

# ============================================================
# 4. WORKED EXAMPLES (3 Detailed Problems)
# ============================================================
st.header("4. Worked Examples (3 Detailed Problems)")

with st.expander("Example 1 – Comparing DOS Shapes (3D vs 2D vs 1D)", expanded=True):
    st.markdown(
    """
    **Problem.**  
    Compare the DOS near the band edge for:

    - a 3D bulk semiconductor,  
    - a 2D quantum well,  
    - a 1D quantum wire,

    assuming the same effective mass.  

    1. Write the qualitative energy dependence of the DOS in each case.  
    2. Which system has the strongest DOS singularity near the band edge (\\(E \\to 0^+\\))?  
    3. How might this affect optical transitions?

    ---
    **Solution.**

    DOS shapes (for \\(E > 0\\), up to constants):

    - **3D bulk:**
    """
    )
    st.latex(r"""
    g_{3D}(E) \propto \sqrt{E}.
    """)

    st.markdown("**2D quantum well:**")
    st.latex(r"""
    g_{2D}(E) = \text{constant}.
    """)

    st.markdown("**1D quantum wire:**")
    st.latex(r"""
    g_{1D}(E) \propto \frac{1}{\sqrt{E}}.
    """)

    st.markdown(
    """
    Near the band edge (small E > 0):

    - 3D: \\(g_{3D}(E) \\to 0\\) as \\(\\sqrt{E}\\).  
    - 2D: \\(g_{2D}(E)\\) is **finite** at the edge.  
    - 1D: \\(g_{1D}(E) \\to \\infty\\) as \\(1/\\sqrt{E}\\): a **van Hove singularity**.

    Therefore:

    1. The strongest singularity occurs in **1D** systems.  
    2. 2D has step-like DOS at each subband edge.  
    3. 3D is the smoothest.

    **Impact on optical transitions**

    Optical absorption is roughly proportional to the product of DOS and
    transition matrix elements. Hence:

    - 1D wires → very **sharp features** at subband edges.  
    - 2D wells → step-like absorption.  
    - 3D bulk → smooth absorption onset.
    """
    )

with st.expander("Example 2 – 0D Quantum Dot: Discrete Levels"):
    st.markdown(
    """
    **Problem.**  
    A small quantum dot has discrete electron energy levels approximately at:

    \\[
    E_1 = 0.10 \\, \\text{eV}, \quad
    E_2 = 0.25 \\, \\text{eV}, \quad
    E_3 = 0.40 \\, \\text{eV}.
    \\]

    1. Sketch qualitatively the DOS \\(g(E)\\).  
    2. Compare this with a 2D quantum well having the same conduction band edge.  
    3. How does this difference affect emission spectra?

    ---
    **Solution.**

    1. For a 0D quantum dot, the DOS is:

       \\[
       g_{0D}(E) = \delta(E - E_1)
                  + \delta(E - E_2)
                  + \delta(E - E_3).
       \\]

       In practice, each δ-function appears as a **narrow peak** due to broadening.

    2. For a 2D quantum well, the DOS above the band edge is a **constant** in each
       subband, giving a **step-like** shape rather than discrete peaks.

    3. Consequently:

       - Quantum dots ⇒ **sharp emission lines** at specific energies (atom-like).  
       - Quantum wells ⇒ **broader spectral features**, because of continuous DOS
         above each subband edge.

    This is why quantum dots are powerful for **color-selective LEDs**, **lasers**
    and **single-photon sources**.
    """
    )

with st.expander("Example 3 – Effect of Temperature on Occupation in 2D"):
    st.markdown(
    """
    **Problem.**  
    Consider a 2D quantum well with a conduction band edge at \\(E = 0\\) and a
    constant DOS for \\(E > 0\\). The Fermi level is \\(E_F = 0.1\\,\\text{eV}\\).

    1. Describe qualitatively what happens to the occupied DOS \\(g(E) f(E)\\) when
       temperature increases from **T = 0 K** to **T = 300 K**.  
    2. What happens to the average electron energy?

    ---
    **Solution.**

    - At **T = 0 K**:
      - f(E) is a **step function**: all states with \\(E < E_F\\) are fully
        occupied (f = 1), all states with \\(E > E_F\\) are empty (f = 0).  
      - With constant DOS, \\(g(E) f(E)\\) is a **flat region** from 0 to \\(E_F\\),
        then 0 above.

    - At **T = 300 K**:
      - f(E) becomes **smooth** around \\(E_F\\).  
      - Some electrons are thermally excited to energies **above** \\(E_F\\), and
        holes appear **below** \\(E_F\\).  
      - \\(g(E) f(E)\\) acquires a **tail** above \\(E_F\\).

    As temperature increases:

    - The **average electron energy increases**.  
    - This affects transport (higher-energy, more mobile electrons) and optical
      properties (broadened absorption edge).
    """
    )

st.markdown("---")

# ============================================================
# 5. MATERIAL EXAMPLES: GaAs, InP, Graphene, CNT
# ============================================================
st.header("5. Material Examples and Technological Applications")

tabs = st.tabs(["GaAs (3D / QW)", "InP (Optoelectronics)", "Graphene (2D)", "CNTs (1D)"])

with tabs[0]:
    st.subheader("GaAs – 3D Bulk and 2D Quantum Wells")

    st.markdown(
    """
    - **GaAs bulk (3D):**  
      - Direct bandgap semiconductor, widely used in **lasers**, **LEDs** and **high-speed electronics**.  
      - DOS near the conduction band edge follows the **3D √E law**, giving a smooth absorption onset.

    - **GaAs / AlGaAs quantum wells (2D):**  
      - Confinement in one direction creates **2D subbands**.  
      - Each subband contributes a **constant DOS** above its edge.  
      - Leads to **step-like DOS** and sharper optical transitions.  
      - Used in **quantum well lasers**, **modulators** and **high-electron-mobility transistors (HEMTs)**.

    By engineering well width and barrier height, we **shape the DOS** and design devices
    with tailored emission wavelengths and threshold currents.
    """
    )

with tabs[1]:
    st.subheader("InP – Optoelectronics and High-Speed Devices")

    st.markdown(
    """
    - **InP** is another direct bandgap semiconductor, important around **1.3–1.55 μm**
      (telecom wavelengths).  
    - In **bulk InP**, DOS is again 3D-like, \\(g_{3D}(E) \\propto \\sqrt{E}\\).

    - In **InP-based quantum wells and superlattices** (e.g. InGaAs/InP):
      - Carriers experience **2D confinement** → constant DOS per subband.  
      - This gives precise control of optical gain spectra in **telecom lasers** and
        **modulators**.

    The combination of InP substrates with quantum wells and heterostructures is
    the basis of **modern optical fiber communication systems**.
    """
    )

with tabs[2]:
    st.subheader("Graphene – 2D Dirac Material")

    st.markdown(
    """
    Graphene is a **2D material** with a linear dispersion relation near the Dirac
    points:

    \\[
    E(k) \approx \hbar v_F |k|.
    \\]

    As a result, the DOS near the Dirac point is approximately:

    \\[
    g(E) \propto |E|.
    \\]

    Key features:

    - Zero bandgap (semi-metal), DOS vanishes at the Dirac point.  
    - Very high carrier mobilities.  
    - Tunable carrier density via gating → **ambipolar transport**.

    Applications:

    - High-frequency transistors  
    - Transparent conductive electrodes  
    - Sensors and flexible electronics

    Even though graphene is 2D, its DOS shape is different from the parabolic 2D
    case because of the **linear dispersion** (Dirac cones), which is an excellent
    example for students to compare with the simple models used here.
    """
    )

with tabs[3]:
    st.subheader("Carbon Nanotubes (CNTs) – 1D Quantum Wires")

    st.markdown(
    """
    Carbon nanotubes (CNTs) are quasi-**1D nanostructures** obtained by rolling
    a graphene sheet into a cylinder.

    - Their electronic structure can be **metallic** or **semiconducting** depending
      on chirality.  
    - In semiconducting CNTs, subbands form along the tube axis, and DOS shows
      characteristic **1D van Hove singularities**:

      \\[
      g_{1D}(E) \propto \frac{1}{\sqrt{E - E_n}} \quad \text{(near each subband edge)}.
      \\]

    Consequences:

    - Optical absorption spectra show **sharp peaks**, each corresponding to
      transitions between subbands (E₁₁, E₂₂, …).  
    - These features make CNTs interesting for **nano-optoelectronic devices**,
      **photodetectors** and **single-photon emitters**.

    CNTs are a beautiful experimental realization of the **1D DOS** model discussed
    in this lecture.
    """
    )

st.markdown("---")

# ============================================================
# 6. SHORT QUIZ (3 QUESTIONS)
# ============================================================
st.header("6. Short Quiz (3 Questions)")

st.markdown("Answer the questions to check your understanding of DOS and occupation.")

st.subheader("Quiz 1")
st.markdown(
"""
For a **3D free electron gas** (bulk), the DOS near the band edge behaves as:

A. \\(g(E) \\propto E^{-1/2}\\)  
B. \\(g(E) \\propto \\sqrt{E}\\)  
C. \\(g(E) = \\text{constant}\\)  
D. \\(g(E) \\propto E^2\\)
"""
)
q1 = st.radio("Your answer for Quiz 1:", ["A", "B", "C", "D"], key="w9q1")

if q1:
    if q1 == "B":
        st.success("Correct! For 3D free electrons, g(E) ∝ √E.")
    else:
        st.error("Not correct. In 3D, DOS increases like the square root of energy.")

st.subheader("Quiz 2")
st.markdown(
"""
Which dimensionality shows a **1/√E divergence** of the DOS at the band edge?

A. 3D bulk  
B. 2D quantum well  
C. 1D quantum wire  
D. 0D quantum dot
"""
)
q2 = st.radio("Your answer for Quiz 2:", ["A", "B", "C", "D"], key="w9q2")

if q2:
    if q2 == "C":
        st.success("Correct! In 1D, g(E) ∝ 1/√E, giving a van Hove singularity.")
    else:
        st.error("Not correct. The 1/√E singularity is characteristic of 1D wires and CNTs.")

st.subheader("Quiz 3")
st.markdown(
"""
At **T = 0 K**, the Fermi–Dirac distribution f(E) is:

A. A smooth sigmoid function centered at E_F.  
B. A step function: 1 for E < E_F, 0 for E > E_F.  
C. Equal to 0 for all E.  
D. Equal to 1/2 for all E.
"""
)
q3 = st.radio("Your answer for Quiz 3:", ["A", "B", "C", "D"], key="w9q3")

if q3:
    if q3 == "B":
        st.success("Correct! At T = 0 K, f(E) is exactly a step at E_F.")
    else:
        st.error("Not correct. At zero temperature, states below E_F are fully occupied, above E_F are empty.")

st.markdown("---")

st.markdown(
"""
### Summary – Week 9

- We defined the **Density of States** \\(g(E)\\) and showed how it depends on **dimension**.  
- We combined DOS with the **Fermi–Dirac distribution** to obtain the **occupied DOS** \\(g(E) f(E)\\).  
- We visualized DOS for 3D, 2D, 1D, and 0D systems and linked these to **quantum wells, wires and dots**.  
- We discussed real materials: **GaAs, InP, Graphene, CNTs**, and how their DOS shapes influence
  optical and electronic properties.

You can now use the sliders to build physical intuition about how confinement and
statistics work together in nanostructures — this will be essential for the coming
weeks on **transport** and **optical processes** in nanoscale devices.
"""
)
