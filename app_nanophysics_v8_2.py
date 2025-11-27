# app_nanophysics_week8.py
# ============================================================
# Nanophysics – Week 8
# 1D Periodic Potential & Band Structure (Kronig–Penney, simplified)
# Streamlit Interactive Lecture Note
# ============================================================

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Streamlit page settings
# ------------------------------------------------------------
st.set_page_config(
    page_title="Nanophysics – Week 8: Periodic Potential & Bands",
    layout="wide"
)

# ------------------------------------------------------------
# Helper functions – Dimensionless Kronig–Penney (delta-barrier limit)
# ------------------------------------------------------------
# We use dimensionless units:
#   ħ = 1, 2m = 1 → m = 1/2, lattice constant a = 1
#
# Kronig–Penney (delta-barrier model):
#   V(x) = (ħ² P / 2m a) Σ δ(x - n a)
#
# Dispersion relation:
#   cos(k a) = cos(q a) + (P / (q a)) sin(q a)
#
# With a = 1, q^2 = E (dimensionless energy) → q = sqrt(E):
#   cos(k) = cos(q) + (P / q) sin(q)
#
# Allowed energies: there exists real k if |RHS(E)| ≤ 1.


def kronig_penney_allowed(P, E_max=20.0, n_E=800):
    """
    Compute allowed energy–k pairs for the simplified Kronig–Penney model.

    Parameters
    ----------
    P     : dimensionless barrier strength
    E_max : maximum energy (dimensionless)
    n_E   : number of energy samples

    Returns
    -------
    E_allowed : 1D array of allowed energies (dimensionless)
    k_allowed : 1D array of corresponding k values (0..π)
    RHS       : RHS(E) values (for diagnostic plots if needed)
    E_all     : full energy grid used
    """
    # Avoid E=0 to prevent division by zero in P/q
    E_all = np.linspace(0.01, E_max, n_E)
    q = np.sqrt(E_all)

    RHS = np.cos(q) + (P / q) * np.sin(q)

    # Allowed when |RHS| <= 1 (since cos(k) must be in [-1,1])
    mask_allowed = np.abs(RHS) <= 1.0

    # Clip RHS to [-1, 1] before arccos to avoid numerical issues
    RHS_clip = np.clip(RHS, -1.0, 1.0)
    k_all = np.arccos(RHS_clip)  # principal branch 0..π

    E_allowed = E_all[mask_allowed]
    k_allowed = k_all[mask_allowed]

    return E_allowed, k_allowed, RHS, E_all


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------
st.title("Nanophysics – Week 8")
st.subheader("1D Periodic Potential & Band Structure (Kronig–Penney Model, Simplified)")

st.markdown("---")

# ============================================================
# 1. LEARNING OBJECTIVES
# ============================================================
st.header("1. Learning Objectives (Haftanın Ana Amaçları)")

st.markdown(
"""
By the end of this week, you should be able to:

1. **Explain** why electrons in a periodic potential form **energy bands** instead of single discrete levels.
2. **Write** the basic dispersion relation of the **Kronig–Penney model** in a dimensionless form.
3. **Identify** allowed energy bands and forbidden **band gaps** from the condition \\(|\\text{RHS}(E)| \\le 1\\).
4. **Simulate** how the band structure changes with the barrier strength \\(P\\) (weak vs strong periodic potential).
5. **Relate** the 1D band structure to **quantum wells, wires and dots** in nanostructures.
6. **Solve** example problems about band edges and the effect of barrier strength on band gaps.
"""
)

st.markdown("---")

# ============================================================
# 2. THEORY: 1D PERIODIC POTENTIAL & KRONIG–PENNEY
# ============================================================
st.header("2. Theory: 1D Periodic Potential and Kronig–Penney Model")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### 2.1 Periodic Potential in a Crystal")

    st.markdown(
    """
    In a 1D crystal, the electron sees a **periodic potential**:

    """
    )
    st.latex(r"""
    V(x + a) = V(x),
    """)

    st.markdown(
    """
    where \\(a\\) is the **lattice constant**.

    Schrödinger equation:

    """
    )

    st.latex(r"""
    \left[
      -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x)
    \right]\psi(x) = E \psi(x).
    """)

    st.markdown(
    """
    For periodic \\(V(x)\\), solutions satisfy **Bloch’s theorem**:

    """
    )

    st.latex(r"""
    \psi_k(x) = e^{ikx} u_k(x), \quad u_k(x + a) = u_k(x).
    """)

    st.markdown(
    """
    This leads to **energy bands** \\(E_n(k)\\) and **gaps** between bands.
    """
    )

with col2:
    st.markdown("### 2.2 Simplified Kronig–Penney Model (Delta Barriers)")

    st.markdown(
    r"""
    We use a simple model: **delta barriers** at each lattice site:

    """
    )
    st.latex(r"""
    V(x) = \sum_{n=-\infty}^{\infty}
           \frac{\hbar^2 P}{2 m a} \, \delta(x - n a),
    """)

    st.markdown(
    """
    where:

    - \\(a\\): lattice constant  
    - \\(P\\): dimensionless barrier strength (controls potential strength)

    Solving Schrödinger’s equation with Bloch’s theorem yields the
    dispersion relation (Kronig–Penney):

    """
    )

    st.latex(r"""
    \cos(k a)
    =
    \cos(q a)
    +
    \frac{P}{q a} \sin(q a),
    """)

    st.markdown(
    r"""
    where:

    - \\(k\\): crystal momentum (Bloch wave number)  
    - \\(q = \sqrt{2mE}/\hbar\\) is the wave number **between** barriers  
    - \\(E\\): electron energy

    **Allowed energies:** for real \\(k\\), we must have:

    """
    )

    st.latex(r"""
    |\cos(k a)| \le 1 \quad \Rightarrow \quad
    \left|\cos(q a) + \frac{P}{q a} \sin(q a)\right| \le 1.
    """)

    st.markdown(
    """
    For a given \\(P\\), this inequality defines **allowed energy bands** (where
    there exists a real k) and **band gaps** (no real k).
    """
    )

st.markdown("### 2.3 Dimensionless Form Used in the Simulation")

st.markdown(
"""
To keep the numerics simple, we set:

- \\(\\hbar = 1\\), \\(2m = 1 \\Rightarrow m = 1/2\\)
- lattice constant \\(a = 1\\)

Then:

- \\(q^2 = E\\) → \\(q = \\sqrt{E}\\) (dimensionless)  
- Dispersion relation becomes:

"""
)

st.latex(r"""
\cos(k) = \cos(\sqrt{E})
         + \frac{P}{\sqrt{E}} \sin(\sqrt{E}).
""")

st.markdown(
"""
For each energy \\(E\\), if

\\[
\left|\cos(\sqrt{E}) + \frac{P}{\sqrt{E}} \sin(\sqrt{E})\right| \le 1,
\\]

then there exists at least one **real** \\(k\\) and \\(E\\) belongs to an
**allowed band**.
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

1. Draw a **schematic periodic potential** in real space.  
2. Compute allowed energies and plot a **band diagram** \\(E(k)\\) in the first Brillouin zone.  
3. Explore how the bands change as we vary **barrier strength** \\(P\\).
"""
)

# ---------------- Sidebar parameters ----------------
st.sidebar.title("Week 8 – Simulation Parameters")

P = st.sidebar.slider(
    "Barrier strength P (dimensionless)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.5
)

E_max = st.sidebar.slider(
    "Maximum energy (dimensionless)",
    min_value=5.0,
    max_value=40.0,
    value=20.0,
    step=5.0
)

n_E = st.sidebar.slider(
    "Number of energy points for sampling",
    min_value=200,
    max_value=1600,
    value=800,
    step=200
)

n_cells = st.sidebar.slider(
    "Number of unit cells to draw (for potential schematic)",
    min_value=3,
    max_value=8,
    value=4,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.info("P parametresini değiştirerek bant aralıklarının nasıl açılıp kapandığını inceleyebilirsin.")

# ---------------- Compute band structure ----------------
E_allowed, k_allowed, RHS_all, E_all = kronig_penney_allowed(P, E_max=E_max, n_E=n_E)

# ============================================================
# 3.1 Real-space periodic potential schematic
# ============================================================
col_pot, col_band = st.columns(2)

with col_pot:
    st.subheader("3.1 Schematic 1D Periodic Potential V(x)")

    # Simple square-barrier periodic potential for plotting only (not used in dispersion)
    a = 1.0
    V0_plot = 1.0  # arbitrary units, just for visualization
    x_plot = np.linspace(0, n_cells * a, 1000)
    V_plot = np.zeros_like(x_plot)

    # Define in each cell: barrier from 0.3a to 0.5a
    for n in range(n_cells):
        x_left = n * a + 0.3 * a
        x_right = n * a + 0.5 * a
        mask_barrier = (x_plot >= x_left) & (x_plot <= x_right)
        V_plot[mask_barrier] = V0_plot

    fig1, ax1 = plt.subplots()
    ax1.plot(x_plot, V_plot)
    ax1.set_xlabel("x (lattice units)")
    ax1.set_ylabel("V(x) (arbitrary)")
    ax1.set_title("Periodic Potential (schematic)")
    ax1.grid(True)
    st.pyplot(fig1)

    st.markdown(
    f"""
    - Lattice constant: **a = 1** (dimensionless units)  
    - Number of cells shown: **{n_cells}**  
    - Barrier strength parameter in the Kronig–Penney model: **P = {P:.1f}**

    Bu grafik sadece periyodik yapıyı **görsel olarak** göstermek içindir;
    bant yapısını hesaplamak için yukarıda verdiğimiz analitik ifadeyi kullanıyoruz.
    """
    )

# ============================================================
# 3.2 Band structure E(k)
# ============================================================
with col_band:
    st.subheader("3.2 Band Diagram E(k) in the First Brillouin Zone")

    fig2, ax2 = plt.subplots()

    if len(E_allowed) > 0:
        # Plot allowed (k,E) and symmetric (-k,E) for first Brillouin zone
        ax2.scatter(k_allowed, E_allowed, s=4)
        ax2.scatter(-k_allowed, E_allowed, s=4)

    ax2.set_xlabel("k (dimensionless, lattice units)")
    ax2.set_ylabel("Energy E (dimensionless)")
    ax2.set_title("Allowed Energy Bands in 1D (Kronig–Penney)")
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(0, E_max)
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown(
    """
    **Gözlemler:**

    - Enerji ekseninde **sürekli bölgeler** (nokta kümeleşmesi) → **bantlar**.  
    - Enerji ekseninde **boş kalan aralıklar** → **yasak bant aralıkları (band gaps)**.  
    - P küçük (zayıf potansiyel) iken bantlar, neredeyse **serbest elektron** eğrisine benzer.  
    - P büyüdükçe, bantlar arasındaki **band gap** genişler.
    """
    )

st.markdown("---")

# ============================================================
# 4. WORKED EXAMPLES (3 Detailed Problems)
# ============================================================
st.header("4. Worked Examples (3 Detailed Problems)")

# Example 1
with st.expander("Example 1 – P = 0: Free Electron Limit", expanded=True):
    st.markdown(
    """
    **Problem.**  
    In the Kronig–Penney model, what happens when the barrier strength
    parameter is \\(P = 0\\)?

    1. What is the dispersion relation in this case?  
    2. Are there band gaps?

    ---
    **Solution.**

    For \\(P = 0\\), the dispersion relation becomes:

    \\[
    \cos(k) = \cos(\sqrt{E}).
    \\]

    This implies:

    \\[
    k = \pm \sqrt{E} + 2 \pi n, \quad n \in \mathbb{Z},
    \\]

    i.e., the energy is effectively:

    \\[
    E = k^2
    \\]

    (up to periodicity in reciprocal space).

    This is the **free electron** dispersion (in our dimensionless units).

    - There is **no periodic scattering**; the electron moves freely.  
    - Therefore, there are **no band gaps**: the energy is continuous.

    In the simulation, if you set **P = 0**, bant diyagramında enerji ekseni
    boyunca boşluk kalmadığını (gap oluşmadığını) görebilirsin.
    """
    )

# Example 2
with st.expander("Example 2 – Opening of the First Band Gap (Moderate P)"):
    st.markdown(
    """
    **Problem.**  
    For a moderate barrier strength, say \\(P = 2\\):

    1. Do you observe a **first band gap** between the lowest and next band?  
    2. How does this compare to \\(P = 0\\)?  
    3. Qualitatively, what happens if we increase P further (e.g., P = 6)?

    ---
    **Solution (qualitative).**

    - At \\(P = 2\\), the periodic potential is **non-zero**. Electrons are
      partially Bragg-scattered at the Brillouin zone boundaries
      (around \\(k = \pm \pi\\)).  
    - This scattering **mixes** states and opens a **gap** at those
      k-values – we see a forbidden energy region between the first
      and second bands.

    Compared to \\(P = 0\\):

    - For \\(P = 0\\), no gap: the dispersion is parabolic (free-electron-like).  
    - For \\(P = 2\\), a small **first band gap** appears.  
    - For \\(P = 6\\), the band gap becomes **much wider**, and the allowed
      bands become more separated.

    In practical nanodevices, stronger periodic modulation (e.g. deeper
    quantum well superlattices) → **larger band gaps** and stronger confinement.
    """
    )

# Example 3
with st.expander("Example 3 – Relation to Quantum Wells, Wires and Dots"):
    st.markdown(
    """
    **Problem.**  
    Explain qualitatively how the 1D band picture from the Kronig–Penney model
    relates to:

    1. **Quantum wells** (e.g., multi-quantum-well structures)  
    2. **Quantum wires**  
    3. **Quantum dots**

    ---
    **Solution (conceptual).**

    - A **single quantum well** has **discrete levels** (like a 1D finite well).  
    - A **periodic array of wells** (superlattice) approximates a **1D crystal**:
      each discrete level of the isolated well **broadens into a band** due to
      coupling between neighboring wells → very similar to the Kronig–Penney bands.

    1. **Quantum wells**:
       - In a superlattice of wells separated by barriers, electronic states form
         **minibands** (narrow bands).
       - Our 1D periodic model is a simplified version of this situation.

    2. **Quantum wires**:
       - Confinement in two directions, periodicity along one direction.  
       - Leads to **1D subbands** dispersing along the wire axis; again,
         a band picture similar to 1D Kronig–Penney applies along the wire.

    3. **Quantum dots**:
       - Confinement in all three directions → **discrete, atom-like levels**,
         not extended bands.
       - However, an **array of coupled quantum dots** can form mini-bands,
         analogous to how discrete atomic levels broaden into bands in a crystal.

    Özetle:  
    - **Tek kuyu → seviye**,  
    - **Kuyu dizisi → bant**,  
    - **Güçlü bağ → geniş bant, zayıf bağ → dar bant**.
    """
    )

st.markdown("---")

# ============================================================
# 5. QUIZ (3 QUESTIONS)
# ============================================================
st.header("5. Short Quiz (3 Questions)")

st.markdown("Answer the questions to check your understanding of band formation.")

# Quiz 1
st.subheader("Quiz 1")
st.markdown(
"""
In the simplified Kronig–Penney model we used (delta barriers):

\\[
\cos(k) =
\cos(\sqrt{E}) + \frac{P}{\sqrt{E}} \sin(\sqrt{E}).
\\]

What is the **condition** for an energy E to belong to an allowed band?

A. \\(\\cos(k) = 0\\)  
B. \\(|\\cos(k)| > 1\\)  
C. \\(\\left|\\cos(\\sqrt{E}) + \\dfrac{P}{\\sqrt{E}} \\sin(\\sqrt{E})\\right| \\le 1\\)  
D. No condition; all E are allowed
"""
)

q1 = st.radio("Your answer for Quiz 1:", ["A", "B", "C", "D"], key="w8q1")

if q1:
    if q1 == "C":
        st.success("Correct! The RHS must be in [−1,1] so that a real k exists.")
    else:
        st.error("Not correct. Allowed energies require |RHS(E)| ≤ 1 so that cos(k) is physical.")

# Quiz 2
st.subheader("Quiz 2")
st.markdown(
"""
How does increasing the barrier strength parameter \\(P\\) qualitatively
affect the band structure?

A. Band gaps become **wider** and bands become more separated.  
B. Band gaps become **narrower**, bands merge.  
C. Band structure disappears; all energies are allowed.  
D. Only the lowest band remains; all others vanish.
"""
)

q2 = st.radio("Your answer for Quiz 2:", ["A", "B", "C", "D"], key="w8q2")

if q2:
    if q2 == "A":
        st.success("Correct! Stronger periodic scattering opens larger band gaps.")
    else:
        st.error("Not correct. Larger P generally increases band gaps and separates bands.")

# Quiz 3
st.subheader("Quiz 3")
st.markdown(
"""
Which statement best describes the relation between **quantum wells**
and **bands** in a periodic structure?

A. A single quantum well already has continuous bands.  
B. A periodic array of wells turns each discrete level into a **band**.  
C. Quantum wells and bands are unrelated concepts.  
D. Bands only appear in 3D crystals, not in 1D structures.
"""
)

q3 = st.radio("Your answer for Quiz 3:", ["A", "B", "C", "D"], key="w8q3")

if q3:
    if q3 == "B":
        st.success("Correct! Coupling of many wells broadens discrete levels into bands.")
    else:
        st.error("Not correct. Periodic coupling of wells is exactly how bands form from discrete levels.")

st.markdown("---")

st.markdown(
"""
### Summary – Week 8

- We introduced **periodic potentials** and the basic idea of **Bloch waves**.  
- We used a simplified **Kronig–Penney model** (delta barriers) in dimensionless units.  
- We derived/used the dispersion relation

  \\[
  \cos(k) = \cos(\sqrt{E}) + \frac{P}{\sqrt{E}}\sin(\sqrt{E}),
  \\]

  and identified **allowed bands** via the condition \\(|\\text{RHS}(E)| \\le 1\\).  
- We saw how **band gaps** appear and grow as the barrier strength \\(P\\) increases.  
- We connected this 1D picture to **quantum wells, superlattices, wires and dots**.  
- We solved **3 worked examples** and checked understanding with **3 quiz questions**.

Artık P, Eₘₐₓ ve örnek sayısını değiştirerek farklı periyodik potansiyel rejimleri için
bant yapılarının nasıl davrandığını sezgisel olarak gösterebilirsin.
"""
)
