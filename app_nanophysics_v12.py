# app_nanophysics_week12.py
# ============================================================
# Nanophysics – Week 12 (Clean LaTeX Rendering)
# Quantum Dots: 0D Confinement, Discrete Levels, 0D DOS,
# Emission Tuning, Purcell Effect, Single-Photon Concept
#
# Lecture notes + worked examples + simulations + graphs + quiz
# ALL equations rendered ONLY with st.latex() for readability
# ============================================================

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Nanophysics – Week 12: Quantum Dots (0D) & Emission Engineering",
    layout="wide"
)

# ----------------------------
# Constants
# ----------------------------
h = 6.626_070_15e-34
hbar = h/(2*np.pi)
c = 2.997_924_58e8
e = 1.602_176_634e-19
m0 = 9.109_383_7015e-31
kB = 1.380_649e-23

# ----------------------------
# Helper functions
# ----------------------------
def particle_in_box_3d_energy_eV(nx, ny, nz, m_rel, L_nm):
    """
    3D infinite box energy:
        E = (h^2 / (8 m L^2)) (nx^2 + ny^2 + nz^2)
    Return energy in eV.
    """
    L = float(L_nm)*1e-9
    m = float(m_rel)*m0
    n2 = float(nx)**2 + float(ny)**2 + float(nz)**2
    EJ = (h**2/(8*m*(L**2))) * n2
    return EJ/e

def list_levels(m_rel, L_nm, nmax=3, Emax_eV=2.0):
    """Generate discrete levels (nx,ny,nz) up to nmax, return sorted list."""
    levels = []
    for nx in range(1, nmax+1):
        for ny in range(1, nmax+1):
            for nz in range(1, nmax+1):
                E = particle_in_box_3d_energy_eV(nx, ny, nz, m_rel, L_nm)
                if E <= Emax_eV:
                    levels.append((E, nx, ny, nz))
    levels.sort(key=lambda x: x[0])
    return levels

def gaussian(x, mu, sigma):
    x = np.array(x, dtype=float)
    return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

def broadened_dos_0d(E_grid, levels_eV, sigma_eV):
    """
    0D DOS as sum of delta peaks broadened by Gaussians:
        D(E) = sum_i g_i * G(E - Ei)
    Here g_i=1 for simplicity.
    """
    D = np.zeros_like(E_grid, dtype=float)
    for Ei in levels_eV:
        D += gaussian(E_grid, Ei, sigma_eV)
    return D

def exciton_transition_energy_eV(Eg_eV, Ee_conf_eV, Eh_conf_eV, Eb_eV):
    """
    Simple QD transition energy:
        E_photon ≈ Eg + Ee_conf + Eh_conf - Eb
    """
    return float(Eg_eV) + float(Ee_conf_eV) + float(Eh_conf_eV) - float(Eb_eV)

def wavelength_nm_from_EeV(E_eV):
    """lambda(nm)=1240/E(eV)"""
    E = float(E_eV)
    return 1240.0/E if E > 0 else np.nan

def purcell_factor(Q, n, V_over_lambda3):
    """
    Purcell factor:
        Fp = (3/(4π^2)) * (Q/V) * (λ/n)^3
    If V is expressed in units of (λ/n)^3, then:
        Fp = (3/(4π^2)) * Q / (V/(λ/n)^3) = const * Q / V_over_lambda3
    """
    return (3.0/(4.0*np.pi**2)) * (float(Q)/float(V_over_lambda3))

# ============================================================
# UI
# ============================================================
st.title("Nanophysics – Week 12")
st.subheader("Quantum Dots (0D): Discrete Levels, 0D DOS, Emission Engineering, Purcell Effect")
st.markdown("---")

# ============================================================
# 1) Learning objectives & relation to Week 11
# ============================================================
st.header("1. Learning Objectives and Relation to Week 11")
st.markdown(
"""
Week 11: excitons, optical gain, and laser threshold (carrier populations + light–matter interaction).

Week 12: we move from 2D quantum wells to **0D quantum dots**:
- fully discrete electronic states,
- 0D density of states (delta-like),
- size-tunable emission,
- strong light–matter coupling in cavities (Purcell enhancement),
- pathway to single-photon sources.

By the end of Week 12, you should be able to:
- Write the particle-in-a-box energy levels for a 0D nanostructure.
- Explain why 0D DOS is discrete and how it differs from 1D/2D/3D.
- Estimate how emission wavelength changes with dot size and effective mass.
- Use the Purcell factor to understand spontaneous emission control.
"""
)

st.markdown("---")

# ============================================================
# 2) Lecture notes (ALL math in st.latex)
# ============================================================
st.header("2. Lecture Notes (All Mathematics Rendered with LaTeX)")

with st.expander("2.1 Quantum dots as 0D confinement (particle-in-a-box model)", expanded=True):
    st.markdown("In a quantum dot, carriers are confined in all three directions, producing fully discrete energy states.")
    st.markdown("A standard first model is the 3D infinite potential box of side length L:")
    st.latex(r"\psi(x,y,z)=\left(\frac{2}{L}\right)^{3/2}\sin\left(\frac{n_x\pi x}{L}\right)\sin\left(\frac{n_y\pi y}{L}\right)\sin\left(\frac{n_z\pi z}{L}\right)")
    st.markdown("The energy eigenvalues are:")
    st.latex(r"E_{n_x n_y n_z}=\frac{h^2}{8m^*L^2}\left(n_x^2+n_y^2+n_z^2\right)")
    st.markdown("Key scaling (very important):")
    st.latex(r"E \propto \frac{1}{m^*L^2}")
    st.latex(r"L\downarrow \Rightarrow E\uparrow \quad\text{(strong confinement, blue shift)}")

with st.expander("2.2 0D density of states (DOS) and spectral signatures", expanded=True):
    st.markdown("For an ideal 0D system, the DOS is a sum of delta functions at discrete energies:")
    st.latex(r"D_{0D}(E)=\sum_i g_i\,\delta(E-E_i)")
    st.markdown("In practice, finite lifetime and disorder broaden each delta peak, often approximated by Gaussians or Lorentzians.")
    st.markdown("A Gaussian-broadened pedagogical form:")
    st.latex(r"D(E)\approx \sum_i g_i\,\frac{1}{\sigma\sqrt{2\pi}}\exp\left[-\frac{(E-E_i)^2}{2\sigma^2}\right]")
    st.markdown("Result: quantum dots show sharp lines/peaks rather than a continuous absorption edge.")

with st.expander("2.3 Quantum-dot emission energy (simple excitonic picture)", expanded=True):
    st.markdown("A minimal emission-energy estimate combines bandgap, confinement, and exciton binding:")
    st.latex(r"E_{\gamma}\approx E_g + E_{e,\mathrm{conf}} + E_{h,\mathrm{conf}} - E_B")
    st.markdown("Convert photon energy to wavelength:")
    st.latex(r"\lambda(\mathrm{nm})\approx \frac{1240}{E_{\gamma}(\mathrm{eV})}")
    st.markdown("In small dots, confinement dominates and emission shifts to shorter wavelengths (blue shift).")

with st.expander("2.4 Purcell effect (controlling spontaneous emission with a cavity)", expanded=True):
    st.markdown("In a resonant optical cavity, spontaneous emission can be enhanced by the Purcell factor:")
    st.latex(r"F_P=\frac{3}{4\pi^2}\left(\frac{\lambda}{n}\right)^3\frac{Q}{V}")
    st.markdown("If the mode volume is expressed in units of (λ/n)^3, define:")
    st.latex(r"\frac{V}{(\lambda/n)^3}\equiv V_{\mathrm{norm}}")
    st.markdown("Then the Purcell factor becomes:")
    st.latex(r"F_P=\frac{3}{4\pi^2}\frac{Q}{V_{\mathrm{norm}}}")
    st.markdown("High Q and small mode volume strongly enhance emission rates and direct emission into a desired mode.")

with st.expander("2.5 Single-photon emission (conceptual)", expanded=False):
    st.markdown("A single quantum dot can emit one photon at a time because it can host discrete excitonic states.")
    st.markdown("A common way to verify single-photon emission is via the second-order correlation:")
    st.latex(r"g^{(2)}(0)<0.5 \quad\Rightarrow\quad \text{single-photon emission}")
    st.markdown("We will not simulate g(2) here, but we will build the physical foundation for it.")

st.markdown("---")

# ============================================================
# 3) Simulations & graphs
# ============================================================
st.header("3. Simulations and Graphs")

st.sidebar.title("Week 12 Controls")

# Material defaults
materials = {
    "Generic": dict(Eg=1.50, me=0.10, mh=0.50, Eb=0.030, n=3.5),
    "GaAs (QD toy)": dict(Eg=1.42, me=0.067, mh=0.45, Eb=0.020, n=3.6),
    "InP (QD toy)": dict(Eg=1.34, me=0.08,  mh=0.60, Eb=0.020, n=3.2),
    "GaN (QD toy)": dict(Eg=3.40, me=0.20, mh=1.00, Eb=0.050, n=2.4),
}
mat = st.sidebar.selectbox("Reference material (toy defaults)", list(materials.keys()))
d = materials[mat]

Eg_eV = st.sidebar.slider("Bandgap Eg (eV)", 0.5, 4.0, float(d["Eg"]), 0.01)
me_rel = st.sidebar.slider("Electron mass me*/m0", 0.02, 2.0, float(d["me"]), 0.01)
mh_rel = st.sidebar.slider("Hole mass mh*/m0", 0.05, 5.0, float(d["mh"]), 0.05)
Eb_eV = st.sidebar.slider("Exciton binding Eb (eV)", 0.0, 0.2, float(d["Eb"]), 0.001)

L_nm = st.sidebar.slider("Dot size L (nm) (box model)", 2.0, 30.0, 8.0, 0.5)
nmax = st.sidebar.slider("Quantum number max (nmax)", 2, 6, 3, 1)
Emax_plot = st.sidebar.slider("Level cutoff (eV) for DOS plot", 0.2, 5.0, 2.0, 0.1)
sigma_eV = st.sidebar.slider("Broadening σ (eV)", 0.001, 0.200, 0.030, 0.001)

st.sidebar.markdown("### Purcell (cavity) parameters")
Q = st.sidebar.slider("Quality factor Q", 100, 200000, 5000, 100)
n_index = st.sidebar.slider("Refractive index n", 1.0, 4.0, float(d["n"]), 0.1)
Vnorm = st.sidebar.slider("Normalized mode volume V / (λ/n)^3", 0.1, 50.0, 1.0, 0.1)

# ------------------------------------------------------------
# 3.1 Discrete energy levels (electron/hole)
# ------------------------------------------------------------
st.subheader("3.1 Discrete Energy Levels in a 0D Box (Quantum Dot Model)")

levels_e = list_levels(me_rel, L_nm, nmax=nmax, Emax_eV=Emax_plot)
levels_h = list_levels(mh_rel, L_nm, nmax=nmax, Emax_eV=Emax_plot)

st.markdown("Energy levels used (infinite 3D box):")
st.latex(r"E_{n_x n_y n_z}=\frac{h^2}{8m^*L^2}\left(n_x^2+n_y^2+n_z^2\right)")

colL, colR = st.columns(2)
with colL:
    st.markdown("**Electron confinement levels**")
    if len(levels_e) == 0:
        st.warning("No electron levels found under the selected cutoff. Increase Emax or nmax.")
    else:
        Ee_vals = [x[0] for x in levels_e[:20]]
        fig, ax = plt.subplots()
        ax.hlines(Ee_vals, 0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("dummy axis")
        ax.set_ylabel("Energy (eV)")
        ax.set_title("Electron confinement energies (lowest levels)")
        ax.grid(True, axis="y")
        st.pyplot(fig)
        st.markdown("Lowest few levels (E, nx, ny, nz):")
        for (E, nx, ny, nz) in levels_e[:10]:
            st.write(f"E = {E:.4f} eV   (n=({nx},{ny},{nz}))")

with colR:
    st.markdown("**Hole confinement levels**")
    if len(levels_h) == 0:
        st.warning("No hole levels found under the selected cutoff. Increase Emax or nmax.")
    else:
        Eh_vals = [x[0] for x in levels_h[:20]]
        fig, ax = plt.subplots()
        ax.hlines(Eh_vals, 0, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("dummy axis")
        ax.set_ylabel("Energy (eV)")
        ax.set_title("Hole confinement energies (lowest levels)")
        ax.grid(True, axis="y")
        st.pyplot(fig)
        st.markdown("Lowest few levels (E, nx, ny, nz):")
        for (E, nx, ny, nz) in levels_h[:10]:
            st.write(f"E = {E:.4f} eV   (n=({nx},{ny},{nz}))")

# ------------------------------------------------------------
# 3.2 0D DOS plot (broadened)
# ------------------------------------------------------------
st.subheader("3.2 0D Density of States (Broadened Delta Peaks)")

st.markdown("Ideal 0D DOS:")
st.latex(r"D_{0D}(E)=\sum_i g_i\,\delta(E-E_i)")
st.markdown("Gaussian broadening used for visualization:")
st.latex(r"D(E)\approx \sum_i g_i\,\frac{1}{\sigma\sqrt{2\pi}}\exp\left[-\frac{(E-E_i)^2}{2\sigma^2}\right]")

Egrid = np.linspace(0, Emax_plot, 1200)
Ee_list = [x[0] for x in levels_e]
Eh_list = [x[0] for x in levels_h]
De = broadened_dos_0d(Egrid, Ee_list, sigma_eV) if len(Ee_list) else np.zeros_like(Egrid)
Dh = broadened_dos_0d(Egrid, Eh_list, sigma_eV) if len(Eh_list) else np.zeros_like(Egrid)

fig, ax = plt.subplots()
ax.plot(Egrid, De, label="Electron 0D DOS (broadened)")
ax.plot(Egrid, Dh, label="Hole 0D DOS (broadened)")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Arbitrary DOS units")
ax.set_title("0D DOS: discrete peaks broadened")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ------------------------------------------------------------
# 3.3 Emission energy and wavelength vs dot size
# ------------------------------------------------------------
st.subheader("3.3 Emission Tuning: Photon Energy and Wavelength vs Dot Size")

st.markdown("Simple excitonic transition energy model:")
st.latex(r"E_{\gamma}\approx E_g + E_{e,\mathrm{conf}} + E_{h,\mathrm{conf}} - E_B")
st.latex(r"\lambda(\mathrm{nm})\approx \frac{1240}{E_{\gamma}(\mathrm{eV})}")

# Use ground-state (1,1,1) for electron and hole
Ee111 = particle_in_box_3d_energy_eV(1, 1, 1, me_rel, L_nm)
Eh111 = particle_in_box_3d_energy_eV(1, 1, 1, mh_rel, L_nm)
Eph = exciton_transition_energy_eV(Eg_eV, Ee111, Eh111, Eb_eV)
lam_nm = wavelength_nm_from_EeV(Eph)

cA, cB, cC, cD = st.columns(4)
with cA:
    st.markdown("Electron ground confinement:")
    st.latex(r"E_{e,111}=\frac{h^2}{8m_e^*L^2}(1^2+1^2+1^2)")
    st.latex(rf"E_{{e,111}}\approx {Ee111:.4f}\,\mathrm{{eV}}")
with cB:
    st.markdown("Hole ground confinement:")
    st.latex(r"E_{h,111}=\frac{h^2}{8m_h^*L^2}(1^2+1^2+1^2)")
    st.latex(rf"E_{{h,111}}\approx {Eh111:.4f}\,\mathrm{{eV}}")
with cC:
    st.markdown("Photon energy:")
    st.latex(r"E_{\gamma}\approx E_g + E_{e,111} + E_{h,111} - E_B")
    st.latex(rf"E_\gamma\approx {Eph:.4f}\,\mathrm{{eV}}")
with cD:
    st.markdown("Photon wavelength:")
    st.latex(r"\lambda(\mathrm{nm})\approx \frac{1240}{E_{\gamma}(\mathrm{eV})}")
    st.latex(rf"\lambda\approx {lam_nm:.1f}\,\mathrm{{nm}}")

# Sweep size
L_sweep = np.linspace(2.0, 30.0, 200)
Ee_s = np.array([particle_in_box_3d_energy_eV(1,1,1, me_rel, L) for L in L_sweep])
Eh_s = np.array([particle_in_box_3d_energy_eV(1,1,1, mh_rel, L) for L in L_sweep])
Eph_s = Eg_eV + Ee_s + Eh_s - Eb_eV
lam_s = 1240.0/np.clip(Eph_s, 1e-6, None)

fig, ax = plt.subplots()
ax.plot(L_sweep, Eph_s, label="Photon energy $E_\\gamma$")
ax.set_xlabel("Dot size L (nm)")
ax.set_ylabel("Photon energy (eV)")
ax.set_title("Emission energy vs dot size (ground-state model)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(L_sweep, lam_s, label="Wavelength $\\lambda$")
ax.set_xlabel("Dot size L (nm)")
ax.set_ylabel("Wavelength (nm)")
ax.set_title("Emission wavelength vs dot size (ground-state model)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ------------------------------------------------------------
# 3.4 Purcell effect simulation
# ------------------------------------------------------------
st.subheader("3.4 Purcell Effect: Emission Enhancement in a Cavity")

st.markdown("Purcell factor:")
st.latex(r"F_P=\frac{3}{4\pi^2}\left(\frac{\lambda}{n}\right)^3\frac{Q}{V}")
st.markdown("Using normalized mode volume:")
st.latex(r"V_{\mathrm{norm}}=\frac{V}{(\lambda/n)^3}")
st.latex(r"F_P=\frac{3}{4\pi^2}\frac{Q}{V_{\mathrm{norm}}}")

Fp = purcell_factor(Q, n_index, Vnorm)
st.latex(rf"F_P \approx {Fp:.2f}")

# Sweep Q for plot
Q_sweep = np.logspace(2, 5, 200)  # 1e2..1e5
Fp_sweep = (3/(4*np.pi**2))*(Q_sweep/Vnorm)

fig, ax = plt.subplots()
ax.semilogx(Q_sweep, Fp_sweep, label="$F_P(Q)$")
ax.set_xlabel("Quality factor Q (log scale)")
ax.set_ylabel("Purcell factor $F_P$")
ax.set_title("Purcell factor vs Q (fixed normalized mode volume)")
ax.grid(True, which="both")
ax.legend()
st.pyplot(fig)

st.markdown("---")

# ============================================================
# 4) Worked Examples (3) — all equations via st.latex
# ============================================================
st.header("4. Worked Examples (All LaTeX)")

with st.expander("Example 1 — Energy scaling with size (blue shift)", expanded=True):
    st.markdown("**Problem:** In a 3D box model, how does the ground-state confinement energy scale with dot size L?")
    st.markdown("**Solution:** For the ground state (1,1,1):")
    st.latex(r"E_{111}=\frac{h^2}{8m^*L^2}(1^2+1^2+1^2)=\frac{3h^2}{8m^*L^2}")
    st.markdown("Thus:")
    st.latex(r"E_{111}\propto \frac{1}{L^2}")
    st.markdown("If the dot size halves \(L\to L/2\), then:")
    st.latex(r"E_{111}(L/2)=\frac{1}{(L/2)^2}E_{111}(L)=4E_{111}(L)")
    st.markdown("So emission energy increases (blue shift) as the dot shrinks.")

with st.expander("Example 2 — Estimate emission wavelength for a given dot size", expanded=False):
    st.markdown("**Problem:** Given \(E_g\), \(m_e^*\), \(m_h^*\), \(L\), and \(E_B\), estimate emission wavelength.")
    st.markdown("**Solution:** Compute confinement energies (ground state) and use:")
    st.latex(r"E_{\gamma}\approx E_g + E_{e,111} + E_{h,111} - E_B")
    st.latex(r"\lambda(\mathrm{nm})\approx \frac{1240}{E_{\gamma}(\mathrm{eV})}")
    st.markdown("Use the app sliders to evaluate numerically; this is exactly what the simulation above does.")

with st.expander("Example 3 — Purcell enhancement with Q and mode volume", expanded=False):
    st.markdown("**Problem:** A cavity has Q and normalized mode volume \(V_{\mathrm{norm}}\). Find \(F_P\).")
    st.markdown("**Solution:**")
    st.latex(r"F_P=\frac{3}{4\pi^2}\frac{Q}{V_{\mathrm{norm}}}")
    st.markdown("So:")
    st.latex(r"Q\uparrow \Rightarrow F_P\uparrow,\qquad V_{\mathrm{norm}}\downarrow \Rightarrow F_P\uparrow")
    st.markdown("High-Q, small-volume cavities maximize emission enhancement.")

st.markdown("---")

# ============================================================
# 5) Quiz (3) — clean LaTeX
# ============================================================
st.header("5. Short Quiz (All LaTeX)")

st.subheader("Quiz 1 — 0D energy scaling")
st.markdown("In a 0D box model, the confinement energy scales with size as:")
st.latex(r"E\propto \frac{1}{L^2}")
q1 = st.radio("Choose:", ["E ∝ L", "E ∝ 1/L", "E ∝ 1/L^2", "E independent of L"], key="w12q1")
if q1:
    if q1 == "E ∝ 1/L^2":
        st.success("Correct.")
    else:
        st.error("Wrong. For particle-in-a-box confinement, E ∝ 1/L².")

st.subheader("Quiz 2 — 0D density of states")
st.markdown("The ideal 0D density of states is:")
st.latex(r"D_{0D}(E)=\sum_i g_i\,\delta(E-E_i)")
q2 = st.radio("Choose:", ["Continuous step", "Square-root", "Delta-like discrete peaks", "Linear ramp"], key="w12q2")
if q2:
    if q2 == "Delta-like discrete peaks":
        st.success("Correct.")
    else:
        st.error("Wrong. 0D DOS is a sum of discrete delta peaks at energy levels.")

st.subheader("Quiz 3 — Purcell factor dependence")
st.markdown("Purcell factor in normalized form:")
st.latex(r"F_P=\frac{3}{4\pi^2}\frac{Q}{V_{\mathrm{norm}}}")
q3 = st.radio("Choose:", ["Fp increases with Q and increases with Vnorm",
                         "Fp decreases with Q and increases with Vnorm",
                         "Fp increases with Q and decreases with Vnorm",
                         "Fp independent of Q and Vnorm"], key="w12q3")
if q3:
    if q3 == "Fp increases with Q and decreases with Vnorm":
        st.success("Correct.")
    else:
        st.error("Wrong. Fp ∝ Q / Vnorm, so higher Q and smaller mode volume increase Fp.")

st.markdown("---")

st.markdown("Week 12 Key Equations Recap:")
st.latex(r"E_{n_x n_y n_z}=\frac{h^2}{8m^*L^2}(n_x^2+n_y^2+n_z^2)")
st.latex(r"D_{0D}(E)=\sum_i g_i\,\delta(E-E_i)")
st.latex(r"E_{\gamma}\approx E_g+E_{e,\mathrm{conf}}+E_{h,\mathrm{conf}}-E_B")
st.latex(r"\lambda(\mathrm{nm})\approx \frac{1240}{E_{\gamma}(\mathrm{eV})}")
st.latex(r"F_P=\frac{3}{4\pi^2}\frac{Q}{V_{\mathrm{norm}}}")
st.latex(r"g^{(2)}(0)<0.5 \Rightarrow \text{single-photon emission (concept)}")
