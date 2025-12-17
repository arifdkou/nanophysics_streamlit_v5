# app_nanophysics_week11_latexclean.py
# ============================================================
# Nanophysics – Week 11 (Clean LaTeX Rendering)
# Excitons, Optical Gain, and Laser Threshold in Nanostructures
# Lecture notes + worked examples + simulations + graphs + quiz
# ALL equations are rendered ONLY with st.latex() for maximum readability
# ============================================================

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Nanophysics – Week 11: Excitons & Optical Gain (Clean LaTeX)",
    layout="wide"
)

# ----------------------------
# Constants (SI)
# ----------------------------
e = 1.602_176_634e-19
eps0 = 8.854_187_8128e-12
h = 6.626_070_15e-34
hbar = h / (2*np.pi)
m0 = 9.109_383_7015e-31
c = 2.997_924_58e8
kB = 1.380_649e-23

a0 = 5.291_772_10903e-11  # Bohr radius (m)
Ry_eV = 13.605693122994   # Rydberg energy (eV)

# ----------------------------
# Helper functions
# ----------------------------
def reduced_mass_rel(me_rel, mh_rel):
    me = float(me_rel); mh = float(mh_rel)
    return (me*mh)/(me+mh)

def exciton_bohr_radius_3d(eps_r, mu_rel):
    return float(eps_r) * (1.0/float(mu_rel)) * a0

def exciton_binding_energy_3d_eV(eps_r, mu_rel):
    return float(mu_rel) * (1.0/(float(eps_r)**2)) * Ry_eV

def exciton_binding_energy_2d_eV(Eb_3d_eV):
    return 4.0 * float(Eb_3d_eV)   # pedagogical approximation

def fermi_dirac(E_eV, Ef_eV, T_K):
    kT_eV = (kB*float(T_K))/e
    E = np.array(E_eV, dtype=float)
    x = (E - float(Ef_eV))/kT_eV
    x = np.clip(x, -200, 200)
    return 1.0/(1.0 + np.exp(x))

def toy_spectrum_shape(E_eV, Eg_eV, Ex_eV, A_cont=1.0, B_ex=2.0, gamma_ex=0.01):
    E = np.array(E_eV, dtype=float)
    Eg = float(Eg_eV)
    cont = np.sqrt(np.clip(E - Eg, 0.0, None))
    lor = (1/np.pi) * (gamma_ex / ((E - Ex_eV)**2 + gamma_ex**2))
    return A_cont*cont + B_ex*lor

def inversion_factor_toy(Etrans_eV, Efc_eV, Efv_eV, T_K):
    # Place bands symmetrically: Ec=+E/2, Ev=-E/2 so Ec-Ev=E
    E = np.array(Etrans_eV, dtype=float)
    Ec = 0.5*E
    Ev = -0.5*E
    fc = fermi_dirac(Ec, Efc_eV, T_K)
    fv = fermi_dirac(Ev, Efv_eV, T_K)
    # net stimulated factor ~ fc + fv - 1
    return fc + fv - 1.0

def threshold_density(alpha_loss_cm, g0_cm2, Ntr_cm3):
    return float(Ntr_cm3) + float(alpha_loss_cm)/float(g0_cm2)

# ============================================================
# UI
# ============================================================
st.title("Nanophysics – Week 11")
st.subheader("Excitons, Optical Gain, and Laser Threshold in Nanostructures (Clean LaTeX)")
st.markdown("---")

# ============================================================
# 1) Objectives / Relation to Week 10
# ============================================================
st.header("1. Learning Objectives and Relation to Week 10")
st.markdown(
"""
Week 10: quantum wells + interband optical transitions (subbands, selection rules, absorption peaks).
Week 11: add electron–hole Coulomb attraction → excitons, then connect to optical gain and laser threshold.
"""
)

st.markdown("Key targets of this week:")
st.markdown("- Compute exciton reduced mass, Bohr radius, binding energy (hydrogenic model).")
st.markdown("- Compare 3D vs 2D excitons and exciton resonance below the band edge.")
st.markdown("- Understand quasi-Fermi levels and qualitative gain condition.")
st.markdown("- Use a toy gain model and a toy laser-threshold model to see parameter impacts.")

st.markdown("---")

# ============================================================
# 2) Lecture Notes (ALL equations via st.latex)
# ============================================================
st.header("2. Lecture Notes (All Math Rendered with LaTeX)")

# 2.1 Excitons
with st.expander("2.1 Exciton basics (hydrogenic model)", expanded=True):
    st.markdown("An exciton is a bound electron–hole pair created by optical excitation.")
    st.markdown("Reduced mass:")
    st.latex(r"\mu = \frac{m_e^* m_h^*}{m_e^* + m_h^*}")
    st.markdown("Effective 3D exciton Bohr radius:")
    st.latex(r"a_B^* = \varepsilon_r \frac{m_0}{\mu}\, a_0")
    st.markdown("Effective 3D exciton binding energy:")
    st.latex(r"E_B^{(3D)} = \left(\frac{\mu}{m_0}\right)\frac{1}{\varepsilon_r^2}\,\mathrm{Ry}")
    st.markdown("Interpretation:")
    st.latex(r"\varepsilon_r \uparrow \Rightarrow E_B \downarrow \quad (\text{more screening})")
    st.latex(r"\mu \uparrow \Rightarrow E_B \uparrow \quad (\text{heavier pair binds stronger})")

# 2.2 3D vs 2D
with st.expander("2.2 3D vs 2D excitons in quantum wells", expanded=True):
    st.markdown("Confinement in quantum wells enhances excitonic effects (pedagogical idealized result).")
    st.latex(r"E_B^{(2D)} \approx 4\,E_B^{(3D)}")
    st.markdown("Exciton resonance relative to bandgap:")
    st.latex(r"E_X \approx E_g - E_B")
    st.markdown("So the exciton peak appears below the band edge by the binding energy.")

# 2.3 Gain condition
with st.expander("2.3 Optical gain and quasi-Fermi levels", expanded=True):
    st.markdown("Under pumping, carriers can be described by quasi-Fermi levels.")
    st.markdown("Fermi–Dirac distribution:")
    st.latex(r"f(E)=\frac{1}{1+\exp\left(\frac{E-E_F}{k_B T}\right)}")
    st.markdown("Qualitative gain condition:")
    st.latex(r"E_{F,c}-E_{F,v}\gtrsim \hbar\omega")
    st.markdown("If this holds over a spectral region, stimulated emission can exceed absorption → gain.")

# 2.4 Laser threshold
with st.expander("2.4 Laser threshold (toy rate-equation view)", expanded=True):
    st.markdown("Minimal rate-equation form:")
    st.latex(r"\frac{dN}{dt}=\frac{I}{qV}-\frac{N}{\tau_n}-G(N)S")
    st.latex(r"\frac{dS}{dt}=\Gamma G(N)S+\beta\frac{N}{\tau_n}-\frac{S}{\tau_p}")
    st.markdown("Linear gain model:")
    st.latex(r"G(N)=g_0\left(N-N_{tr}\right)")
    st.markdown("Threshold condition (toy):")
    st.latex(r"G(N_{th})=\alpha_{\mathrm{loss}}")
    st.latex(r"N_{th}=N_{tr}+\frac{\alpha_{\mathrm{loss}}}{g_0}")

st.markdown("---")

# ============================================================
# 3) Simulations & Graphs
# ============================================================
st.header("3. Simulations and Graphs")

st.sidebar.title("Week 11 Controls")

materials = {
    "Generic": dict(eps_r=12.0, me=0.10, mh=0.50, Eg=1.50),
    "GaAs": dict(eps_r=12.9, me=0.067, mh=0.45, Eg=1.42),
    "InP": dict(eps_r=12.5, me=0.08,  mh=0.60, Eg=1.34),
    "GaN": dict(eps_r=9.5,  me=0.20,  mh=1.00, Eg=3.40),
    "MoS2 (toy)": dict(eps_r=6.0, me=0.45, mh=0.55, Eg=1.90),
}
mat = st.sidebar.selectbox("Reference material", list(materials.keys()))
d = materials[mat]

eps_r = st.sidebar.slider("εr", 2.0, 25.0, float(d["eps_r"]), 0.1)
me_rel = st.sidebar.slider("me*/m0", 0.02, 2.00, float(d["me"]), 0.01)
mh_rel = st.sidebar.slider("mh*/m0", 0.05, 5.00, float(d["mh"]), 0.05)
Eg_eV = st.sidebar.slider("Eg (eV)", 0.5, 4.0, float(d["Eg"]), 0.01)
T_K = st.sidebar.slider("T (K)", 4, 600, 300, 1)

st.sidebar.markdown("Quasi-Fermi levels (toy)")
Efc = st.sidebar.slider("EFc (eV)", -2.0, 2.0, 0.30, 0.01)
Efv = st.sidebar.slider("EFv (eV)", -2.0, 2.0, -0.30, 0.01)

st.sidebar.markdown("Laser threshold (toy)")
alpha_loss = st.sidebar.slider("α_loss (cm^-1)", 1.0, 100.0, 20.0, 1.0)
g0 = st.sidebar.slider("g0 (cm^2)", 1e-18, 2e-16, 5e-17, 1e-18)
Ntr = st.sidebar.slider("Ntr (cm^-3)", 1e17, 5e19, 1e18, 1e17)

# 3.1 Exciton numbers
st.subheader("3.1 Exciton Parameters (Computed)")

mu_rel = reduced_mass_rel(me_rel, mh_rel)
aB_nm = exciton_bohr_radius_3d(eps_r, mu_rel) * 1e9
Eb3 = exciton_binding_energy_3d_eV(eps_r, mu_rel)
Eb2 = exciton_binding_energy_2d_eV(Eb3)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("Reduced mass:")
    st.latex(r"\frac{\mu}{m_0}=\frac{(m_e^*/m_0)(m_h^*/m_0)}{(m_e^*/m_0)+(m_h^*/m_0)}")
    st.latex(rf"\frac{{\mu}}{{m_0}}\approx {mu_rel:.4f}")
with c2:
    st.markdown("Bohr radius:")
    st.latex(r"a_B^*=\varepsilon_r\frac{m_0}{\mu}a_0")
    st.latex(rf"a_B^*\approx {aB_nm:.2f}\,\mathrm{{nm}}")
with c3:
    st.markdown("Binding energy:")
    st.latex(r"E_B^{(3D)}=\left(\frac{\mu}{m_0}\right)\frac{1}{\varepsilon_r^2}\mathrm{Ry}")
    st.latex(rf"E_B^{{(3D)}}\approx {Eb3:.4f}\,\mathrm{{eV}}")
    st.latex(rf"E_B^{{(2D)}}\approx 4E_B^{{(3D)}}\approx {Eb2:.4f}\,\mathrm{{eV}}")

st.markdown("Thermal comparison at your temperature:")
kT_eV = (kB*T_K)/e
st.latex(r"k_B T = \frac{k_B T}{q}\ \mathrm{(eV)}")
st.latex(rf"k_B T \approx {kT_eV:.4f}\,\mathrm{{eV}}")
st.latex(rf"\frac{{E_B^{{(2D)}}}}{{k_B T}}\approx {Eb2/kT_eV:.2f}")

# 3.2 Resonance position
st.subheader("3.2 Exciton Resonance Position")

Ex_2D = Eg_eV - Eb2
st.latex(r"E_X\approx E_g - E_B^{(2D)}")
st.latex(rf"E_X\approx {Ex_2D:.4f}\,\mathrm{{eV}}")

# 3.3 Toy spectrum + inversion
st.subheader("3.3 Toy Spectrum: Continuum + Exciton, Controlled by Quasi-Fermi Levels")

gamma_ex = st.slider("Exciton linewidth γ (eV)", 0.001, 0.050, 0.010, 0.001)
A_cont = st.slider("Continuum weight A", 0.1, 5.0, 1.0, 0.1)
B_ex = st.slider("Exciton weight B", 0.0, 10.0, 2.0, 0.1)

E = np.linspace(max(0.0, Eg_eV-0.25), Eg_eV+0.60, 900)
S = toy_spectrum_shape(E, Eg_eV, Ex_2D, A_cont=A_cont, B_ex=B_ex, gamma_ex=gamma_ex)
F = inversion_factor_toy(E, Efc, Efv, T_K)
net = S * F

st.markdown("Toy model equations used for plots:")
st.latex(r"S(E)=A\sqrt{\max(E-E_g,0)} + B\cdot\frac{1}{\pi}\frac{\gamma}{(E-E_X)^2+\gamma^2}")
st.latex(r"F(E)\approx f_c(E_c)+f_v(E_v)-1")
st.latex(r"f(E)=\frac{1}{1+\exp\left(\frac{E-E_F}{k_B T}\right)}")

fig1, ax1 = plt.subplots()
ax1.plot(E, S, label="S(E)")
ax1.axvline(Eg_eV, linestyle="--", alpha=0.7, label="Eg")
ax1.axvline(Ex_2D, linestyle="--", alpha=0.7, label="Ex")
ax1.set_xlabel("Photon energy E (eV)")
ax1.set_ylabel("Arbitrary units")
ax1.set_title("Toy spectral shape: continuum + exciton resonance")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(E, net, label="S(E)·F(E)")
ax2.axhline(0.0, linestyle="--", alpha=0.7)
ax2.axvline(Eg_eV, linestyle="--", alpha=0.7, label="Eg")
ax2.set_xlabel("Photon energy E (eV)")
ax2.set_ylabel("Net response (toy units)")
ax2.set_title("Toy net response: sign changes with quasi-Fermi separation")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

st.markdown("Qualitative gain condition:")
st.latex(r"E_{F,c}-E_{F,v}\gtrsim \hbar\omega")

# 3.4 Threshold
st.subheader("3.4 Toy Laser Threshold")

Nth = threshold_density(alpha_loss, g0, Ntr)
st.latex(r"N_{th}=N_{tr}+\frac{\alpha_{\mathrm{loss}}}{g_0}")
st.latex(rf"N_{{th}}\approx {Nth:.3e}\,\mathrm{{cm}}^{{-3}}")

N_vals = np.linspace(Ntr*0.8, max(Nth*1.2, Ntr*1.6), 500)
G_vals = g0*(N_vals - Ntr)

fig3, ax3 = plt.subplots()
ax3.plot(N_vals, G_vals, label="G(N)")
ax3.axhline(alpha_loss, linestyle="--", alpha=0.7, label="α_loss")
ax3.axvline(Nth, linestyle="--", alpha=0.7, label="Nth")
ax3.set_xlabel("Carrier density N (cm^-3)")
ax3.set_ylabel("Gain/Loss (cm^-1)")
ax3.set_title("Threshold where gain meets losses (toy)")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)

st.markdown("---")

# ============================================================
# 4) Worked Examples (all equations via st.latex)
# ============================================================
st.header("4. Worked Examples (All LaTeX)")

with st.expander("Example 1 — Exciton radius and binding (step-by-step)", expanded=True):
    st.markdown("Given: εr = 12.9, me* = 0.067 m0, mh* = 0.45 m0.")
    st.markdown("Step 1: reduced mass")
    st.latex(r"\frac{\mu}{m_0}=\frac{(0.067)(0.45)}{0.067+0.45}\approx 0.0583")
    st.markdown("Step 2: Bohr radius")
    st.latex(r"a_B^*=\varepsilon_r\frac{m_0}{\mu}a_0=\varepsilon_r\frac{1}{(\mu/m_0)}a_0")
    st.latex(r"a_B^*\approx 12.9\cdot\frac{1}{0.0583}\cdot 0.0529\,\mathrm{nm}\approx 11.7\,\mathrm{nm}")
    st.markdown("Step 3: Binding energy")
    st.latex(r"E_B^{(3D)}=\left(\frac{\mu}{m_0}\right)\frac{1}{\varepsilon_r^2}\mathrm{Ry}")
    st.latex(r"E_B^{(3D)}\approx 0.0583\cdot\frac{1}{(12.9)^2}\cdot 13.6\,\mathrm{eV}\approx 4.8\,\mathrm{meV}")

with st.expander("Example 2 — Exciton resonance wavelength", expanded=False):
    st.markdown("Given: Eg = 1.50 eV, Eb(2D) = 0.040 eV.")
    st.latex(r"E_X\approx E_g-E_B^{(2D)}=1.50-0.040=1.46\,\mathrm{eV}")
    st.latex(r"\lambda\approx \frac{1240}{E_X(\mathrm{eV})}\approx \frac{1240}{1.46}\approx 849\,\mathrm{nm}")

with st.expander("Example 3 — Threshold density from linear gain", expanded=False):
    st.markdown("Given: g0 = 5×10^-17 cm^2, Ntr = 1×10^18 cm^-3, α_loss = 20 cm^-1.")
    st.latex(r"g_0(N_{th}-N_{tr})=\alpha_{\mathrm{loss}}")
    st.latex(r"N_{th}=N_{tr}+\frac{\alpha_{\mathrm{loss}}}{g_0}")
    st.latex(r"N_{th}=10^{18}+\frac{20}{5\times 10^{-17}}=1.4\times 10^{18}\,\mathrm{cm^{-3}}")

st.markdown("---")

# ============================================================
# 5) Quiz (math via st.latex where needed)
# ============================================================
st.header("5. Short Quiz (All LaTeX)")

st.subheader("Quiz 1")
st.markdown("Which change increases the 3D exciton binding energy?")
st.latex(r"E_B^{(3D)}=\left(\frac{\mu}{m_0}\right)\frac{1}{\varepsilon_r^2}\mathrm{Ry}")
q1 = st.radio("Choose:", ["Increase εr", "Decrease εr", "Decrease μ", "Increase εr and decrease μ"], key="q1")
if q1:
    if q1 == "Decrease εr":
        st.success("Correct.")
    else:
        st.error("Wrong. Smaller εr means less screening → stronger binding.")

st.subheader("Quiz 2")
st.markdown("Where is the exciton resonance relative to the bandgap?")
st.latex(r"E_X\approx E_g-E_B")
q2 = st.radio("Choose:", ["Above Eg", "At Eg", "Below Eg by Eb", "Unrelated"], key="q2")
if q2:
    if q2 == "Below Eg by Eb":
        st.success("Correct.")
    else:
        st.error("Wrong. Exciton peak is below Eg by Eb.")

st.subheader("Quiz 3")
st.markdown("Laser threshold for linear gain model:")
st.latex(r"G(N)=g_0(N-N_{tr}),\qquad G(N_{th})=\alpha_{\mathrm{loss}}")
q3 = st.radio("Choose:", ["Nth=Ntr-α/g0", "Nth=Ntr+α/g0", "Nth=α·g0", "Nth=Ntr/g0"], key="q3")
if q3:
    if q3 == "Nth=Ntr+α/g0":
        st.success("Correct.")
    else:
        st.error("Wrong. From g0(Nth-Ntr)=α → Nth=Ntr+α/g0.")

st.markdown("---")
st.markdown("Week 11 key equations recap:")
st.latex(r"\mu=\frac{m_e^*m_h^*}{m_e^*+m_h^*}")
st.latex(r"a_B^*=\varepsilon_r\frac{m_0}{\mu}a_0")
st.latex(r"E_B^{(3D)}=\left(\frac{\mu}{m_0}\right)\frac{1}{\varepsilon_r^2}\mathrm{Ry}")
st.latex(r"E_B^{(2D)}\approx 4E_B^{(3D)}")
st.latex(r"E_X\approx E_g-E_B")
st.latex(r"E_{F,c}-E_{F,v}\gtrsim \hbar\omega")
st.latex(r"N_{th}=N_{tr}+\frac{\alpha_{\mathrm{loss}}}{g_0}")
