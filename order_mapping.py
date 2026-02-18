import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="2D Order Mapping", layout="wide")

# =====================================================
# CONSTANTS & FUNCTIONS
# =====================================================
px = py = 13.68e-6
m_range = np.arange(-8, 9)
n_range = np.arange(-8, 9)

def get_mapping(alpha_deg, beta_deg, wl_um, tilt_deg):
    k = 2 * np.pi / (wl_um * 1e-6)
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    
    # 1. Incident Vector
    kin = np.array([
        -k * np.sin(alpha) * np.cos(beta),
        -k * np.sin(alpha) * np.sin(beta),
        -k * np.cos(alpha)
    ])
    
    # 2. Specular Reflection (Blaze Center)
    t_rad = np.deg2rad(tilt_deg)
    # Mirror normal vector based on tilt
    norm = np.array([np.sin(t_rad)/np.sqrt(2), np.sin(t_rad)/np.sqrt(2), np.cos(t_rad)])
    kout_spec = kin - 2 * np.dot(kin, norm) * norm
    spec_tx = np.degrees(np.arctan2(kout_spec[0], kout_spec[2]))
    spec_ty = np.degrees(np.arctan2(kout_spec[1], kout_spec[2]))
    
    # 3. All Diffraction Orders
    tx_list, ty_list = [], []
    for m in m_range:
        for n_val in n_range:
            kx = kin[0] + 2*np.pi*m/px
            ky = kin[1] + 2*np.pi*n_val/py
            v_sq = k**2 - kx**2 - ky**2
            if v_sq > 0:
                kz = np.sqrt(v_sq)
                tx_list.append(np.degrees(np.arctan2(kx, kz)))
                ty_list.append(np.degrees(np.arctan2(ky, kz)))
                
    return tx_list, ty_list, spec_tx, spec_ty

# =====================================================
# STREAMLIT UI - LAYOUT
# =====================================================
st.title("2D Order Mapping Simulator")

# Reserve space for the plot at the top
plot_spot = st.empty()

st.write("---")
st.subheader("Parameters")

# Columns for sliders at the bottom
c1, c2 = st.columns(2)

with c1:
    s_tilt = st.slider('Mirror Tilt (deg)', -12.0, 12.0, 0.0, 0.1)
    s_alpha = st.slider('Alpha (Incident θ)', 0.0, 80.0, 45.0, 1.0)

with c2:
    s_beta = st.slider('Beta (Azimuth φ)', 0.0, 360.0, 45.0, 1.0)
    s_wl = st.slider('Wavelength (μm)', 0.3, 1.0, 0.905, 0.005)

# =====================================================
# GENERATE PLOT
# =====================================================
tx_l, ty_l, stx, sty = get_mapping(s_alpha, s_beta, s_wl, s_tilt)

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(tx_l, ty_l, s=15, color='red', alpha=0.5, label='Diffraction Orders (m,n)')
ax.plot(stx, sty, 'bo', alpha=0.6, markersize=8, label='Blaze Center (Specular)')

ax.set_xlabel('$θ_x$ (deg)')
ax.set_ylabel('$θ_y$ (deg)')
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')

# Send plot to the reserved spot at the top
plot_spot.pyplot(fig)
