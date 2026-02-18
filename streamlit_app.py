import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set page config for a cleaner look in Notion
st.set_page_config(page_title="DMD Diffraction Simulation", layout="wide")

# =====================================================
# CONSTANTS & CORE FUNCTIONS
# =====================================================
px = py = 13.68e-6
ax_dim = px * 0.96
ay_dim = py * 0.96
M_count = N_count = 10
Ngrid = 400 # Reduced slightly for snappier web performance
θmax = np.deg2rad(30)

def sinc(u):
    return np.where(u == 0, 1, np.sin(u) / (u + 1e-20))

def lattice(M, q, p):
    return np.sin(M * q * p / 2) / (np.sin(q * p / 2) + 1e-20)

def get_geometry(alpha_deg, beta_deg, L_val, lam):
    k_val = 2 * np.pi / lam
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    
    kin = np.array([
        k_val * np.sin(alpha) * np.cos(beta),
        k_val * np.sin(alpha) * np.sin(beta),
        k_val * np.cos(alpha)
    ])
    
    n0 = np.array([0, 0, 1])
    kout0 = kin - 2 * np.dot(kin, n0) * n0
    z_h = kout0 / np.linalg.norm(kout0)
    
    tmp = np.array([1e-10, 1e-10, 1])
    yh = np.cross(tmp, z_h)
    yh /= np.linalg.norm(yh)
    xh = -np.cross(yh, z_h)
    
    return kin, z_h, xh, yh, L_val, k_val

# =====================================================
# STREAMLIT UI - SIDEBAR
# =====================================================
st.sidebar.header("Simulation Controls")

tilt_val = st.sidebar.slider('Mirror Tilt (deg)', -12.0, 12.0, 0.0, 0.5)
alpha_val = st.sidebar.slider('Alpha (Incident Angle)', 0, 80, 45)
beta_val = st.sidebar.slider('Beta (Azimuth)', 0, 360, 45)
L_val = st.sidebar.slider('Distance L (m)', 0.005, 0.1, 0.02, 0.005)
wl_um = st.sidebar.slider('Wavelength (μm)', 0.3, 1.0, 0.905, 0.005)

# =====================================================
# CALCULATION LOGIC
# =====================================================
# Pre-calc grid
ux = np.linspace(-θmax, θmax, Ngrid)
uy = np.linspace(-θmax, θmax, Ngrid)
UX, UY = np.meshgrid(ux, uy)

wl_m = wl_um * 1e-6
ki, zh, xh, yh, dist, k_now = get_geometry(alpha_val, beta_val, L_val, wl_m)

# Directions
kvecs = (zh[None,None,:] + UX[:,:,None]*xh + UY[:,:,None]*yh)
kvecs /= np.linalg.norm(kvecs, axis=2)[:,:,None]
kvecs *= k_now

qx = kvecs[:,:,0] - ki[0]
qy = kvecs[:,:,1] - ki[1]

# Mirror Tilt
theta_t = np.deg2rad(tilt_val)
n = np.array([np.sin(theta_t)/np.sqrt(2), np.sin(theta_t)/np.sqrt(2), np.cos(theta_t)])
q_blaze_x = -2 * (ki @ n) * n[0]
q_blaze_y = -2 * (ki @ n) * n[1]

# Intensity
P = sinc((qx - q_blaze_x) * ax_dim / 2) * sinc((qy - q_blaze_y) * ay_dim / 2)
Latt = lattice(M_count, qx, px) * lattice(N_count, qy, py)
I = np.abs(P * Latt)**2

# Screen Projection
den = (kvecs[:,:,0]*zh[0] + kvecs[:,:,1]*zh[1] + kvecs[:,:,2]*zh[2])
scale = dist * k_now / den
rx, ry, rz = scale * kvecs[:,:,0] / k_now, scale * kvecs[:,:,1] / k_now, scale * kvecs[:,:,2] / k_now
X_proj = (rx*xh[0] + ry*xh[1] + rz*xh[2]) * 1000 # to mm
Y_proj = (rx*yh[0] + ry*yh[1] + rz*yh[2]) * 1000 # to mm

# =====================================================
# PLOTTING
# =====================================================
st.title("Interactive DMD Diffraction Simulation")

fig, axp = plt.subplots(figsize=(8, 6))
im = axp.imshow(I, extent=[X_proj.min(), X_proj.max(), Y_proj.min(), Y_proj.max()], 
                origin='lower', cmap='jet', vmin=0, vmax=2000)
axp.set_xlabel("x (mm)")
axp.set_ylabel("y (mm)")
fig.colorbar(im, ax=axp, label="Intensity")

st.pyplot(fig)
