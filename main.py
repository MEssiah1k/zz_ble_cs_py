import numpy as np
import matplotlib.pyplot as plt

c = 299_792_458.0

def wrap_to_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def steering_tw(f_list, d):
    """Two-way steering for distance spectrum: exp(-j*4*pi*f*d/c)."""
    return np.exp(+1j * (4*np.pi * f_list * d / c))

def simulate_phi_sum_onetone(f, dA, d_others, a_others,
                            sigma_phi_rad, sigma_n, rng):
    """
    V2.0 one-tone simulation:
    - phase diff 1: Tx->A measured at A: wrap(2*pi*f*dA/c + (phiA-phiT) + n1)
    - phase diff 2: Tx receives sum of returns from A and others:
        y = sum_k alpha_k * exp(j*(2*pi*f*dk/c + (phiT - phik))) + w
      phase2 = wrap(angle(y))
    - phi_sum = wrap(phase1 + phase2)

    d_others: list of distances for interferers (B,C,D,...), not including A
    a_others: list of linear amplitudes for interferers, same length as d_others
    """
    # Random PLL initial phases for THIS tone
    phiT = rng.uniform(0, 2*np.pi)
    phiA = rng.uniform(0, 2*np.pi)

    # Phase diff 1 (Tx->A at A)
    n1 = rng.normal(0, sigma_phi_rad)
    phase1 = wrap_to_pi(2*np.pi*f*dA/c + (phiA - phiT) + n1)

    # Return sum at Tx: include A + interferers
    # A return amplitude set to 1 (can change if you want)
    aA = 1.0
    # A's return term phase (relative to Tx LO)
    termA = aA * np.exp(1j * (2*np.pi*f*dA/c + (phiT - phiA)))

    y = termA

    # Interferers
    for dk, ak in zip(d_others, a_others):
        phik = rng.uniform(0, 2*np.pi)  # each terminal has its own PLL phase
        y += ak * np.exp(1j * (2*np.pi*f*dk/c + (phiT - phik)))

    # Add complex noise at Tx
    w = (rng.normal(0, sigma_n) + 1j*rng.normal(0, sigma_n))
    y += w

    phase2 = wrap_to_pi(np.angle(y))
    phi_sum = wrap_to_pi(phase1 + phase2)
    return phi_sum

def bartlett_spectrum_from_phi_sum(phi_sum_vec, f_list, d_grid):
    """
    Build a 'range spectrum' from phi_sum across tones:
      z_i = exp(j*phi_sum_i)
      P(d) = | sum_i z_i * exp(-j*4*pi*f_i*d/c) |^2
    """
    z = np.exp(1j * phi_sum_vec)  # phase-only complex vector

    P = np.zeros_like(d_grid, dtype=np.float64)
    for i, d in enumerate(d_grid):
        s = steering_tw(f_list, d)
        P[i] = np.abs(np.vdot(s, z))**2  # <s, z> magnitude squared
    return P

def estimate_distance_from_spectrum(P, d_grid):
    return d_grid[np.argmax(P)]

def mod_range(x, period):
    return np.mod(x, period)

# ----------------------------
# V2.0 Settings
# ----------------------------
rng = np.random.default_rng(1)

# Frequency plan
f_start = 2.402e9
df = 2e6
N = 5
f_list = f_start + df*np.arange(N)

# Synthetic ambiguity period (~75m for df=2MHz)
d_syn = c/(2*df)

# Distance search grid for spectrum
d_step = 0.05
d_grid = np.arange(0.0, d_syn, d_step)

# Noise / phase error
sigma_deg = 5.0
sigma_phi = np.deg2rad(sigma_deg)
sigma_n = 0.05  # complex noise at Tx during return superposition

# Multi-receiver setup (interferers B,C,D,...)
M_interferers = 2  # number of interferers besides A

# Fix interferer distances (recommended: fixed geometry)
# You can also randomize per trial if you want.
d_others_fixed = np.array([8.0, 33.0, 52.0, 70.0])[:M_interferers]

# Interferer amplitudes (linear). Example: random between -3dB to -15dB.
atten_db = rng.uniform(3.0, 15.0, size=M_interferers)
a_others_fixed = 10**(-atten_db/20)

# ----------------------------
# Figure 1: sweep dA, estimate by spectrum peak (mod d_syn)
# ----------------------------
d_true = np.linspace(0, float(d_syn), 1501)
d_hat = np.zeros_like(d_true)

for k, dA in enumerate(d_true):
    # For each distance point, generate phi_sum across N tones
    phi_sum_vec = np.array([
        simulate_phi_sum_onetone(f, dA, d_others_fixed, a_others_fixed,
                                 sigma_phi, sigma_n, rng)
        for f in f_list
    ])

    P = bartlett_spectrum_from_phi_sum(phi_sum_vec, f_list, d_grid)
    d_hat[k] = estimate_distance_from_spectrum(P, d_grid)

d_true_mod = mod_range(d_true, d_syn)
d_hat_mod  = mod_range(d_hat,  d_syn)

plt.figure()
plt.scatter(d_true_mod, d_hat_mod, s=6, color="tab:blue", alpha=0.6,
            label=f"estimated (V2.0, 1→many, N={N})")
plt.plot(d_true_mod, d_true_mod, "--", color="tab:orange", linewidth=2, label="true (mod)")
plt.xlabel(f"True distance mod {d_syn:.1f} m (m)")
plt.ylabel(f"Estimated distance mod {d_syn:.1f} m (m)")
plt.title(f"V2.0 | 1→many superposed returns (PLL random per tone), Δf={df/1e6:.1f} MHz, σφ={sigma_deg:.1f}°")
plt.grid(True)
plt.legend()

# ----------------------------
# Figure 2: fixed dA=20m, show range spectrum P(d)
# ----------------------------
d0 = 20.0
phi_sum_vec = np.array([
    simulate_phi_sum_onetone(f, d0, d_others_fixed, a_others_fixed,
                             sigma_phi, sigma_n, rng)
    for f in f_list
])

P = bartlett_spectrum_from_phi_sum(phi_sum_vec, f_list, d_grid)
P_norm = P / (np.max(P) + 1e-12)

plt.figure()
plt.plot(d_grid, P_norm, color="tab:blue", linewidth=1.5, label="range spectrum (normalized)")
plt.axvline(d0, color="tab:orange", linestyle="--", linewidth=2, label="true d(A)")
# Optional: show interferer distances as reference
for i, dk in enumerate(d_others_fixed):
    plt.axvline(dk, color="tab:green", linestyle=":", linewidth=1.2, label="interferers" if i == 0 else None)

plt.xlabel("Distance axis (m)")
plt.ylabel("Estimated energy (normalized)")
plt.title(f"V2.0 | Spectrum at d(A)={d0:.1f} m with {M_interferers} interferers (superposed returns)")
plt.grid(True)
plt.legend()

plt.show()
