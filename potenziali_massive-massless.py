# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:28:33 2024

@author: Pietro
"""

import math 
import numpy as np
import matplotlib.pyplot as plt

# Constants
M = 1  # Black hole mass (arbitrary units)
L = 3*math.sqrt(2)  # Angular momentum per unit mass

# Define effective potential
def V_eff(r, M, L):
    return (1 - 2 * M / r) * (L**2 / r**2)

# r range (avoiding Schwarzschild radius 2M)
r = np.linspace(2.1, 15, 500)

# Compute effective potential
V = V_eff(r, M, L)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(r, V, label=f"$L/M = {L/M}$", color="blue")
plt.axvline(x=2 * M, color='red', linestyle='--', label="$r_H$ (Orizzonte degli Eventi)")
plt.axvline(x=3, color='green', linestyle='--', label="Photon Sphere ($r=3M$)")
plt.axhline(0, color='black', linestyle='-', linewidth=0.8)

# Labels and title
plt.title("Potenziale Efficace $V_{eff}(r)$ per un Buco Nero di Schwarzschild", fontsize=14)
plt.xlabel("$r/M$", fontsize=12)
plt.ylabel("$V_{eff}$", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()


#%%


import numpy as np
import matplotlib.pyplot as plt

# Constants
M = 1  # Black hole mass (arbitrary units)
L = 4  # Angular momentum per unit mass

# Define effective potential for massive particles
def V_eff_massive(r, M, mu, L):
    return (1 - 2 * M / r) * (mu**2 + L**2 / r**2)

# r range (avoiding Schwarzschild radius 2M)
r = np.linspace(2.1, 15, 500)

# Different values of mu
mu_values = [0.01, 0.3, 0.99]

# Plot
plt.figure(figsize=(10, 6))

for mu in mu_values:
    V_massive = V_eff_massive(r, M, mu, L)
    plt.plot(r, V_massive, label=f"$\mu = {mu}$")

# Add vertical lines
plt.axvline(x=2 * M, color='red', linestyle='--', label="$r_H$ (Orizzonte degli Eventi)")
plt.axvline(x=3, color='green', linestyle='--', label="Photon Sphere ($r=3M$)")
plt.axvline(x=6, color='purple', linestyle='--', label="ISCO ($r=6M$)")
plt.axhline(0, color='black', linestyle='-', linewidth=0.8)

# Labels and title
plt.title("Potenziale Efficace $V_{eff}(r)$ per un Buco Nero di Schwarzschild", fontsize=14)
plt.xlabel("$r/M$", fontsize=12)
plt.ylabel("$V_{eff}$", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.show()



#%%

import numpy as np
import matplotlib.pyplot as plt

# Parametri per simulare un segnale EMRI-like
t_emri = np.linspace(0, 50, 5000)  # Tempo più lungo
frequency_base = 1  # Frequenza di base in Hz
modulation_frequency = 0.1  # Frequenza di modulazione

# Simulazione del segnale EMRI-like senza decadimento dell'ampiezza
h_emri = np.cos(2 * np.pi * frequency_base * t_emri) * \
         np.cos(2 * np.pi * modulation_frequency * t_emri)

# Plot del segnale EMRI-like
plt.figure(figsize=(12, 6))
plt.plot(t_emri, h_emri, label='Segnale EMRI-like (senza decadimento)')
plt.title("Simulazione di un segnale EMRI (senza decadimento)")
plt.xlabel("Tempo [s]")
plt.ylabel("Ampiezza")
plt.grid(True)
plt.legend()
plt.show()


#%%


import numpy as np
import matplotlib.pyplot as plt

# Costanti fisiche
G = 6.67430e-11  # Costante gravitazionale (m^3 kg^-1 s^-2)
c = 3e8  # Velocità della luce (m/s)

# Parametri
r = 1e21  # Distanza dall'osservatore (in m)
mu = 1e30  # Massa ridotta (in kg)
R = 1e9  # Raggio orbitale (in m)
theta = np.pi / 4  # Angolo di osservazione
f_gw = 1e-3  # Frequenza dell'onda gravitazionale (Hz)
t = np.linspace(0, 10, 1000)  # Tempo (in s)

# Calcolo delle forme d'onda gravitazionali
h_plus = (4 * G * mu * R**2 / (r * c**4)) * (1 + np.cos(theta)**2) * np.cos(2 * np.pi * f_gw * t)
h_cross = (4 * G * mu * R**2 / (r * c**4)) * np.cos(theta) * np.sin(2 * np.pi * f_gw * t)

# Applichiamo il fattore di scala per rendere il grafico più leggibile
scaling_factor = 1e17  # Fattore di scala
h_plus_scaled = h_plus * scaling_factor
h_cross_scaled = h_cross * scaling_factor

# Aggiorniamo la frequenza per rendere le oscillazioni più visibili
f_gw = 1  # Frequenza dell'onda gravitazionale in Hz

# Ricalcoliamo le forme d'onda
h_plus = (4 * G * mu * R**2 / (r * c**4)) * (1 + np.cos(theta)**2) * np.cos(2 * np.pi * f_gw * t)
h_cross = (4 * G * mu * R**2 / (r * c**4)) * np.cos(theta) * np.sin(2 * np.pi * f_gw * t)

# Applichiamo il fattore di scala
h_plus_scaled = h_plus * scaling_factor
h_cross_scaled = h_cross * scaling_factor

# Plot separato per h_+(t)
plt.figure(figsize=(12, 6))
plt.plot(t, h_plus_scaled, label=r"$h_+(t)$ (scalato)", color='blue')
plt.title("Forma d'onda gravitazionale $h_+(t)$ (frequenza maggiore)")
plt.xlabel("Tempo [s]")
plt.ylabel("Ampiezza (scalata)")
plt.grid(True)
plt.legend()
plt.show()

# Plot separato per h_×(t)
plt.figure(figsize=(12, 6))
plt.plot(t, h_cross_scaled, label=r"$h_\times(t)$ (scalato)", color='orange')
plt.title("Forma d'onda gravitazionale $h_\times(t)$ (frequenza maggiore)")
plt.xlabel("Tempo [s]")
plt.ylabel("Ampiezza (scalata)")
plt.grid(True)
plt.legend()
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Crea la figura e un assieme di assi 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Generiamo una griglia di punti (X, Z) e ricaviamo Y dal piano x - y + 4 = 0 => y = x + 4
X_vals = np.linspace(-2, 6, 20)   # range scelto per X
Z_vals = np.linspace(-2, 4, 20)   # range scelto per Z
X, Z = np.meshgrid(X_vals, Z_vals)
Y = X + 4  # dal piano x - y + 4 = 0 => y = x + 4

# Disegniamo la superficie del piano (porzione)
ax.plot_surface(X, Y, Z, alpha=0.5)

# Punto S(3, 2, 1)
Sx, Sy, Sz = 3, 2, 1
ax.scatter(Sx, Sy, Sz, s=50)  # disegna il punto sorgente

# Miglioriamo la visualizzazione
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("Piano x - y + 4 = 0 e sorgente S(3,2,1)")

plt.show()












