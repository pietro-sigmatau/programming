# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:04:06 2025

@author: Pietro
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded







# Parametri
M = 1.0  # Massa del buco nero
l = 2  # Numero quantico del momento angolare
omega = 0.5  # Frequenza della perturbazione
N = 5000  # Numero di punti di discretizzazione
r_min, r_max = 2.1 * M, 50 * M  # Limiti in r
r = np.linspace(r_min, r_max, N)
dr = r[1] - r[0]

# Coordinata tortoise
r_star = r + 2 * M * np.log(r / (2 * M) - 1)

# Potenziale Zerilli
n = (l - 1) * (l + 2) / 2
V_zerilli = (2 * (r - 2 * M) / (r**3 * (n * r + 3 * M)**2)) * (
    n**2 * (n + 1) * r**3 + 3 * M * n**2 * r**2 + 9 * M**2 * n * r + 9 * M**3
)
V_zerilli = np.nan_to_num(V_zerilli, nan=0.0, posinf=0.0, neginf=0.0)

# Matrice tridiagonale per l'Hamiltoniana
diagonal = 2 / dr**2 + V_zerilli - omega**2
off_diagonal = -1 / dr**2 * np.ones(N - 1)

bands = np.zeros((3, N))
bands[0, 1:] = off_diagonal  # Superdiagonale
bands[1, :] = diagonal  # Diagonale principale
bands[2, :-1] = off_diagonal  # Sottodiagonale

# Soluzione dell'equazione di Zerilli
Z = solve_banded((1, 1), bands, np.zeros(N))
Z[0] = np.exp(1j * omega * r_star[0])  # Condizione al bordo sinistra
Z[-1] = np.exp(-1j * omega * r_star[-1])  # Condizione al bordo destra

# Normalizzazione
norm_factor = np.sqrt(np.sum(np.abs(Z)**2) * dr)
if norm_factor == 0:
    norm_factor = 1.0
Z = Z / norm_factor

# Onde incidente, riflessa e trasmessa
region_left = r_star < r_star[N // 3]
region_right = r_star > r_star[2 * N // 3]

Z_incident = np.exp(1j * omega * r_star[region_left])
Z_reflected = Z[region_left] - Z_incident
R = np.sum(np.abs(Z_reflected)**2) / np.sum(np.abs(Z_incident)**2)

Z_transmitted = Z[region_right]
T = np.sum(np.abs(Z_transmitted)**2) / np.sum(np.abs(Z_incident)**2)

# Output
print(f"Riflettività: R = {R:.6f}")
print(f"Trasmissività: T = {T:.6f}")

# Grafico
plt.figure(figsize=(10, 6))
plt.plot(r_star, V_zerilli, label="Potenziale Zerilli")
plt.plot(r_star, np.real(Z), label="Re(Z)", color="blue")
plt.plot(r_star, np.imag(Z), label="Im(Z)", color="orange")
plt.title("Soluzione dell'equazione di Zerilli")
plt.xlabel("r*")
plt.ylabel("Z(r*)")
plt.legend()
plt.grid()
plt.show()


