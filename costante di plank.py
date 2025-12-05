# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:54:10 2024

@author: Pietro
"""

import numpy as np

# Dati forniti
V0_values = [0.53, 0.63, 1.15, 1.35, 1.65]  # Valori di V0
lambda_values_nm = [580, 546, 436, 405, 365]  # Lunghezze d'onda in nanometri

# Carica elementare in Coulomb
e = 1.6e-19

# Convertire i valori di V0 da volt a electronvolt
eV0_values = np.array(V0_values) * e

# Convertire le lunghezze d'onda da nanometri a metri
lambda_values_m = np.array(lambda_values_nm) * 1e-9

# Velocità della luce in metri al secondo
c = 3e8

# Calcolare il vettore delle frequenze v in Hertz
v_values = c / lambda_values_m

# Calcolare la regressione lineare
h, W = np.polyfit(v_values, eV0_values, 1)

# Stampa dei risultati
print("Stima della costante di Planck (h):", h)
print("Stima della funzione di lavoro (W):", W)


#%%



import numpy as np
import matplotlib.pyplot as plt

# Dati forniti
V0_values = [0.53, 0.63, 1.15, 1.35, 1.65]  # Valori di V0
lambda_values_nm = [580, 546, 436, 405, 365]  # Lunghezze d'onda in nanometri

# Carica elementare in Coulomb
e = 1.6e-19

# Convertire i valori di V0 da volt a electronvolt
eV0_values = np.array(V0_values) * e

# Convertire le lunghezze d'onda da nanometri a metri
lambda_values_m = np.array(lambda_values_nm) * 1e-9

# Velocità della luce in metri al secondo
c = 3e8

# Calcolare il vettore delle frequenze v in Hertz
v_values = c / lambda_values_m

# Calcolare la regressione lineare
h, W = np.polyfit(v_values, eV0_values, 1)

# Creare il modello lineare
x = np.linspace(min(v_values), max(v_values), 100)
y = h * x + W

# Creare il grafico
plt.figure(figsize=(10, 6))
plt.scatter(v_values, eV0_values, color='blue', label='Dati sperimentali')
plt.plot(x, y, color='red', label=f'Regressione lineare: h={h:.2e}, W={W:.2e}')
plt.xlabel('Frequenza (Hz)')
plt.ylabel('Energia cinetica dell\'elettrone moltiplicata per V0 (J)')
plt.title('Stima della costante di Planck e della funzione di lavoro')
plt.legend()
plt.grid(True)
plt.show()
