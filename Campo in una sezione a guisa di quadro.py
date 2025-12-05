# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:00:46 2024

@author: Pietro
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametri della griglia e del tempo
Lx = 10.0  # Lunghezza della sezione quadrata in x
Ly = 10.0  # Lunghezza della sezione quadrata in y
Nx = 100   # Numero di punti della griglia in x
Ny = 100   # Numero di punti della griglia in y
c = 1.0    # Velocità della propagazione dell'onda
dt = 0.01  # Passo temporale
Tmax = 2.0 # Tempo massimo di simulazione

x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)

# Definizione del campo iniziale A(x, y, z=0, t=0)
A0 = np.exp(-((X/2)**2 + (Y/2)**2)) * np.cos(2*np.pi*X) * np.cos(2*np.pi*Y)

# Funzione per aggiornare il campo nel tempo (soluzione numerica dell'equazione d'onda)
def update_field(A, dt, c, Nx, Ny):
    A_new = np.copy(A)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            A_new[i, j] = (2*A[i, j] - A[i, j] + c**2 * dt**2 * (
                (A[i+1, j] - 2*A[i, j] + A[i-1, j]) +
                (A[i, j+1] - 2*A[i, j] + A[i, j-1])
            ))
    return A_new

# Animazione del campo
fig, ax = plt.subplots()
img = ax.imshow(A0, extent=(-Lx/2, Lx/2, -Ly/2, Ly/2), cmap='viridis')
ax.set_title('Propagazione del Campo A(x, y, z=0, t)')
ax.set_xlabel('x')
ax.set_ylabel('y')

def animate(t):
    global A0
    A0 = update_field(A0, dt, c, Nx, Ny)
    img.set_array(A0)
    return [img]

ani = animation.FuncAnimation(fig, animate, frames=int(Tmax/dt), interval=50, blit=True)
plt.show()
