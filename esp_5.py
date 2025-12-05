# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:51:00 2023

@author: Pietro
"""

import numpy as np
import matplotlib.pyplot as plt

# Dati forniti
frequenze = [100, 200, 500, 1000, 2000, 5000, 10000]  # Hz
ampiezze_dB = [-20, -15, -10, -5, 0, 5, 10]  # Amplificazione in dB

# Calcolare omega (frequenza angolare) nel range specificato
omega = 2 * np.pi * np.logspace(np.log10(min(frequenze)), np.log10(max(frequenze)), num=1000)

# Interpolare le ampiezze dB ai punti omega
amplificazione_dB_interp = np.interp(omega, 2 * np.pi * np.array(frequenze), ampiezze_dB)

# Creare il grafico dell'amplificazione in decibel in funzione di omega
plt.figure(figsize=(10, 6))

plt.semilogx(omega, amplificazione_dB_interp, label='Amplificazione (dB)')

plt.xlabel('Frequenza (rad/s)')
plt.ylabel('Amplificazione (dB)')
plt.title('Grafico dell\'Amplificazione (dB) in Funzione di Frequenza')
plt.legend()
plt.grid(True)
plt.show()


#%%


import numpy as np
import matplotlib.pyplot as plt

# Dati di esempio
omega = [3141, 6283, 628318, 2513275, 2886779]  # sostituisci con i tuoi dati di pulsazione angolare
amplificazione = [2,9, 2.92, 3.04, 2.72, 2.52]  # sostituisci con i tuoi dati di amplificazione

# Calcola l'amplificazione in dB
amplificazione_dB = [20 * np.log10(A) for A in amplificazione]

# Crea il grafico
plt.figure
plt.semilogx(omega, amplificazione_dB, marker='o', linestyle='-', color='b')

# Etichette e titoli
plt.xlabel('Pulsazione Angolare (rad/s)')
plt.ylabel('Amplificazione (dB)')
plt.title('Grafico Studio in Frequenza')

# Mostra il grafico
plt.grid(True)
plt.show()



#%%

import numpy as np
import matplotlib.pyplot as plt

# Dati di esempio
omega = [3141, 6283, 628318, 2513275, 2886779]  # sostituisci con i tuoi dati di pulsazione angolare
amplificazione_dB_calcolate = [9.42, 9.46, 9.72, 8.91, 8.09]  # dati calcolati in decibel

# Crea il grafico
plt.figure(figsize=(10, 6))
plt.plot(omega, amplificazione_dB_calcolate, marker='o', linestyle='-', color='b')

# Etichette e titoli
plt.xlabel('Pulsazione Angolare (rad/s)')
plt.ylabel('Amplificazione (dB)')
plt.title('Grafico Studio in Frequenza')

# Mostra il grafico
plt.grid(True)
plt.show()





#%%



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Dati specifici
omega = [3141, 6283, 628318, 2513275, 2886779]  # sostituisci con i tuoi dati di pulsazione angolare
amplificazione_dB_calcolate = [9.42, 9.46, 9.72, 8.91, 8.09]  # dati calcolati in decibel

# Crea il grafico con scala semilogaritmica sull'asse delle ascisse
plt.figure(figsize=(10, 6))
plt.semilogx(omega, amplificazione_dB_calcolate, marker='o', linestyle='-', color='b', label='Dati')

# Calcola il fit lineare nella regione interessante
regione_interessante = np.array(omega[2:])  # Prendi dati da 600.000 in poi
fit_lineare = linregress(np.log10(regione_interessante), amplificazione_dB_calcolate[2:])
pendenza = fit_lineare.slope

# Plotta il fit lineare
plt.plot(regione_interessante, fit_lineare.intercept + pendenza * np.log10(regione_interessante), '--', color='r', label='Fit Lineare')

# Etichette e titoli
plt.xlabel('Pulsazione Angolare (rad/s)')
plt.ylabel('Amplificazione (dB)')
plt.title('Grafico Studio in Frequenza (Scala Semilogaritmica)')

# Aggiungi legenda
plt.legend()

# Mostra il grafico
plt.grid(True)
plt.show()

# Mostra la pendenza calcolata
print(f'La pendenza calcolata è {pendenza:.2f} dB/decade')



#%%






import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Dati specifici
omega_terza_parte = [3141, 6283, 376991, 628318, 1256637]
amplificazione_dB_terza_parte = [9.83, 9.43, 20.25, 24.02, 29.48]

# Crea il grafico con scala semilogaritmica sull'asse delle ascisse
plt.figure(figsize=(10, 6))
plt.semilogx(omega_terza_parte, amplificazione_dB_terza_parte, marker='o', linestyle='-', color='b', label='Dati')

# Calcola il fit lineare
fit_lineare_terza_parte = linregress(np.log10(omega_terza_parte), amplificazione_dB_terza_parte)
pendenza_terza_parte = fit_lineare_terza_parte.slope

# Plotta il fit lineare
plt.plot(omega_terza_parte, fit_lineare_terza_parte.intercept + pendenza_terza_parte * np.log10(omega_terza_parte), '--', color='r', label='Fit Lineare')

# Etichette e titoli
plt.xlabel('Pulsazione Angolare (rad/s)')
plt.ylabel('Amplificazione (dB)')
plt.title('Grafico Studio in Frequenza (Scala Semilogaritmica) - Terza Parte')

# Aggiungi legenda
plt.legend()

# Mostra il grafico
plt.grid(True)
plt.show()

# Mostra la pendenza calcolata
print(f'La pendenza calcolata è {pendenza_terza_parte:.2f} dB/decade')







#%%


import numpy as np
import mathplotplib.pyplot as plt

def f(zeta):
    return 1/(np.sqrt((np.pi/3)-(zeta*zeta)))

hz=[0.1, 0.01, 0.05, 0.001, 0.005]

alpha=np.arange(0.1,1.0,0.1)
ris=np.zeros()

for u in (0,4):
    a=0
    N=10000
    hz=hz[u]




#%%



import numpy as np
import matplotlib.pyplot as plt

def f(zeta):
    return 1/(np.sqrt((np.pi/3)-zeta**2))

alpha=np.arange(0.1,1.1,.1)
hz=[0.1, 0.05, 0.01, 0.001, 0.005]
In=np.empty(50) #vettore vuoto di 50 elementi

S=0

N=10000 #tipico N per i trapezi
a=0

for i in range(0,10):         #non è per integrare con i trapezi, è esterno perché varia alpha ogni volta che si applica l'integrazione con trapezi
    b=alpha[i]
    for j in range(0,5):
        h=hz[j]
        S=f(a)+f(b)
        x=a
        for k in range(1,N):
            x=x+h
            S=S+(2*f(x))
            In[i]=h/(2*S)

        
for i in range(0,50):
    print(In[i])















