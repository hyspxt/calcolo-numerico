import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

"""
Ripetere l’esercizio precedente su un dataset scaricabile al seguente indirizzo:
https://www.kaggle.com/sakshamjn/heightvsweight-for-linear-polynomial-regression
contenente i dati riguardanti il peso e l’altezza di 71 individui.
Una volta scaricato, per caricarlo su Spyder utilizzare la libreria pandas, nello
specifico la funzione pandas.read csv che fornisce come output il dataset.
Questo dovrà essere poi convertito in un numpy array.
"""

data = pd.read_csv("HeightVsWeight.csv")
data = np.array(data)

x = data[:, 0]
y = data[:, 1]

n = 7 # Grado del polinomio
'''
Qui in realtá non c'é molto da dire é facilmente verificabile che un polinomio di grado maggiore si adatta meglio ai punti
e quini la precisione della funzione approssimante é maggiore.
'''
N = x.size

A = np.zeros((N, n+1))

for i in range(n+1):
    A[:, i] = x**i

''' RISOLUZIONE CON EQUAZIONI NORMALI'''

ATA = A.T @ A
ATy = A.T @ y

L = scipy.linalg.cholesky(ATA, lower=True)
alpha1 = scipy.linalg.solve(L, ATy, lower=True) # L y = AT b -> y
alpha_normali = scipy.linalg.solve(L.T, alpha1) # LT x = y -> x

''' RISOLUZIONE CON SVD '''

U, s, VT = scipy.linalg.svd(A)

print('Shape of U:', U.shape) # Matrice ortogonale numero punti x numero punti
print('Shape of s:', s.shape) # VETTORE di valori singolari con gradopolinomio valori
print('Shape of V:', VT.shape) # Matrice ortogonale gradopolinomio x gradopolinomio

alpha_svd = np.zeros(s.shape)

for j in range(n+1):
  uj = U[:, j]
  vj = VT[j, :]
  alpha_svd = alpha_svd + (np.dot(uj, y) * vj) / s[j]
  

''' VISUALIZZAZIONE DEI RISULTATI '''

def p(alpha, x):
  A = np.zeros((len(x), len(alpha)))
  for i in range(len(alpha)):
      A[:, i] = x**i
  y = np.dot(A, alpha)
  return y


'''CONFRONTO GRAFICO '''
x_plot = np.linspace(10, 80, 100)  # C'é da aumentare lo spazio di approssimazione.
y_normali = p(alpha_normali, x_plot)
y_svd = p(alpha_svd, x_plot)

plt.figure(figsize=(20, 10))

# PLOT con Eq. Normali
plt.subplot(1, 2, 1)
plt.plot(x, y, 'or') # Plot dei punti 
plt.plot(x_plot, y_normali) # Plot funzione approssimante
plt.title('Approssimazione tramite Eq. Normali')

# PLOT con Eq. Normali
plt.subplot(1, 2, 2)
plt.plot(x, y, 'or')
plt.plot(x_plot, y_svd)
plt.title('Approssimazione tramite SVD')

plt.show()