import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

data = pd.read_csv("HeightVsWeight.csv")
data = np.array(data)
print(data.shape)

x = data[:, 0]
y = data[:, 1]

print(x.shape)
print(y.shape)

n = 5
N = x.size

A = np.zeros((N, n+1))

for i in range(n+1):
    A[:, i] = x**i

''' RISOLUZIONE CON EQUAZIONI NORMALI'''

# calcoliamo la matrice del sistema e il termine noto a parte
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
x_plot = np.linspace(10, 80, 100)  # C'é aumentare lo spazio di approssimazione.
y_normali = p(alpha_normali, x_plot)
y_svd = p(alpha_svd, x_plot)

plt.figure(figsize=(20, 10))

# PLOT con Eq. Normali
plt.subplot(1, 2, 1)
plt.plot(x, y, 'or')
plt.plot(x_plot, y_normali)
plt.title('Approssimazione tramite Eq. Normali')

# PLOT con Eq. Normali
plt.subplot(1, 2, 2)
plt.plot(x, y, 'or')
plt.plot(x_plot, y_svd)
plt.title('Approssimazione tramite SVD')

plt.show()