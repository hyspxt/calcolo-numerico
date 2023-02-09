import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

"""
1. Calcolare il polinomio di grado n = 5 che approssimi i seguenti dati:
{(1.0, 1.18), (1.2, 1.26), (1.4, 1.23), (1.6, 1.37), (1.8, 1.37), (2.0, 1.45), (2.2, 1.42),
(2.4, 1.46), (2.6, 1.53), (2.8, 1.59), (3.0, 1.50)}
2. Risolvere il problema ai minimi quadrati sia con le equazioni normali che con la SVD (vedi A.1 e A.2).
iii. Valutare graficamente i polinomi di approssimazione e confrontare gli errori commessi dai due metodi sul set di punti.
"""

n = 7 # Grado del polinomio approssimante

'''
Qui in realtá non c'é molto da dire é facilmente verificabile che un polinomio di grado maggiore si adatta meglio ai punti
e quini la precisione della funzione approssimante é maggiore.
'''


x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])

print('Shape of x:', x.shape)
print('Shape of y:', y.shape, "\n")

N = x.size # Numero dei dati (punti da approssimare mediante il polinomio)

A = np.zeros((N, n+1)) # La matrice ha come righe il numero di punti = 11 e colonne il grado del polinomio

for i in range(n+1):
    A[:,i] = x**i  # Vuol dire che la colonna i ha i valori dei punti elevati alla i

print("A = \n", A)

''' Risoluzione tramite equazioni normali'''

# Si calcolano matrice del sistema e termine noto 
ATA = A.T @ A
ATy = A.T @ y

L = scipy.linalg.cholesky(ATA, lower=True) # Cholesky genera una matrice triangolare inferiore, per questo lower = true
alpha1 = scipy.linalg.solve(L, ATy, lower=True) # L y = AT b -> y
alpha_normali = scipy.linalg.solve(L.T, alpha1) # LT x = y -> x

print("alpha_normali = \n", alpha_normali)


'''Risoluzione tramite SVD'''

U, s, VT = scipy.linalg.svd(A) 

print('Shape of U:', U.shape) # Matrice ortogonale numero punti x numero punti
print('Shape of s:', s.shape) # VETTORE di valori singolari con gradopolinomio valori
print('Shape of V:', VT.shape) # Matrice ortogonale gradopolinomio x gradopolinomio

alpha_svd = np.zeros(s.shape) # Vettore dei coefficienti alpha del polinomio approssimante

for j in range(n+1): # N + 1 perché se il grado del polinomio é n, allora ci sono n+1 colonne
  uj = U[:, j]    # uj é uguale alla j-esima colonna di U
  vj = VT[j, :]   # vj é uguale alla j-esima riga di VT
                  # Sono entrambi VETTORI.
  alpha_svd = alpha_svd + (np.dot(uj, y) * vj) / s[j]

print("Alpha_svd = ", alpha_svd)   # Il risultato di una molt. di vettori é ovviamente un vettore

'''Verifica e confronto delle soluzioni'''

# Funzione per valutare il polinomio p, in un punto x, dati i coefficienti alpha
def p(alpha, x):
  A = np.zeros((len(x), len(alpha)))
  for i in range(len(alpha)):
      A[:, i] = x**i
  y = np.dot(A, alpha)
  return y

'''CONFRONTO ERRORI SUI DATI '''
y1 = p(alpha_normali, x)
y2 = p(alpha_svd, x)

err1 = np.linalg.norm (y-y1, 2) 
err2 = np.linalg.norm (y-y2, 2) 
print ('Errore di approssimazione con Eq. Normali: ', err1)
print ('Errore di approssimazione con SVD: ', err2)

"""
Sebbene siano quasi identici, SVD presenta un errore leggermente minore in quanto é piú stabile come metodo
e infatti genera soluzioni stabili anche con input poco precisi.
"""

'''CONFRONTO GRAFICO '''
x_plot = np.linspace(1, 3, 100) 

y_normali = p(alpha_normali, x_plot)
y_svd = p(alpha_svd, x_plot)

plt.figure(figsize=(20, 10))

# PLOT con Eq. Normali
plt.subplot(1, 2, 1)
plt.plot(x, y, ' or')
plt.plot(x_plot, y_normali)
plt.title('Approssimazione tramite Eq. Normali')

# PLOT con Eq. Normali
plt.subplot(1, 2, 2)
plt.plot(x, y, 'or')
plt.plot(x_plot, y_svd)
plt.title('Approssimazione tramite SVD')

plt.show()