import numpy as np
import numpy.matlib
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
import matplotlib.pyplot as plt

import time

'''
METODI ITERATIVI
'''

"""
Scrivi le funzioni: Jacobi(A, b, x0, maxit, tol, xTrue) e GaussSeidel(A, b, x0, maxit, tol, xTrue) per 
implementare i metodi di Jacobi e GaussSeidel per la risoluzione di sistemi lineari con matrice a diagonale
dominante. In particolare:
    
    - x0 sia l'iterato iniziale;
    
    - la condizione d'arresto sia dettata dal numero massimo di iterazione consentite maxit e dalla 
    tolleranza tol sulla differenza relativa tra due iterati successivi.
    
    - si preveda in input la soluzione esatta xTrue per calcolare l'errore relativo ad ogni iterazione.

Entambe le funzioni restituiscono in output:
    - la soluzione x
    - il numero k di iterazioni effettuate
    - il vettore relErr di tutti gli errori relativi.

"""


def Jacobi(A,b,x0,maxit,tol, xTrue):
    
  n = np.size(x0) # Numero di colonne di x0  
  ite = 0 # Contatore per il numero di iterazioni
  x = np.copy(x0)
  norma_it = 1 + tol # La prima iterata di errore non é possibile calcolarla, 
                     # quindi, si inizializza con tolleranza + 1 che, é > tol 
  
  relErr = np.zeros((maxit, 1)) # Errore relativo tra sol. calcolata ed esatta
  errIter = np.zeros((maxit, 1)) # Errore iterativo tra sol. di due iterate successive
  relErr[0] = np.linalg.norm(xTrue-x0)/np.linalg.norm(xTrue)
  
  while (ite < maxit - 1 and norma_it > tol):
    x_old = np.copy(x) # Iterata precedente
    for i in range(0,n):
      #x[i]=(b[i]-sum([A[i,j]*x_old[j] for j in range(0,i)])-sum([A[i, j]*x_old[j] for j in range(i+1,n)]))/A[i,i]
      x[i] = (b[i]-np.dot(A[i,0:i],x_old[0:i])-np.dot(A[i,i+1:n],x_old[i+1:n])) / A[i,i]
    ite=ite+1
    norma_it = np.linalg.norm(x_old-x) / np.linalg.norm(x_old)
    relErr[ite] = np.linalg.norm(xTrue-x) / np.linalg.norm(xTrue)
    errIter[ite-1] = norma_it
    
  relErr=relErr[:ite]
  errIter=errIter[:ite]  
  return [x, ite, relErr, errIter]


def GaussSeidel(A, b, x0, maxit, tol, xTrue):
    
  n = np.size(x0)
  ite = 0
  x = np.copy(x0)
  norma_it = 1 + tol
  
  relErr = np.zeros((maxit, 1))
  errIter = np.zeros((maxit, 1))
  relErr[0] = np.linalg.norm(xTrue-x0) / np.linalg.norm(xTrue)
  
  while (ite < maxit and norma_it > tol):
      x_old = np.copy(x)
      for i in range(0,n):
          x[i] = (b[i] - np.dot(A[i,0:i],x[0:i])-np.dot(A[i,i+1:n],x_old[i+1:n])) / A[i,i]
      ite = ite + 1
      norma_it = np.linalg.norm(x_old-x) / np.linalg.norm(x_old)
      relErr[ite] = np.linalg.norm(xTrue-x) / np.linalg.norm(xTrue)
      errIter[ite-1] = norma_it      
      
  relErr = relErr[:ite]
  errIter = errIter[:ite]  
  return [x, ite, relErr, errIter]

def itConverges(A):
   return max(abs(np.linalg.eigvalsh(A))) < 1


####### Testing #######
""" 
1. Considerare la precedente matrice tridiagonale (dell'file LAB_2_diretti.py') per N = 100
"""

n = 100
A = 9 * np.eye(n) + np.diag(-4 * np.ones(n - 1), k = -1) + np.diag(-4 * np.ones(n - 1), k = +1) 
xTrue = np.ones((n,1))
b = np.matmul(A, xTrue)

print('\n A:\n',A)
print('\n xTrue:\n',xTrue)
print('\n b:\n',b)


""" 
2. Verificare, calcolando il raggio spettrale della matrice, la convergenza dei metodi.
"""

D = 9 * np.eye(n)
E = np.diag(-4 * np.ones(n - 1), k = -1)
F = np.diag(-4 * np.ones(n - 1), k = +1) 

# JACOBI
G = np.linalg.inv(D)
itConverges(G)
if itConverges(G):
    print('Jacobi converge in quanto il raggio spettrale é < 1 ')
else:
    print('Jacobi NON converge in quanto il raggio spettrale é < 1 ')

# GAUSS-SEIDEL
G = np.linalg.inv(D-E)
if itConverges(G):
    print('Gauss-Seidel converge in quanto il raggio spettrale é < 1 ')
else: 
    print('Gauss-Seidel NON converge in quanto il raggio spettrale é >= 1 ') 

"""
3. Eseguire i calcoli con tolleranza 1.e-8
"""

#metodi iterativi
x0 = np.zeros((n,1))
x0[0] = 1
maxit = 100
tol = 10 ** - 8

(xJacobi, kJacobi, relErrJacobi, errIterJacobi) = Jacobi(A, b, x0, maxit, tol, xTrue) 
(xGS, kGS, relErrGS, errIterGS) = GaussSeidel(A, b, x0, maxit, tol, xTrue) 

print('\nSoluzione calcolata da Jacobi:' )
for i in range(n):
    print('%0.2f' %xJacobi[i])

print('\nSoluzione calcolata da Gauss Seidel:' )
for i in range(n):
    print('%0.2f' %xGS[i])

# CONFRONTI
# Confronto grafico degli errori di Errore Relativo

rangeJabobi = range (0, kJacobi)
rangeGS = range(0, kGS)


"""
4. Riportare in un grafico l'errore relativo (in norma 2) di entrambi i metodi al variare del numero
   di iterazioni per N fissato. Scegliere almeno due valori di N.'
"""

plt.figure()
plt.plot(rangeJabobi, relErrJacobi, label='Jacobi', color='blue', linewidth=1, marker='o'  )
plt.plot(rangeGS, relErrGS, label='Gauss Seidel', color = 'red', linewidth=2, marker='.' )
plt.legend(loc='upper right')
plt.xlabel('iterations')
plt.ylabel('Relative Error')
plt.title('Comparison of the different algorithms')
plt.show()

####### Comportamento al variare di N ########  

dim = np.arange(5, 100, 5)

ErrRelF_J = np.zeros(np.size(dim))
ErrRelF_GS = np.zeros(np.size(dim))

ite_J = np.zeros(np.size(dim))
ite_GS = np.zeros(np.size(dim))

# Misura dei tempi di esecuzione dei vari metodi
elapsedTimeJ = np.zeros(np.size(dim))
elapsedTimeGS = np.zeros(np.size(dim))
elapsedTimeCholesky = np.zeros(np.size(dim))
elapsedTimeLU = np.zeros(np.size(dim))

i = 0

for n in dim:
    
    #creazione del problema test
    A = 9 * np.eye(n) + np.diag(-4 * np.ones(n - 1), k = 1) + np.diag(-4 * np.ones(n - 1), k = -1)  
    xTrue = np.ones((n,1))
    b = np.matmul(A, xTrue)
    
    x0 = np.zeros((n, 1))
    x0[0] = 1
    
    
    ######### Metodi Diretti ############
    #### Cholesky ####
    st = time.time()
    L = scipy.linalg.cholesky(A, lower = True)
    y = scipy.linalg.solve(L, b)
    my_x = scipy.linalg.solve(L.T, y)
    et = time.time()
    elapsedTimeCholesky[i] = et - st
    
    #### LU PIVOTING ####
    # Ax = b <--> piv_LU * x = b
    st = time.time()
    lu, piv = LUdec.lu_factor(A)
    my_x = scipy.linalg.lu_solve((lu, piv), b)
    et = time.time()
    elapsedTimeLU[i] = et - st
    
    #metodi iterativi
    
    st = time.time()
    (xJ, kJ, relErrJ, errIterJ) = Jacobi(A, b, x0, maxit, tol, xTrue) 
    et = time.time()
    elapsedTimeJ[i] = et - st 
    
    st = time.time()
    (xGS, kGS, relErrGS, errIterGS) = GaussSeidel(A, b, x0, maxit, tol, xTrue) 
    et = time.time()
    elapsedTimeGS[i] = et - st
    
    #errore relativo finale
    ErrRelF_J[i] = relErrJ[-1]
    ErrRelF_GS[i] = relErrGS[-1]
    
    #iterazioni
    ite_J[i] = kJ
    ite_GS[i]= kGS

    i = i+1
    
"""
5. Riportare in un grafico l'errore relativo finale dei metodi al variare della dimensione N del sistema.'
"""

# errore relativo finale dei metodi al variare della dimensione N
plt.figure()
plt.semilogy(dim, ErrRelF_J, label='Jacobi', color='purple', linewidth=1, marker='o')
plt.semilogy(dim, ErrRelF_GS, label='GaussSeidel', color='cyan', linewidth=1, marker='o')
plt.legend(loc='upper right')
plt.xlabel('N value')
plt.ylabel('Relative Error')
plt.title('Algorithms Relative Error on N')
plt.show()

"""
6. Riportare in un grafico il numero di iterazioni di entrambi i metodi al variare di N.'
"""

#numero di iterazioni di entrambi i metodi al variare di N
plt.figure()
plt.semilogy(dim, ite_J, label='Jacobi', color='purple', linewidth=1, marker='o')
plt.semilogy(dim, ite_GS, label='GaussSeidel', color='cyan', linewidth=1, marker='o')
plt.legend(loc='upper right')
plt.xlabel('N dimension')
plt.ylabel('Number of iterations')
plt.title('Algorithms iterations on N')
plt.show()

"""
7. Riportare in un grafico il tempo impiegato dai metodi di Jacobi, Gauss-Seidel, LU, Cholesky al variare di N.
"""
# errore relativo finale dei metodi al variare della dimensione N
plt.figure()
plt.plot(dim, elapsedTimeJ, label='Jacobi', color='purple', linewidth=1, marker='o')
plt.plot(dim, elapsedTimeGS, label='GaussSeidel', color='cyan', linewidth=1, marker='o')
plt.plot(dim, elapsedTimeCholesky, label='Cholesky', color='orange', linewidth=1, marker='o')
plt.plot(dim, elapsedTimeLU, label='LU+Pivoting', color='green', linewidth=1, marker='.')
plt.legend(loc='upper right')
plt.xlabel('N value')
plt.ylabel('Execution Time')
plt.title('Different algorithms Execution time on N')
plt.show()

