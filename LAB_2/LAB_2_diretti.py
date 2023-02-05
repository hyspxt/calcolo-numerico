import numpy as np
import numpy.matlib
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec
import matplotlib.pyplot as plt

'''
METODI DIRETTI
'''

def es1(n):
    
    """
    1. Risoluzione di sistemi lineari con matrice generica.
    Scrivere uno script Python che:
        
        a) crea un problema test di dimensione variabile n la cui soluzione esatta sia il vettore x di tutti
        elementi unitari e b il termine noto ottenuto moltiplicando la matrice A per la soluzione x.
        
        b) calcola il numero di condizione (o una stima di esso).
    
        c) risolve il sistema lineare Ax = b con la fattorizzazione LU con pivoting.
    
    Problema Test: Un matrice di numeri casuali A generata con la funzione randn di Matlab (n variabile tra 10-1000)
        
    """
    
    A = np.matlib.rand((n, n))
    x = np.ones((n, 1))
    b = np.dot(A, x)
    
    condA = np.linalg.cond(A, p=2)
    print('Condiz. di A = ', condA)
    
    # Ax = b <--> piv_LU * x = b
    lu, piv = LUdec.lu_factor(A)
    my_x = scipy.linalg.lu_solve((lu, piv), b)
    print('Norm = ', scipy.linalg.norm(x - my_x, 2))


# Testing      
es1(10)
es1(100)
es1(1000)

def es2(n, A):
    
    """
    2. Risoluzione di sistemi lineari con matrice simmetrica e definita positiva.
    Scrivere uno script Python:
        a) crea un problema test di dimensione variabile n la cui soluzione esatta sia il vettore x di tutti 
        elementi unitari e b il termine noto ottenuto moltiplicando la matrice A per la soluzione x.
        
        b) calcola il numero di condizione (o una stima di esso)
    
        c) risolve il sistema lineare Ax = b con la fattorizzazione di Cholesky
    
    Problema Test: 
        - matrice di Hilbert di dimensione n (con n variabile fra 2 e 15)
        
        - matrice tridiagonale simmetrica e definita positiva avente sulla diagonale elementi uguiali a 9
        e quelli sopra e sottodiagonali uguali a -4    
    """
    
    x = np.ones((n, 1))
    b = np.dot(A, x)
    
    condA = np.linalg.cond(A, p=2)
    print('Condiz. di A = ', condA)

    L = scipy.linalg.cholesky(A, lower = True)
    B = np.matmul(L, np.transpose(L))
    
    errRel = scipy.linalg.norm(A - B, 'fro')
    print('Relative error= ', errRel)
    
    y = scipy.linalg.solve(L, b)
    my_x = scipy.linalg.solve(L.T, y)
    norm = scipy.linalg.norm(x - my_x, 2)
    print('Norma = ', norm)
    
    return(condA, norm)

####### Testing #######

FIXED_STOP = 12

# Testing Hilbert
print('Caso Hilbert: ')
KA_Hilb = np.zeros((FIXED_STOP-1, 1)) # Inizializzazione degli array di condizionamento e errore rel
Err_Hilb = np.zeros((FIXED_STOP-1, 1))

for n in np.arange(2, FIXED_STOP): # Si verificano le condizioni dell'errore e del condizionamento al variare di n
                                    # Ovvero la dimensione della matrice.
    A_Hilb = scipy.linalg.hilbert(n)
    (KA_Hilb[n - 2], Err_Hilb[n - 2]) = es2(n, A_Hilb)
    
# Testing Matrice Tridiagonale
print('Caso Tridiagonale: ')
KA_TriD = np.zeros((FIXED_STOP-1, 1))
Err_TriD = np.zeros((FIXED_STOP-1, 1))

for n in np.arange(2, FIXED_STOP):
    A_TriD = np.diag(9 * np.ones(n)) + np.diag(-4 * np.ones(n-1), k = -1) + np.diag(-4 * np.ones(n-1), k = 1)
    (KA_TriD[n - 2], Err_TriD[n - 2]) = es2(n, A_TriD)
    

###### PLOTTING ######

# Disegna il grafico del numero di condizione in funzione della dimensione del sistema. Caso Hilbert.
points = FIXED_STOP - 1
dim_Matr_x = np.linspace(2, FIXED_STOP, points)
plt.plot(dim_Matr_x, KA_Hilb)
plt.title('HILBERT, Condizionamento')
plt.xlabel('Dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()

# Disegna il grafico dell'errore in norma 2 in funzione della dimensione del sistema. Caso Hilbert.
plt.plot(dim_Matr_x, Err_Hilb)
plt.title('HILBERT, Errore relativo')
plt.xlabel('Dimensione matrice: n')
plt.ylabel('Err = ||my_x-x|| / ||x||')
plt.show()

# Disegna il grafico del numero di condizione in funzione della dimensione del sistema. Caso Tridiagonale.
points = FIXED_STOP
plt.plot(dim_Matr_x, KA_TriD)
plt.title('TRIDIAGONALE, Condizionamento')
plt.xlabel('Dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()

# Disegna il grafico dell'errore in norma 2 in funzione della dimensione del sistema. Caso Tridiagonale.
plt.plot(dim_Matr_x, Err_TriD)
plt.title('TRIDIAGONALE, Errore relativo')
plt.xlabel('Dimensione matrice: n')
plt.ylabel('Err = ||my_x-x|| / ||x||')
plt.show()     

"""
1. Utilizzando i grafici richiesti: spiegare l’andamento dell’errore rispetto al numero di condizione della
matrice, l’andamento del tempo di esecuzione rispetto alla dimensione del sistema in relazione alla
complessità computazioneale degli algoritmi utilizzati.

2. Discutere la differenza di errore e tempo di esecuzione ottenuti con i metodi diretti e iterativi.
"""

    