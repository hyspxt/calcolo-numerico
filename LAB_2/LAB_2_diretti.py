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
    
    A = np.matlib.rand((n, n))    # Si crea una matrice con valori random di dimensione nxn
    x = np.ones((n, 1))  # Matrice di uni di dimensione nx1 NON É UN VETTORE
    b = np.dot(A, x)  # A @ x
    
    condA = np.linalg.cond(A, p=2)
    print('Condiz. di A = ', condA)
    
    # Ax = b <--> piv_L * x = b
    lu, piv = LUdec.lu_factor(A)
    my_x = scipy.linalg.lu_solve((lu, piv), b)

    print('Norm = ', scipy.linalg.norm(x - my_x, 2))  # errore relativo
    print ('\n')
    
# Testing      
es1(10)
es1(100)
es1(1000)

'''
Il numero di condizione é ovviamente variabile ad ogni esecuzione perché si é composta la matrice con numeri casuali, 
ma in generale, piú la dimensione del sistema aumenta, piú il numero di condizionamento aumenta. In questo caso, a differenza
delle apparenze, non aumenta in modo particolarmente elevato poiché la fattorizzazione LR CON PIVOTING é in realtá un metodo 
abbastanza stabile e non si fa influenzare particolarmente da problemi di condizionamento (a differenza della sua controparte senza pivoting).
A tal proposito, in quanto la dimensione del sistema é caratterizzata dalla forma n^p (con p norma) -> problema ben condizionato.

Per quanto riguarda la soluzione, il sistema di partenza Ax = b puó essere risolto risolvendo i due sistemi 
triangolari in ordine:     Ly = Pb     e     Ux = y
dove la matrice A nxn NON SINGOLARE é fattorizzabile con pivoting (PA = LU), con P matrice di permutazione,
L matrice triangolare inferiore con tutti 1 sulla diagonale e U triangolare superiore non singolare. 

'''




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

    # A = L * L^T
    L = scipy.linalg.cholesky(A, lower = True)
    
    # Per Cholesky si risolveono in ordine i sistemi Ly = b e L.Tx = y
    y = scipy.linalg.solve(L, b)
    my_x = scipy.linalg.solve(L.T, y)
    norm = scipy.linalg.norm(x - my_x, 2)
    print('Norma = ', norm)
    print('\n')
    
    return(condA, norm)

####### Testing #######

FIXED_STOP = 12

# Testing Hilbert
print('Caso Hilbert: -----------')
KA_Hilb = np.zeros((FIXED_STOP-2, 1)) # Inizializzazione degli array di condizionamento e errore rel
Err_Hilb = np.zeros((FIXED_STOP-2, 1))

for n in np.arange(2, FIXED_STOP): # Si verificano le condizioni dell'errore e del condizionamento al variare di n
                                    # Ovvero la dimensione della matrice.
    A_Hilb = scipy.linalg.hilbert(n)
    (KA_Hilb[n - 2], Err_Hilb[n - 2]) = es2(n, A_Hilb)
    
# Testing Matrice Tridiagonale
print('Caso Tridiagonale: ----------')
KA_TriD = np.zeros((FIXED_STOP-2, 1))
Err_TriD = np.zeros((FIXED_STOP-2, 1))

for n in np.arange(2, FIXED_STOP):
    A_TriD = np.diag(9 * np.ones(n)) + np.diag(-4 * np.ones(n-1), k = -1) + np.diag(-4 * np.ones(n-1), k = 1) # K e la posizione di cui si slitta rispetto alla diagonale
    (KA_TriD[n - 2], Err_TriD[n - 2]) = es2(n, A_TriD)
    

###### PLOTTING ######

# Disegna il grafico del numero di condizione in funzione della dimensione del sistema. Caso Hilbert.
points = FIXED_STOP - 2
dim_Matr_x = np.linspace(2, FIXED_STOP, points)
plt.semilogy(dim_Matr_x, KA_Hilb)      # Cambiare nel caso con plot al posto di semilogy
plt.title('HILBERT, Condizionamento')
plt.xlabel('Dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()

# Disegna il grafico dell'errore in norma 2 in funzione della dimensione del sistema. Caso Hilbert.
plt.semilogy(dim_Matr_x, Err_Hilb)
plt.title('HILBERT, Errore relativo')
plt.xlabel('Dimensione matrice: n')
plt.ylabel('Err = ||my_x-x|| / ||x||')
plt.show()

'''
Nel caso della fattorizzazione di Cholesky applicata su matrice di HILBERT, il numero di condizionamento
assume valori elevatissimi anche al passare dalla dimensione 2 alla dimensione 3 della matrice. Questo perché
la matrice di Hilbert é fortemente ill-conditioned: una minima variazione nei valori iniziali causa una grande
variazione nella soluzione finale, ció si applica anche all' errore relativo che come si vede dal grafico 
scala velocemente (peggiora) diversi ordini di grandezza.

'''

# Disegna il grafico del numero di condizione in funzione della dimensione del sistema. Caso Tridiagonale.
points = FIXED_STOP
plt.semilogy(dim_Matr_x, KA_TriD)
plt.title('TRIDIAGONALE, Condizionamento')
plt.xlabel('Dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()

# Disegna il grafico dell'errore in norma 2 in funzione della dimensione del sistema. Caso Tridiagonale.
plt.semilogy(dim_Matr_x, Err_TriD)
plt.title('TRIDIAGONALE, Errore relativo')
plt.xlabel('Dimensione matrice: n')
plt.ylabel('Err = ||my_x-x|| / ||x||')
plt.show()     

'''
Una matrice tridiagonale simmetrica e positiva invece é decisamente ben condizionata, in quanto tende ad essere
molto meno sensibile rispetto alle variazioni dei dati, sia dal punto di vista del condizionamento, che come
si vede dal grafico non cambia praticmamente nemmeno ordine di grandezza e l'errore relativo che ne cambia appena 1.'
In generale tutti i tipi di matrice tridiagonale sono considerate ben condizionate.
'''



    