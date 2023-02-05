import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

"""
Esercizio 3. Per ognuna delle seguenti funzioni:
    - f(x) = x exp(x) con x -> [-1, 1]
    
    - f(x) = 1 / (1 + 25 * x) con x -> [-1, 1]
    
    - f(x) = sin(5x) + 3x con x -> [1, 5]
    
Assegnati m punti equispaziati, con m fissato.
1) Per ciascun valore di n {1, 2, 3, 5, 7} creare una figura con il grafico della funzione esatta f(x) 
    insieme a quello del polinomio approssimante p(x). Evidenziare gli m punti noti.

2) Per ciascun valore di n {1, 2, 3, 5, 7} riportare il valore dell' errore commesso nel punto x = 0
    
3) Calcolare la norma 2 dell' errore di approssimazione, commesso sugli m nodi, per ciascun valore di {1, 5, 7}.
"""

m =  50 # Il numero di punti é fissato. Si puó aumentare in modo da visualizzare meglio la funzione, in particolare la seconda.
        # Maggior numero di punti = piú precisione del polinomio approssimante.
        # Con 50 punti si vedono correttamente le funzioni, ma la seconda risulta visivamente fuorviante.


# Funzione per valutare il polinomio p, in un punto x, dato il vettore dei coefficienti alpha
def p(alpha, x):
  A = np.zeros((len(x), len(alpha)))
  for i in range(len(alpha)):
      A[:, i] = x**i
  y = np.dot(A, alpha)
  return y

def approxPoly_neq(nDati, gradoP, x, y):
    """
    Il polinomio approssimante viene calcolato mediante il metodo delle equazioni normali.
    """
    A = np.zeros((nDati, gradoP + 1))
    for i in range(gradoP+1):
        A[:, i] = x**i
    
    ATA = A.T @ A
    ATy = A.T @ y
    # Si decompone con Cholesky
    L = scipy.linalg.cholesky(ATA, lower = True)
    alpha1 = scipy.linalg.solve(L, ATy, lower = True)
    alpha_normali = scipy.linalg.solve(L.T, alpha1)
    return alpha_normali
    
def approxPoly_svd(nDati, gradoP, x, y):
    """
    Il polinomio approssimante viene calcolato mediante il metodo SVD.
    """
    A = np.zeros((nDati, gradoP + 1))
    for i in range(gradoP+1):
        A[:, i] = x**i
    
    U, s, VT = scipy.linalg.svd(A)
    alpha_svd = np.zeros(s.shape)

    for j in range(gradoP+1):
      uj = U[:, j]
      vj = VT[j, :]
      alpha_svd = alpha_svd + (np.dot(uj, y) * vj) / s[j]      
    return alpha_svd

    
def polyPlot(f, start, end):
    
    x_plot = np.linspace(start, end, m)
    y_plot = f(x_plot)       # Lo spazio di plotting della y dipende ovviamente dalla funzione che si considera.
    
    plt.figure()
    plt.plot(x_plot, y_plot, 'red', label = 'function f', linestyle = '--')
    plt.title("Function")
    
    for n, color in [(1, 'blue'),(2, 'green'),(3, 'purple'),(5, 'pink'),(7, 'orange')]:
        
        alpha = approxPoly_neq(m, n, x_plot, y_plot)
        y_p = p(alpha, x_plot)
        
        ''' 
        1) Per ciascun valore di n {1, 2, 3, 5, 7} creare una figura con il grafico della funzione esatta f(x) 
           insieme a quello del polinomio approssimante p(x). Evidenziare gli m punti noti.
        '''
        plt.plot(x_plot, y_p, color, label = f'polinomio grado {n}', marker = '.') # Per vedere meglio la funzione volendo si puó 
                                                                                   # togliere il marker.     
        
        ''' 
        2) Per ciascun valore di n {1, 2, 3, 5, 7} riportare il valore dell' errore commesso nel punto x = 0
        '''
        
        err = f(0) - p(alpha, np.array([0])) # Vettore contente un solo valore, cioé l' unico punto x = 0.
        print(f'Polinomio di grado {n} -> errore {err}', '\n')
        
        ''' 
        3) Calcolare la norma 2 dell' errore di approssimazione, commesso sugli m nodi, per ciascun valore di {1, 5, 7}.
        '''
        for i in [1, 5, 7]:
            if i == n:   # Non c'é da valutare la norma per tutti i gradi del polinomio
                         # Quindi la si calcola solo se l' iterazione é uguale al grado 
                norm = np.linalg.norm(y_plot - y_p)  # Norma = norm(sol.esatta - sol)
                print (f'Polinomio di grado {n} -> norma {norm}', '\n')
        plt.legend()
    plt.show()
    
f1 = lambda x: x * np.exp(x)
f2 = lambda x: 1 / (1 + (25 * x))
f3 = lambda x: np.sin(5 * x) + 3 * x

print("Funzione 1: ")
polyPlot(f1, -1, 1)

print("Funzione 2: ")
polyPlot(f2, -1, 1)
print("Funzione 3: ")
polyPlot(f3, 1, 5)

# f = lambda x: ...
# x = np.linspace(1, 5, N)
# y = f(x)