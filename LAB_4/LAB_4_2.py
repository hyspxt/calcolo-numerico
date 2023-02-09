"""
Scrivere una funzione che implementi il metodo delle approssimazioni successive per il calcolo dello zero di una funzione f(x) per x âˆˆ R
n prendendo come input una funzione per l'aggiornamento:
- g1(x) = x - f(x)e^{x/2}
- g2(x) = x - f(x)e^{-x/2}
- g3(x) = x - f(x)/f'(x)

Testare il risolutore per risolvere f(x) = e^x - x^2
la cui soluzione esatta é = 0.7034674. In particolare:

    1. La funzione deve calcolare l' errore |x_k - x^* | ad ogni iterazione.
    2. Disegnare il grafico della funzione f nellâ€™intervallo I = [1, 1] e verificare che x_k sia lo zero di f in [âˆ’1, 1].
    3. Calcolare lo zero della funzione utilizzando tutte le funzioni precedentemente scritte.
    4. Confrontare l' accuratezza delle soluzioni trovate e il numero di iterazioni effettuate.
    5. Plottare l' errore al variare delle iterazioni per tutte le funzioni.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

''' Metodo delle approssimazioni successive'''
def succ_app(f, g, tolf, tolx, maxit, xTrue, x0=0):
  
  err=np.zeros(maxit+1, dtype=np.float64)          # Il metodo delle approssimazioni successive é simile al Newton dell'es precedente.
  vecErrore=np.zeros(maxit+1, dtype=np.float64)
  
  i = 0
  err[0]= tolx + 1
  vecErrore[0] = abs(x0 - xTrue)
  x = x0

  while (abs(f(x)) > tolf and i < maxit): 
    x_appr = g(x)          # x_{k+1} - g(x_k)
    err[i] = abs(x_appr - x)
    vecErrore[i] = abs(x - xTrue)
    i = i + 1
    x = x_appr
    
  err = err[0:i] 
  vecErrore = vecErrore[0:i] 
  return (x, i, err, vecErrore) 


'''creazione del problema'''
f = lambda x: np.exp(x) - x**2
df = lambda x: np.exp(x) - 2 * x

g1 = lambda x: x-f(x) * np.exp(x / 2)
g2 = lambda x: x-f(x) * np.exp(-x / 2)
g3 = lambda x: x-f(x) / df(x)

xTrue = -0.7034674
fTrue = f(xTrue)
print('fTrue = ', fTrue)

tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 0


''' Grafico funzione in [-1, 1]'''
x_plot = np.linspace(-1, 1, 101)
y_plot = f(x_plot)
plt.plot(x_plot, y_plot)
plt.plot(xTrue, fTrue, 'r*')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Funzione in [-1, 1]')
plt.show()

'''Calcolo soluzione cin g1, g2 e g3'''

(x1, i1, diffErr1, err1) = succ_app(f, g1, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g1 \n x =',x1,'\n iter_new=', i1)
print('\n')

(x2, i2, diffErr2, err2) = succ_app(f, g2, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g2 \n x =',x2,'\n iter_new=',i2)
print('\n')

(x3, i3, diffErr3, err3) = succ_app(f, g3, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g3 \n x =',x3,'\n iter_new=',i3)
print('\n')


''' Grafico Errore vs Iterazioni'''
# g1
iterazioni_g1 = np.arange(0, i1)
plt.plot(iterazioni_g1, err1)
plt.xlabel('Iterazioni')
plt.ylabel('Errore')
plt.title('Metodo delle approssimazioni successive con g1')
plt.show()

'''
Il metodo delle approssimazioni successive non é detto che converga a causa di una possibile scelta inadeguata
della stima iniziale o proprietá della funzione di punto fisso. 

In generale peró, il metodo si comporta abbastanza similmente al metodo di Newton. In questo caso sono richieste
un maggior numero di iterate a differenza del caso g3.
'''

# g2
iterazioni_g2 = np.arange(0, i2)    
plt.plot(iterazioni_g2, err2)
plt.xlabel('Iterazioni')
plt.ylabel('Errore')
plt.title('Metodo delle approssimazioni successive con g2')
plt.show()

'''
# Il grafico risulta in quel modo proprio perché la funzione non gode della proprietá di contrazione. 
Ovvero col procedere delle iterazioni, la distanza tra la stima attuale e la soluzione si riduca di un 
fattore costante compreso tra 0 e 1.

A tal proposito, dato che la funzione non gode di tale proprietá, il metodo non converge.
'''

# g3
iterazioni_g3 = np.arange(0, i3)
plt.plot(iterazioni_g3, err3)
plt.xlabel('Iterazioni')
plt.ylabel('Errore')
plt.title('Metodo delle approssimazioni successive con g3')
plt.show()