import numpy as np
import math
import matplotlib.pyplot as plt

### CALCOLARE LO ZERO DI UNA FUNZIONE ###
"""
Scrivere una funzione che implementi il metodo di bisezione e una funzione per il metodo di Newton per il 
calcolo dello zero di una funzione f(x) per x R^n. Testare entrambi i risolutori per risolvere:

    - f(x) = e^x - x^2
    
la cui soluzione é x^* = -0.7034674. In particolare:
    1) Le due funzioni devono calcolare l'errore |x_k - x^*| ad ogni iterazione.
    2) Disegnare il grafico di f nell' intervallo I = [-1, 1] e verificare che x^* sia lo zero di f in [-1, 1]
    3) Calcolare lo zero della funzione utilizzando entrambe le funzioni precedentemente scritte.
    4) Confrontare l'accuratezza delle soluzioni trovate e il numero di iterazioni effettuate dai solutori.
    5) Plottare l'errore al variare delle iterazioni per entrambi i metodi.
    
"""


''' Metodo di Bisezione'''
def bisezione(a, b, f, tolx, xTrue):
    
  k = math.ceil(math.log((b-a)/tolx, 2))     # Numero minimo di iterazioni per avere un errore minore di tolX
                                           # Per ottenerlo, la formula é k => log_2((b-a)/epsilon)
                                           # dove epsilon é la tolleranza ammessa (quindi tolx)
  vecErrore = np.zeros( (k,1) )
  
  if f(a)*f(b) > 0:    # La funzione rimane sempre positiva o negativa -> non ci sono zeri di funzione.
    print("La funzione non cambia segno in questo intervallo.")
    return None
    
  for i in range(1,k):  # Metodo di bisezione
      
      if (b - a) < tolx:       # L' intervallo é piú piccolo della tolleranza.
        print('Errore: l\'intervallo è troppo piccolo ')
        return ('Errore', i, k, vecErrore)
     
      c = (a + b)/2   
      vecErrore[i-1] = abs(c - xTrue) # Si compone il vettore degli errori a partire dalla richiesta 1 
                                      # L'indice é i - 1 poiché le iterazioni partono da 1 e non da 0.
                                      # Si vede bene nel grafico il perché di questa cosa, provando a metterlo a i.
     
      if abs(f(c)) < tolx :           # Se f(c) è molto vicino a 0, ma comunque inferiore della tolleranza 
          break                       # diamo per buono c e ci siamo assicurati che converga.
      else:
        if f(c) > 0:     # Il nuovo intervallo sará [a, c]
          b = c
        else:
          a = c          # Il nuovo intervallo sará [c, b]
          
  return (c, i, k, vecErrore)

      
''' Metodo di Newton'''

def newton( f, df, tolf, tolx, maxit, xTrue, x0=0):
  
  err=np.zeros(maxit, dtype=float)              # Si tratta di due differenti tipi di errore. Il primo é la differenza tra
  vecErrore=np.zeros( (maxit,1), dtype=float)   # |x_{k + 1} - x_k|, mentre il secondo é il medesimo vecErr descritto nel metodo precedente. 
  
  i = 0
  err[0] = tolx + 1     # L'errore iniziale é dato dalla tolleranza ammessa + 1
  vecErrore[0] = np.abs(x0-xTrue)
  x = x0                # Valore della prima iterazione = 0, o stima iniziale

  while ( abs(f(x)) > tolf and i < maxit): 
      
    x_newton = x - f(x) / df(x)
    err[i] = abs(x_newton - x)
    vecErrore[i] = abs(x - xTrue)
    
    i = i + 1
    x = x_newton        # Ripetiamo l'esecuzione ponendo x come il valore di x_newton calcolato in questa iterazione.
    
    
  err = err[0:i]                # Gi errori si aggiornano ogni volta man mano che si procede nelle iterazioni.
  vecErrore = vecErrore[0:i]
  return (x, i, err, vecErrore)  


'''creazione del problema'''
f = lambda x: np.exp(x) - x**2 
df = lambda x: np.exp(x) - 2 * x 
xTrue = -0.7034674
fTrue = f(xTrue)
print (fTrue)

a=-1.0
b=1.0
tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 0

''' Grafico funzione in [a, b]'''
x_plot = np.linspace(a, b, 101)
f_plot = f(x_plot)

plt.plot(x_plot, f_plot)
plt.plot(xTrue, fTrue, 'r*')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.title('function')
plt.show()

''' Calcolo soluzione tramite Bisezione e Newton'''
(x_bi, i_bi, k_bi, err_bi) = bisezione(a, b, f, tolx, xTrue)
print('Metodo di bisezione \n x =',x_bi,'\n iter_bise=', i_bi + 1, '\n iter_max=', k_bi)
print('\n')


(x_ne, i_ne, errDiff_ne, err_ne) = newton(f, df, tolf, tolx, maxit, xTrue) 
print('Metodo di Newton \n x =',x_ne,'\n iter_new=', i_ne, '\n err_new=', err_ne)
print('\n')


''' Grafico Errore vs Iterazioni'''
iterazioni_bisezione = np.arange(0, i_bi + 1)
plt.plot(iterazioni_bisezione, err_bi) # Cambiare nel caso con semilogy
plt.xlabel('Iterazioni')
plt.ylabel('Errore')
plt.title('Bisezione method')
plt.show()


iterazioni_newton = np.arange(0, i_ne)
plt.plot(iterazioni_newton, err_ne)
plt.xlabel('Iterazioni')
plt.ylabel('Errore')
plt.title('Newton method')
plt.show()


'''
Per quanto riguarda il metodo di bisezione, esso richiede un numero decisamente maggiore di iterate rispetto al 
metodo di Newton, che ha convergenza quadratica.
Oltre a questo, l'errore relativo nel metodo di bisezione descresce ad un ritmo bene o male costante, 
in generale dimezzandosi ad ogni iterazione, sebbene ci siano alcune punte che ne indicano l' aumento: ció é
raro ma puó accadere in casi di problemi di precisione numerica o di approssimazione dei valori intermedi.
    
In Newton invece l' errore relativo diminuisce molto rapidamente nelle prima iterazioni, ma puó oscillare se la
prima stima é molto lontana dalla soluzione. In generale peró, decresce molto piú velocemente rispetto al metodo
di bisezione.
'''
