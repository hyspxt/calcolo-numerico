"""
Confrontare e commentare le prestazioni dei tre metodi con le seguenti funzioni
• f(x) = x^3 + 4xcos(x) − 2  nell’intervallo [0, 2], con g(x) = (2−x^3) / 4∗cos(x)
• f(x) = x − x^{1/3} − 2     nell’intervallo [3, 5], con g(x) = x^{1/3} + 2
Suggerimento: confronta il numero di iterazioni, i tempi di esecuzione e i risultati ottenuti. 
Analizza la dipendenza dai parametri, dagli intervalli o dalle funzioni.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import time

''' Metodo di Bisezione'''
def bisezione(a, b, f, tolx, xTrue):
    
  k = math.ceil(math.log((b-a)/tolx, 2))     # Numero minimo di iterazioni per avere un errore minore di tolX
                                           # Per ottenerlo, la formula é k => log_2((b-a)/epsilon)
                                           # dove epsilon é la tolleranza ammessa (quindi tolx)
  vecErrore = np.zeros( (k,1) )
  
  if f(a)*f(b) > 0:    # La funzione rimane sempre positiva o negativa -> non ci sono zeri di funzione.
    print("La funzione non cambia segno in questo intervallo.")
    return ('Nessun punto stazionario', 0, 0, vecErrore)
    
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

def newton( f, df, tolf, tolx, maxit, xTrue, x0 = 0.5):
  
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


''' Metodo delle approssimazioni successive'''
def succ_app(f, g, tolf, tolx, maxit, xTrue, x0 = 0.5):
  
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

###### Confronto dei tre metodi ######
## Creazione del problema ##

tolx = 10**(-10)  # Dati uguali per tutti i problemi
tolf = 10**(-6)
maxit = 100
x0 = 0.5        # Qui si puó mettere ció che si vuole, ma 0.5 velocizza un po' le iterazioni

xTrue = 0.5368384275286979   # La radice esatta é calcolata simbolicamente in base ai risultati di uno dei tre metodi.
xTrue2 = 3.521379706915468

"""
Funzione 1:
"""
f1 = lambda x: x ** 3 + 4 * x * np.cos(x) - 2
df1 = lambda x: 3 * (x ** 2 ) + 4 * np.cos(x) - 4 * x * np.sin(x)
g1 = lambda x: (2 - x ** 3) / (4 * np.cos(x))

# Plotting funzione 1
a_f1 = 0
b_f1 = 2

x_plot = np.linspace(a_f1, b_f1, 100)
y_plot = f1(x_plot)
plt.plot(x_plot, y_plot)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Funzione f1 in [0, 2]')
plt.show()
    
# Confronto metodi funzione 1

elapsedTime_f1 = np.zeros(3)

st = time.time()
(x_bi, i_bi, k_bi, err_bi) = bisezione(a_f1, b_f1, f1, tolx, xTrue)
et = time.time()
elapsedTime_f1[0] = et - st 
   

st = time.time()
(x_ne, i_ne, errDiff_ne, err_ne) = newton(f1, df1, tolf, tolx, maxit, xTrue) 
et = time.time()
elapsedTime_f1[1] = et - st      

st = time.time()
(x_sa, i_sa, errDiff_sa, err_sa) = succ_app(f1, g1, tolf, tolx, maxit, xTrue, x0)
et = time.time()
elapsedTime_f1[2] = et - st      

###### Plotting Dati Funzione 1 ######

iterazioni_bisezione = np.arange(0, i_bi + 1)
iterazioni_newton = np.arange(0, i_ne)
iterazioni_succappr = np.arange(0, i_sa)

### Bisezione ###
plt.plot(iterazioni_bisezione, err_bi)
plt.xlabel('Numero di iterazioni')
plt.ylabel('Errore')
plt.title('Bisezione method on f1')
plt.show()

### Newton ###
plt.plot(iterazioni_newton, err_ne)
plt.xlabel('Numero di iterazioni on f1')
plt.ylabel('Errore')
plt.title('Newton method')
plt.show()

### Approssimazioni successive ###
plt.plot(iterazioni_succappr, err_sa)
plt.xlabel('Numero di iterazioni')
plt.ylabel('Errore')
plt.title('Approssimazioni successive method on f1')
plt.show()

# Confronto prestazioni, numero di iterazioni e risultati
print('Metodo di Bisezione \n x = ', x_bi, '\n iter_bisezion = ', i_bi, '\n iter_max = ', k_bi, '\n Tempo di esecuzione = ', elapsedTime_f1[0], 'seconds')
print('\n')
print('Metodo di Newton \n x =',x_ne,'\n iter_new = ', i_ne, '\n err_new = ', err_ne , '\n Tempo di esecuzione = ', elapsedTime_f1[1], 'seconds')
print('\n')
print('Metodo approssimazioni successive \n x =',x_sa,'\n iter_new=', i_sa, '\n err_sa = ', err_sa , '\n Tempo di esecuzione = ', elapsedTime_f1[2], 'seconds')
print('\n')


"""
Funzione 2:
"""

f2 = lambda x: x - np.power(x, 1/3) - 2
df2 = lambda x: 1 - (1/3) * np.power(x, -2/3)
g2 = lambda x: x ** (1 / 3) + 2

# Plotting funzione 1
a_f2 = 3
b_f2 = 5

x_plot2 = np.linspace(a_f2, b_f2, 100)
y_plot2 = f2(x_plot2)
plt.plot(x_plot2, y_plot2, 'g')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Funzione f2 in [3, 5]')
plt.show()
    
# Confronto metodi funzione 1

elapsedTime_f2 = np.zeros(3)

st = time.time()
(x_bi2, i_bi2, k_bi2, err_bi2) = bisezione(a_f2, b_f2, f2, tolx, xTrue2)
et = time.time()
elapsedTime_f2[0] = et - st 
   

st = time.time()
(x_ne2, i_ne2, errDiff_ne2, err_ne2) = newton(f2, df2, tolf, tolx, maxit, xTrue2) 
et = time.time()
elapsedTime_f2[1] = et - st      

st = time.time()
(x_sa2, i_sa2, errDiff_sa2, err_sa2) = succ_app(f2, g2, tolf, tolx, maxit, xTrue2, x0)
et = time.time()
elapsedTime_f2[2] = et - st      

###### Plotting Dati Funzione 1 ######

iterazioni_bisezione2 = np.arange(0, i_bi2 + 3)
iterazioni_newton2 = np.arange(0, i_ne2)
iterazioni_succappr2 = np.arange(0, i_sa2)

### Bisezione ###
plt.plot(iterazioni_bisezione2, err_bi2, 'g')
plt.xlabel('Numero di iterazioni')
plt.ylabel('Errore')
plt.title('Bisezione method on f2')
plt.show()

### Newton ###
plt.plot(iterazioni_newton2, err_ne2, 'g')
plt.xlabel('Numero di iterazioni on f2')
plt.ylabel('Errore')
plt.title('Newton method')
plt.show()

### Approssimazioni successive ###
plt.plot(iterazioni_succappr2, err_sa2, 'g')
plt.xlabel('Numero di iterazioni')
plt.ylabel('Errore')
plt.title('Approssimazioni successive method on f2')
plt.show()

# Confronto prestazioni, numero di iterazioni e risultati
print('Metodo di Bisezione \n x = ', x_bi2, '\n iter_bisezion = ', i_bi2, '\n iter_max = ', k_bi2, '\n Tempo di esecuzione = ', elapsedTime_f2[0], 'seconds')
print('\n')
print('Metodo di Newton \n x =',x_ne2,'\n iter_new = ', i_ne2, '\n err_new = ', err_ne2 , '\n Tempo di esecuzione = ', elapsedTime_f2[1], 'seconds')
print('\n')
print('Metodo approssimazioni successive \n x =',x_sa2,'\n iter_new=', i_sa2, '\n err_sa = ', err_sa2 , '\n Tempo di esecuzione = ', elapsedTime_f2[2], 'seconds')
print('\n')


