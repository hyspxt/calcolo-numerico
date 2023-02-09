"""
Minimizzare la funzione che implementi il metodo del gradiente con step size variabile
 f(x) definita come: f(x) = || x - b ||^2_2 + lambda || x ||^2_2
dove:
    - x, b sono vettori di R^n
    
    - b = (1,...,1)
    
    - lambda > 0 , lambda vettore di R
In particolare, per n fissato:
1. Testare differenti valori di lambda.
2. Plottare, al variare delle iterazioni, la funzione obiettivo, l’errore e la norma del gradiente.
"""

import numpy as np
import matplotlib.pyplot as plt

def next_step(x,grad, lambd): # backtracking procedure for the choice of the steplength
    alpha = 1.1
    rho = 0.5
    c1 = 0.25
    p = -grad
    j = 0
    jmax = 10    
    while (f(x + alpha*p, lambd) > f(x, lambd) + c1*alpha*grad.T@ p) and (j < jmax) :  # Si procede finché la condizione di Armijo é rispettata.
        alpha = rho * alpha        # Alpha si dimezza
        j = j + 1
    if(j >= jmax):     # Se numero di iterazioni supera iterazioni_max
        return -1
    return alpha


def minimize(x0, x_true, lambd, mode, step, MAXITERATION, ABSOLUTE_STOP): 
    
    x=np.zeros((2,MAXITERATION)) 
    norm_grad_list=np.zeros((1,MAXITERATION)) # Norma gradienti -> ||grad f(x_k)|| k = 0,1..
    function_eval_list=np.zeros((1,MAXITERATION)) # ||f(x_k)|| 
    error_list=np.zeros((1,MAXITERATION)) # ||x_k - x_true||
    
    k = 0
    x_last  = np.array([x0[0],x0[1]])
    x[:,k] = x_last   # Il primo elemento del vettore x
    function_eval_list[:,k] = f(x0, lambd)  # Lista di valutazione dei valori della funzione nel punto x0
    error_list[:,k] = np.linalg.norm(x0-x_true)  # Lista di errori 
    norm_grad_list[:,k] = np.linalg.norm(grad_f(x0, lambd)) # Lista di numeri di condizionamento
     
    while (np.linalg.norm(grad_f(x_last, lambd))>ABSOLUTE_STOP and k < MAXITERATION ):
        
        k = k + 1  # Si aumenta subito l' iterazione poiché abbiamo giá riempito tutti i vettori nelle 
                   # prime posizioni.
                   
        grad = grad_f(x_last, lambd) # Si calcola il gradiente della funzione nel punto considerato. Nell' algoritmo 
                                # grad é il valore p_k
        
        # backtracking step
        step = next_step(x_last, grad, lambd) # Si procede con l' algoritmo di backtracking, che restituisce
                                       # il nuovo valore alpha_k che rappresenta la lunghezza del passo.
        if(step==-1):
            print('non converge')
            return
    
        x_last = x_last - step * grad # Si fa la sottrazione perché l'algoritmo dei gradienti prevede che 
                                        # p_k = -grad f (x_k). step é sempre la lunghezza del passo.
      
        x[:,k] = x_last  # L'elemento k del vettore é l'ultimo calcolato
        function_eval_list[:,k] = f(x_last, lambd)  # Valore della funzione nel punto x_last
        error_list[:,k] = np.linalg.norm(x_last - x_true) # Errore
        norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last, lambd)) # Condizionamento

    function_eval_list = function_eval_list[:, :k]  # Gli elementi vuoti (quelli fuori dal range (iterazioni
                                                    # - maxIter)) dei vettori vengono riempiti con i valori giá
                                                    # calcolati.
    error_list = error_list[:, :k]
    norm_grad_list = norm_grad_list[:, :k]
    
    print('Iterations = ',k)
    print('Last guess (ultima x calcolata): x=(%f,%f)'%(x[0,k-1],x[1,k-1]))

    if mode == 'plot_history': # Si determina se salvare x o meno. Viene fatto a causa del costo che puó eventualmente
                               # causare tale salvataggio a livello computazionale. 
        return (x_last,norm_grad_list, function_eval_list, error_list, k, x)
    else:
        return (x_last,norm_grad_list, function_eval_list, error_list, k)
    
    


def f(x, lambd): # X è un vettore in R^2. Non é necessario posizionare due variabili, come specificato in SUperfici.py
          # si puó usare il metodo meshgrid
    b = np.ones( (n, ) )    
    return np.linalg.norm(x - b) ** 2 + lambd * np.linalg.norm(x) ** 2  # La funzione f(x). x[0] rappresenta la x e x[1] la y.
 
def grad_f(x, lambd):
    b = np.ones( (n, ) )    
    return 2 * (x - b) + 2 * lambd * x # Gradiente di f(x)

step=0.1
MAXITERATIONS=1000
ABSOLUTE_STOP=1.e-5

x0 = np.array((3,2))
n = 2  # Dimensione fissata dello spazio vettoriale
x_true = np.array([0,0])
mode = 'plot_history'
b = np.ones( (n, ) )    # b é un vettore di soli uni delle dimensione

lambd = 0.3  # Lambda é uno scalare e al suo variare aumentano / diminuiscono i punti 

(x_last, norm_grad_list, function_eval_list, error_list, k, x) = minimize(x0, x_true,lambd, mode, step,MAXITERATIONS,ABSOLUTE_STOP)

'''plots'''
x_iter = np.arange(1, k+1).reshape(norm_grad_list.shape) # Si fa il reshape delle iterazioni per poterle plottare

#Errore vs Iterazioni
plt.figure()
plt.plot(x_iter, error_list, color='red', marker='.', markersize=2)
plt.xlabel('Iterazioni');
plt.ylabel('Errore');
plt.title('Iterazioni vs Errore di f')

#Iterazioni vs Funzione Obiettivo
plt.figure()
plt.plot(x_iter, function_eval_list, color='green', marker='.', markersize=2)
plt.xlabel('Iterazioni')
plt.ylabel('funzione obbiettivo')
plt.title('Iterazioni vs Funzione Obiettivo')

# Iterazioni vs Norma Gradiente
plt.figure()
plt.plot(x_iter, norm_grad_list, color='blue', marker='.', markersize=2)
plt.xlabel('Iterazioni')
plt.ylabel('Norma Gradiente')
plt.title('Iterazioni vs Norma Gradiente di f')


'''
Al variare di lambda, variano anche il numero di iterazioni (e quindi i punti).
Non é una cosa regolare e aumentare lambda non garantisce che aumentano le iterazioni.
Personalmente, per una maggior precisione della funzione minimizzata (piú punti) il valore é = 0.3
(nell' intervallo 0.1')

Lambda é il fattore di penalizzazione applicato alla soluzione.

'''





