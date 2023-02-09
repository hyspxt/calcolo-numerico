"""
Scrivere una funzione che implementi il metodo del gradiente con step size a_k
variabile, calcolato secondo la procedura di backtracking ad ogni iterazione kesima.
Testare la function per minimizzare f(x) definita come: f(x) = 10(x - 1)^2 + (y -’ 2)^2
In particolare:
1. Plottare la superficie f(x) con plt.plot surface().
2. Plottare le curve di livello (plt.contour()) e le iterate calcolate dal metodo.
3. Plottare, al variare delle iterazioni, la funzione obiettivo, lâ€™errore e lanorma del gradiente.
"""

import numpy as np
import matplotlib.pyplot as plt

def next_step(x,grad): # backtracking procedure for the choice of the steplength
    alpha = 1.1
    rho = 0.5
    c1 = 0.25
    p = -grad
    j = 0
    jmax = 10    
    while (f(x + alpha*p) > f(x) + c1*alpha*grad.T@ p) and (j < jmax) :  # Si procede finché la condizione di Armijo Ã© rispettata.
        alpha = rho * alpha        # Alpha si dimezza
        j = j + 1
    if(j >= jmax):     # Se numero di iterazioni supera iterazioni_max
        return -1
    return alpha


def minimize(x0, x_true, mode, step, MAXITERATION, ABSOLUTE_STOP): 
    
    x=np.zeros((2,MAXITERATION)) 
    norm_grad_list=np.zeros((1,MAXITERATION)) # Norma gradienti -> ||grad f(x_k)|| k = 0,1..
    function_eval_list=np.zeros((1,MAXITERATION)) # ||f(x_k)|| 
    error_list=np.zeros((1,MAXITERATION)) # ||x_k - x_true||
    
    k = 0
    x_last  = np.array([x0[0],x0[1]])
    x[:,k] = x_last   # Il primo elemento del vettore x
    function_eval_list[:,k] = f(x0)  # Lista di valutazione dei valori della funzione nel punto x0
    error_list[:,k] = np.linalg.norm(x0-x_true)  # Lista di errori 
    norm_grad_list[:,k] = np.linalg.norm(grad_f(x0)) # Lista di numeri di condizionamento
     
    while (np.linalg.norm(grad_f(x_last))>ABSOLUTE_STOP and k < MAXITERATION ):
        
        k = k + 1  # Si aumenta subito l' iterazione poichÃ© abbiamo giÃ¡ riempito tutti i vettori nelle 
                   # prime posizioni.
                   
        grad = grad_f(x_last) # Si calcola il gradiente della funzione nel punto considerato. Nell' algoritmo 
                                # grad Ã© il valore p_k
        
        # backtracking step
        step = next_step(x_last, grad) # Si procede con l' algoritmo di backtracking, che restituisce
                                       # il nuovo valore alpha_k che rappresenta la lunghezza del passo.
        if(step==-1):
            print('non converge')
            return
    
        x_last = x_last - step * grad # Si fa la sottrazione perchÃ© l'algoritmo dei gradienti prevede che 
                                        # p_k = -grad f (x_k). step Ã© sempre la lunghezza del passo.
      
        x[:,k] = x_last  # L'elemento k del vettore Ã© l'ultimo calcolato
        function_eval_list[:,k] = f(x_last)  # Valore della funzione nel punto x_last
        error_list[:,k] = np.linalg.norm(x_last - x_true) # Errore
        norm_grad_list[:,k] = np.linalg.norm(grad_f(x_last)) # Condizionamento

    function_eval_list = function_eval_list[:, :k]  # Gli elementi vuoti (quelli fuori dal range (iterazioni
                                                    # - maxIter)) dei vettori vengono riempiti con i valori giÃ¡
                                                    # calcolati.
    error_list = error_list[:, :k]
    norm_grad_list = norm_grad_list[:, :k]
    
    print('Iterations = ',k)
    print('Last guess (ultima x calcolata): x=(%f,%f)'%(x[0,k-1],x[1,k-1]))

    if mode == 'plot_history': # Si determina se salvare x o meno. Viene fatto a causa del costo che puÃ³ eventualmente
                               # causare tale salvataggio a livello computazionale. 
        return (x_last,norm_grad_list, function_eval_list, error_list, k, x)
    else:
        return (x_last,norm_grad_list, function_eval_list, error_list, k)


'''creazione del problema'''

x_true = np.array([1,2])

def f(x): # X Ã¨ un vettore in R^2. Non Ã© necessario posizionare due variabili, come specificato in SUperfici.py
          # si puÃ³ usare il metodo meshgrid
    return 10*(x[0] - 1)**2 + (x[1] - 2 )**2 # La funzione f(x). x[0] rappresenta la x e x[1] la y.
 
def grad_f(x):
    return np.array([ 20*x[0] - 20 , 2*x[1] - 4]) # Gradiente di f(x)

step=0.1
MAXITERATIONS=1000
ABSOLUTE_STOP=1.e-5
mode='plot_history'
x0 = np.array((3,-5))


(x_last, norm_grad_list, function_eval_list, error_list, k, x) = minimize(x0, x_true, mode, step,MAXITERATIONS,ABSOLUTE_STOP)

v_x0 = np.linspace(-5,5,500)
v_x1 = np.linspace(-5,5,500)
xv = [x0v,x1v] = np.meshgrid(v_x0, v_x1) 
z = f(xv)
   
'''superficie'''
plt.figure()

ax = plt.axes(projection='3d')
ax.plot_surface(x0v, x1v, z, cmap='ocean')
ax.set_title('Surface plot')
plt.show()

'''contour plots'''
if mode=='plot_history':
   contours = plt.contour(x0v, x1v, z, levels = 40) # Si plottano le curve di livello di livello.
   plt.plot(x[0], x[1], 'o', markersize = 3) # Plottiamo le x soluzioni
   plt.plot(x_true[0], x_true[1], 'x' ) # Soluzioni ottime
   plt.title('Curve di livello + iterate')

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
Il valore dell'errore decresce poiché utilizziamo un metodo iterativo per arrivare alla soluzione, minimizzando
la funzione.

Stessa cosa accade per il valore della funzione obiettivo, che é proprio in fondo ció che vogliamo minimizzare
quindi é corretto che decresca.

La norma del gradiente di f segue di pari passo ció che é scritto subito sopra.

'''








