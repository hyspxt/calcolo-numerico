import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data, metrics
from scipy import signal
from numpy import fft
from scipy.optimize import minimize


"""
Degradare due nuove immagini applicando, mediante le funzioni gaussian kernel()
psf fft(), l’operatore di blur con parametri:
• σ = 0.5 dimensione 5 × 5
• σ = 1 dimensione 7 × 7
• σ = 1.3 dimensione 9 × 9
ed aggiungendo rumore gaussiano con deviazione standard (0, 0.05].
i. Ripetere gli esercizi 2 e 3 con le nuove immagini.
ii. Ripetere gli esercizi 2 e 3 sostituendo il metodo del gradiente coniugato
con il metodo del gradiente da voi implementato nello scorso laboratorio.
"""

def next_step(x, grad, f0): # backtracking procedure for the choice of the steplength
    alpha = 1.1
    rho = 0.5
    c1 = 0.25
    p = -grad
    j = 0
    jmax = 100 # In origine era = 10, aumentare per non incorrere nella non convergenza
    while (f0(x + alpha*p) > f0(x) + c1*alpha*grad.T@ p) and (j < jmax) :  # Si procede finché la condizione di Armijo é rispettata.
        alpha = rho * alpha        # Alpha si dimezza
        j = j + 1
    if(j >= jmax):     # Se numero di iterazioni supera iterazioni_max
        return -1
    return alpha
def CG_minimize(f0, grad_f0, x0, x_true , step, MAXITERATION, ABSOLUTE_STOP): 
    
    k = 0

    x_last_matrix = np.copy(x0) # Qui, dato che operiamo sull' immagine ci affidiamo alla matrice
    x_last = np.reshape(x_last_matrix, x0.shape[0] * x0.shape[1])
    
    PSNR_vector = np.zeros((MAXITERATION))
    MSE_vector = np.zeros((MAXITERATION))
    
    while (np.linalg.norm(grad_f0(x_last))>ABSOLUTE_STOP and k < MAXITERATION ):
                    
        grad = grad_f0(x_last) # Si calcola il gradiente della funzione nel punto considerato. Nell' algoritmo 
                                # grad é il valore p_k
        
        # backtracking step
        step = next_step(x_last, grad, f0) # Si procede con l' algoritmo di backtracking, che restituisce
                                       # il nuovo valore alpha_k che rappresenta la lunghezza del passo.
        
        if( step==-1 ):
            print('non converge')
            return
    
        x_last = x_last - step * grad # Si fa la sottrazione perché l'algoritmo dei gradienti prevede che 
                                        # p_k = -grad f (x_k). step é sempre la lunghezza del passo.
      
        x_last_matrix = np.reshape(x_last, x0.shape)
        PSNR_vector[k] = metrics.peak_signal_noise_ratio(x_true, x_last_matrix)
        MSE_vector[k] = metrics.mean_squared_error(x_true, x_last_matrix)
        #print(f'iterazione k = {k}, abbiamo PNSR = {iter_PSNR[k]} e MSE = {iter_MSE[k]}')
        
        k = k + 1  # Si aumenta subito l' iterazione poiché abbiamo giá riempito tutti i vettori nelle 
                   # prime posizioni.
    
    print('Iterations = ',k)

    PSNR_vector = PSNR_vector[:k] # Prendiamo l' ultimo elemento dei vettori, ovvero quello corrispondente alla ultima iterazione.
    MSE_vector  = MSE_vector[:k]    

    return (x_last, k, PSNR_vector, MSE_vector)    
    
np.random.seed(0)

# Crea un kernel Gaussiano di dimensione kernlen e deviazione standard sigma
def gaussian_kernel(kernlen, sigma):
    x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)    
    # Kernel gaussiano unidmensionale
    kern1d = np.exp(- 0.5 * (x**2 / sigma))
    # Kernel gaussiano bidimensionale
    kern2d = np.outer(kern1d, kern1d)
    # Normalizzazione
    return kern2d / kern2d.sum()

# Esegui l'fft del kernel K di dimensione d agggiungendo gli zeri necessari 
# ad arrivare a dimensione shape
def psf_fft(K, d, shape):
    # Aggiungi zeri
    K_p = np.zeros(shape)
    K_p[:d, :d] = K

    # Sposta elementi
    p = d // 2
    K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)

    # Esegui FFT
    K_otf = fft.fft2(K_pr)
    return K_otf

# Moltiplicazione per A
def A(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(K * x))

# Moltiplicazione per A trasposta
def AT(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(np.conj(K) * x))

image = data.moon() # Immagine 1
# plt.imshow(image)

image2 = data.horse() # Immagine 2
# plt.imshow(image2)

normalized_image = skimage.img_as_float(image) # La normalizziamo nell' intervallo [0, 1]. In questo modo
                                               # garantiamo che i pixel siano rappresentati come float.
normalized_image2 = skimage.img_as_float(image2)                                               
                                            
m, n = normalized_image.shape
o, p = normalized_image2.shape

""" 
I parametri sono: 
    - Kernel 5x5 con sigma = 0.5
    - 7x7 con sigma = 1
    0 9x9 con sigma = 1.3
"""
# Creazione del blur gaussiano
K = gaussian_kernel(5, 0.5)
K = psf_fft(K, 5, normalized_image.shape)

K2 = gaussian_kernel(7, 1)
K2 = psf_fft(K2, 7, normalized_image2.shape)

# Generazione del noise gaussiano
sigma =  0.02
noise = np.random.normal(0, sigma, size = normalized_image.shape)
noise2 = np.random.normal(0, sigma, size = normalized_image2.shape)

# Applicazione di blur e noise
b = A(normalized_image, K) + noise
b2 = A(normalized_image2, K2) + noise2

PSNR = metrics.peak_signal_noise_ratio(normalized_image, b)  # Si calcolano PSNR e MSE ( + é alto il PSNR + l'immagine é buona)
MSE = metrics.mean_squared_error(normalized_image, b)
print('PSNR Prima immagine = ', PSNR)
print('MSE Prima immagine = ', MSE)

PSNR2 = metrics.peak_signal_noise_ratio(normalized_image2, b2)  # Si calcolano PSNR e MSE ( + é alto il PSNR + l'immagine é buona)
MSE2 = metrics.mean_squared_error(normalized_image2, b2)
print('PSNR Seconda immagine = ', PSNR2)
print('MSE Seconda immagine = ', MSE2)

##########

plt.figure(figsize = (20,10))     # Plotting dell' immagine originale
axis1 = plt.subplot(1, 2, 1)
axis1.imshow(normalized_image, cmap = 'gray')  # In scala di grigi si vedono meglio le sfocature
plt.title('Immagine 1 originale', fontsize = 18)

axis2 = plt.subplot(1, 2, 2)     # Plotting dell'immagine corrotta
axis2.imshow(b, cmap = 'gray')
plt.title(f'Immagine 1 degradata (PSNR: {PSNR: .2f}, MSE: {MSE: .5f})', fontsize=20)
plt.show()    
    
plt.figure(figsize = (20,10))     # Plotting dell' immagine originale
axis1 = plt.subplot(1, 2, 1)
axis1.imshow(normalized_image2, cmap = 'gray')  # In scala di grigi si vedono meglio le sfocature
plt.title('Immagine 2 originale', fontsize = 18)

axis2 = plt.subplot(1, 2, 2)     # Plotting dell'immagine corrotta
axis2.imshow(b2, cmap = 'gray')
plt.title(f'Immagine 2 degradata (PSNR: {PSNR2: .2f}, MSE: {MSE2: .5f})', fontsize=20)
plt.show()    
 
#######
    
def f(x): # 1/2 || Ax -b ||^2
    x_resized = np.reshape(x, (m, n)) # Viene effettuato il reshape della soluzione x per evitare gli errori con il kernel dell' immagine (la matrice).
    result = (0.5) * ( np.sum( np.square( (A(x_resized, K) - b ) )))
    return result

def grad_f(x):  # Gradiente della funzione, che in quanto ha una sola variabile, corrisponde alla derivata (vettore unielemento).
    x_resized = np.reshape(x, (m, n))
    result = AT(A(x_resized, K), K) -AT(b, K)
    result = np.reshape(result, m*n)   # Rispettivamente larghezza e e altezza dell' immagine. Cosí il gradiente fa tutta la matrice.
    return result
       
def f2(x): # 1/2 || Ax -b ||^2
    x_resized = np.reshape(x, (o, p)) # Viene effettuato il reshape della soluzione x per evitare gli errori con il kernel dell' immagine (la matrice).
    result = (0.5) * ( np.sum( np.square( (A(x_resized, K2) - b2 ) )))
    return result

def grad_f2(x):  # Gradiente della funzione, che in quanto ha una sola variabile, corrisponde alla derivata (vettore unielemento).
    x_resized = np.reshape(x, (o, p))
    result = AT(A(x_resized, K2), K2) -AT(b2, K2)
    result = np.reshape(result, o*p)   # Rispettivamente larghezza e e altezza dell' immagine. Cosí il gradiente fa tutta la matrice.
    return result   

######## SOLUZIONE NAIVE

step = 0.1
x0 = b  # Immagine da ricostruire
x2 = b2

ABSOLUTE_STOP = 1.e-4

deblurredPSNR = np.zeros((5, ))    # Cambiare 5 se si vogliono ottenere un numero di valori di PSNR e 
deblurredMSE = np.zeros((len(deblurredPSNR), ))       # MSE differenti.

deblurredPSNR2 = np.zeros((5, ))    # Cambiare 5 se si vogliono ottenere un numero di valori di PSNR e 
deblurredMSE2 = np.zeros((len(deblurredPSNR2), ))       # MSE differenti.

i = 0

for max_it in [10, 25, 50, 75, 100]:
    (x_last, iterazioni, PSNR, MSE) = CG_minimize(f, grad_f, x0, normalized_image, step, max_it, ABSOLUTE_STOP)
    (x_last2, iterazioni2, PSNR_second, MSE_second) = CG_minimize(f2, grad_f2, x2, normalized_image2, step, max_it, ABSOLUTE_STOP)
    deblur_image = np.reshape(x_last, (m, n))   # Nuova immagine prodotta
    deblur_image2 = np.reshape(x_last2, (o, p))   # Nuova immagine prodotta
    deblurredPSNR[i] = metrics.peak_signal_noise_ratio(normalized_image, deblur_image)
    deblurredMSE[i] = metrics.mean_squared_error(normalized_image, deblur_image)
    deblurredPSNR2[i] = metrics.peak_signal_noise_ratio(normalized_image2, deblur_image2)
    deblurredMSE2[i] = metrics.mean_squared_error(normalized_image2, deblur_image2)
    i = i + 1
    
## IMMAGINE 1
plt.figure(figsize = (20,10))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(b, cmap = 'gray')
plt.title('Immagine 1 degradata', fontsize=20)

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(deblur_image, cmap = 'gray')
plt.title(f'Immagine 1 ricostruita (PSNR: {deblurredPSNR[4]: .2f}, MSE: {deblurredMSE[4]: .7f})', fontsize=20)
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredPSNR)
plt.xlabel('Iterazioni')
plt.ylabel('PSNR')
plt.title('PSNR immagine 1 al variare delle iterazioni, naive')
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredMSE)
plt.xlabel('Iterazioni')
plt.ylabel('MSE')
plt.title('MSE immagine 1 al variare delle iterazioni, naive')
plt.show() 

## IMMAGINE 2
plt.figure(figsize = (20,10))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(b2, cmap = 'gray')
plt.title('Immagine 2 degradata', fontsize=20)

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(deblur_image2, cmap = 'gray')
plt.title(f'Immagine 2 ricostruita (PSNR: {deblurredPSNR2[4]: .2f}, MSE: {deblurredMSE2[4]: .7f})', fontsize=20)
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredPSNR2)
plt.xlabel('Iterazioni')
plt.ylabel('PSNR')
plt.title('PSNR immagine 2 al variare delle iterazioni, naive, naive')
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredMSE2)
plt.xlabel('Iterazioni')
plt.ylabel('MSE')
plt.title('MSE immagine 2 al variare delle iterazioni, naive')
plt.show() 

###### CON REGOLARIZZAZIONE DI TIKHONOV

lamb = 15 # Lambda é il fattore di penalitá applicata alla soluzione: piú é alto piú l' immagine peggiora.
          # 0.01 é il valore che minimizza il psnr. Aumentandolo, aumentano anche il numero di iterazioni.

def f_regolarized(x): # f regolarizzata con Tikhonov
    x_resize = np.reshape(x, (m, n))
    result = (0.5) * np.linalg.norm(A(x_resize, K) - b, 2) ** 2 + lamb * (0.5) * np.linalg.norm(x_resize, 2) ** 2  
    return result

def grad_f_regolarized(x):
    x_resize = np.reshape(x, (m, n))
    result = AT(A(x_resize, K), K) - AT(b, K) + lamb * x_resize
    result = np.reshape(result, m*n)   
    return result

def f2_regolarized(x): # f regolarizzata con Tikhonov
    x_resize = np.reshape(x, (o, p))
    result = (0.5) * np.linalg.norm(A(x_resize, K2) - b2, 2) ** 2 + lamb * (0.5) * np.linalg.norm(x_resize, 2) ** 2  
    return result

def grad_f2_regolarized(x):
    x_resize = np.reshape(x, (o, p))
    result = AT(A(x_resize, K2), K2) - AT(b2, K2) + lamb * x_resize
    result = np.reshape(result, o*p)   
    return result

###############

step = 0.1
x0 = b  # Immagine da ricostruire
x2 = b2

ABSOLUTE_STOP = 1.e-4

deblurredPSNR = np.zeros((5, ))    # Cambiare 5 se si vogliono ottenere un numero di valori di PSNR e 
deblurredMSE = np.zeros((len(deblurredPSNR), ))       # MSE differenti.

deblurredPSNR2 = np.zeros((5, ))    # Cambiare 5 se si vogliono ottenere un numero di valori di PSNR e 
deblurredMSE2 = np.zeros((len(deblurredPSNR2), ))       # MSE differenti.

i = 0

for max_it in [5, 10, 15, 20, 25]:
    (x_last, iterazioni, PSNR, MSE) = CG_minimize(f_regolarized, grad_f_regolarized, x0, normalized_image, step, max_it, ABSOLUTE_STOP)
    (x_last2, iterazioni2, PSNR_second, MSE_second) = CG_minimize(f2_regolarized, grad_f2_regolarized, x2, normalized_image2, step, max_it, ABSOLUTE_STOP)
    deblur_image = np.reshape(x_last, (m, n))   # Nuova immagine prodotta
    deblur_image2 = np.reshape(x_last2, (o, p))   # Nuova immagine prodotta
    deblurredPSNR[i] = metrics.peak_signal_noise_ratio(normalized_image, deblur_image)
    deblurredMSE[i] = metrics.mean_squared_error(normalized_image, deblur_image)
    deblurredPSNR2[i] = metrics.peak_signal_noise_ratio(normalized_image2, deblur_image2)
    deblurredMSE2[i] = metrics.mean_squared_error(normalized_image2, deblur_image2)
    i = i + 1

## IMMAGINE 1
plt.figure(figsize = (20,10))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(b, cmap = 'gray')
plt.title('Immagine 1 degradata', fontsize=20)

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(deblur_image, cmap = 'gray')
plt.title(f'Immagine 1 ricostruita (PSNR: {deblurredPSNR[4]: .2f}, MSE: {deblurredMSE[4]: .7f})', fontsize=20)
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredPSNR)
plt.xlabel('Iterazioni')
plt.ylabel('PSNR')
plt.title('PSNR immagine 1 al variare delle iterazioni, regolarizzata')
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredMSE)
plt.xlabel('Iterazioni')
plt.ylabel('MSE')
plt.title('MSE immagine 1 al variare delle iterazioni, regolarizzata')
plt.show() 

## IMMAGINE 2
plt.figure(figsize = (20,10))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(b2, cmap = 'gray')
plt.title('Immagine 2 degradata', fontsize=20)

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(deblur_image2, cmap = 'gray')
plt.title(f'Immagine 2 ricostruita (PSNR: {deblurredPSNR2[4]: .2f}, MSE: {deblurredMSE2[4]: .7f})', fontsize=20)
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredPSNR2)
plt.xlabel('Iterazioni')
plt.ylabel('PSNR')
plt.title('PSNR immagine 2 al variare delle iterazioni, regolarizzata')
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredMSE2)
plt.xlabel('Iterazioni')
plt.ylabel('MSE')
plt.title('MSE immagine 2 al variare delle iterazioni, regolarizzata')
plt.show() 

"""
Il numero di iterazioni dipende dai tre fattori: lambda, range di maxit e j_max nel metodo next_step.
L' obiettivo é quello di trovare un giusto compromesso.
"""