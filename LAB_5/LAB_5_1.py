import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data, metrics
from scipy import signal
from numpy import fft
from scipy.optimize import minimize


"""
A. DEBLUR IMMAGINI.

Il problema di deblur consiste nella ricostruzione di un immagine a partire da un dato acquisito mediante il seguente modello:
 - b = Ax + η (1)
dove b rappresenta l’immagine corrotta, x l’immagine originale che vogliamo ricostruire, A l’operatore che applica il blur Gaussiano 
ed η il rumore additivo con distribuzione Gaussiana di media 0 e deviazione standard σ.

Esercizio 1)
i. Caricare l’immagine camera() dal modulo skimage.data, rinormalizzandola nel range [0, 1].
ii. Applicare un blur di tipo gaussiano con deviazione standard 3 il cui kernel ha dimensioni 24 × 24 
utilizzando le funzioni fornite gaussian kernel(), psf fft() ed A().
iii. Aggiungere rumore di tipo gaussiano, con deviazione standard 0.02, usando la funzione np.random.normal().
iv. Calcolare il Peak Signal Noise Ratio (PSNR) ed il Mean Squared Error (MSE) tra l’immagine degradata 
e l’immagine originale usando le funzioni peak signal noise ratio e mean squared error 
disponibili nel modulo skimage.metrics
"""

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

image = data.camera() # Si carica l'immagine
plt.imshow(image)
normalized_image = skimage.img_as_float(image) # La normalizziamo nell' intervallo [0, 1]. In questo modo
                                               # garantiamo che i pixel siano rappresentati come float.
m, n = normalized_image.shape

# Creazione del blur gaussiano
K = gaussian_kernel(24, 3)
K = psf_fft(K, 24, normalized_image.shape)

# Generazione del noise gaussiano
sigma =  0.02
noise = np.random.normal(0, sigma, size = normalized_image.shape)

# Applicazione di blur e noise
b = A(normalized_image, K) + noise

PSNR = metrics.peak_signal_noise_ratio(normalized_image, b)  # Si calcolano PSNR e MSE ( + é alto il PSNR + l'immagine é buona)
MSE = metrics.mean_squared_error(normalized_image, b)
print('PSNR = ', PSNR)
print('MSE = ', MSE)

plt.figure(figsize = (20,10))     # Plotting dell' immagine originale
axis1 = plt.subplot(1, 2, 1)
axis1.imshow(normalized_image, cmap = 'gray')  # In scala di grigi si vedono meglio le sfocature
plt.title('Immagine originale', fontsize = 18)

axis2 = plt.subplot(1, 2, 2)     # Plotting dell'immagine corrotta
axis2.imshow(b, cmap = 'gray')
plt.title(f'Immagine degradata (PSNR: {PSNR: .2f}, MSE: {MSE: .7f})', fontsize=20)
plt.show()

"""
B. SOLUZIONE NAIVE

Esercizio 2.
1. Utilizzando il metodo del gradiente coniugato implementato dalla funzione minimize, calcolare la soluzione naive.
2. Analizza l' andamento del PSNR e dell' MSE al variare del numero di iterazioni.
"""

# É necessario invertire il processo di degradazione per effettuare la ricostruzione.

def f(x): # 1/2 || Ax -b ||^2
    x_resized = np.reshape(x, (m, n)) # Viene effettuato il reshape della soluzione x per evitare gli errori con il kernel dell' immagine (la matrice).
    result = (0.5) * ( np.sum( np.square( (A(x_resized, K) - b ) )))
    return result

def grad_f(x):  # Gradiente della funzione, che in quanto ha una sola variabile, corrisponde alla derivata (vettore unielemento).
    x_resized = np.reshape(x, (m, n))
    result = AT(A(x_resized, K), K) -AT(b, K)
    result = np.reshape(result, m*n)   # Rispettivamente larghezza e e altezza dell' immagine. Cosí il gradiente fa tutta la matrice.
    return result

x0 = b  # Immagine da ricostruire
deblurredPSNR = np.zeros((5, ))    # Cambiare 5 se si vogliono ottenere un numero di valori di PSNR e 
deblurredMSE = np.zeros((len(deblurredPSNR), ))       # MSE differenti.
i = 0

for max_it in [10, 25, 50, 75, 100]:
    result = minimize(f, x0, method='CG', jac = grad_f, options={'maxiter':max_it, 'return_all':True}) # Minimizzazione (trovare il min x) con gradienti coniugati
    deblur_image = np.reshape(result.x, (m, n))   # Nuova immagine prodotta
    deblurredPSNR[i] = metrics.peak_signal_noise_ratio(normalized_image, deblur_image)
    deblurredMSE[i] = metrics.mean_squared_error(normalized_image, deblur_image)
    i = i + 1

plt.figure(figsize = (20,10))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(b, cmap = 'gray')
plt.title('Immagine degradata', fontsize=20)

ax2 = plt.subplot(1, 2, 2)
ax2.imshow(deblur_image, cmap = 'gray')
plt.title(f'Immagine ricostruita (PSNR: {deblurredPSNR[4]: .2f}, MSE: {deblurredMSE[4]: .7f})', fontsize=20)
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredPSNR)
plt.xlabel('Iterazioni')
plt.ylabel('PSNR')
plt.title('PSNR al variare delle iterazioni, naive')
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredMSE)
plt.xlabel('Iterazioni')
plt.ylabel('MSE')
plt.title('MSE al variare delle iterazioni, naive')
plt.show()

"""
Per ridurre gli effetti del rumore nella ricostruzione `e necessario introdurre un termine di 
regolarizzazione di Tikhonov (non specifico i dati del problema ma si possono trovare in LAB5.pdf).

i. Utilizzando il metodo del gradiente coniugato implementato dalla funzione minimize, calcolare la
 soluzione regolarizzata.
ii. Analizza l’andamento del PSNR e dell’MSE al variare del numero di iterazioni.
iii. Facendo variare λ, analizzare come questo influenza le prestazioni del metodo analizzando le immagini.
iv. Attraverso test sperimentali individuare il valore di λ che minimizza il PSNR.

"""

lamb = 0.1 # Lambda é il fattore di penalitá applicata alla soluzione: piú é alto piú l' immagine peggiora.
          # 0.01 é il valore che minimizza il psnr. Aumentandolo, aumentano anche il numero di iterazioni.

def f2(x): # f regolarizzata con Tikhonov
    x_resize = np.reshape(x, (m, n))
    result = (0.5) * np.linalg.norm(A(x_resize, K) - b, 2) ** 2 + lamb * (0.5) * np.linalg.norm(x_resize, 2) ** 2  
    return result

def grad_f2(x):
    x_resize = np.reshape(x, (m, n))
    result = AT(A(x_resize, K), K) - AT(b, K) + lamb * x_resize
    result = np.reshape(result, m*n)   
    return result

x0 = b
deblurredPSNR_2 = np.zeros((5, ))    # Cambiare 5 se si vogliono ottenere un numero di valori di PSNR e 
deblurredMSE_2 = np.zeros((len(deblurredPSNR_2), ))        # MSE differenti.
i = 0

for max_ite in [10, 25, 50, 75, 100]:
    result = minimize(f2, x0, method='CG', jac = grad_f2, options={'maxiter':max_ite, 'return_all':True})
    deblur_image_2 = np.reshape(result.x, (m, n))
    deblurredPSNR_2[i] = metrics.peak_signal_noise_ratio(normalized_image, deblur_image_2)
    deblurredMSE_2[i] = metrics.mean_squared_error(normalized_image, deblur_image_2)
    i = i + 1

plt.figure(figsize = (20,10))
ax3 = plt.subplot(1, 2, 1)
ax3.imshow(b, cmap = 'gray')
plt.title('Immagine degradata', fontsize=20)

ax4 = plt.subplot(1, 2, 2)
ax4.imshow(deblur_image_2, cmap = 'gray')
plt.title(f'Immagine ricostruita (PSNR: {deblurredPSNR_2[4]: .2f}, MSE: {deblurredMSE_2[4]: .7f})', fontsize=20)
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredPSNR_2)
plt.xlabel('Iterazioni')
plt.ylabel('PSNR')
plt.title('PSNR al variare delle iterazioni, regolarizzata')
plt.show()

plt.figure()
plt.plot([10, 25, 50, 75, 100], deblurredMSE_2)
plt.xlabel('Iterazioni')
plt.ylabel('MSE')
plt.title('MSE al variare delle iterazioni, regolarizzata')
plt.show()