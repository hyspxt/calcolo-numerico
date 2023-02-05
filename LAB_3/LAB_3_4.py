import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from skimage import data

"""
Compressione di un immagine tramite SVD.

Utilizzando la libreria skimage, nello specifico il modulo data, caricare e visualizzare un'immagine A in 
scala di grigio di dimensione m x n.

    1) Calcolare la matrice A_pp = sum_{i = 1}^p (u_i * v_i^t * s_i) dove p <= rango(A)

    2) Visualizzare l'immagine A_p

    3) Calcolare l'errore relativo ||A - A_p||_2 / ||A||_2

    4) Calcolare il fattore di compressione c_p = 1/ p * min (m, n) - 1

    5) Calcolare e plottare l' errore relativo e il fattore di compressione al variare di p.

"""

A = data.coins()       # Se si vuole provare con altre immagini, provare con data.moon(), data.brick(), 
                         # data.coins() e data.camera()

print(type(A))
print(A.shape)

plt.imshow(A, cmap='gray')
plt.show()

"""
   1) Calcolare la matrice A_pp = sum_{i = 1}^p (u_i * v_i^t * s_i) dove p <= rango(A)
"""

A_p = np.zeros(A.shape)
p_max = 10
U, s, VT = scipy.linalg.svd(A)

for i in range(p_max):
  ui = U[:, i]
  vi = VT[i, :]
  A_p = A_p + (np.outer(ui, vi) * s[i])  # Per creare la matrice correttamente é necessario fare il prodotto esterno 
                                         # dei due vettori ui e vi. Il prodotto esterno di due vettori crea una matrice 
                                         # in cui ogni elemento della riga i e colonna j é dato dal prodotto tra l'elemento
                                         # i-esimo del vettore ui e l' elemento j-esimo del vettore vi. Si invece moltiplica e basta.
                                         # ESEMPIO
                                         # Se abbiamo ui = [1, 2, 3] e vi = [4, 5, 6] allora avremo una matrice:
                                         # [[4, 5, 6],
                                         # [8, 10, 12],
                                         # [12, 15, 18]}.              
                                        
"""
    2) Visualizzare l'immagine A_p
"""

print("A_p = \n", A_p)
plt.imshow(A_p, cmap='gray')
plt.show()

"""
    3) Calcolare l'errore relativo ||A - A_p||_2 / ||A||_2
"""
err_rel = np.linalg.norm(A - A_p, ord = 2) / np.linalg.norm(A, ord = 2)

"""
    4) Calcolare il fattore di compressione c_p = 1/ p * min (m, n) - 1
"""

m = U.size    # Servono per calcolare il fattore di compressione.
n = VT.size
c = min(m, n) / p_max - 1 

print('\n')
print('L\'errore relativo della ricostruzione di A è', err_rel)
print('Il fattore di compressione è c=', c)


plt.figure(figsize=(20, 10))

fig1 = plt.subplot(1, 2, 1)
fig1.imshow(A, cmap='gray')
plt.title('True image')

fig2 = plt.subplot(1, 2, 2)
fig2.imshow(A_p, cmap='gray')
plt.title('Reconstructed image with p =' + str(p_max))

plt.show()

dim = np.arange(2, 20, 2)
relErr = np.zeros(np.size(dim))
c_p = np.zeros(np.size(dim))

# al variare di p
"""
    5) Calcolare e plottare l' errore relativo e il fattore di compressione al variare di p.
"""
j = 0
for p_max in dim:      # P_max indica il numero di diadi che comporranno la matrice.
                        # una diade é una matrice [n, 1] e NON é un vettore.
    
    A_p = np.zeros(A.shape)
    U, s, VT = scipy.linalg.svd(A)

    for i in range(p_max):
      ui = U[:, i]
      vi = VT[i, :]
      A_p = A_p + (np.outer(ui, vi) * s[i])
      
    m = U.size    # Servono per calcolare il fattore di compressione.
    n = VT.size
    relErr[j] = np.linalg.norm(A - A_p, ord = 2) / np.linalg.norm(A, ord = 2)
    c_p[j] = min(m, n) / p_max - 1 
    print('L\'errore relativo della ricostruzione di A è', relErr[j])
    print('Il fattore di compressione è c=', c_p[j])
    j = j + 1
    
plt.plot(dim, relErr)
plt.title('Relative Error on p dimension')
plt.xlabel('p dimension')
plt.ylabel('Relative Error')
plt.show()

plt.plot(dim, c_p)  # É normale che il fattore di compressione diminuisca col procedere delle iterazioni:
                    # un valore molto alto significa che l' immagine compressa é molto differente da quella di partenza
plt.title('Compression on p dimension')
plt.xlabel('p dimension')
plt.ylabel('Compression')
plt.show()
