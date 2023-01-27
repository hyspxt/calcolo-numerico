#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Exercise on machine precision

def ex1():
    """
    Execute the following code
        import sys
        help(sys.float_info)
        print(sys.float_info)
    and understand the meaning of max, max_exp and max_10_exp 
    """
    import sys
    help(sys.float_info)
    print(sys.float_info)
    # max: is the maximum representable finite float
    # max_exp: maximum int e such that radix**(e-1) is representable
    # max_10_exp:maximum int e such that 10**e is representable 
    
    
def ex2():
    """
    Write a code to compute the machine precision epsilon in (float) default precision
    with a WHILE construct. Compute also the mantissa digits number.
    """
    epsilon = 1.0 
    mantissa_digit = 1
    while 1.0 + epsilon / 2.0 > 1.0:
        epsilon = epsilon / 2.0
        mantissa_digit += 1
    print("Epsilon = " + str(epsilon))
    print("Mantissa digits = " + str(mantissa_digit))
    # L'idea é che continuiamo a dimezzare epsilon ad ogni ciclo in modo tale da determinare
    # quanto piccolo puó diventare prima che sommarlo al valore 1 sia cosí irrilevante da non contribuire
    # a rendere vero lo statement del while(1.0 + epsilon / 2.0 > 1.0), trovando cosí la precisione di macchina
    # Si noti il fatto che epsilon corrisponde alla distanza tra 1 e il primo numero successivo rappresentabile in float
    
    
def ex3():
    """
    Import NumPy (import numpy as np) and exploit the funtions float16 and float32 in the 
    WHILE statement and see the differences. Check the result of print(np.finfo(float).eps).
    """    
    print("float16: ")
    epsilon = np.float16(1.0) 
    mantissa_digit = 1
    while np.float16(1.0) + epsilon / np.float16(2.0) > np.float16(1.0):
        epsilon = epsilon / np.float16(2.0)
        mantissa_digit += 1
    print("Epsilon = " + str(epsilon))
    print("Mantissa digits = " + str(mantissa_digit))
    
    print("float32: ")
    epsilon = np.float32(1.0) 
    mantissa_digit = 1
    while np.float32(1.0) + epsilon / np.float32(2.0) > np.float32(1.0):
        epsilon = epsilon / np.float32(2.0)
        mantissa_digit += 1
    print("Epsilon = " + str(epsilon))
    print("Mantissa digits = " + str(mantissa_digit))
    
    print("np.finfo(float).eps = "+str(np.finfo(float).eps))
    
    ### Exercises with matplotlib
    # Matplotlib is a plotting library for the Python programming language and its numerical
    # mathematics extension NumPy, from https://matplotlib.org/  
    
def ex4():
    """
    Create a figure combining together the cosine and sine curves, from 0 to 10:
    - Add a legend
    - Add a title
    - Change the default colors
    """
    
    x_sin = np.linspace(0,10)
    y_sin = np.sin(x_sin)
    
    x_cos = np.linspace(0, 10)
    y_cos = np.cos(x_cos)
    
    
    plt.subplots(constrained_layout = True)[1].secondary_xaxis(0.5); # Si aggiunge l' asse x
    plt.title("Sin and cosine from 0 to 10")  #Titolo
    plt.plot(x_sin,y_sin,color='blue', linestyle='-')
    plt.plot(x_cos,y_cos,color='red', linestyle='--')
    plt.legend(['sin', 'cosin']) 
    plt.show()
    
    ### Fibonacci and approximations ###
    
def ex5(n):
    """ 
    Write a script that, given an input number n, computes the numbers of the fibonacci sequence that are less than n
    """
    
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    count = 2
    while a + b < n :
        b += a
        a = b - a
        count += 1
    return count

def ex6():
    """
    Write a code computing, for a natural number k, the ratio r(k) = F(k+1) / F(k), 
    where F(k) are the Fibonacci numbers. 
    Verify that, for a large k, {r(k)}k converges to the value phi = 1+sqrt(5)/2 
    Create a plot of the error (with respect to phi)
    """ 

    arange = np.arange(50) # Creiamo una lista con 50 valori, da 0 a 49
    plt.plot(arange, [relative_error(i) for i in arange])  
    # Plottiamo i valori arange nelle ascisse
    # relative_error e l' errore relativo al valore i-esimo di arange calcolato in r(k)
    plt.legend(['Relative error'])
    plt.show()

    # Quello che fa la funzione sostanzialmente é tracciare un grafico e plottarne i punti dove:
    # l'asse x é un insieme di valori da 0 a 49, mentre la y é l' errore relativo di r(k), che viene calcolato
    # mediante la costante phi. Al crescere di k, r(k) viene approssimato ad una maggiore precisione, riducendo
    # l'errore relativo.

def relative_error(k):
    phi = (1.0 + 5 ** 0.5) / 2.0 # ** é l' operazione di elevamento a potenza
    return abs(r(k) - phi) / phi

def r(k):
    if k <= 0:
        return 0
    if k == 1:
        return 1
    a, b = 0, 1
    for _ in range(k): # _ é l' operatore che indica che si prende l' ultimo risultato
        b += a 
        a = b - a
    print (b / a)
    return b / a

