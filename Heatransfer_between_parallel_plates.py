# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:41:35 2021

@author: Felix
"""
#import symengine as se
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


"Channel Dimensions:"
h = 0.015 # in meters 
R = h/2
l = 200# length

"Fluid Information:"
roh = 1.1644 #Density
cp = 1.01*10**3 
lamb = 0.026

"Dimensionless Boundaryconditions:"
CW1 = 0 # Berandungsgeschwindigkeit "unten" in m/s
CW2 = 0 # Berandungsgeschwindigkeit "oben" in m/s
C   = 0.1 # durchschnitts Geschwindigkeit (aus Massenstrom)
TW1 = 273.15 # Wandtemperatur "oben" in Kelvin
TW2 = 283.15 # Wandtemperatur "unten" in Kelvin
T0  = 293.15 # Anfangstempertur des Fluid in Kelvin

RBx0 = 1# - TW1/T0
x0 = [0]

RByp1 = 0.
RByn1 = 0.
yn1 = -1
yp1 = 1
"-----------------------------------------------------------------------"

"Dimensionless Variables:"
a = lamb/cp/roh
Pe = C*R/a

"-----------------------------------------------------------------------"

"Velocity Profile:"
w = (CW1+CW2)/2
dw = CW2-CW1 

a = w/C-1
b = dw/(3*C)
d = 1 - w/(3*C)

def cProf (y):
    return (a*y**2+b*y+d)*C*3/2
"----------------------------------------------------------------------"
"Size of Power Series"

ssize = 200 #Bei ssize = 141 für NofB = 26 genau, bei ssize = 143 für NofB = 100 genau
    

"----------------------------------------------------------------------"
"Varables to Ease the programm"

NofB = 100
BsI = np.float64(0.0001)
CI = 1000 
stol = BsI/CI



"----------------------------------------------------------------------"
"Power Series:"

def k(beta, k0, k1):
    k2 = -d*k0*beta**2/2
    k3 =-(b*k0+d*k1)*beta**2/6
    kn = [k0, k1, k2, k3]
    for n in range(ssize-4):
        kn4 = -beta**2*(a*kn[n]+b*kn[n+1]+d*kn[n+2])/((n+4)*(n+3))
        kn.append(kn4)
    return kn

"Funktion B1"
def kB1 (beta):
    return k(beta, 1, 0)


def B1 (beta, y):
    B1.Bs =[]
    B = 0.
    kn1 = kB1(beta)
    for n in range (ssize):
        B1.Bs.append(kn1[n]*y**n)
        B += kn1[n]*y**n
    return B


"Funktion B2"
def kB2 (beta):
    return k(beta, 0, 1)
    

def B2 (beta ,y):
    B2.Bs = []
    B = 0.
    kn2 = kB2(beta)
    for n in range (ssize):
        B2.Bs.append(kn2[n]*y**n)
        B += kn2[n]*y**n
    return B

def Bn(beta ,y):
    out = 0
    if BC.choice == 1:
        out = B2(beta , y)
    elif BC.choice == 2:
        out = B1(beta , y)
    elif out == 3:
        out = B1(beta , y) + B2(beta , y)*(BC.B2p+BC.B2n)/(BC.B1p+BC.B1n)
    else:
        out = B1(beta , y)*BC.G[0] + B2(beta , y)*BC.G[1]
    return out

"----------------------------------------------------------------------"

"Nullstellen der Reihe bestimmen:"  

"Equation after applying Boundary Conditions to Power Series for Eigenvalue calculation:"
def BC (beta):
    BC.B1p = B1(beta , yp1)
    BC.B1n = B1(beta , yn1)
    BC.B2p = B2(beta , yp1)
    BC.B2n = B2(beta , yn1)
    B_list = [[BC.B1p,BC.B2p],[BC.B1n,BC.B2n]]
    B = np.array(B_list)
    R = np.array([RByp1, RByn1])
    BC.G = np.linalg.inv(B).dot(R)
    out = 0.
    BC.choice = 0
    if (BC.B1p+BC.B1n)**2 < 1e-12:
        out  = BC.B2n + BC.B2p
        BC.choice = 1
    elif  (BC.B2p+BC.B2n)**2 < 1e-12:
        out = BC.B1p + BC.B1n
        BC.choice = 2
    else:
        out = BC.B1p + BC.B2p*(BC.B2p+BC.B2n)/(BC.B1p+BC.B1n)
        BC.choice = 3
    return out


"Eigenvalue calculation:"

def EVs():
    EV = []     #EV = Eigenvalues
    beta = 0.   # Eigenvalue
    lv = 0.     #last value
    i = 0
    while i < NofB:
        v = BC(beta)     # v = value
        if v*lv < 0 or v**2 < 1e-24: # Bisection Method to improve detected Eigenvalues 
            xl = beta - BsI
            xr = beta
            while (np.abs(xl-xr)) >= stol:
                c = (xl+xr)/2.0
                prod = BC(xl)*BC(c)
                if prod > stol:
                    xl = c
                else:
                    if prod < stol:
                        xr = c
            EV.append(c)
            print('calculated beta', i,' = ', beta)
            i = i+1
        lv = v
        beta += BsI
    return EV

"----------------------------------------------------------------------"
"An bestimmen:"

"Differentiation der Eigenwertfunktion nach beta"

def dB1 (beta):
    B1 = 0. 
    kn1 = kB1(beta)
    for n in range (ssize):
        B1 += kn1[n]
    return B1


def dB2 (beta):
    B2 = 0 
    kn2 = kB2(beta)
    for n in range (ssize):
        B2 += kn2[n]
    return B2

def dBC (beta):
    out = 0.
    if BC.choice == 1:
        out = dB2(beta)
    elif BC.choice == 2:
        out = dB1(beta)
    else:
        out = dB1(beta) + dB2(beta)*(BC.B2p+BC.B2n)/(BC.B1p+BC.B1n)
    return out

def An (beta):
    b = tf.constant(beta)
    with tf.GradientTape() as t:
        t.watch(b)
        B = dBC(b)
    dB = t.gradient(B,b)
    Ans = []
    for i in range(NofB):
        Bi = dB[i]
        A = -2.0/(Bi*beta[i])
        Ans.append(A)
    return Ans

"----------------------------------------------------------------------"
"Lösen der zweiten DGL"

def Zn (x, beta):
    sx = np.size(x)
    Zi = []
    for i in range(sx) :
        xi = x[i]        
        Z = tf.exp(-2*xi*beta**2/(3*Pe))
        Zi.append(Z)
    return Zi

"----------------------------------------------------------------------"
"Berechnen der dimlosen Temperatur"

def Tet (beta, x, y):
    #beta = EVs()
    print('done betas!')
    print(beta)
    A = An(beta)
    print('done As!')
    print(A)
    result = 0
    for i in range(NofB):
        Zi = Zn(x[0], beta[i]) 
        Ai = A[i]
        Bi = Bn(beta[i] , y)
        Tn = Ai*Bi*Zi
        result += Tn
    print (BC.choice)
    return result

"----------------------------------------------------------------------"
"Temperature Profile:"

def TVW(y):     # Tempratrueprofile between Walls by Wall temeratures
    return ((TW2 - TW1)*y + TW1 + TW2)/2


"----------------------------------------------------------------------"
"Bestimmen des wahren Temperaturverlaufs."

def T(teta, tw):
    return teta*T0 + tw

"----------------------------------------------------------------------"
"Wärmeströme bestimmen"

"Ableitung von B nach y"

def dB1y (beta, y):
    B1 = 0. 
    kn1 = kB1(beta)
    for n in range (ssize-1):
        B1 += kn1[n+1]*(n+1)*y**n
    return B1


def dB2y (beta, y):
    B2 = 0. 
    kn2 = kB2(beta)
    for n in range (ssize-1):
        B2 += kn2[n+1]*(n+1)*y**n
    return B2

def dBy (beta,y):
    out = 0.
    if BC.choice == 1:
        out = dB2y(beta,y)
    elif BC.choice == 2:
        out = dB1y(beta,y)
    else:
        out = dB1y(beta,y) + dB2y(beta,y)*(BC.B2p+BC.B2n)/(BC.B1p+BC.B1n)
    return out

"Abgeleitetes Teta"

def dTetay (x, y):
    result = 0
    A = An(beta)
    for i in range(NofB):
        Zi = Zn(x[0], beta[i]) 
        Ai = A[i]
        Bi = dBy(beta[i] , y)
        Tn = Ai*Bi*Zi
        result += Tn
    return result    


"Abgeleiteter Wandtemp.-verlauf"

def dTVW ():
    return (TW2-TW1)/2

"Wärmestrom über Wand"
def q(x,y):
    dT = dTetay(x, y)
    dtvw = dTVW()
    return  -(dT*T0 + dtvw)*lamb

def q1 (x):
   Q = q(x, -1)
   return Q

def q2 (x):
    Q = q(beta, x, 1)
    return Q


"----------------------------------------------------------------------"
"Abbildungen reproduzieren:"

y = np.linspace(-R, R, 100)
x = np.linspace(0, l, 100)

X, Y = np.meshgrid(x, y)

Xed = X/l # X entdimensioniert
Yed = Y/R # Y entdimentsioniert

c = cProf(Yed)
tw = TVW(Yed)


fig = plt.figure()
plt.plot(c, Y)
plt.show()


fig = plt.figure()
plt.plot(Y, tw)
plt.show()


beta = EVs()


Z = Tet(beta, Xed, Yed) 
print(Z)
t = T(Z, tw)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z, cmap='jet')
ax.grid(True)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,t, cmap='jet')

ax.grid(True)
plt.show()

Q11 = q1(Xed)
print(Q11)
print(np.size(Q11))
Q1 = np.float64(Q11)

fig = plt.figure()
plt.plot(Xed, Q11)
plt.show()

fig = plt.figure()
#
