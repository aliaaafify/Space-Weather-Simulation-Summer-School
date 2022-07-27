#!/usr/bin/env python
"""
Solution of a 1D Poisson equation: -u_xx = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = (3*x + x^2)*exp(x)

Analytical solution: -x*(x-1)*exp(x)

Finite differences (FD) discretization: second-order diffusion operator
"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
#%matplotlib qt
plt.close()

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)
order = 2

"System matrix and RHS term"
A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
F = 2*(2*x**2 + 5*x -2)*np.exp(x)

# boundary condition at x=0
A[0,:] = np.concatenate(([1],np.zeros(N)))
F[0] = 0

# boundary condition at x=1
if order == 1:
    A[N,:] = (1/Dx)*(np.concatenate((np.zeros(N-1),[-1,1])))
    F[N] = 0
if order == 2:
    A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-2),[1/2,-2,3/2]))
    F[N] = 0


"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = U #np.concatenate(([0],U,[0]))
ua = 2*x*(3-2*x)*np.exp(x)

"Plotting solution"
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
#plt.axis([0, 1,0, 1])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

print(0.10508/0.388099)
print(0.027332/0.10508)

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
#%matplotlib qt
plt.close()

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)
xN = np.linspace(0,1+Dx,N+2)

order = 1

if order < 2:
    "System matrix and RHS term"
    A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
    F = 2*(2*x**2 + 5*x -2)*np.exp(x)
    # boundary condition at x=0
    A[0,:] = np.concatenate(([1],np.zeros(N)))
    F[0] = 0
    A[N,:] = (1/Dx)*(np.concatenate((np.zeros(N-1),[-1,1])))
    F[N] = 0
else:
    "System matrix and RHS term"
    A = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
    F = 2*(2*xN**2 + 5*xN -2)*np.exp(xN)
    # boundary condition at x=0
    A[0,:] = np.concatenate(([1],np.zeros(N+1)))
    F[0] = 0
    # boundary condition at x=1
    A[N+1,:] = (1/(2*Dx))*(np.concatenate((np.zeros(N-1),[-1,0,1])))
    F[N+1] = 0

"Solution of the linear system AU=F"
U = np.linalg.solve(A,F)
u = U[0:N+1] #np.concatenate(([0],U,[0])) #numerical sol
ua = 2*x*(3-2*x)*np.exp(x) #exact value

"Plotting solution"
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
#plt.axis([0, 1,0, 1])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

print(0.10508/0.388099)
print(0.027332/0.10508)

#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    #We have to load this
from math import pi
#%matplotlib qt
plt.close()

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)


"time parameters"
dt = 1/24
time = np.arange(0,3+dt,dt)
nt = len(time)

order = 2

if order < 2:
    U = np.zeros((N+1,nt))
    U[:,0] = x*(3-2*x)*np.exp(x)+u0
else:
    xN = np.linspace(x,[x[N]+Dx])
    U = np.zeros((N+2,nt))
    U[:,1] = xN*(3-2*xN)*np.exp(xN)+u0
    
for it in range(nt):
    
    if order < 2:
        "System matrix and RHS term"
        A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
        F = 2*(2*x**2 + 5*x -2)*np.exp(x)
        
        "temporal term"
        A = A + (1/dt)*np.diag(np.ones(N+1))
        F = F + U[:,it]/dt
        
        "boundary at x=0"
        A[0,:] = np.concatenate(([1],np.zeros(N)))
        F[0] = 0
        
        "boundary at x=1"
        A[N,:] = (1/Dx)*(np.concatenate((np.zeros(N-1),[-1,1])))
        F[N] = 0
    else:
        "System matrix and RHS term"
        A = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
        F = 2*(2*xN**2 + 5*xN -2)*np.exp(xN)
        
        "temporal term"
        A = A + (1/dt)*np.diag(np.ones(N+2))
        F = F + U[:,it]/dt
        
        "boundary at x=0"
        A[0,:] = np.concatenate(([1],np.zeros(N+1)))
        F[0] = 0
        
        "boundary at x=1"
        A[N+1,:] = (1/(2*Dx))*(np.concatenate((np.zeros(N-1),[-1,0,1])))
        F[N+1] = 0
        
    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1]= u

"Plotting solution"
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
plt.legend(fontsize=12,loc='upper left')
plt.grid()
#plt.axis([0, 1,0, 1])
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"animation"
#fig = plt.figure()
#ax = plt.axes(xlim = (0, 1),ylim = (u0,u0+6))
