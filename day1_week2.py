#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:47:09 2022

@author: aliaaafify
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_fun1(x):
    "function to get a value of cosx + xsinx"
    f1=np.cos(x)+ x*np.sin(x)
    return f1

def plot_fun2(x0):
    "function to get a value of f' (derivative)" 
    f2=x*np.cos(x0)
    return f2

x = np.linspace(-6,6,1000) # genrate a set of independant variables
f1= plot_fun1(x) # denpendent variables
f2= plot_fun2(x) # derivative


fig,ax = plt.subplots()
ax.plot(x,f1,'b-')
ax.plot(x,f2,'-r')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.legend([r'$y$', r'$\dot y$'],fontsize=16)
ax.grid(True)
ax.legend()

#%%

import numpy as np
import matplotlib.pyplot as plt

#forward
def fun1(xx):
    "function to get a value of cosx + xsinx"
    f_1 = np.cos(xx)+ xx*np.sin(xx)
    return f_1


h = 0.25
x_0 = -6
x_fin = 6
y_dot_forward = np.array([]) #initializing step points 
x_forw = np.array([x_0]) #initialize solutin array
x_1 = x_0

while x_1 <= x_fin:
    #evaluation
    f_1 = fun1(x_1)
    f_1_h = fun1(x_1+h)
    slope1 = (f_1_h - f_1) / h
    #iteration
    x_1 = x_1 + h
    #storage
    x_forw = np.append(x_forw,x_1) #approximation for the derivative
    y_dot_forward = np.append(y_dot_forward,slope1)
    
    
fig,ax = plt.subplots()
ax.plot(x,f2,'b-')

ax.plot(x_forw[:-1],y_dot_forward,'-r')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.legend([r'$\dot y$ real', r'$\dot y$ forward'],fontsize=16)
ax.grid(True)
ax.legend()
#import sys
#sys.exit()

#%%
#backword

import numpy as np
import matplotlib.pyplot as plt

#forward
def fun1(xx):
    "function to get a value of cosx + xsinx"
    f_1 = np.cos(xx)+ xx*np.sin(xx)
    return f_1

h = 0.25
x_0 = -6
x_fin = 6
y_dot_backword = np.array([]) #initializing step points 
x_back = np.array([x_0]) #initialize solutin array
x_1 = x_0

while x_1 <= x_fin:
    #evaluation
    f_1 = fun1(x_1)
    f_1_h = fun1(x_1-h)
    slope2 = (f_1 - f_1_h) / h
    #iteration
    x_1 = x_1 + h
    #storage
    x_back = np.append(x_back,x_1) #approximation for the derivative
    y_dot_backword = np.append(y_dot_backword,slope2)
    
fig,ax = plt.subplots()
ax.plot(x,f2,'b-')

ax.plot(x_forw[:-1],y_dot_forward,'-r')

ax.plot(x_back[:-1],y_dot_backword,'-k')

ax.set_xlabel('x')
ax.set_ylabel('y')
fig.legend([r'$\dot y$ real', r'$\dot y$ forward'],fontsize=16)
ax.grid(True)
ax.legend()
#%%
#central

import numpy as np
import matplotlib.pyplot as plt


#xx = np.linspace(0,6,1000) # genrate a set of independant variables
h = 0.25
x_0 = -6
x_fin = 6
y_dot_central = np.array([]) #initializing step points
x_cen = np.array([x_0]) #initialize solutin array
x_1 = x_0

while x_1 <= x_fin:
    #evaluation
    f_1 = fun1(x_1)  #f_k
    f_1_h_plus = fun1(x_1+h) #f_k+1
    f_1_h_minus = fun1(x_1-h) #f_k-1
    slope3 = (f_1_h_plus - f_1_h_minus) / (2*h) #(f_k+1 - f_k-1) / 2*h
    #iteration
    x_1 = x_1 + h
    #storage
    x_cen = np.append(x_cen,x_1) #approximation for the derivative in x
    y_dot_central = np.append(y_dot_central,slope3)  #approximation for the derivative in y
    
fig,ax = plt.subplots()

ax.plot(x,f2,'b-') #plot the real value of the derivative

ax.plot(x_forw[:-1],y_dot_forward,'-r') #plot the forward value of the derivative

ax.plot(x_back[:-1],y_dot_backword,'-k') #plot the backword value of the derivative

ax.plot(x_cen[:-1],y_dot_central,'-g') #plot the central value of the derivative

ax.set_xlabel('x')
ax.set_ylabel('y')
fig.legend([r'$\dot y$ real', r'$\dot y$ forward',r'$\dot y$ backword',r'$\dot y$ central'],fontsize=16)
ax.grid(True)
ax.legend()

#%%
#integration

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def RHS(x, t):
    """ODE right hand side"""
    Y=-2*x
    return Y

t0=0 #intial time
tf=2 #final time
y0=3 #intial condition

"Evaluate the exact solution"
time = np.linspace(t0,tf) 
y_true = odeint(RHS,y0,time)

fig1 = plt.figure()
plt.plot(time,y_true,'k-',linewidth=2)
plt.ylabel(r'$y(t)$')
plt.xlabel('time')
plt.legend(['Truth'])
plt.grid(True)

#sz=np.array([])
#for step_size in sz:
    
step_size = 0.2

"First order Runge-Kutta or Eular Method"
timeline = np.array([t0])
sol_rk1 = np.array([y0])
x_appr = timeline
y_appr = sol_rk1
y = y0

while t0 <= tf-step_size: # (tf-step_size) to not finsih one step after
    
    #evaluation
    f_RHS1 = RHS(y0,t0)*step_size  #h*f'_k
    y1 = y0 + f_RHS1 #f_k+1
    #storage
    t0=t0+step_size
    x_appr = np.append(x_appr,t0) #approximation for the integration in x
    y_appr = np.append(y_appr,y1)  #approximation for the integration in y
    #iteration (intialize next step)
    y0=y1
    t=t0
    
plt.plot(x_appr,y_appr,'o-r',linewidth=2) # plotting x and x from the approximation function
plt.legend([r'$dy/dx$ truth', r'$dy/dx$ appro'],fontsize=16)

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def RHS(x, t):
    """ODE right hand side"""
    Y=-2*x
    return Y

"second order Runge-Kutta or Eular Method"
t0=0 #intial time
tf=2 #final time
y0=3 #intial condition

timeline = np.array([t0])
sol_rk1 = np.array([y0])
x_appr = timeline
y_appr = sol_rk1
y = y0
step_size = 0.2
while t0 <= tf-step_size: # (tf-step_size) to not finsih one step after
    
    #evaluation
    f_RHS1 = y0 + (RHS(y0,t0)*(step_size/2))  #second arrgument
    f_RHS2 = t0 + (step_size/2) #first arrgument
    f_RHS3 = step_size*(RHS(f_RHS1,f_RHS2))
    y1 = y0 + f_RHS3
    #storage
    t0=t0+step_size
    x_appr = np.append(x_appr,t0) #approximation for the integration in x
    y_appr = np.append(y_appr,y1)  #approximation for the integration in y
    #iteration (intialize next step)
    y0=y1
    t=t0
    
plt.plot(x_appr,y_appr,'o-b',linewidth=2) # plotting x and x from the approximation function
plt.legend([r'$dy/dx$ truth', r'$1st$ -Runge-Kutta ', r'$2st -Runge-Kutta'],fontsize=16)

#%%
"fourth order Runge-Kutta  or Eular Method"
t0=0 #intial time
tf=2 #final time
y0=3 #intial condition

timeline = np.array([t0])
sol_rk1 = np.array([y0])
x_appr = timeline
y_appr = sol_rk1
y = y0
step_size = 0.2


while t0 <= tf-step_size: # (tf-step_size) to not finsih one step after
    
    #evaluation of fourth order of Runge-Kutta
    k1 = RHS(y0,t0)  #first step of Runge-Kutta
    k2 = RHS(y0+(k1*(step_size/2)),t0+(step_size/2)) #second step of Runge-Kutta
    k3 = RHS(y0+(k2*(step_size/2)),t0+(step_size/2)) #third step of Runge-Kutta
    k4 = RHS(y0+(k2*step_size),t0+step_size) #fourth step of Runge-Kutta

    y1 = y0 + ((k1+(2*k2)+(2*k3)+k4)/6) * step_size # the full approximation of Runge-Kutta 4th order
    
    #storage
    t0=t0+step_size
    x_appr = np.append(x_appr,t0) #approximation for the integration in x
    y_appr = np.append(y_appr,y1)  #approximation for the integration in y
    
    #iteration (intialize next step)
    y0=y1
    t=t0
    
plt.plot(x_appr,y_appr,'o-g',linewidth=2) # plotting x and x from the approximation function
plt.legend([r'$truth$', r'1st of Runge-Kutta', r'2nd of Runge-Kutta',r'4th of Runge-Kutta'],fontsize=16)

#%% 


 g = 9.8 #gravity constant
 l = 3 #lenght of the pendulum

 import numpy as np
 from scipy.integrate import odeint
 import matplotlib.pyplot as plt

 def pendulum_dynamics(theta,t):
     """the dynamics for the pendulum dynamics"""
     theta_dot = np.zeros(2)
     theta_dot[0] = theta_dot[0]
     theta_dot[1] = - (g/l)*np.sin(theta_dot[0])
     return theta_dot

 #def pendulum_dynamics_damp(theta,t):
 #    """the dynamics for the pendulum dynamics adding damping"""
 #    theta_dot = np.zero(2)
 #    theta_dot[0] = theta[0]
 #    theta_dot[1] = - (g/l)*np.sin(theta[0]) + damp[]
 #    return theta_dot
    
# t0=0
# tf=15
# theta = np.array([np.pi/3,0])
# t = np.linspace(t0,tf,1000)
# solution = odeint(pendulum_dynamics,theta,t)
# step_size = 0.2

# #def RK2 (func, y0,t) # (tf-step_size) to not finsih one step after
# #    n = len(t)
# #    y = np.zeros(n,len(t))
#     # y[0]=y0
#     # for i in range(n-1):
#     # #evaluation of fourth order of Runge-Kutta
#     # k1 = pendulum_dynamics(y[i],t0)  #first step of Runge-Kutta
#     # k2 = pendulum_dynamics(y[i]+(k1*(step_size/2)),t0+(step_size/2)) #second step of Runge-Kutta
#     # k3 = pendulum_dynamics(y[i]+(k2*(step_size/2)),t0+(step_size/2)) #third step of Runge-Kutta
#     # k4 = pendulum_dynamics(y[i]+(k2*step_size),t0+step_size) #fourth step of Runge-Kutta

#     # y1 = y0 + ((k1+(2*k2)+(2*k3)+k4)/6) * step_size # the full approximation of Runge-Kutta 4th order
    
#     # #storage
#     # t0=t0+step_size
#     # x_appr = np.append(x_appr,t0) #approximation for the integration in x
#     # y_appr = np.append(y_appr,y1)  #approximation for the integration in y
    
#     # #iteration (intialize next step)
#     # y0=y1
#     # t=t0
    
# fig = plt.figure()
# plt.subplot(2,1,1)
# plt.plot(x_appr,solution,'b-')
# plt.set_xlabel('x')
# plt.set_ylabel('y')
# plt.legend([r'$y$',],fontsize=16)
# plt.grid(True)

# plt.subplot(2,1,2)
# plt.plot(t,solution,'r-')
# plt.set_xlabel('x')
# plt.set_ylabel('y')
# plt.legend([r'$y$',],fontsize=16)
# plt.grid(True)
#%%

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

sigma=10
ro=28
beta=8/3
t0=0
tf=20
t = np.linspace(t0,tf,1000)
#x = np.array([5,5,5]) #3d array
x = 5 * np.ones(3)  #another way to write a 3d array

def lorenz63(x,t,sigma,ro,beta):
    """lorenz system function"""
    xdot = sigma*(x[1]-x[0])
    ydot = x[0]*(ro-x[2])-x[1]
    zdot = (x[0]*x[1]) - (beta*x[2])
    return xdot, ydot, zdot

solu_lorenz=odeint(lorenz63,x,t,args=(sigma,ro,beta))

fid1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(solu_lorenz[:,0],solu_lorenz[:,1],solu_lorenz[:,2],'b')

#%%

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

sigma=10
ro=28
beta=8/3
t0=0
tf=20
t = np.linspace(t0,tf,1000)

x1 = 20 * 2 * (np.random.rand(20)-0.5)
x2 = 30 * 2 * (np.random.rand(20)-0.5)
x3 = 50 * np.random.rand(20)

def lorenz63(x,t,sigma,ro,beta):
    """lorenz system function"""
    xdot = sigma * (x[1]-  x[0])
    ydot = x[0] * (ro-x[2]) - x[1]
    zdot = (x[0]*x[1]) - (beta*x[2])
    return xdot, ydot, zdot

fid1 = plt.figure()
ax = plt.axes(projection='3d')
for i in range(20):
    x0v = np.array([x1[i],x2[i],x3[i]])
    solu_lorenz1=odeint(lorenz63,x0v,t,args=(sigma,ro,beta))    
    ax.plot3D(solu_lorenz1[:,0],solu_lorenz1[:,1],solu_lorenz1[:,2])
    
