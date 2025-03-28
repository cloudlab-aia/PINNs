# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 19:02:39 2023

@author: Usuario
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


# Considero MASA = 1 sin pérdida de generalidad
# Mirar osciladores como: forzado, forzado y caos, van der Pol, de torsion
# Pequeñas oscilaciones
# Oscilador cuántico, 1D y 2D

# =============================================================================
# Oscilador Armónico Simple
# =============================================================================

def oscilador_simple(w0,t):
    
    # Ec. dif.: m*(d^2x/dt^2) = -k*x
    # x(0) = 1, dx/dt = 0
    # Solución: A*cos(phi + wt)
    
    phi = 0 # dx/dt = 0 entonces resuelves la ec. y te da que phi = 0, pi, multiplo de estos
    A = 1/(np.cos(phi))
    
    cos = torch.cos(phi + w0*t)
    
    x = A*cos
    return x

# =============================================================================
# Oscilador Armónico Sobreamortiguado
# =============================================================================

def oscilador_sobre(d, w0, t):
    
    # Ec. dif.: m(d^2x/dt^2) + mu(dx/dt) + kx = 0
    # IC: x(0) = 1 , dx/dt = 0
    # Con las condiciones iniciales se saca:
    # r1,r2 = -mu/2 +- sqrt((mu/2)^2 - k)
    # A1 = -r2/(r1-r2)
    # A2 = r1/(r1-r2)
    # Solcuión:  x = A1*exp(r1*t) + A2*exp(r2*t)
    
    assert d > w0 # Condicion sobreamortiguado
    
    alfa = np.sqrt(d**2-w0**2)
    
    r1 = -d + alfa
    r2 = -d - alfa
    
    A1 = -r2/(r1-r2)
    A2 = r1/(r1-r2)
    
    exp1 = torch.exp(-d*t)
    exp2 = torch.exp(alfa*t)
    exp3 = torch.exp(-alfa*t)
    
    x = exp1*(A1*exp2 + A2*exp3)
    
    return x


# =============================================================================
# Oscilador Armónico Crítico
# =============================================================================
    

def oscilador_critico(d,t):
    
    # Ec. dif.: m(d^2x/dt^2) + mu(dx/dt) + kx = 0
    # IC: x(0) = 1 , dx/dt = 0
    # Con las condiciones iniciales se saca:
    
    # assert d == w0 # COndicion critica
    
    A1 = 1 # Cond. inic. x(0) = 1
    A2 = d # Cond. inic. dx/dt = 0
    
    exp = torch.exp(-d*t)
    
    x = exp*(A1 + t*A2)
    
    return x

# =============================================================================
# Oscilador Armónico Subamortiguado
# =============================================================================

def oscilador_amort(d, w0, t):
    
    # Ec. dif.: m(d^2x/dt^2) + mu(dx/dt) + kx = 0
    # IC: x(0) = 1 , dx/dt = 0
    # d<w0, d = mu/(2m) , w0 = sqrt(k/m) # Subamortiguado
    # Solución: x(t) = exp(-dt)(2Acos(phi+wt)) con w = sqrt(w0^2-d^2)
    
    assert d < w0 # Condicion subamortiguado
    
    w = np.sqrt(w0**2 - d**2)
    
    phi = np.arctan(-d/w) # fase inicial con dx/dt = 0
    A = 1/(2*np.cos(phi)) # x(0) = 1
    
    cos = torch.cos(phi + w*t)
    exp = torch.exp(-d*t)
    
    x  = exp*2*A*cos
    
    return x


# =============================================================================
# Oscilador numérico 1
# =============================================================================

def osc_numerico_1(y, t, k):
    x, v = y
    dydt = [v, -k*x]
    return dydt


# =============================================================================
# Oscilador numérico 2
# =============================================================================

def osc_numerico_2(y, t, mu, k):
    x, v = y
    dxdt = v
    dvdt = -mu*v - k*x
    return [dxdt, dvdt]


# =============================================================================
# PINN
# =============================================================================

# Se puede poner otra red neuronal en lugar de esta (la mía)

class PINN(nn.Module):
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        
        activation = nn.Tanh # Funcion de activacion
        
        self.fcs = nn.Sequential(*[nn.Linear(N_INPUT, N_HIDDEN),
                        activation()]) # input layer
        
        self.fch = nn.Sequential(*[nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)]) # hidden layers
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT) # output layer
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


# =============================================================================
# El problema que se quiere resolver
# =============================================================================

'''
Los datos de entrenamiento se generan en una parte de la solución, para ver si 
se adapta correctamente a la solución.
'''
# Parametros, suponiendo masa m=1, obviamente influyen en los resultados...

d, w0 = 2, 20 # delta y omega inicial, que dependen de mu, m, k del muelle, vale para el subamortiguado y simple, al simple le da igual todo
#d, w0 = 10, 1 # Condicion para el caso Sobre amortiguado // adimensional y rad/s 
#d, w0 = 10, 10 # Condicion para el criticamente amortiguado
mu, k = 2*d, w0**2 # mu kg/s y k N/m

# La solucion se coge para todo el problema, elegir la x del problema
t = torch.linspace(0,1,500).view(-1,1)
#x = oscilador_simple(w0, t).view(-1,1) # Simple
#x = oscilador_sobre(d, w0, t).view(-1,1) # Sobreamortiguado
#x = oscilador_critico(d, t).view(-1,1) # Critico
x = oscilador_amort(d, w0, t).view(-1,1) # Subamortiguado, 1e-4, 10000 epochs o 1e-3 y 20000 epochs
print(t.shape, x.shape)

# Se cogen unos puntos al inicio del dominio. SOLO COGER ESTOS x,t PARA EL PRIMER CASO
# COMPARANDO CON LA RED NEURONAL EN EL OSCILADOR SUBAMORTIGUADO, EL RESTO SIN
# DATOS DE AYUDA EXTRA, YA QUE SI NO, PARECE QUE SE NECESITE LA SOL. ANLITICA.
#t_data = t[0:200:20]
#x_data = x[0:200:20]

# Condicinoes iniciales
t_data = t[0]
x_data = x[0]
print(t_data.shape, x_data.shape)

y0 = [1,0] # posicion inicial y velocidad inicial, c.i. para el odeint
t_num = np.linspace(0,1,500)

solucion1 = odeint(osc_numerico_1, y0, t_num, args=(k,)) # osc arm simple
x_num1 = solucion1[:,0]

solucion2 = odeint(osc_numerico_2, y0, t_num, args=(mu, k)) # oscilador amortiguado
x_num2 = solucion2[:,0]

# Graficar la solución Analítica junto con la numérica en dos subplots como abajo
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))

ax1.grid()
ax1.set_xlabel('t (s)')
ax1.set_ylabel('x (m)')
ax1.set_title('Solución Analítica')
ax1.plot(t, x, color="grey", label="Solución Analítica")
ax1.scatter(t_data, x_data, color="orange", label="Datos de entrenamiento")
ax1.legend()
ax2.grid()
ax2.set_xlabel('t (s)')
ax2.set_ylabel('x (m)')
ax2.set_title('Solución Numérica')
ax2.plot(t_num, x_num2, color="black", label="Solución Numérica")
ax2.scatter(t_data, x_data, color="orange", label="Datos de entrenamiento")
ax2.legend()



# =============================================================================
# Red neuronal
# ============================================================================= 
'''
Primero se entrena el modelo con los datos de entrenamiento, para que la
NN se ajuste a los datos de entrenamiento, y utilice esos datos de entrena
miento para aprender y ajustarse solo a esos datos, no a toda la solucion
analitica, porque si no, obviamente que se ajusta. Lo que se quiere ver
es que si se ajusta solo a unos puntos, no generaliza la forma de la ecuacion
diferencial. Por eso se ajusta y aprende solo de los datos de entrenamiento
remarcados. Luego en el bucle, se evalua la red neuronal para todo el dominio,
y se comprueba que no se ajusta fuera de ellos.
'''
'''
# Entrenamiento de la red neuronal sin tener en cuenta la EDO
torch.manual_seed(123)

model = PINN(1,1,32,3) # Arquitectura de la red
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

loss_values = []
n_epochs = 17500

#fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))


for epoch in range(n_epochs):
    
    optimizer.zero_grad()

    x_pred = model(t_data)
    
    loss = torch.mean((x_pred-x_data)**2)# use mean squared error MSE
    loss_values.append(loss.item())
    loss.backward()
    
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
        
    # plot the result as training progresses
    if (epoch) % 10 == 0: 
        
        x_pred = model(t).detach()

        ax1.clear()
        ax2.clear()
        ax1.set_title('Solución')
        ax2.set_title('Pérdida')
        ax1.set_xlabel('t')
        ax2.set_xlabel('Epoch')
        ax1.set_ylabel('u')
        ax2.set_ylabel('Loss')
        ax2.set_xlim((0, n_epochs))
        #ax2.set_ylim((0, loss_values[0]))

        ax1.plot(t, x, c='grey', linewidth=2, alpha=0.8, label='Solución Analítica')
        ax1.plot(t,x_pred, color="blue", linewidth=4, alpha=0.8, label="Red Neuronal")
        ax1.scatter(t_data, x_data, c='orange', label='Data')
        
        #ax2.set_yscale('log')
        ax2.plot(range(epoch+1), loss_values, 'b-') # Pérdida total
        # Se pueden graficar todas las pérdidas...
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.pause(0.1)
        #plt.show()
'''


        
# =============================================================================
# PINN
# =============================================================================

'''
Idea de las PINNs es evaluar la varianza residual de la EDP en diferentes puntos
del dominio, para aprender la forma de la ecuacion diferencial, y luego evaluar
la red en todo el modelo. Se calcula en 30 puntos uniformes.
'''
'''
La clave es evaluar el modelo segun el tipo de perdida. Tienes unos datos
de entrenamiento, evaluas la PINN en esos datos y haces la perdida para aprender.
Tienes la EDP, evaluas la PINN en unos puntos del dominio uniformes para que
sigan la forma de la EDP, y haces la perdida para aprender, que es la misma red.
Y asi con todas las perdidas, teniendo en cuenta tambien contorno e iniciales,
aunque estas se pueden incluir en los datos de x_data, en ese termino tienes los
datos conocidos, boundary y las initial conditionsa. Asi, tienes todas las
funciones de coste para hacer el coste total de la PINN, y que aprenda como 
debe aprender una PINN. En cada caso, se evalua modelo y loss, y luego se 
juntan todos. Ahora tiene sentido otros programas donde en las funciones
evaluan unos puntos para cada perdida.
'''

'''
¡¡¡MUY IMPORTANTE!!!

Hay que multiplicarle un térnmino de 1e-4 al error de la varianza residual de la
EDP par que converja y no se quede a 0. Si se ponen terminos tipo 1e-3 o superiores
como 1e-2 o se quita, tardan mucho en convergen o directamente se quedan en 0.
Si se pone 1e-5, hay partes que convergen mas rapido que otras, pero sigue convergiendo.
'''

# Puntos para entrenar la PINN
t_pinn = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)

# Entrenamiento de la red neuronal sin tener en cuenta la EDO
torch.manual_seed(123)

model = PINN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

loss_values = []
n_epochs = 17500 # 25000 para el mas

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.set_title('Solución PINN')

ax2.set_title('Función de Pérdida')
ax1.set_xlabel('t (s)')
ax2.set_xlabel('Número de épocas')
ax1.set_ylabel('x (m)')
ax2.set_ylabel('Loss')
ax2.set_xlim((0, n_epochs))

for epoch in range(n_epochs):
    
    optimizer.zero_grad()
    
    # Pérdida de los datos de entrenamiento conocidos, que incluye las condiciones de contorno
    # DESCOMENTAR ESTO PARA EL CASO DIDACTICO PRIMERO DE LA OTRA RED A LA QUE SE LE PASAN DATOS
    #x_pred = model(t_data)
    #loss1 = torch.mean((x_pred-x_data)**2) # Error MSE
  
    # Pérdida de la ec. diferencial.
    x_pred_pinn = model(t_pinn)
    
    dx  = torch.autograd.grad(x_pred_pinn, t_pinn, torch.ones_like(x_pred_pinn), create_graph=True)[0] # dx/dt
    dx2 = torch.autograd.grad(dx,  t_pinn, torch.ones_like(dx),  create_graph=True)[0] # d^2x/dt^2
    
    # Cambio de osciladores
    
    #EDO = dx2 + k*x_pred_pinn # Varainza residual de la EC. Dif. Simple, poner 1e-3
    EDO = dx2 + mu*dx + k*x_pred_pinn # Varainza residual de la EC. Dif. amortiguada
    
    loss2 = (1e-3)*torch.mean(EDO**2)
    
    # Pérdida de los datos de entrenamiento conocidos, que incluye las condiciones de contorno
    x_pred = model(t)   
    # CONDICIONES INICIALES, MUY IMPORTANTE
    loss0 = torch.mean((x_pred[0]-torch.ones(1))**2)
    loss1 = torch.mean((dx[0]-torch.zeros(1))**2) # Error MSE
    
    # Función de pérdida de la PINN total
    loss = loss0 + loss1 + loss2 # Se tienen en cuenta todas las pérdidas
    loss_values.append(loss.item())
    loss.backward()
    
    optimizer.step()
    
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
    
    # plot the result as training progresses
    if (epoch) % 1000 == 0: 
        
        x_pred = model(t).detach()
        tp = t_pinn.detach()
        
        ax1.grid()
        ax1.plot(t, x, c='grey', linewidth=2, alpha=0.8, label='Solución Analítica')
        ax1.plot(t,x_pred, color="blue", linewidth=4, alpha=0.8, label="PINN")
        ax1.scatter(t_data, x_data, c='orange', label='Data')
        ax1.scatter(tp, -0*torch.ones_like(tp), c='purple', label='Posiciones de los datos de la PINN')
        
        #ax2.set_yscale('log')
        ax2.plot(range(epoch+1), loss_values, 'b-') # Pérdida total
        # Se pueden graficar todas las pérdidas...
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.pause(0.1)
        ax1.clear()
        ax2.clear()
        #plt.show()

    



# Se evalua el modelo y se grafica
'''
x_pred = model(t).detach()
t_pinn = torch.linspace(0,1,30).view(-1,1)
tp = t_pinn.detach()

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.set_title('Solución PINN')
ax1.grid()
ax2.set_title('Función de Pérdida')
ax1.set_xlabel('t (s)')
ax2.set_xlabel('Número de épocas')
ax1.set_ylabel('x (m)')
ax2.set_ylabel('Loss')
ax2.set_xlim((0, n_epochs))

ax1.plot(t, x, c='grey', linewidth=2, alpha=0.8, label='Solución Analítica')
ax1.plot(t,x_pred, color="blue", linewidth=4, alpha=0.8, label="PINN")
ax1.scatter(t_data, x_data, c='orange', label='Data')
ax1.scatter(tp, -0*torch.ones_like(tp), c='purple', label='Posiciones de los datos de la PINN')

#ax2.set_yscale('log')
ax2.plot(range(epoch+1), loss_values, 'b-') # Pérdida total
# Se pueden graficar todas las pérdidas...
ax1.legend(loc='best')
ax2.legend(loc='best')
'''
# =============================================================================
# Referencias
# =============================================================================

# https://github.com/benmoseley/harmonic-oscillator-pinn/blob/main/Harmonic%20oscillator%20PINN.ipynb
# https://beltoforion.de/en/harmonic_oscillator/
# https://es.wikipedia.org/wiki/Oscilador_armónico#cite_note-4

