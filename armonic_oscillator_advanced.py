# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:59:48 2023

@author: Usuario
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Considero MASA = 1 sin pérdida de generalidad
# Mirar osciladores como: forzado, forzado y caos, van der Pol, de torsion
# Pequeñas oscilaciones
# Oscilador cuántico, 1D y 2D

# =============================================================================
# Oscilador Armónico Forzado
# =============================================================================

def oscilador_forzado(w,w0,d,f0,t):
    
    # Ec. dif.: m(d^2x/dt^2) + mu(dx/dt) + kx = f(t)
    # f(t) = f0*cos(wt)
    # IC: x(0) = 0 , dx/dt = 0
    # Con las condiciones iniciales se saca:
    # C1 = C2 = -(A*cos(-phi))/2
    # A1 = -r2/(r1-r2)
    # A2 = r1/(r1-r2)
    # Solcuión:  x = A*cos(w*t-d) + exp(-bt)*(C1*exp(sqrt(b^2-w0^2*t)) + C2*exp(-sqrt(b^2-w0^2*t)))
    
    A = np.sqrt(f0**2/((w0**2-w**2)**2+4*d**2*w**2))
    phi = np.arctan((2*d*w)/(w0**2-w**2))
    cos1 = torch.cos(w*t-phi)
    cos2 = np.cos(-phi)
    cos3 = torch.cos(np.sqrt(w0**2-d**2)*t)
    exp = torch.exp(-d*t)
    
    x = A*cos1-A*cos2*exp*cos3
    
    return x


def f(f0,w,t):
    cos = torch.cos(w*t)
    return f0*cos


# =============================================================================
# Oscilador numérico 3
# =============================================================================

def osc_numerico_3(y, t, mu, k, f0, w):
    x, v = y
    dxdt = v
    dvdt = f0*np.cos(w*t) - mu*v - k*x
    return [dxdt, dvdt]


# =============================================================================
# PINN
# =============================================================================

# Se puede poner otra red neuronal en lugar de esta (la mía)

class PINN(nn.Module):
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
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
# Oscilador forzado
w = 2*np.pi
w0 = 5*w
d = w0/20
f0 = 1000

mu, k = 2*d, w0**2

# La solucion se coge para todo el problema, elegir la x del problema
t = torch.linspace(0,2.5,500).view(-1,1)
x = oscilador_forzado(w, w0, d, f0, t).view(-1,1) # Oscilador forzado
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

y0 = [0,0] # posicion inicial y velocidad inicial, c.i. para el odeint
t_num = np.linspace(0,2.5,500)

solucion3 = odeint(osc_numerico_3, y0, t_num, args=(mu, k, f0, w))
x_num3 = solucion3[:,0]


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
ax2.plot(t_num, x_num3, color="black", label="Solución Numérica")
ax2.scatter(t_data, x_data, color="orange", label="Datos de entrenamiento")
ax2.legend()
     
        
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
t_pinn = torch.linspace(0,2.5,200).view(-1,1).requires_grad_(True) # Probar con 50

# Entrenamiento de la red neuronal sin tener en cuenta la EDO
torch.manual_seed(123)

model = PINN(1,1,64,4)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

loss_values = []
n_epochs = 150000 # con 150000 va bien, se puede probar con menos

# fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))


for epoch in range(n_epochs):

    optimizer.zero_grad()
  
    # Pérdida de la ec. diferencial.
    x_pred_pinn = model(t_pinn)
    
    dx  = torch.autograd.grad(x_pred_pinn, t_pinn, torch.ones_like(x_pred_pinn), create_graph=True)[0] # dx/dt
    dx2 = torch.autograd.grad(dx,  t_pinn, torch.ones_like(dx),  create_graph=True)[0] # d^2x/dt^2
    
    # Cambio de osciladores
    
    #EDO = dx2 + k*x_pred_pinn # Varainza residual de la EC. Dif. Simple
    EDO = dx2 + mu*dx + k*x_pred_pinn - f(f0,w,t_pinn) # Varainza residual de la EC. Dif. forzada
    
    loss2 = (2e-4)*torch.mean(EDO**2)

    # Pérdida de los datos de entrenamiento conocidos, que incluye las condiciones de contorno
    x_pred = model(t)   
    # CONDICIONES INICIALES, MUY IMPORTANTE
    loss0 = torch.mean((x_pred[0]-torch.zeros(1))**2)
    loss1 = torch.mean((dx[0]-torch.zeros(1))**2) # Error MSE
    
    # Función de pérdida de la PINN total
    loss = loss0 + loss1 + loss2 # Se tienen en cuenta todas las pérdidas
    loss_values.append(loss.item())
    loss.backward()
    
    optimizer.step()
    
    '''
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
    
    # plot the result as training progresses
    if (epoch) % 100 == 0: 
        
        x_pred = model(t).detach()
        tp = t_pinn.detach()
        
        ax1.clear()
        ax2.clear()
        ax1.set_title('Solución')
        ax1.grid()
        ax2.set_title('Pérdida')
        ax1.set_xlabel('t')
        ax2.set_xlabel('Epoch')
        ax1.set_ylabel('u')
        ax2.set_ylabel('Loss')
        ax2.set_xlim((0, n_epochs))
        #ax2.set_ylim((0, loss_values[0]))

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
        #plt.show()
    '''
        

# Se evalua el modelo y se grafica

x_pred = model(t).detach()
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
    
        
# =============================================================================
# Referencias
# =============================================================================

# https://cocalc.com/share/public_paths/c2bc8c88b2cb2b3ebfcd560dae5526a4b0f8a5aa
