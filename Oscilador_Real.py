# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:20:17 2023

@author: Usuario
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker

# Use GPU if available

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print("Using {} device".format(device))

# =============================================================================
# Oscilador armónico simple
# =============================================================================

# Ecuación: m*(d2y/dt2) = -k*y
# Solución: A*cos(wt+phi)
# Parámetros físicos
A=1
w=1
phi=0
m=1
k=1
u0 = 0
v0 = 0
Nu = 2 # Numero de puntos de condiciones de contorno
Nt = 100 # numero de puntos de entrenamiento de la PDE
Nd = 10 # numero de puntos de entrenamiento de los datos aleatorios

# Funciones de utilidad: analiticas y de contorno

def analitica(t):
    return A*torch.cos(w*t+phi)


# =============================================================================
# PINN
# =============================================================================

# Definimos la PINN con las funciones de pérdida
input_size = 1 # Segun le paso (x), (t) o (x,t) es 1 neurona o 2 o las que sean (,,,)
hidden_size = 32
output_size = 1

class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.loss_function = nn.MSELoss()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)  
    
    def forward(self, u):
        #x = x.unsqueeze(dim=1)  # Agregar dimensión adicional a x para tener (batch_size,1)
        #t = t.unsqueeze(dim=1)  # Agregar dimensión adicional a t para tener (batch_size,1)
        #inputs = torch.cat([x, t],axis=1)  # Concatenar las variables x y t
        #inputs = torch.hstack((x,t))
        u = nn.functional.relu(self.fc1(u))
        u = nn.functional.relu(self.fc2(u))
        u = nn.functional.relu(self.fc3(u))
        u = nn.functional.relu(self.fc4(u))
        u = self.fc5(u)
        return u

    'LOSS FUNCTIONS'

    # Loss BC
    '''
    def lossBOUND(self,x_boundary):
      # CC:
      loss_BC=self.loss_function(self.forward(x_boundary),analitica(x_boundary))
      
      return loss_BC
    '''
    # Loss IC
    def lossIC(self,x_ic):
      # CI: u(t=0) = u0
      #x_ic = u0
      '''
      loss_IC1 = self.loss_function(self.forward(x_ic[0]),torch.ones_like(1)*u0)
      loss_IC2 = self.loss_function(self.forward(x_ic[-1]),torch.ones_like(1)*v0) 
      loss_IC = loss_IC1 + loss_IC2
      '''
      loss_IC = self.loss_function(self.forward(x_ic),torch.ones_like(x_ic)*analitica(x_ic))
      
      return loss_IC
  
    # Loss PDE (f) (varianza residual)
    def lossPDE(self,x_train):
      g = x_train.clone()
      g.requires_grad=True #Enable differentiation
      
      f = self.forward(g)
      
      f_t = torch.autograd.grad(f, g, torch.ones_like(f),retain_graph=True, create_graph=True)[0] # Repasar esto
      f_tt = torch.autograd.grad(f_t, g, torch.ones_like(f_t),retain_graph=True, create_graph=True)[0]
      loss_PDE = self.loss_function(m*f_tt,-k*f) # La ecuacion diferencial EDP
      
      return loss_PDE
  
    # Loss Datos
    def lossDATA(self, x_data): # Si se diferenciara x_analytic, cambiar x_train por x_analytic
        loss_anal = self.loss_function(self.forward(x_data),analitica(x_data))
        return loss_anal
    
    # LOSS TOTAL
    def loss(self,x_ic,x_train,x_data):
        #loss_bc = self.lossBOUND(x_boundary)
        loss_ic = self.lossIC(x_ic)
        loss_pde = self.lossPDE(x_train)
        loss_data = self.lossDATA(x_data)
      
        return (1e-4)*loss_pde + loss_ic + loss_data  #+ loss_bc


# =============================================================================
# El problema que se quiere resolver
# =============================================================================

# Datos de entrada. POSIBLE NORMALIZACION, ENTRE 0 Y 1 PARA QUE DE MEJOR

batch_size = 300
dominio = torch.linspace(0,3*np.pi,batch_size).view(-1,1) # en realidad es tiempo de 0 a 3*pi
y = analitica(dominio).view(-1,1) # esto es espacio x(t)
print(dominio.shape)

# Condicinoes iniciales
Bc1 = dominio[0,:]
Bc2 = dominio[-1,:]
y_Bc1 = y[0,:]
y_Bc2 = y[-1,:]

# Total Training points BC1+BC2
all_train=torch.vstack([Bc1,Bc2]) # Puntos de las condiciones de contorno
y_all_train=torch.vstack([y_Bc1,y_Bc2])

#Select Nu points
idx = np.random.choice(all_train.shape[0], Nu, replace=False) 
x_ic = all_train[idx]
 
x_train = Bc1 + (Bc2-Bc1)*torch.rand(1,Nt) # Muestreo aleatorio uniforme
x_data = (Bc1 + (Bc2-Bc1)*torch.rand(1,Nd)).view(-1,1)
x_train = torch.vstack((x_train.view(-1,1),x_ic)) # Con el view(-1,1) lo pongo en la dimension correcta
print(x_ic)
print(x_train.shape)
print(x_data)
print(x_ic.shape)

# gráfica solucion analitica
plt.figure()
plt.grid()
plt.plot(dominio, y, label="Solucion analitica")
plt.scatter(all_train, y_all_train, color="orange", label="Condiciones iniciales")
plt.legend()
plt.show()

# =============================================================================
# Entrenamiento
# =============================================================================

model = PINN(input_size, hidden_size, output_size)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 40000
loss_values = []


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
plt.grid()

for epoch in range(n_epochs):
    # CAMBIAR LOS DATOS DE ENTRADA CADA CIERTAS EPOCAS PARA EVITAR OVERFITTING

    # Reiniciar los gradientes
    optimizer.zero_grad()

    # Evaluar la red en los datos de entrenamiento. Probar luego diferentes estrategias de ML 
    u_pred = model(x_train)
    
    # Evaluar la función analítica en los mismos datos
    u_analytic = analitica(x_train)

    # Calcular la pérdida
    #loss = criterion(u_pred, u_analytic) # Solo incluye los datos conocidos, faltan las otras perdidas
    loss = model.loss(all_train,x_train,x_data)
    loss_values.append(loss.item())

    # Calcular los gradientes y actualizar los pesos
    loss.backward()
    optimizer.step()

    # Imprimir la pérdida en cada época
    if epoch % 1000 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
        
    # Animación de la convergencia de la solución pINN hacia la solución analítica
    if epoch % 1000 == 0:
        u_pred = u_pred.detach()
        u_analytic = u_analytic.detach()
        x_plot = x_train.detach()
        x_data_plot = x_data.detach()

        ax1.clear()
        ax2.clear()
        ax1.set_title('Solución')
        ax1.grid()
        ax2.set_title('Pérdida')
        ax1.set_xlabel('t')
        ax2.set_xlabel('Epoch')
        ax1.set_ylabel('u')
        ax2.set_ylabel('Loss')
        #ax1.set_xlim((-np.pi, np.pi))
        #ax1.set_ylim((-1.5, 1.5))
        #ax1.set_zlim((-1.5,1.5))
        ax2.set_xlim((0, n_epochs))
        #ax2.set_ylim((0, loss_values[0]))

        ax1.scatter(x_plot, u_analytic, c='grey', linewidth=2, alpha=0.8, label='Analítica') # Como son random, el plot queda mal, mejor scatter
        ax1.scatter(x_plot,u_pred, color="blue", linewidth=4, alpha=0.8, label="PINN")
        ax1.scatter(all_train, y_all_train, c='orange', label='IC')
        ax1.scatter(x_data_plot, analitica(x_data_plot), c='orange', label = 'Data')
        ax1.scatter(x_plot, -0*torch.ones_like(x_plot), c='purple', label='Posiciones de los datos de la PINN')
        
        #ax2.set_yscale('log')
        ax2.plot(range(epoch+1), loss_values, 'b-') # Pérdida total
        # Se pueden graficar todas las pérdidas...
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.pause(0.1)
        #plt.show()

























    