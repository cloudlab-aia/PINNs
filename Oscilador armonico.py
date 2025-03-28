# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:59:26 2023

@author: David Muñoz Hernández
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
# Oscilador armónico
# =============================================================================

# EDP del oscilador armónico

def pinn_oscillator(x, t):
    # Variables de la ecuación
    omega = 1  # frecuencia natural del oscilador armónico
    c = 0.5    # coeficiente de amortiguamiento
    k = 1     # constante de elasticidad del resorte

    # Definir la variable de entrada
    u = torch.tensor([x, t], requires_grad=True) # Esto no esta bien

    # Derivadas parciales de la solución respecto a x y t
    du = torch.autograd.grad(u.sum(), u, create_graph=True)
    d2u = torch.autograd.grad(du[0].sum(), u, create_graph=True)

    # Ecuación diferencial parcial
    f = d2u[0][1] + 2 * c * omega * du[0][1] + omega**2 * u[0] - k * u[1]

    return f


# Solución analítica

def analytical_solution(x, t):
    return torch.cos(x) * torch.cos(t)


# Definimos los datos de entrenamiento, separar en training, test y validación cruzada.
n_train = 5000
batch_size = 100

x_train = np.random.uniform(low=-np.pi, high=np.pi, size=(batch_size)) # De -pi a pi en el espacio
# Incluir los valores -π y π en x_train
# x_train[0] = -np.pi
# x_train[-1] = np.pi
t_train = np.random.uniform(low=0, high=2*np.pi, size=(batch_size)) # De 0 a 2*pi en el tiempo
x_train = x_train.reshape((batch_size, 1))
t_train = t_train.reshape((batch_size, 1))
u_train = np.concatenate((x_train,t_train), axis=1)
# Hay que hacer un u para cada condicion, para pasarselo al modelo, ya que no se usa x,t sino u.
# esto es para las funciones de boundary y de ic. Condiciones de contorno
x_boundary1 = (np.ones(batch_size)*-np.pi).reshape((batch_size,1))
u_boundary1 = np.concatenate((x_boundary1,t_train), axis=1)
x_boundary2 = (np.ones(batch_size)*np.pi).reshape((batch_size,1))
u_boundary2 = np.concatenate((x_boundary2,t_train), axis=1)
t_0 = np.zeros(batch_size).reshape((batch_size,1))
u_ic = np.concatenate((x_train,t_0), axis=1)

# Convertimos los datos a tensores de PyTorch
x_train = torch.from_numpy(x_train).float()
t_train = torch.from_numpy(t_train).float()
u_train = torch.from_numpy(u_train).float()
u_boundary1 = torch.from_numpy(u_boundary1).float()
u_boundary2 = torch.from_numpy(u_boundary2).float()
u_ic = torch.from_numpy(u_ic).float()
'''
X,T = torch.meshgrid(x_train,t_train)
y_real = analytical_solution(X, T)
plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x_train,t_train,y_real)
'''
# Definimos la red neuronal
input_size = 2
hidden_size = 20
output_size = 1

class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, u):
        #x = x.unsqueeze(dim=1)  # Agregar dimensión adicional a x para tener (batch_size,1)
        #t = t.unsqueeze(dim=1)  # Agregar dimensión adicional a t para tener (batch_size,1)
        #inputs = torch.cat([x, t],axis=1)  # Concatenar las variables x y t
        #inputs = torch.hstack((x,t))
        u = nn.functional.relu(self.fc1(u))
        u = nn.functional.relu(self.fc2(u))
        u = nn.functional.relu(self.fc3(u))
        u = self.fc4(u)
        return u


# Definimos las condiciones de contorno e iniciales para la ecuación. Esto define el problema a resolver

def pinn_boundary(u_boundary1, u_boundary2):
    # Condiciones de contorno en el extremo izquierdo (x=-π)
    bc1 = model(u_boundary1)#.squeeze().item()
    # EL squeeze() es para eliminar cualquier dimension adicional en el tensor que no sea necesaria
    # Así el tensor tiene un solo elemento antes de convertirlo a un escalar de python
    # Condiciones de contorno en el extremo derecho (x=π)
    bc2 = model(u_boundary2)#.squeeze().item()

    return bc1, bc2


def initial_conditions(u_ic):
    # Condiciones de contorno en el tiempo inicial (t=0)
    ic = model(u_ic)#.squeeze().item()

    return ic
    
    
# Definimos la función de pérdida para la PINN  del oscilador (ARREGLAR ESTO, HACER EN LA PROPIA ÉPOCA)
  
def pinn_loss(model, x, t, u_train, u_analytic, u_boundary1, u_boundary2, u_ic):
    
    x.requires_grad_(True)
    t.requires_grad_(True)
    u_train.requires_grad_(True)
    u_pred = model(u_train)
    # Derivadas parciales
    grad_outputs = torch.ones_like(u_pred)  # Tensor de unos con las mismas dimensiones que u_pred
    
    #dif_u = torch.autograd.grad(u_pred, u_train, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, allow_unused=True)[0]
    #print(dif_u)
    #u_x = dif_u[:,0].view(batch_size,1)
    #u_t = dif_u[:,1].view(batch_size,1)
    #print(u_t)
    u_x = torch.autograd.grad(u_pred, x.view(1,batch_size), grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    u_t = torch.autograd.grad(u_pred, t.view(1,batch_size), grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    #dif2_u = torch.autograd.grad(dif_u, u_train, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, allow_unused=True)[0]
    
    #u_xx = torch.autograd.grad(u_x, x.view(1,batch_size), grad_outputs=grad_outputs, create_graph=True, allow_unused=True)[0]
    
    # Ecuacion diferencial
    f = u_t + u_pred #- u_xx 
    
    # Pérdida de los datos de entrenamiento
    loss_u = torch.mean((u_pred - u_analytic)**2) 
    # Pérdida de las condicinoes de contorno
    bc1_pred, bc2_pred = pinn_boundary(u_boundary1, u_boundary2)
    loss_bc = torch.mean((bc1_pred - 0.0) ** 2) + torch.mean((bc2_pred - 0.0) ** 2)
    # Pérdida de las condiciones iniciales
    ic_pred = initial_conditions(u_ic)
    loss_ic = torch.mean((ic_pred - 0.0) ** 2)
    # Pérdida de la varianza residual de la EDP. Cumplimiento de la EDP
    loss_f = torch.mean(f**2)
    
    # Pérdida total
    loss = loss_f + loss_bc + loss_ic + loss_u
    
    return loss
    

# Entrenamiento de la red neuronal
model = PINN(input_size, hidden_size, output_size)

criterion = nn.MSELoss() #esto es la funcion de coste de aliexpress, sin mi funcion
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 10000
loss_values = []


fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
x_train2 = x_train.detach().numpy() 
t_train2 = t_train.detach().numpy() 
X, T = np.meshgrid(x_train2, t_train2)
# Hacer una gráfica en 3D con x,t,u no solo con x,u


for epoch in range(n_epochs):
    # CAMBIAR LOS DATOS DE ENTRADA CADA CIERTAS EPOCAS PARA EVITAR OVERFITTING
    '''
    #Cada 30000 epocas, la entrenamos con datos aleatorios de entrada
    if (epoch+1)%30000==0:
        X_numpy = np.random.rand(batch_size) #np.linspace(0, 10, 100) #np.random.rand(100)*10 
        X_numpy[0] = 0
        X_numpy[-1] = 1
        X = torch.from_numpy(X_numpy.astype(np.float32))
        X = X.view(X.shape[0], 1) 
        X.requires_grad=True
        
        for i in range(len(X_numpy)):
            rho1[i] = rho(X_numpy[i])
        RHO = torch.from_numpy(rho1.astype(np.float32))
        RHO = RHO.view(RHO.shape[0], 1)
        f_numpy=Analytic(X_numpy,1,0.4,0.6)
    '''
    # Reiniciar los gradientes
    optimizer.zero_grad()

    # Evaluar la red en los datos de entrenamiento. Probar luego diferentes estrategias de ML 
    u_pred = model(u_train)
    
    # Evaluar la función analítica en los mismos datos
    u_analytic = analytical_solution(x_train,t_train)

    # Calcular la pérdida
    #loss = criterion(u_pred, u_analytic) # Solo incluye los datos conocidos, faltan las otras perdidas
    loss = pinn_loss(model, x_train, t_train, u_train, u_analytic, u_boundary1, u_boundary2, u_ic)
    loss_values.append(loss.item())

    # Calcular los gradientes y actualizar los pesos
    loss.backward()
    optimizer.step()

    # Imprimir la pérdida en cada época
    if epoch % 500 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
        
    # Animación de la convergencia de la solución pINN hacia la solución analítica
    if epoch % 500 == 0:
        u_pred = u_pred.detach().numpy()
        u_analytic = u_analytic.detach().numpy()

        ax1.clear()
        ax2.clear()
        ax1.set_title('Solución')
        ax2.set_title('Pérdida')
        ax1.set_xlabel('x')
        ax2.set_xlabel('Epoch')
        ax1.set_ylabel('t')
        ax1.set_zlabel('u')
        ax2.set_ylabel('Loss')
        ax1.set_xlim((-np.pi, np.pi))
        #ax1.set_ylim((-1.5, 1.5))
        #ax1.set_zlim((-1.5,1.5))
        ax2.set_xlim((0, n_epochs))
        #ax2.set_ylim((0, loss_values[0]))
        
        ax1.scatter(x_train2, t_train2, u_pred, cmap='viridis') # mirar alternativas como wireframe o contourf
        ax1.scatter(x_train2, t_train2, u_analytic, cmap='inferno')

        '''
        ax1.plot(x_train, u_pred, 'b-', label='PINN')
        ax1.plot(x_train, u_analytic, 'r--', label='Analytic')
        '''
        #ax2.set_yscale('log')
        ax2.plot(range(epoch+1), loss_values, 'b-') # Pérdida total
        # Se pueden graficar todas las pérdidas...
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.pause(0.1)
        #plt.show()

    
'''    
# Se evalua la red, hacer una separación en datos de test y training  
with torch.no_grad():
    y_test_pred = PINN(x_test)
    test_loss = loss_fn(y_test_pred, y_test)

print(f"Test Loss: {test_loss.item():.4f}")
'''    
    
# =============================================================================
# Se grafican los resultados
# =============================================================================
    
# Se puede graficar después del entrenamiento y no en cada iteración del bucle
    
    
    
    
    
    
    
    