import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def compute_integrale_rectangle(x, y):
    nbi = len(x)
    integrale = 0.
    for i in range(nbi-1):
        integrale = integrale + y[i]*(x[i+1]-x[i])  
    return integrale

def plot_integrale(x, y):
    nbi = len(x)
    for i in range(nbi-1):
        # dessin du rectangle
        x_rect = [x[i], x[i], x[i+1], x[i+1], x[i]] # abscisses des sommets
        y_rect = [0   , y[i], y[i]  , 0     , 0   ] # ordonnees des sommets
        plt.plot(x_rect, y_rect,"r")
    plt.plot(x,y,'bo-')
    plt.show()

def f(x):
    return np.cos(x)

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

xmin = 0
xmax = 3*np.pi/2
nbx = 20
x = np.linspace(xmin,xmax,nbx)
y = f(x)
nbr = int(nbx/SIZE)

for i in range(SIZE):
    if RANK == SIZE - 1 :
        x1 = x[nbr*(SIZE-1):]
        y1 = y[nbr*(SIZE-1):]
    elif RANK == i:
        x1 = x[nbr*i:nbr*(i+1) +1]
        y1 = y[nbr*i:nbr*(i+1) +1]

sendbuf = compute_integrale_rectangle(x1,y1)
print("I am Processor ",RANK," My value is ",sendbuf)

integral_val = COMM.reduce(sendbuf,op=MPI.SUM,root=0)

if RANK == 0 :
    print(integral_val)
    plot_integrale(x,y)






