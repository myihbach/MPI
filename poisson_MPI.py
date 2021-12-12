#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:34:21 2020



/*
 *   Solving the Poisson's equation discretized on the [0,1]x[0,1] domain
 *   using the finite difference method and a Jacobi's iterative solver.
 *
 *   Delta u = f(x,y)= 2*(x*x-x+y*y -y)
 *   u equal 0 on the boudaries
 *   The exact solution is u = x*y*(x-1)*(y-1)
 *
 *   The u value is :
 *    coef(1) = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
 *    coef(2) = 1./(hx*hx)
 *    coef(3) = 1./(hy*hy)
 *
 *    u(i,j)(n+1)= coef(1) * (  coef(2)*(u(i+1,j)+u(i-1,j)) &
 *               + coef(3)*(u(i,j+1)+u(i,j-1)) - f(i,j))
 *
 *   ntx and nty are the total number of interior points along x and y, respectivly.
 * 
 *   hx is the grid spacing along x and hy is the grid spacing along y.
 *    hx = 1./(ntx+1)
 *    hy = 1./(nty+1)
 *
 *   On each process, we need to:
 *   1) Split up the domain
 *   2) Find our 4 neighbors
 *   3) Exchange the interface points
 *   4) Calculate u
 *
 */
"""
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
import meshio
from psydac.ddm.partition import mpi_compute_dims



comm = MPI.COMM_WORLD
nb_procs = comm.Get_size()
rank = comm.Get_rank()

a,b,c,d = 0,1,0,1 # The domain [0,1]x[0,1]

nb_neighbours = 4
N = 0
E = 1
S = 2
W = 3

neighbour = np.zeros(nb_neighbours, dtype=np.int8)
ntx = 10
nty = 10

Nx = ntx+2
Ny = nty+2

npoints  =  [ntx, nty]
p1 = [2,2]
P1 = [False, False]
reorder = True


coef = np.zeros(3)
''' Grid spacing '''
hx = 1./(ntx+1.);
hy = 1./(nty+1.);

''' Equation Coefficients '''
coef[0] = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy);
coef[1] = 1./(hx*hx);
coef[2] = 1./(hy*hy);

def create_2d_cart(npoints, p1, P1, reorder):
    
    # Store input arguments                                                                                                                                                                                                                                               
    npts    = tuple(npoints)
    pads    = tuple(p1)
    periods = tuple(P1)
    reorder = reorder    
    nprocs, block_shape = mpi_compute_dims(nb_procs, npts, pads )
    
    dims = nprocs
    
    if (rank == 0):
        print("Execution poisson with",nb_procs," MPI processes\n"
               "Size of the domain : ntx=",npoints[0], " nty=",npoints[1],"\n"
               "Dimension for the topology :",dims[0]," along x", dims[1]," along y\n"
               "-----------------------------------------")  
    
    # Creation of the Cartesian topology

    cart2d = comm.Create_cart(dims=dims, periods= periods, reorder=reorder) 
    return dims, cart2d

def create_2dCoords(cart2d, npoints, dims):

    ''' Create 2d coordinates of each process'''
    # the lenght of the domaine of each proc
    l_x , l_y = (b-a)/dims[0] , (d-c)/dims[1] 
    # coordinates of proc
    x , y = cart2d.Get_coords(rank)
    # intervals of proc
    sx , ex = a+x*l_x , a+(x+1)*l_x
    sy , ey = c+y*l_y , c+(y+1)*l_y

    #print("Rank in the topology :",rank," Local Grid Index :", sx, " to ",ex," along x, ",sy, " to", ey," along y")
    return sx, ex, sy, ey

def create_neighbours(cart2d):

    ''' Get my northern and southern neighbours '''
    neighbour[2] , neighbour[0] = cart2d.Shift(direction=1,disp=1)
    ''' Get my western and eastern neighbours '''
    neighbour[3] , neighbour[1]= cart2d.Shift(direction=0,disp=1)

    #print("Process", rank," neighbour: N", neighbour[N]," E",neighbour[E] ," S ",neighbour[S]," W",neighbour[W])
    return neighbour



''' Exchange the points at the interface '''
#def communications(u, sx, ex, sy, ey, type_column, type_ligne):
def communications():   
    ''' Send to neighbour N and receive from neighbour S '''
    if neighbour[N] != -2:
        comm.send(u_new[2,:],dest=neighbour[N])
        u_new[0,:] = comm.recv(source=neighbour[N])

    ''' Send to neighbour S and receive from neighbour N '''
    if neighbour[S] != -2:
        comm.send(u_new[-3,:],dest=neighbour[S])
        u_new[-1,:] = comm.recv(source=neighbour[S])

    ''' Send to neighbour E and receive from neighbour W '''
    if neighbour[E] != -2:
        comm.send(u_new[:,-3],dest=neighbour[E])
        u_new[:,-1] = comm.recv(source=neighbour[E])

    ''' Send to neighbour W  and receive from neighbour E '''
    if neighbour[W] != -2:
        comm.send(u_new[:,2],dest=neighbour[W])
        u_new[:,0] = comm.recv(source=neighbour[W])

"""
## Update boundaries condition
def update_boundaries(u):
    if neighbour[N] == -2:
        u[1,:] = 0
    if neighbour[S] == -2:
        u[-2,:] = 0
    if neighbour[E] == -2:
        u[:,-2] = 0
    if neighbour[W] == -2:
        u[:,1] = 0
    return u"""


def initialization(Nx,Ny):
    ''' Grid spacing in each dimension'''
    ''' Solution u and u_new at the n and n+1 iterations '''

    SIZE = (Ny,Nx)
    u       = np.zeros(SIZE,dtype=np.float64)
    u_new   = np.zeros(SIZE,dtype=np.float64)
    u_exact , f = compute_u_exact_f()

    '''Initialition of rhs f and exact soluction '''
    return u, u_new, u_exact ,f

def compute_u_exact_f():
    u_exact = np.zeros((Ny,Nx),dtype=np.float64)
    f = np.zeros((Ny,Nx),dtype=np.float64)
    x = np.linspace(sx,ex,ntx)
    y = np.linspace(ey,sy,nty) ## switch sy with ey to remain the cartesian perspective of y 
    for i in range(nty):
        for j in range(ntx):
            f[i+1,j+1] = 2 * ( x[j]**2 - x[j] + y[i]**2 - y[i] )
            u_exact[i+1,j+1] = x[j]*y[i]*(x[j]-1)*(y[i]-1)

    return u_exact , f


def computation():
    #Compute the new value of u 
    
    for i in range(1,Ny-1):
        for j in range(1,Nx-1):
            u_new[i,j] = coef[0] * ( coef[1] * ( u[i+1,j] + u[i-1,j] ) + coef[2]*( u[i,j+1] + u[i,j-1] ) - f[i,j])
    
    #u_new[:,:] = update_boundaries(u_new)    
    if neighbour[N] == -2 :  u_new[1,:]  = 0
    if neighbour[S] == -2 :  u_new[-2,:] = 0
    if neighbour[E] == -2 :  u_new[:,-2] = 0
    if neighbour[W] == -2 :  u_new[:,1]  = 0
    

"""def output_results(u, u_exact):
    print("Exact Solution u_exact - Computed Solution u - difference")
    for itery in range(sy, ey+1, 1):
        print(u_exact[IDX(1, itery)], '-', u[IDX(1, itery)], u_exact[IDX(1, itery)]-u[IDX(1, itery)] );
"""

#Calcul for the global error (maximum of the locals errors)
def global_error():
    errors = abs(u[1:-2,1:-2]-u_new[1:-2,1:-2])
    return errors.max()


def plot_2d(u):

    x = np.linspace(sx,ex,ntx)
    y = np.linspace(ey,sy,nty)

    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = fig.gca(projection='3d')                      
    X, Y = np.meshgrid(x, y)      

    ax.plot_surface(X, Y, u, cmap=cm.viridis)
    
    plt.show()


dims, cart2d   = create_2d_cart(npoints, p1, P1, reorder)
neighbour      = create_neighbours(cart2d)
sx, ex, sy, ey = create_2dCoords(cart2d, npoints, dims)
u, u_new, u_exact , f = initialization(Nx,Ny)


''' Time stepping '''
it = 0
convergence = False
it_max = 10000
eps = 2.e-6

''' Elapsed time '''
t1 = MPI.Wtime()

#import sys; sys.exit()
while (not(convergence) and (it < it_max)):
    it = it+1;
    u[:,:] = u_new[:,:]
    ''' Computation of u at the n+1 iteration '''
    computation()
    '''sychronisation'''
    comm.Barrier()
    ''' Exchange of the interfaces at the n iteration '''
    communications()
    ''' Computation of the global error '''
    local_error = global_error();
    diffnorm = comm.allreduce(np.array(local_error), op=MPI.MAX )   
    ''' Stop if we obtained the machine precision '''
    convergence = (diffnorm < eps)
    
    ''' Print diffnorm for process 0 '''
    
    if ((rank == 0) and ((it % 500) == 0)):
        print("Local error : ",local_error)
        print("Global error : ",diffnorm)
        print("Iteration", it);
        
''' Elapsed time '''
t2 = MPI.Wtime()

if (rank == 0):
    ''' Print convergence time for process 0 '''
    print("Convergence after",it, 'iterations in', t2-t1,'secs')

    ''' Compare to the exact solution on process 0 '''
    #output_results(u, u_exact)
    print(u_new[1:-2,1:-2].round(3))
    print("=======================")
    print(u_exact[1:-2,1:-2].round(3))
    #plot_2d(u_exact[1:-2,1:-2])

    """ print(u_new[1:10,-12:-2].round(3))
    print("=======================")
    print(u_exact[1:10,-12:-2].round(3))"""
