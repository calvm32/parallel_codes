import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from schur_complement import *

"""
Solve the 1D Poisson equation - lap(u) = f,
with Dirichlet boundary conditions
"""

# set up laplacian matrix
def create_laplacian_matrix(Nx, hx):
    A = -2 * np.eye(Nx) + np.diag(np.ones(Nx-1), 1) + np.diag(np.ones(Nx-1), -1)
    A /= hx**2 # scale by the grid spacing^2

    return A

def serial_poisson(Lx, Nx, f):
    """
    use 1d backward differentiation to solve the poisson equation
    """
    # length between elements
    hx = Lx / (Nx - 1)
    x = np.linspace(0, Lx, Nx)

    A = create_laplacian_matrix(Nx, hx)

    # ----------------
    # solve the system
    # ----------------

    f_vec = f.flatten()
    u_vec = np.linalg.solve(A, f_vec)
    u = u_vec.reshape(Nx)

    return u

def parallel_poisson(comm, rank, size, Lx, Nx, f):
    """
    use 1d backward differentiation to solve the poisson equation
    use Schur complement to write the system as [[A11, A12], [A21, A22]]
        - Aii = local interior points
        - Aij = Aji^T = interior and interface relations
        - Ann = interface relations
    """
    # length between elements
    hx = Lx / (Nx - 1)

    # local grid size
    chunk_size = ceil(Nx / size)
    start = rank * chunk_size
    end = min(start + chunk_size, Nx)
    local_N = end - start - 2 # exclude the two endpoints = interface nodes
    block_size_list = [] # to perform parallel Schur
    C = [0] # final block
    
    # local interior points
    if local_N > 0: # need at least 3 points for interior
        Aii = create_laplacian_matrix(local_N, hx)
        bi = f[start-1:end-1]
    else: # small chunk -> just return f
        Aii = np.array([[1]])
        bi = f[start:end]

    # treat boundaries as interface nodes for now
    Fi_left = np.zeros(local_N)
    Fi_right = np.zeros(local_N)
    if start != 0:
        Fi_left[0] = 1.0 / hx**2
    if end != Nx:
        Fi_right[-1] = 1.0 / hx**2

    Ai_inv_Fi = np.linalg.solve(Aii, Fi)
    Ai_inv_bi = np.linalg.solve(Aii, bi)

    # solve for interface nodes
    Si = Fi.T @ Ai_inv_Fi
    zi = Ei @ Ai_inv_bi

    # sum interface node solutions globally
    S_global = comm.allreduce(Si, op=MPI.SUM)
    z_global = comm.allreduce(zi, op=MPI.SUM)

    S = C - S_global
    z = bS - z_global

    # solve for interior nodes
    if rank == 0:
        xS = np.linalg.solve(S, z)
    xS = comm.bcast(xS, root=0)
    local_u = Ai_inv_bi - Ai_inv_Fi@xS

    # gather solns
    global_u = None
    if rank == 0:
        global_u = np.zeros(Nx)
    comm.Gather(local_u, global_u, root=0)

    return global_u

def main():

    # set up mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ----------------------
    # define problem + solve
    # ----------------------

    Lx = 1.0 # domain size
    Nx = 50 # number of pts

    # source term
    f = np.zeros(Nx)

    if rank == 0: # serial solution for comparison
        u_serial = serial_poisson(Lx, Nx, f)
        print("Serial solution:", u_serial)

    u_parallel = parallel_poisson(comm, rank, size, Lx, Nx, f)

    # --------
    # plotting
    # --------

    if rank == 0:
        x = np.linspace(0, Lx, Nx)
        plt.plot(x, u_parallel, label='Parallel (local solves)')
        plt.plot(x, u_serial, '--', label='Serial')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('1D Poisson equation')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()