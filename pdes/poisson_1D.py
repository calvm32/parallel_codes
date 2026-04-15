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
    """
    1D backwards differentiation tridiagonal matrix
    """
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

    chunk_size = ceil(Nx / size) # sort of requires Nx >> size
    start = rank * chunk_size
    end = min(start + chunk_size, Nx)
    local_N = end - start - 2 # exclude the two endpoints = interface nodes
    k = size+1
    xS = np.zeros(2)
    print(f"Rank {rank}: local_N = {local_N}")

    if local_N <= 0: #bookkeeping for trivial blocks, altho try not to let those happen
        Aii = None
        bi = None
        Fi = None
        Ai_inv_Fi = None
        Ai_inv_bi = None
        Si = np.zeros((2,2))
        zi = np.zeros(2)
    else:
        Aii = create_laplacian_matrix(local_N, hx)
        bi = f[start+1:end-1]

        # -------------------------------
        # interior/interface interactions
        # -------------------------------
        Fi = np.zeros((local_N, 2))
        if start != 0:
            Fi[0, 0] = 1.0 / hx**2
        if end != Nx:
            Fi[-1, 1] = 1.0 / hx**2

        Ai_inv_Fi = np.linalg.solve(Aii, Fi)
        Ai_inv_bi = np.linalg.solve(Aii, bi)

        # -------------------------
        # solve for interface nodes
        # -------------------------
        Ei = Fi.T
        Si = Ei @ Ai_inv_Fi
        zi = Ei @ Ai_inv_bi
        
    # sum interface node solutions globally
    S_global = comm.allreduce(Si, op=MPI.SUM)
    z_global = comm.allreduce(zi, op=MPI.SUM)
    print(f"Rank {rank}: S_global = {S_global}")

    # ------------------------
    # solve for interior nodes
    # ------------------------
    if rank == 0:
        # interface bookkeeping
        C = np.zeros((k, k))
        bS = np.zeros(k)

        S_global = np.zeros((k, k))
        z_global = np.zeros(k)
        
        # piece S and z back from local contributions
        rows = [rank, rank+1]
        for i_local, i_global in enumerate(rows):
            for j_local, j_global in enumerate(rows):
                S_global[i_global, j_global] += Si[i_local, j_local]

        for i_local, i_global in enumerate([rank, rank+1]):
            z_global[i_global] += zi[i_local]

        S = C - S_global
        z = bS - z_global
        xS = np.linalg.solve(S, z)

    xS = comm.bcast(xS, root=0)

    xS_local = np.array([xS[rank], xS[rank+1]])
    local_u = Ai_inv_bi - Ai_inv_Fi@xS_local
    print(f"Rank {rank}: local_u = {local_u}")

    # gather solns
    counts = comm.gather(len(local_u), root=0)

    if rank == 0:
        displs = np.cumsum([0] + counts[:-1])
        global_u = np.zeros(sum(counts))
    else:
        displs = None

    comm.Gatherv(local_u, (global_u, counts, displs, MPI.DOUBLE), root=0)

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
    f = np.ones(Nx)

    if rank == 0: # serial solution for comparison
        u_serial = serial_poisson(Lx, Nx, f)
        print("Serial solution:", u_serial)

    u_parallel = parallel_poisson(comm, rank, size, Lx, Nx, f)

    # --------
    # plotting
    # --------

    if rank == 0:
        x = np.linspace(0, Lx, Nx)
        plt.plot(x, u_parallel, label='Parallel')
        plt.plot(x, u_serial, '--', label='Serial')
        plt.plot(x, abs(u_serial - u_parallel), '--', label='Error')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('1D Poisson equation')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()