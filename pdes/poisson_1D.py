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

    chunk_size = ceil(Nx / size) # best if Nx >> size
    start = rank * chunk_size
    end = min(start + chunk_size, Nx)

    # -------------------------
    # GLOBAL interface indexing
    # -------------------------

    # interface nodes are the boundaries between subdomains
    is_left = (start != 0)
    is_right = (end != Nx)

    # local interior nodes (strict)
    I_global = np.arange(start + 1, end - 1)

    # local unknown vector (full local stencil block)
    n_local = end - start
    A_local = create_laplacian_matrix(n_local, hx)

    # local RHS
    b_local = f[start:end].copy()

    # ------------------------
    # LOCAL interface handling
    # ------------------------
    S_local_idx = []
    S_global_idx = []

    if is_left:
        S_local_idx.append(0)
        S_global_idx.append(start)

    if is_right:
        S_local_idx.append(n_local - 1)
        S_global_idx.append(end - 1)

    S_local_idx = np.array(S_local_idx, dtype=int)
    S_global_idx = np.array(S_global_idx, dtype=int)

    I_local_idx = np.array([i for i in range(n_local) if i not in S_local_idx])

    # ------------
    # Local blocks
    # ------------

    Aii = A_local[np.ix_(I_local_idx, I_local_idx)]
    Fi  = A_local[np.ix_(I_local_idx, S_local_idx)]
    Ei  = A_local[np.ix_(S_local_idx, I_local_idx)]
    Cii = A_local[np.ix_(S_local_idx, S_local_idx)]

    bi = b_local[I_local_idx]
    bS_local = b_local[S_local_idx] if len(S_local_idx) > 0 else np.zeros(0)

    Ai_inv_bi = np.linalg.solve(Aii, bi)
    Ai_inv_Fi = np.linalg.solve(Aii, Fi)

    Si = Cii - Ei @ Ai_inv_Fi
    zi = bS_local - Ei @ Ai_inv_bi

    # mash together contributions and combine them globally
    S_contrib = np.zeros((Nx, Nx))
    z_contrib = np.zeros(Nx)

    for local_i, global_i in enumerate(S_global_idx):
        z_contrib[global_i] += zi[local_i]
        for local_j, global_j in enumerate(S_global_idx):
            S_contrib[global_i, global_j] += Si[local_i, local_j]

    S_global = comm.allreduce(S_contrib, op=MPI.SUM)
    z_global = comm.allreduce(z_contrib, op=MPI.SUM)

    # ------------------
    # solve Schur system
    # ------------------
    if rank == 0:
        S = S_global.copy()
        z = z_global.copy()

        # Dirichlet BCs
        S[0, :] = 0
        S[0, 0] = 1
        z[0] = 0

        S[-1, :] = 0
        S[-1, -1] = 1
        z[-1] = 0

        xS = np.linalg.solve(S, z)
    else:
        xS = None

    xS = comm.bcast(xS, root=0)

    if len(I_local_idx) > 0:
        local_u = Ai_inv_bi - Ai_inv_Fi @ xS[S_global_idx]
    else:
        local_u = np.array([])

    # bather + rebuild
    counts = comm.gather(len(local_u), root=0)

    if rank == 0:
        displs = np.cumsum([0] + counts[:-1])
        global_u_interior = np.zeros(sum(counts))
    else:
        displs = None
        global_u_interior = None

    comm.Gatherv(local_u, (global_u_interior, counts, displs, MPI.DOUBLE), root=0)

    if rank == 0:
        u_full = np.zeros(Nx)

        offset = 0
        for r in range(size):
            start_r = r * chunk_size
            end_r = min((r + 1) * chunk_size, Nx)

            # fill interior
            inner_len = counts[r]
            u_full[start_r+1:end_r-1] = global_u_interior[offset:offset+inner_len]
            offset += inner_len

        # fill interfaces
        u_full[:] = xS

        return u_full

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
        #print("Serial solution:", u_serial)

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
        plt.savefig("parallel_vs_serial.png", dpi=200, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()