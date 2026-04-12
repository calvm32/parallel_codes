import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from .schur_complement import *

# set up laplacian matrix
def create_laplacian_matrix(Nx, Ny, hx, hy):
    # 1D Laplacian in x direction
    Ax = -4 * np.eye(Nx) + np.diag(np.ones(Nx-1), 1) + np.diag(np.ones(Nx-1), -1)
    Ax /= hx**2 # Scaling by the grid spacing squared

    # 1D Laplacian in y direction
    Ay = -4 * np.eye(Ny) + np.diag(np.ones(Ny-1), 1) + np.diag(np.ones(Ny-1), -1)
    Ay /= hy**2 # Scaling by the grid spacing squared

    I_Ny = np.eye(Ny)
    I_Nx = np.eye(Nx)
    
    A = np.kron(I_Ny, Ax) + np.kron(Ay, I_Nx)
    return A

def serial_poisson(Lx, Ly, Nx, Ny, f):
    """
    use 2d backward differentiation to solve the laplace equation
    """
    # setup heights for matrix
    hx, hy = Lx / (Nx - 1), Ly / (Ny - 1)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    create_laplacian_matrix(Nx, Ny, hx, hy)

    # ----------------
    # solve the system
    # ----------------

    f_vec = f.flatten()
    u_vec = np.linalg.solve(A, f_vec)
    u = u_vec.reshape((Nx, Ny))

    return u

def parallel_poisson(comm, rank, size, Lx, Ly, Nx, Ny, f):
    """
    use 2d backward differentiation to solve the laplace equation
    use Schur complement to write the system as [[A11, A12], [A21, A22]]
        - A11 = local interior points
        - A12, A21 = local boundary points
        - A22 = neighboring points
    """
    # setup heights for matrix
    hx, hy = Lx / (Nx - 1), Ly / (Ny - 1)

    # local grid size
    chunk_size_x = Nx // size
    start_x = rank * chunk_size_x
    end_x = min(start_x + chunk_size_x, Nx)

    # create A11 and solve A11*u_i = f_i
    local_A = create_laplacian_matrix(end_x - start_x, Ny, hx, hy)
    local_f = f[start_x:end_x, :].flatten()
    local_u = np.linalg.solve(local_A, local_f)
    
    # update boundaries A12, A21
    if rank < size - 1: # all but final rank updates next
        comm.send(local_u[-1, :], dest=rank + 1)
    if rank > 0:
        comm.send(local_u[0, :], dest=rank - 1)

    if rank > 0: # all but first rank updates prev
        neighbor_data = comm.recv(source=rank - 1)
    if rank < size - 1:
        neighbor_data = comm.recv(source=rank + 1)

    print(neighbor_data)

    # After computing the interior solution, apply Schur complement to update boundary terms
    # Schur complement updates the boundary: u_b = A22^-1 * A21 * u_i
    # (You would need to set up A21 and A22, which come from the communication step)

    # gather solns
    global_u = None
    if rank == 0:
        global_u = np.zeros((Nx, Ny))
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

    Lx, Ly = 1.0, 1.0 # domain size
    Nx, Ny = 50, 50 # number of pts

    # source term
    f = np.zeros((Nx, Ny))
    f[int(Nx/4), int(Ny/4)] = 100 # Adding a point source at (1/4, 1/4)

    u = serial_poisson(Lx, Ly, Nx, Ny, f)
    u = parallel_poisson(comm, rank, size, Lx, Ly, Nx, Ny, f)

    # --------
    # plotting
    # --------

    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, u.T, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('Solution to Poisson equation')

    plt.show()

if __name__ == "__main__":
    main()