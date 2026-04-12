import numpy as np
from mpi4py import MPI
from numpy.linalg import solve

def parallel_schur(A, block1_size, block2_size, comm, rank, size):
    
    A11 = A[:block1_size, :block1_size]
    A12 = A[:block1_size, block2_size:]
    A21 = A[block2_size:, :block1_size]
    A22 = A[block2_size:, block2_size:]
    
    # Schur complement S = A22 - A21*inv(A11)*A12
    X = solve(A11, A12)
    rows_per_proc = A21.shape[0] // size # split up rows of A21 ?
    columns = []

    # divide up into rows
    start_row = rank*rows_per_proc #?? unclear
    end_row = min((rank+1)* rows_per_proc, A21.shape[0])
    local_result = A21[start_row:end_row] @ X
    local_result_flat = local_result.flatten()
    local_count = len(local_result_flat)

    # Gather counts and displacements
    counts = comm.gather(local_count, root=0)  # number of elements each rank will send

    if rank == 0:
        displacements = np.insert(np.cumsum(counts[:-1]), 0, 0)
        global_result_flat = np.empty(sum(counts), dtype=A.dtype)
    else:
        displacements = None
        global_result_flat = None

    # Use Gatherv for variable-length data
    comm.Gatherv(sendbuf=local_result_flat, recvbuf=(global_result_flat, counts, displacements, MPI.DOUBLE), root=0)

    if rank == 0:
        global_result = global_result_flat.reshape(A21.shape[0], A12.shape[1])
        global_S = A22 - global_result
        return global_S
    else:
        return None


def main():

    # set up mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # test random matrix w size N
    A = None
    N = 6
    block1_size = 3
    block2_size = N - block1_size

    # make the matrix and broadcast it to all processors
    if rank == 0:
        A = np.random.rand(N, N)
    A = comm.bcast(A, root=0) 

    S = parallel_schur(A, block1_size, block2_size, comm, rank, size)

    if rank == 0:
        print("parallel Schur complement:\n", S)

if __name__ == "__main__":
    main()
