import numpy as np
from mpi4py import MPI

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
    end_row = (rank+1)* rows_per_proc
    if end_row >= A21.shape[0]:
        end_row = A21.shape[0]

    for row in range(start_row, end_row + 1):
        # calculate A21 * A11_inv * A12
        column = A21[row-1]*X
        columns.append(column)

    local_result = np.array(local_result) # shape: (num_local_rows, A12.shape[1])
    local_result_flat = local_result.flatten() # flatten before sending

    # on root, preallocate flattened array
    if rank == 0:
        global_result_flat = np.empty(A21.shape[0] * A12.shape[1], dtype=A.dtype)
    else:
        global_result_flat = None

    comm.Gather(local_result_flat, global_result_flat, root=0)

    if rank == 0:
        global_result = global_result_flat.reshape(A21.shape[0], A12.shape[1])

    if rank == 0:
        global_S = A22 - global_result # should be cheap enough
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
