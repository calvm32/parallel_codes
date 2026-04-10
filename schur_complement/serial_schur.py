import numpy as np

def serial_schur(A, block1_size, block2_size):
    
    A11 = A[:block1_size, :block1_size]
    A12 = A[:block1_size, block2_size:]
    A21 = A[block2_size:, :block1_size]
    A22 = A[block2_size:, block2_size:]
    
    # Schur complement S = A22 - A21*inv(A11)*A12
    A11_inv = np.linalg.inv(A11)
    S = A22 - A21 @ A11_inv @ A12

    return S


def main():
    # test random matrix w size N
    N = 6
    block1_size = 3
    block2_size = N - block1_size
    A = np.random.rand(N, N)

    S = serial_schur(A, block1_size, block2_size)

    print("serial Schur complement:\n", S)

if __name__ == "__main__":
    main()