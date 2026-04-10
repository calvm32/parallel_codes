import multiprocessing
import os

def hello_task(number):
    print(f"Hello, I am process {number} (OS PID: {os.getpid()})")

if __name__ == "__main__":
    num_procs = 8
    
    with multiprocessing.Pool(processes=num_procs) as pool:
        pool.map(hello_task, range(num_procs))
