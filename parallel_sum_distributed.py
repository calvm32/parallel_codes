import multiprocessing
import random
import os

def hello_task(rank):
    """Simple Hello World task."""
    print(f"[Process {rank}] Hello! I am operating on OS PID: {os.getpid()}")

def sum_task(data_chunk, rank):
    """Performs a local sum on a provided list chunk."""
    local_sum = sum(data_chunk)
    print(f"[Process {rank}] Received chunk: {data_chunk} | Local Sum: {local_sum}")
    return local_sum

if __name__ == "__main__":
    num_procs = 8
    items_per_proc = 10
    total_elements = num_procs * items_per_proc

    # 1. Generate the global list in the Parent process
    global_data = [random.randint(1, 100) for _ in range(total_elements)]
    
    print(f"--- Parent Process ---")
    print(f"Generated Global List: {global_data}")
    print(f"Serial Global Sum: {sum(global_data)}")
    print(f"----------------------\n")

    # 2. Prepare the chunks (Slicing the list)
    # This creates a list of lists: [ [1...10], [11...20], ... ]
    chunks = [global_data[i:i + items_per_proc] for i in range(0, total_elements, items_per_proc)]

    # 3. Initialize the Pool
    with multiprocessing.Pool(processes=num_procs) as pool:
        
        # Run Hello Task
        pool.map(hello_task, range(num_procs))
        print("")

        # 4. Run the Parallel Sum
        # starmap allows us to pass multiple arguments (the chunk AND the rank)
        local_sums = pool.starmap(sum_task, zip(chunks, range(num_procs)))

    # 5. Final Reduction (The "MPI_SUM" equivalent)
    global_parallel_sum = sum(local_sums)
    
    print(f"\n----------------------")
    print(f"Final Parallel Global Sum: {global_parallel_sum}")
    print(f"Verification: {'Success' if global_parallel_sum == sum(global_data) else 'Failure'}")
