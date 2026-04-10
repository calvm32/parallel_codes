#!/bin/bash
#SBATCH --job-name=python_parallel
#SBATCH --output=/work/larios/alarios/res.out
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:01:00

# --- Environment Setup ---
module purge                # Optional: Clears existing modules to avoid conflicts
module load python/3.9

# --- Execution ---
python3 parallel_hello.py
