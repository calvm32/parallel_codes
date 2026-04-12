# ---------------------------------------
# Stokes equations in a lid-driven cavity
# ---------------------------------------

from firedrake import *
from firedrake.petsc import PETSc

N = 64
mesh = UnitSquareMesh(N, N)

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)

a = (inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q))*dx

L = inner(Constant((0, 0)), v) * dx

# -------------------
# boundary conditions
# -------------------

# velocity BCs
bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),       # u = (1,0) on lid
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]  # 0 on walls

up = Function(Z)

# pressure defined up to constant, which is handled w/ a nullspace
nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# ------------
# direct solve
# ------------

# w/ sparse direct solver MUMPS
# direct factorization instead of iterative krylov subspace method (NO preconditioners)
parameters = {
    "ksp_type": "gmres",                    # solver type = generalized minimal residual, 
    "mat_type": "aij",                      # matrix factor type = general sparse (parallel) matrix w/ info about "adjacent rows w/ identical nonzero structure"
    "pc_type": "lu",                        # preconditioner = LU decomposition, which easily approximates the inverse
    "pc_factor_mat_solver_type": "mumps"    # direct solver provided = mumps (MUltifrontal Massively Parallel Solver)
}

solve(a == L, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters)

# ---------------
# iterative solve
# ---------------

# Schur complement preconditioner w/ unassembled matrices
parameters = {
    "mat_type": "matfree",                                      # unassembled matrix type

    # use Schur compliment factorization to approximate the inverse
    "ksp_type": "gmres",                                        # basic configuration is GMRES (generalized minimal residual) solver
    "ksp_monitor_true_residual": None,                          # monitor convergence
    "ksp_view": None,                                           # view Krylov solver object
    "pc_type": "fieldsplit",                                    # split into velocity and pressure fields
    "pc_fieldsplit_type": "schur",                              # makes preconditioner use a schur factorization to solve the block matrix
    "pc_fieldsplit_schur_fact_type": "diag",                    # retains the diagonals of the factorization (default is full)

    # velocity block: approximate the inverse of thevector laplacian using a single multigrid V-cycle.::
    "fieldsplit_0_ksp_type": "preonly",                         # apply preconditioner ONCE (no krylov iterations)
    "fieldsplit_0_pc_type": "python",                           # allows preconditioner directly from python code (we'll offload the work to firedrake)
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",     # constructs pc by assembling sparse matrix and 
    "fieldsplit_0_assembled_pc_type": "hypre",                  # Hypre algebraic multigrid processor is applied to **assembled operators** rather than matfree forms

    # schur complement block: approximate the inverse of the schur complement w/ a pressure mass inverse
    "fieldsplit_1_ksp_type": "preonly",                         # apply preconditioner ONCE (no krylov iterations)
    "fieldsplit_1_pc_type": "python",                           # allows preconditioner directly from python code (we'll offload the work to firedrake)
    "fieldsplit_1_pc_python_type": "firedrake.MassInvPC",       # inverts the mass matrix in the provided space, used to handle viscosity

    # The mass inverse is dense, and therefore approximated with ILU
    "fieldsplit_1_Mp_mat_type": "aij",                          # general sparse (parallel) matrix w/ info about "adjacent rows w/ identical nonzero structure"
    "fieldsplit_1_Mp_pc_type": "ilu"                            # incomplete LU factorization
 }

up.assign(0)
solve(a == L, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters)

u, p = up.subfunctions
u.rename("Velocity")
p.rename("Pressure")

VTKFile("stokes.pvd").write(u, p)