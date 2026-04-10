# ----------------------------
# Poisson equation on a square
# ----------------------------

from firedrake import *

def build_problem(mesh_size, parameters, aP=None, block_matrix=False):
	"""
	aP: optional preconditioning
	block_matrix: monolithic or 2x2 block
	""" 
	mesh = UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)

	Sigma = FunctionSpace(mesh, "RT", 1)
	V = FunctionSpace(mesh, "DG", 0)
	W = Sigma * V

	sigma, u = TrialFunctions(W)
	tau, v = TestFunctions(W)

	rg = RandomGenerator() # hold the forcing term
	f = rg.uniform(V)

	# weak form:
	a = dot(sigma, tau)*dx + div(tau)*u*dx + div(sigma)*v*dx
	L = -f*v*dx

	if aP is not None:
		aP = aP(W)

	parameters['pmat_type'] = 'nest' if block_matrix else 'aij'

	w = Function(W)
	vpb = LinearVariationalProblem(a, L, w, aP=aP)
	solver =  LinearVariationalSolver(vpb, solver_parameters=parameters)

	return solver, w

# -----------
# "naive" pcs
# -----------

parameters = {
    "ksp_type": "gmres",
    "ksp_gmres_restart": 100,
    "ksp_rtol": 1e-8,
    "pc_type": "ilu",
    }

print("Naive preconditioning")
for n in range(8):
    solver, w = build_problem(n, parameters, block_matrix=False)
    solver.solve()
    print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

# ---------------
# exact schur pcs
# ---------------

parameters = {
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-8,

    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",

    # find exactly A^-1
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "ilu",
    "fieldsplit_0_ksp_rtol": 1e-12,

    # find exactly S^-1
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_pc_type": "none",
    "fieldsplit_1_ksp_rtol": 1e-12,
}

print("Exact full Schur complement")
for n in range(8):
    solver, w = build_problem(n, parameters, block_matrix=True)
    solver.solve()
    print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

# ---------------------
# approximate schur pcs
# ---------------------

parameters = {
    "ksp_type": "fgmres",
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "ilu",
    "fieldsplit_0_ksp_rtol": 1e-12,
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_ksp_rtol": 1e-12,

    # note S_p = -C * diag(A^-1) * B is a good approximate for S (same spectrum) but sparse
    # construct S_p using diagonal of A
    "pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_1_pc_type": "hypre"
}

print("Schur complement with S_p")
for n in range(8):
    solver, w = build_problem(n, parameters, block_matrix=True)
    solver.solve()
    print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

# This is much better, the problem takes much less time to solve and
# when observing the iteration counts for inverting :math:`S` we can see
# why.
#
# ============== ==================
#  Mesh elements CG iterations on S
# ============== ==================
#    2                  2
#    8                  8
#    32                 17
#    128                18
#    512                19
#    2048               19
#    8192               19
#    32768              19
# ============== ==================
#
# We can now think about backing off the accuracy of the inner solves.
# Effectively computing a worse approximation to :math:`P` that we hope
# is faster, despite taking more GMRES iterations.  Additionally we can
# try dropping some terms in the factorisation of :math:`P`, by adjusting
# ``pc_fieldsplit_schur_fact_type`` from ``full`` to one of ``upper``,
# ``lower``, or ``diag`` we make the preconditioner slightly worse, but
# gain because we require fewer applications of :math:`A^{-1}`.  For our
# problem where computing :math:`A^{-1}` is cheap, this is not a great
# problem, however for many fluids problems :math:`A^{-1}` is expensive
# and it pays to experiment.
#
# For example, we might wish to try a full factorisation, but
# approximate :math:`A^{-1}` by a single application of ILU(0) and
# :math:`S^{-1}` by a single multigrid V-cycle on :math:`S_p`.  To do
# this, we use the following set of parameters. ::

parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_1_pc_type": "hypre"
}

# Note how we can switch back to GMRES here, our inner solves are linear
# and so we no longer need a flexible Krylov method. ::

print("Schur complement with S_p and inexact inner inverses")
for n in range(8):
    solver, w = build_problem(n, parameters, block_matrix=True)
    solver.solve()
    print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

# This results in the following GMRES iteration counts
#
# ============== ==================
#  Mesh elements  GMRES iterations
# ============== ==================
#    2                  2
#    8                  9
#    32                 11
#    128                13
#    512                13
#    2048               12
#    8192               12
#    32768              12
# ============== ==================
#
# and the solves take only a few seconds.
#
# Providing the Schur complement approximation
# ++++++++++++++++++++++++++++++++++++++++++++
#
# Instead of asking PETSc to build an approximation to :math:`S` which
# we then use to solve the problem, we can provide one ourselves.
# Recall that :math:`S` is spectrally a Laplacian only in a
# discontinuous space.  A natural choice is therefore to use an interior
# penalty DG formulation for the Laplacian term on the block of the scalar
# variable. We can provide it as an :class:`~.AuxiliaryOperatorPC` via a python preconditioner. Note that the ```form``` method in ```AuxiliaryOperatorPC``` takes the test functions as the first argument and the trial functions as the second argument, which is the reverse of the usual convention. ::

class DGLaplacian(AuxiliaryOperatorPC):
    def form(self, pc, v, u):
        W = u.function_space()
        n = FacetNormal(W.mesh())
        alpha = Constant(4.0)
        gamma = Constant(8.0)
        h = CellSize(W.mesh())
        h_avg = (h('+') + h('-'))/2
        a_dg = -(inner(grad(u), grad(v))*dx \
            - inner(jump(u, n), avg(grad(v)))*dS \
            - inner(avg(grad(u)), jump(v, n), )*dS \
            + alpha/h_avg * inner(jump(u, n), jump(v, n))*dS \
            - inner(u*n, grad(v))*ds \
            - inner(grad(u), v*n)*ds \
            + (gamma/h)*inner(u, v)*ds)
        bcs = None
        return (a_dg, bcs)

parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": __name__+ ".DGLaplacian",
    "fieldsplit_1_aux_pc_type": "hypre"
}

print("DG approximation for S_p")
for n in range(8):
    solver, w = build_problem(n, parameters, aP=None, block_matrix=False)
    solver.solve()
    print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())

# This actually results in slightly worse convergence than the diagonal
# approximation we used above.
#
# ============== ==================
#  Mesh elements  GMRES iterations
# ============== ==================
#     2                 2
#     8                 9
#     32                12
#     128               13
#     512               14
#     2048              13
#     8192              13
#     32768             13
# ============== ==================
#
# Block diagonal preconditioners
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# An alternate approach to using a Schur complement is to use a
# block-diagonal preconditioner.  To do this, we note that the
# mesh-dependent ill conditioning of linear operators comes from working
# in the wrong norm.  To convert into working in the correct norm, we
# can precondition our problem using the *Riesz map* for the spaces.
# For details on the mathematics behind this approach see for example
# :cite:`Kirby:2010`.
#
# We are working in a space :math:`W \subset H(\text{div}) \times L^2`,
# and as such, the appropriate Riesz map is just :math:`H(\text{div})`
# inner product in :math:`\Sigma` and the :math:`L^2` inner product in
# :math:`V`.  As was the case for the DG Laplacian, we do this by
# providing a function that constructs this operator to our
# ``build_problem`` function. ::

def riesz(W):
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    return (dot(sigma, tau) + div(sigma)*div(tau) + u*v)*dx

# Now we set up the solver parameters.  We will still use a
# ``fieldsplit`` preconditioner, but this time it will be additive,
# rather than a Schur complement. ::

parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "additive",

# Now we choose how to invert the two blocks.  The second block is easy,
# it is just a mass matrix in a discontinuous space and is therefore
# inverted exactly using a single application of zero-fill ILU. ::

#
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "ilu",

# The :math:`H(\text{div})` inner product is the tricky part. For a
# first attempt, we will invert it with a direct solver.  This is a reasonable
# option up to a few tens of thousands of degrees of freedom. ::

#
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
}


print("Riesz-map preconditioner")
for n in range(8):
	solver, w = build_problem(n, parameters, aP=riesz, block_matrix=True)
	solver.solve()
	print(w.function_space().mesh().num_cells(), solver.snes.ksp.getIterationNumber())
