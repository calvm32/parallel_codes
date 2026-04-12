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

for n in range(8):
    solver, w = build_problem(n, parameters, block_matrix=True)
    solver.solve()