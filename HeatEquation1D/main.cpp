static char help[] = "To solve the transient heat equation in a one-dimensional domain.\n\n";

#include <petscksp.h>

int BEuler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, KSP ksp, PC pc,
	PetscReal delta_t, PetscReal time_end, PetscReal M );

int AddBC( MPI_Comm comm, Mat A, Vec uu, Vec ff, PetscInt M, PetscReal hbc,
	PetscInt head_bc, PetscInt tail_bc, 
	PetscReal gg_h, PetscReal gg_t, PetscReal hh_h, PetscReal hh_t);

int main( int argc, char **argv )
{
	MPI_Comm	comm;
	KSP		ksp;
	PC		pc;

	// initialization
	PetscInitialize(&argc, &argv, (char*)0, help);
	comm = PETSC_COMM_WORLD;

	// variables
	PetscInt	M = 10;
	PetscInt        max_its = 100;
	PetscReal       tol = 1.0e-5;
	PetscReal	kappa = 1.0;	// thermal conductivity
	PetscReal	rho = 1.0;	// density
	PetscReal	cc = 1.0;	// specific heat capacity
	PetscReal	delta_t = 0.1;
	PetscReal	delta_x = 0.1;
	PetscReal	time_end = 1.0;
	PetscReal       gg_h = 0.0;	// prescribed temperature at x=0
	PetscReal       gg_t = 0.0;	// prescribed temperature at x=1
	PetscReal       hh_h = 1.0;	// heat flux at x=0
	PetscReal       hh_t = 1.0;	// heat flux at x=1
	PetscInt	head_bc = 0;	// boundary condition at x=0: 0 for prescribed temperature, 1 for heat flux.
	PetscInt	tail_bc = 0;	// boundary condition at x=0: 0 for prescribed temperature, 1 for heat flux.

	// get option
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-space_mesh_size",	&M, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-max_its",	&max_its, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-tol",	&tol, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-kappa",	&kappa, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-rho",	&rho, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-capacity",	&cc, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-time_step",	&delta_t, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-space_step",	&delta_x, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-time_end",	&time_end, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-gg_head",	&gg_h, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-gg_tail",	&gg_t, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-hh_head",	&hh_h, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-hh_tail",	&hh_t, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-head_bc",	&head_bc, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-tail_bc",	&tail_bc, PETSC_NULL);

	// parameter check

	// constant scalar
	const double para1 = (delta_t * kappa) / (delta_x * delta_x * rho * cc);
	const double para2 = delta_t / (rho * cc);
	PetscReal hbc = (delta_x * hh_t) / kappa;
	const double pi = 3.141592653589793;
	const double ll = 1.0;

	// spacial coordinates
	double * coor_x = new double[M+1](); 
	for(int ii=1; ii<M+1; ii++) {
		coor_x[ii] = coor_x[ii-1] + delta_x;
	}
	
	// heat source
	double * h_src = new double[M+1]();
	for(int ii=0; ii<M+1; ii++) {
		h_src[ii] = para2 * sin(ll * pi * coor_x[ii]);
	}

	Vec	ff;
	VecCreate(comm, &ff);
	VecSetSizes(ff, PETSC_DECIDE, M+1);
	VecSetFromOptions(ff);
	for(PetscInt ii=0; ii<M+1; ii++) {
		VecSetValues(ff, 1, &ii, &h_src[ii], INSERT_VALUES);
	}

	VecAssemblyBegin(ff);
	VecAssemblyEnd(ff);

//	VecView(ff, PETSC_VIEWER_STDOUT_(comm));

	// matrix A
	Mat	A;
	MatCreate(comm, &A);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M+1, M+1);
	MatSetFromOptions(A);
	MatMPIAIJSetPreallocation(A, 3, NULL, 3, NULL);
	MatSeqAIJSetPreallocation(A, 3, NULL);

	PetscInt	rstart, rend, m, n;
	MatGetOwnershipRange(A, &rstart, &rend);
	MatGetSize(A, &m, &n);
	for (PetscInt ii=rstart; ii<rend; ii++) {
		PetscInt	index[3] = {ii-1, ii, ii+1};
		PetscScalar	value[3] = {-1.0*para1, 2.0*para1+1.0, -1.0*para1};

		if (ii == 0) {
			MatSetValues(A, 1, &ii, 2, &index[1], &value[1], INSERT_VALUES);
		}
		else if (ii == n-1) {
			MatSetValues(A, 1, &ii, 2, index, value, INSERT_VALUES);
		}
		else {
			MatSetValues(A, 1, &ii, 3, index, value, INSERT_VALUES);
		}
	}

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	
//	MatView(A, PETSC_VIEWER_STDOUT_WORLD);

	// initial condition: u_0 = exp(x)
	double * u_0 = new double[M+1]();
	for(int ii=0; ii<M+1; ii++) {
		u_0[ii] = exp(coor_x[ii]);
	}

	Vec	uu;
	VecDuplicate(ff, &uu);
	for(PetscInt ii=0; ii<M+1; ii++) {
		VecSetValues(uu, 1, &ii, &u_0[ii], INSERT_VALUES);
	}

	VecAssemblyBegin(uu);
	VecAssemblyEnd(uu);

//	VecView(uu, PETSC_VIEWER_STDOUT_(comm));

	// boundary conditions
	AddBC( comm, A, uu, ff, M, hbc, head_bc, tail_bc, gg_h, gg_t, hh_h, hh_t);

	// solver
	KSPCreate(comm, &ksp);
	Vec	u_new;
	VecDuplicate(uu, &u_new);

	// implicit solver
	BEuler( comm, A, u_new, uu, ff, ksp, pc, delta_t, time_end, M );

	VecView(u_new, PETSC_VIEWER_STDOUT_(comm));

	// destory
	KSPDestroy(&ksp);
	VecDestroy(&uu);
	VecDestroy(&u_new);
	VecDestroy(&ff);
	MatDestroy(&A);

	PetscFinalize();
	delete [] coor_x; delete [] h_src; delete [] u_0;
	return 0;
}

int BEuler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, KSP ksp, PC pc,
	PetscReal delta_t, PetscReal time_end, PetscReal M )
{
	PetscInt	N = time_end / delta_t;
	PetscInt	its;
	PetscReal	rnorm;
	PetscReal	time = 0.0;

	// ksp set
	KSPSetOperators(ksp, A, A);
	KSPSetType(ksp, KSPGMRES);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCJACOBI);
	KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetFromOptions(ksp);

	// solver
	for(int ii=1; ii<N+1; ii++) {
		time += delta_t;
		VecAXPY(b, 1.0, f);
		KSPSolve(ksp, b, x);
		VecCopy(x, b);
		KSPMonitor(ksp, its, rnorm);
		PetscPrintf(comm, "======> time_index: %D\t time: %g\n", ii, (double)time);
		PetscPrintf(comm, "        KSP its: %D\t r_norm: %g\n", its, (double)rnorm);
		PetscPrintf(comm, "        solver done.\n");
	}

	return 0;
}

int AddBC( MPI_Comm comm, Mat A, Vec uu, Vec ff, PetscInt M, PetscScalar hbc, 
	PetscInt head_bc, PetscInt tail_bc, 
	PetscReal gg_h, PetscReal gg_t, PetscReal hh_h, PetscReal hh_t)
{
	if (!head_bc) {
		PetscInt	row = 0;
		PetscInt	col[2] = {0, 1};
		PetscScalar	value[2] = {1.0, 0.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		VecSetValues(uu, 1, &row, &gg_h, INSERT_VALUES);
		VecAssemblyBegin(uu);
		VecAssemblyEnd(uu);

		PetscPrintf(comm, "======> Essential BC at x=0: prescribed temperature %g\n", gg_h);
	}
	else {
		PetscInt        row = 0;
		PetscInt        col[2] = {0, 1};
		PetscScalar     value[2] = {1.0, -1.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		VecSetValues(uu, 1, &row, &hbc, INSERT_VALUES);
		VecAssemblyBegin(uu);
		VecAssemblyEnd(uu);

		PetscPrintf(comm, "======> Natural BC at x=0: heat flux %g\n", hh_h);
	}
	if (!tail_bc) {
		PetscInt	row = M;
		PetscInt	col[2] = {M-1, M};
		PetscScalar	value[2] = {0.0, 1.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		VecSetValues(uu, 1, &row, &gg_t, INSERT_VALUES);
		VecAssemblyBegin(uu);
		VecAssemblyEnd(uu);
		PetscPrintf(comm, "======> Essential BC at x=0: prescribed temperature %g\n", gg_t);
	}
	else {
		PetscInt        row = M;
		PetscInt        col[2] = {M-1, M};
		PetscScalar     value[2] = {1.0, -1.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		VecSetValues(uu, 1, &row, &hbc, INSERT_VALUES);
		VecAssemblyBegin(uu);
		VecAssemblyEnd(uu);

		PetscPrintf(comm, "======> Natural BC at x=1: heat flux %g\n", hh_t);
	}

	VecView(uu, PETSC_VIEWER_STDOUT_(comm));
	VecView(ff, PETSC_VIEWER_STDOUT_(comm));
	MatView(A, PETSC_VIEWER_STDOUT_WORLD);

	return 0;
}

