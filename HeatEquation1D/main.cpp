static char help[] = "To solve the transient heat equation in a one-dimensional domain.\n\n";

#include <petscksp.h>

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

	// parameter check

	// constant scaler
	const double para1 = (delta_t * kappa) / (delta_x * delta_x * rho * cc);
	const double para2 = delta_t / (rho * cc);
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

	VecView(ff, PETSC_VIEWER_STDOUT_(comm));

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
		PetscInt    index[3] = {ii-1, ii, ii+1};
		PetscScalar value[3] = {-1.0*para1, 2.0*para1+1.0, -1.0*para1};
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
	
	MatView(A, PETSC_VIEWER_STDOUT_WORLD);

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

	VecView(uu, PETSC_VIEWER_STDOUT_(comm));

//	// ksp solver
//	KSPCreate(comm, &ksp);
//	KSPSetOperators(ksp, A, A);
//	KSPGetPC(ksp, &pc);
//	PCSetType(pc, PCJACOBI);
//	KSPSetTolerances(ksp, 1.e-2/m, 1.e-50, PETSC_DEFAULT, PETSC_DEFAULT);
//	KSPSetFromOptions(ksp);
//
//	// solve
//        PetscInt        its = 0;
//        PetscReal       yoldNorm = 0.0, ynewNorm = 0.0;
//        PetscScalar     y = 0.0;
//        PetscReal       err = 10.0;
//        while (its < max_its && err > tol) {
//                VecCopy(ynew, yold);
//		VecCopy(znew, zold);
//
//                KSPSolve(ksp, zold, ynew);
//
//                VecNorm(ynew, NORM_2, &ynewNorm);
//                VecNorm(yold, NORM_2, &yoldNorm);
//
//                y = 1.0 / ynewNorm;
//                VecAXPBY(znew, y, 0.0, ynew);
//
//                err = PetscAbsReal(yoldNorm - ynewNorm);
//                its++;
//        }
//
//	PetscReal       lamda;
//	KSPSolve(ksp, znew, ynew);
//        VecTDot(ynew, znew, &lamda);
//	lamda = 1.0 / lamda;
//
//	// output
//	PetscPrintf(comm, "Error: %g,\t Iterations: %D\n", (double)err, its);
//        PetscPrintf(comm, "The smallest eigenvalue: %g\n", (double)lamda);

	// destory
//	KSPDestroy(&ksp);
	VecDestroy(&uu);
	VecDestroy(&ff);
	MatDestroy(&A);

	PetscFinalize();
	delete [] coor_x; delete [] h_src; delete [] u_0;
	return 0;
}
