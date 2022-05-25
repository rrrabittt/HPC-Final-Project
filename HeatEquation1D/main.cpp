static char help[] = "To solve the transient heat equation in a one-dimensional domain.\n\n";

#include <petscksp.h>

int main( int argc, char **argv )
{
	MPI_Comm	comm;
	Vec		zold, znew, yold, ynew;
	Mat		A;
	KSP		ksp;
	PC		pc;

	// initialization
	PetscInitialize(&argc, &argv, (char*)0, help);
	comm = PETSC_COMM_WORLD;

	// variables
	PetscInt	m = 5;
	PetscInt        max_its = 100;
        PetscReal       tol = 1.0e-5;
	PetscReal	kappa = 1.0;	// thermal conductivity
	PetscReal	rho = 1.0;	// density
	PetscReal	cc = 1.0;	// specific heat capacity
	PetscReal	delta_t = 1.0e-2;
	PetscReal	delta_x = 1.0e-2;

	// get option
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-mat_size", &m, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-max_its", &max_its, PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-tol", &tol, PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-kappa", &kappa, PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-rho", &rho, PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-capacity", &cc, PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-time_step", &delta_t, PETSC_NULL);
        PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-space_step", &delta_x, PETSC_NULL);

	// constant scaler
	const double para = (delta_t * kappa) / (delta_x * delta_x * rho * cc);

	// generate vector and matrix
	VecCreate(comm, &znew);
	VecSetSizes(znew, PETSC_DECIDE, m);
	VecSetFromOptions(znew);
	VecDuplicate(znew, &zold);
        VecDuplicate(znew, &yold);
        VecDuplicate(znew, &ynew);
	VecSet(znew, 0.0);
        VecSet(ynew, 0.0);

	VecSetValue(znew, 0, 1.0, INSERT_VALUES);

	VecAssemblyBegin(znew);
	VecAssemblyEnd(znew);

	VecView(znew, PETSC_VIEWER_STDOUT_(comm));

	MatCreate(comm, &A);
	MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, m);
	MatSetFromOptions(A);
	MatMPIAIJSetPreallocation(A, 3, NULL, 3, NULL);
	MatSeqAIJSetPreallocation(A, 3, NULL);

        PetscInt	rstart, rend, M, N;
        MatGetOwnershipRange(A, &rstart, &rend);
        MatGetSize(A, &M, &N);
        for (PetscInt i=rstart; i<rend; i++) {
                PetscInt    index[3] = {i-1, i, i+1};
		PetscScalar value[3] = {-1.0*para, 2.0*para+1.0, -1.0*para};
                if (i == 0) {
                        MatSetValues(A, 1, &i, 2, &index[1], &value[1], INSERT_VALUES);
                }
		else if (i == N-1) {
			MatSetValues(A, 1, &i, 2, index, value, INSERT_VALUES);
		}
		else {
                        MatSetValues(A, 1, &i, 3, index, value, INSERT_VALUES);
		}
        }

        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	
	MatView(A, PETSC_VIEWER_STDOUT_WORLD);

	MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE);

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
//
//	// destory
//	KSPDestroy(&ksp);
	VecDestroy(&zold);
        VecDestroy(&znew);
        VecDestroy(&yold);
        VecDestroy(&ynew);
        MatDestroy(&A);

	PetscFinalize();
	return 0;
}
