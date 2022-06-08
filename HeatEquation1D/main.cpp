static char help[] = "To solve the transient heat equation in a one-dimensional domain.\n\n";

#include <petscksp.h>
#include "petscviewerhdf5.h"

// solver tools
void FEuler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, PetscInt M,
	PetscReal delta_t, PetscReal time_start, PetscReal time_end, PetscInt head_bc, PetscInt tail_bc,
	PetscReal hbc_h, PetscReal hbc_t, PetscReal gg_h, PetscReal gg_t,
	PetscBool is_record, PetscInt record_frq );

void BEuler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, KSP ksp, PC pc,
	PetscReal delta_t, PetscReal time_start, PetscReal time_end, PetscInt M, PetscInt head_bc, PetscInt tail_bc,
	PetscReal hbc_h, PetscReal hbc_t, PetscReal gg_h, PetscReal gg_t,
	PetscBool is_record, PetscInt record_frq );

void MatBC( MPI_Comm comm, Mat A, PetscInt M, PetscInt head_bc, PetscInt tail_bc );

// HDF5 tools
void HDF5_Write();

void HDF5_Read();

int main( int argc, char **argv )
{
	// initialization
	PetscInitialize(&argc, &argv, (char*)0, help);
	MPI_Comm comm = PETSC_COMM_WORLD;

	// variables
	PetscInt	M = 10;
	PetscReal	kappa = 1.0;	// thermal conductivity
	PetscReal	rho = 1.0;	// density
	PetscReal	cc = 1.0;	// specific heat capacity
	PetscReal	delta_t = 0.1;
	PetscReal	delta_x = 0.1;
	PetscReal	time_start = 0.0;
	PetscReal	time_end = 1.0;
	PetscReal       gg_h = 0.0;	// prescribed temperature at x=0
	PetscReal       gg_t = 0.0;	// prescribed temperature at x=1
	PetscReal       hh_h = 1.0;	// heat flux at x=0
	PetscReal       hh_t = 1.0;	// heat flux at x=1
	PetscInt	head_bc = 0;	// boundary condition at x=0: 0 for prescribed temperature, 1 for heat flux.
	PetscInt	tail_bc = 0;	// boundary condition at x=0: 0 for prescribed temperature, 1 for heat flux.
	PetscBool	is_explicit = PETSC_FALSE;
	PetscBool	is_record = PETSC_TRUE;
	PetscInt	record_frq = 10;
	PetscBool	is_restart = PETSC_FALSE;


	// get option
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-mesh_size",	&M, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-kappa",	&kappa, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-rho",	&rho, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-capacity",	&cc, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-time_step",	&delta_t, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-time_end",	&time_end, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-gg_head",	&gg_h, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-gg_tail",	&gg_t, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-hh_head",	&hh_h, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-hh_tail",	&hh_t, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-head_bc",	&head_bc, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-tail_bc",	&tail_bc, PETSC_NULL);
	PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-is_explicit",	&is_explicit, PETSC_NULL);
	PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-is_record",	&is_record, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-record_frq",	&record_frq, PETSC_NULL);
	PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-is_restart",	&is_restart, PETSC_NULL);

	// hdf5 read
	Vec	u_re;
	VecCreate(comm, &u_re);
	VecSetSizes(u_re, PETSC_DECIDE, M+1);
	VecSetFromOptions(u_re);
	if (is_restart) {
		PetscPrintf(comm, "********** Restart: data loading from hdf5file... **********\n");
		Vec	temp;
		VecCreate(comm, &temp);
		VecSetSizes(temp, PETSC_DECIDE, 3);
		VecSetType(temp, VECSEQ);

		PetscViewer	viewer;
		PetscViewerHDF5Open(comm, "SOL_re", FILE_MODE_READ, &viewer);
		PetscObjectSetName((PetscObject)u_re, "current_u");
		VecLoad(u_re, viewer);
		PetscObjectSetName((PetscObject)temp, "parameters");
		VecLoad(temp, viewer);
		PetscViewerDestroy(&viewer);

		PetscScalar	*value;
		VecGetArray(temp, &value);
		VecView(temp, PETSC_VIEWER_STDOUT_(comm));
		M		= value[0];
		delta_t		= value[1];
		time_start	= value[2];
		PetscPrintf(comm, "-mesh_size %D\n", M);
		PetscPrintf(comm, "-time_step %g\n", delta_t);
		PetscPrintf(comm, "-time_start %g\n", time_start);

		VecDestroy(&temp);
	}

	// parameter check
	delta_x = 1.0 / M;

	// constant scalar
	const double para1 = (delta_t * kappa) / (delta_x * delta_x * rho * cc);
	const double para2 = delta_t / (rho * cc);
	PetscReal hbc_h = (delta_x * hh_h) / kappa;
	PetscReal hbc_t = (delta_x * hh_t) / kappa;
	const double pi = PETSC_PI; //3.141592653589793;
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
		PetscScalar	value[3];
		if (is_explicit) {
			value[0] = 1.0*para1;
			value[1] = 1.0-2.0*para1;
			value[2] = 1.0*para1;
		}
		else {
			value[0] = -1.0*para1;
			value[1] = 2.0*para1+1.0;
			value[2] = -1.0*para1;
		}

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

	if (is_restart) VecCopy(u_re, uu);

	// boundary conditions
	MatBC( comm, A, M, head_bc, tail_bc );

	// solver
	KSP		ksp;
	PC		pc;
	KSPCreate(comm, &ksp);
	Vec	u_new;
	VecDuplicate(uu, &u_new);
	
	if (is_explicit) {
		// explicit solver
		FEuler( comm, A, u_new, uu, ff, M, delta_t, time_start, time_end, head_bc, tail_bc, hbc_h, hbc_t, gg_h, gg_t,
				is_record, record_frq );
	}
	else {
		// implicit solver
		BEuler( comm, A, u_new, uu, ff, ksp, pc, delta_t, time_start, time_end, M, head_bc, tail_bc, hbc_h, hbc_t,
			       	gg_h, hh_t, is_record, record_frq );
	}

	VecView(u_new, PETSC_VIEWER_STDOUT_(comm));

	// destory
	KSPDestroy(&ksp);
	VecDestroy(&uu);
	VecDestroy(&u_new);
	VecDestroy(&u_re);
	VecDestroy(&ff);
	MatDestroy(&A);

	PetscFinalize();
	delete [] coor_x; delete [] h_src; delete [] u_0;
	return 0;
}

void BEuler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, KSP ksp, PC pc,
	PetscReal delta_t, PetscReal time_start, PetscReal time_end, PetscInt M, PetscInt head_bc, PetscInt tail_bc,
	PetscReal hbc_h, PetscReal hbc_t, PetscReal gg_h, PetscReal gg_t,
	PetscBool is_record, PetscInt record_frq )
{
	PetscInt	N = (time_end - time_start) / delta_t;
	PetscInt	its;
	PetscInt	zero = 0;
	PetscReal	rnorm;
	PetscReal	time = time_start;

	// ksp set
	KSPSetOperators(ksp, A, A);
	KSPSetType(ksp, KSPGMRES);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCJACOBI);
	KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetFromOptions(ksp);

	// solver
	PetscPrintf(comm, "======> time: %g\n", (double)time);
	for(int ii=1; ii<N+1; ii++) {
		time += delta_t;
		VecAXPY(b, 1.0, f);
		if (head_bc) {
			VecSetValues(b, 1, &zero, &hbc_h, INSERT_VALUES);
		}
		else {
			VecSetValues(b, 1, &zero, &gg_h, INSERT_VALUES);
		}
		if (tail_bc) {
			VecSetValues(b, 1, &M, &hbc_t, INSERT_VALUES);
		}
		else {
			VecSetValues(b, 1, &M, &gg_h, INSERT_VALUES);
		}
		VecAssemblyBegin(b);
		VecAssemblyEnd(b);
		KSPSolve(ksp, b, x);
		VecCopy(x, b);
		KSPMonitor(ksp, its, rnorm);
		PetscPrintf(comm, "        step solver done.\n");
		PetscPrintf(comm, "        KSP its: %D\t r_norm: %g\n", its, (double)rnorm);
		PetscPrintf(comm, "======> time: %g\n", (double)time);

		// hdf5 write
		if (is_record && ii%record_frq==0) {
			PetscViewer	viewer;
			char file[10] = "SOL_";
			char num[4];
			sprintf(num, "%d", ii);
			strcat(file, num);
			PetscViewerHDF5Open(comm, file, FILE_MODE_WRITE, &viewer);
			PetscObjectSetName((PetscObject)x, "current_u");
			VecView(x, viewer);
			Vec	temp;
			VecCreate(comm, &temp);
			VecSetSizes(temp, PETSC_DECIDE, 3);
			VecSetType(temp, VECSEQ);
			PetscScalar	value[3];
			PetscInt	index[3];
			value[0] = M; value[1] = delta_t; value[2] = time;
			index[0] = 0; index[1] = 1; index[2] = 2;
			VecSetValues(temp, 3, index, value, INSERT_VALUES);
			VecAssemblyBegin(temp);
			VecAssemblyEnd(temp);
			PetscObjectSetName((PetscObject)temp, "parameters");
			VecView(temp, viewer);
			PetscViewerDestroy(&viewer);
			VecDestroy(&temp);
		}
	}
}

void FEuler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, PetscInt M,
	PetscReal delta_t, PetscReal time_start, PetscReal time_end, PetscInt head_bc, PetscInt tail_bc,
	PetscReal hbc_h, PetscReal hbc_t, PetscReal gg_h, PetscReal gg_t,
	PetscBool is_record, PetscInt record_frq )
{
	PetscInt	N = (time_end - time_start) / delta_t;
	PetscInt	zero = 0;
	PetscReal	time = time_start;

	for(int ii=1; ii<N+1; ii++) {
		time += delta_t;
		VecAXPY(b, 1.0, f);
		if (head_bc) {
			VecSetValues(b, 1, &zero, &hbc_h, INSERT_VALUES);
		}
		else {
			VecSetValues(b, 1, &zero, &gg_h, INSERT_VALUES);
		}
		if (tail_bc) {
			VecSetValues(b, 1, &M, &hbc_t, INSERT_VALUES);
		}
		else {
			VecSetValues(b, 1, &M, &gg_h, INSERT_VALUES);
		}
		VecAssemblyBegin(b);
		VecAssemblyEnd(b);
		MatMult(A, b, x);
		VecCopy(x, b);

		// hdf5 write
		if (is_record && ii%record_frq==0) {
			PetscViewer	viewer;
			char file[10] = "SOL_";
			char num[4];
			sprintf(num, "%d", ii);
			strcat(file, num);
			PetscViewerHDF5Open(comm, file, FILE_MODE_WRITE, &viewer);
			PetscObjectSetName((PetscObject)x, "current_u");
			VecView(x, viewer);
			Vec	temp;
			VecCreate(comm, &temp);
			VecSetSizes(temp, PETSC_DECIDE, 3);
			VecSetType(temp, VECSEQ);
			PetscScalar	value[3];
			PetscInt	index[3];
			value[0] = M; value[1] = delta_t; value[2] = time;
			index[0] = 0; index[1] = 1; index[2] = 2;
			VecSetValues(temp, 3, index, value, INSERT_VALUES);
			VecAssemblyBegin(temp);
			VecAssemblyEnd(temp);
			PetscObjectSetName((PetscObject)temp, "parameters");
			VecView(temp, viewer);
			PetscViewerDestroy(&viewer);
			VecDestroy(&temp);
		}
	}
	PetscPrintf(comm, "======> solver done <======\n");
}

void MatBC( MPI_Comm comm, Mat A, PetscInt M, PetscInt head_bc, PetscInt tail_bc )
{
	if (!head_bc) {
		PetscInt	row = 0;
		PetscInt	col[2] = {0, 1};
		PetscScalar	value[2] = {1.0, 0.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		PetscPrintf(comm, "======> Essential BC at x=0: prescribed temperature.\n");
	}
	else {
		PetscInt	row = 0;
		PetscInt	col[2] = {0, 1};
		PetscScalar	value[2] = {1.0, -1.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		PetscPrintf(comm, "======> Natural BC at x=0: heat flux.\n");
	}
	if (!tail_bc) {
		PetscInt	row = M;
		PetscInt	col[2] = {M-1, M};
		PetscScalar	value[2] = {0.0, 1.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		PetscPrintf(comm, "======> Essential BC at x=0: prescribed temperature.\n");
	}
	else {
		PetscInt	row = M;
		PetscInt	col[2] = {M-1, M};
		PetscScalar	value[2] = {-1.0, 1.0};

		MatSetValues(A, 1, &row, 2, col, value, INSERT_VALUES);
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		PetscPrintf(comm, "======> Natural BC at x=1: heat flux.\n");
	}
//	MatView(A, PETSC_VIEWER_STDOUT_WORLD);
}
