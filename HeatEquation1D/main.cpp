static char help[] = "To solve the transient heat equation in a one-dimensional domain.\n\n";

#include <petscksp.h>
#include "hdf5.h"

// solver tools
void FEuler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, PetscInt M,
	PetscReal delta_t, PetscReal time_end, PetscInt head_bc, PetscInt tail_bc,
	PetscReal hbc_h, PetscReal hbc_t, PetscReal gg_h, PetscReal gg_t );

void BEuler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, KSP ksp, PC pc,
	PetscReal delta_t, PetscReal time_end, PetscInt M, PetscInt head_bc, PetscInt tail_bc,
	PetscReal hbc_h, PetscReal hbc_t, PetscReal gg_h, PetscReal gg_t );

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
	PetscReal	time_end = 1.0;
	PetscReal       gg_h = 0.0;	// prescribed temperature at x=0
	PetscReal       gg_t = 0.0;	// prescribed temperature at x=1
	PetscReal       hh_h = 1.0;	// heat flux at x=0
	PetscReal       hh_t = 1.0;	// heat flux at x=1
	PetscInt	head_bc = 0;	// boundary condition at x=0: 0 for prescribed temperature, 1 for heat flux.
	PetscInt	tail_bc = 0;	// boundary condition at x=0: 0 for prescribed temperature, 1 for heat flux.
	PetscBool	is_explicit = PETSC_FALSE;

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
	MatBC( comm, A, M, head_bc, tail_bc );

	// solver
	KSP		ksp;
	PC		pc;
	KSPCreate(comm, &ksp);
	Vec	u_new;
	VecDuplicate(uu, &u_new);
	
	if (is_explicit) {
		// explicit solver
		FEuler( comm, A, u_new, uu, ff, M, delta_t, time_end, head_bc, tail_bc, hbc_h, hbc_t, gg_h, gg_t );
	}
	else {
		// implicit solver
		BEuler( comm, A, u_new, uu, ff, ksp, pc, delta_t, time_end, M, head_bc, tail_bc, hbc_h, hbc_t, gg_h, hh_t );
	}

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

void BEuler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, KSP ksp, PC pc,
	PetscReal delta_t, PetscReal time_end, PetscInt M, PetscInt head_bc, PetscInt tail_bc,
	PetscReal hbc_h, PetscReal hbc_t, PetscReal gg_h, PetscReal gg_t )
{
	PetscInt	N = time_end / delta_t;
	PetscInt	its;
	PetscInt	zero = 0;
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
	PetscPrintf(comm, "======> time_index: %D\t time: %g\n", (int)zero, (double)time);
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
//		VecView(x, PETSC_VIEWER_STDOUT_(comm));
		KSPMonitor(ksp, its, rnorm);
		PetscPrintf(comm, "        step solver done.\n");
		PetscPrintf(comm, "        KSP its: %D\t r_norm: %g\n", its, (double)rnorm);
		PetscPrintf(comm, "======> time_index: %D\t time: %g\n", ii, (double)time);
	}
}

void FEuler( MPI_Comm comm, Mat A, Vec x, Vec b, Vec f, PetscInt M,
	PetscReal delta_t, PetscReal time_end, PetscInt head_bc, PetscInt tail_bc,
	PetscReal hbc_h, PetscReal hbc_t, PetscReal gg_h, PetscReal gg_t )
{
	PetscInt	N = time_end / delta_t;
	PetscInt	zero = 0;
	PetscReal	time = 0.0;

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

void HDF5_Write_Solution(Vec uu, PetscInt M, const double para1, const double para2, const double ll,
		PetscInt head_bc, PetscInt tail_bc, PetscReal hbc_h, PetscReal hbc_t, PetscReal gg_h, PetscReal gg_t,
		PetscReal delta_t, PetscReal delta_x )
{
	// write current solution
	hid_t	file_id;
	hid_t	dataset_uu, dataset_para1, dataset_para2, dataset_M, dataset_delx, dataset_delt, dataset_ll;
	hid_t	dataset_headbc, dataset_tailbc, dataset_hbch, dataset_hbct, dataset_ggh, dataset_ggt;
	hid_t	dataspace_uu, dataspace_scalar;

	hsize_t	dim_uu[1]; dim_uu[0] = M;
	hsize_t	dim_scalar[1]; dim_scalar[0] = 1;

	PetscScalar	*array;
	VecGetArray(uu, &array);
	const double delx = delta_x;

	char file[10] = "SOL_";
	char num[10] = "0000";
	strcat(file, num);
	file_id = H5Fcreate(file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	dataspace_uu = H5Screate_simple(1, dim_uu, NULL);
	dataspace_scalar = H5Screate_simple(1, dim_scalar, NULL);

	dataset_uu	= H5Dcreate2(file_id, "/Current_solution", H5T_NATIVE_DOUBLE, dataspace_uu, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_para1	= H5Dcreate2(file_id, "/Para1", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_para2	= H5Dcreate2(file_id, "/Para2", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_M	= H5Dcreate2(file_id, "/Matrix_size", H5T_NATIVE_INT, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_delx	= H5Dcreate2(file_id, "/Delta_x", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_delt	= H5Dcreate2(file_id, "/Delta_t", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_ll	= H5Dcreate2(file_id, "/Heat_src_para", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_headbc	= H5Dcreate2(file_id, "/Head_BC", H5T_NATIVE_INT, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_tailbc	= H5Dcreate2(file_id, "/Tail_BC", H5T_NATIVE_INT, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_hbch	= H5Dcreate2(file_id, "/Head_hbc", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_hbct	= H5Dcreate2(file_id, "/Tail_hbc", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_ggh	= H5Dcreate2(file_id, "/Head_gbc", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	dataset_ggt	= H5Dcreate2(file_id, "/Tail_gbc", H5T_NATIVE_DOUBLE, dataspace_scalar, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	

	H5Dwrite(dataset_uu, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, array);
	H5Dwrite(dataset_para1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &para1);
	H5Dwrite(dataset_para2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &para2);
	H5Dwrite(dataset_M, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &M);
	H5Dwrite(dataset_delx, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &delta_x);
	H5Dwrite(dataset_delt, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &delta_t);
	H5Dwrite(dataset_ll, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ll);
	H5Dwrite(dataset_headbc, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &head_bc);
	H5Dwrite(dataset_tailbc, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tail_bc);
	H5Dwrite(dataset_hbch, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &hbc_h);
	H5Dwrite(dataset_hbct, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &hbc_t);
	H5Dwrite(dataset_ggh, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &gg_h);
	H5Dwrite(dataset_ggt, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &gg_t);

	H5Dclose(dataset_uu);
	H5Dclose(dataset_para1);
	H5Dclose(dataset_para2);
	H5Dclose(dataset_M);
	H5Dclose(dataset_delx);
	H5Dclose(dataset_delt);
	H5Dclose(dataset_ll);
	H5Dclose(dataset_headbc);
	H5Dclose(dataset_tailbc);
	H5Dclose(dataset_hbch);
	H5Dclose(dataset_hbct);
	H5Dclose(dataset_ggh);
	H5Dclose(dataset_ggt);
	H5Sclose(dataspace_uu);
	H5Sclose(dataspace_scalar);

	H5Fclose(file_id);
}

void HDF5_Read();
