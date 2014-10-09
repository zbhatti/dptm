#include "dpMatrixMultiplication.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpMatrixMultiplication::dpMatrixMultiplication(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;

	name = "MatrixMultiplication";
	kernelString = "\n"
		"__kernel void mmul( \n" 
		"const int Mdim, \n" 
		"const int Ndim,  \n" 
		"const int Pdim, \n" 
		"__global float* A, \n" 
		"__global float* B,  \n" 
		"__global float* C){ \n" 
		"	int k; \n" 
		"	int i = get_global_id(0); //get column \n" 
		"	int j = get_global_id(1); //get row \n" 
		"	float tmp = 0.0f; \n" 
		"	for (k=0; k<Pdim; k++)  \n" 
		"		tmp += A[j*Pdim+k] * B[k*Mdim+i];  \n" 
		"	C[j*Mdim+i] = tmp; \n" 
		"} \n";
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "mmul", &err); clErrChk(err);
}

void dpMatrixMultiplication::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = xLocal;
	localSize[1] = yLocal;
	localSize[2] = 1;
	
	for (int i =0; pow(2,i)*pow(2,i)*sizeof(float)/(float) 1048576 <= dataMB;i++){
		N = pow(2,i);
		P = pow(2,i);
		M = pow(2,i);
	}
	
	N=(int)sqrt(1048576*dataMB/sizeof(float));
	P=N;
	M=N;
	
	//calculating total data as MB of matrix A only. MBof(B) <= MBof(A)
	MB=(N*P*sizeof(float))/1048576;
	
}

void dpMatrixMultiplication::init(){
	szA = N*P;
	szB = P*M;
	szC = N*M;
	
	dataParameters.push_back(N);
	dataParameters.push_back(P);
	dataParameters.push_back(M);
	
	dataNames.push_back("N");
	dataNames.push_back("P");
	dataNames.push_back("M");
	
	A = new float[szA];
	B = new float[szB];
	C = new float[szC];
	if (!A || !B || !C)
		fprintf(stderr,"error in dynamic allocation");
	
	generateMatrix(A,N,P);
	generateMatrix(B,P,M);

}

void dpMatrixMultiplication::memoryCopyOut(){

	a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * szA, NULL, &err); clErrChk(err);
	b_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * szB, NULL, &err); clErrChk(err);
	c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * szC, NULL, &err); clErrChk(err);

	clErrChk(clEnqueueWriteBuffer(queue, a_in, CL_TRUE, 0, sizeof(float) * szA, A, 0, NULL, NULL)); 
	clErrChk(clEnqueueWriteBuffer(queue, b_in, CL_TRUE, 0, sizeof(float) * szB, B, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpMatrixMultiplication::plan(){
	clErrChk(clSetKernelArg(kernel, 0, sizeof(int), &M)); 
	clErrChk(clSetKernelArg(kernel, 1, sizeof(int), &N)); 
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &P)); 
	clErrChk(clSetKernelArg(kernel, 3, sizeof(cl_mem), &a_in)); 
	clErrChk(clSetKernelArg(kernel, 4, sizeof(cl_mem), &b_in)); 
	clErrChk(clSetKernelArg(kernel, 5, sizeof(cl_mem), &c_out));
	globalSize[0] = (size_t) M; 
	globalSize[1] = (size_t) N;
}

int dpMatrixMultiplication::execute(){
	err=clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
	clErrChk(err);
	if(err<0)
		return -1;
	clErrChk(clFinish(queue));
	return 0;
}

void dpMatrixMultiplication::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(float) * szC, C, 0, NULL, NULL ));
	clErrChk(clFinish(queue));
}

void dpMatrixMultiplication::cleanUp(){
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(a_in));
	clErrChk(clReleaseMemObject(b_in));
	clErrChk(clReleaseMemObject(c_out));
	delete[] A;
	delete[] B;
	delete[] C;
}

void dpMatrixMultiplication::generateMatrix(float *A, int height, int width){
	int i, j;
	srand(time(NULL));
	for (j=0; j<height; j++){//rows in A
		for (i=0; i<width; i++){//cols in A
			A[i + width*j] = rand() / (RAND_MAX/99999.9 + 1);
		}
	}
}