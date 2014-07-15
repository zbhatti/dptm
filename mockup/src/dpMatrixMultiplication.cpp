#include "dpMatrixMultiplication.hpp"
#include "helperFunctions/errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpMatrixMultiplication::dpMatrixMultiplication(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;
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
	
}

void dpMatrixMultiplication::init(int xLocal, int yLocal, int zLocal){
	
	N = 4096;
	P = 4096;
	M = 1024;
	szA = N*P;
	szB = P*M;
	szC = N*M;
	
	A = new float[szA];
	B = new float[szB];
	C = new float[szC];
	if (!A || !B || !C)
		fprintf(stderr,"error in dynamic allocation");
	
	localSize[0] = xLocal;
	localSize[1] = yLocal;
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "mmul", &err); clErrChk(err);
	a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * szA, NULL, &err); clErrChk(err);
	b_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * szB, NULL, &err); clErrChk(err);
	c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * szC, NULL, &err); clErrChk(err);
	
}

void dpMatrixMultiplication::memoryCopyOut(){
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
	globalSize[0] = (size_t) M; globalSize[1] = (size_t) N;
}

void dpMatrixMultiplication::execute(){
	clErrChk(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpMatrixMultiplication::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, c_out, CL_TRUE, 0, sizeof(float) * szC, C, 0, NULL, NULL ));
	clErrChk(clFinish(queue));
}

void dpMatrixMultiplication::cleanUp(){
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseMemObject(a_in));
	clErrChk(clReleaseMemObject(b_in));
	clErrChk(clReleaseMemObject(c_out));
	delete[] A;
	delete[] B;
	delete[] C;
}

void dpMatrixMultiplication::generateMatrix(float A[], int height, int width){
	int i, j;
	srand(time(NULL));
	for (j=0; j<height; j++){//rows in A
		for (i=0; i<width; i++){//cols in A
			A[i + width*j] = rand() / (RAND_MAX/99999.9 + 1);
		}
	}
}