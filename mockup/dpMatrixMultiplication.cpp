#include "dpMatrixMultiplication.hpp"
#include "helperFunctions/errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }



//source: http://www.cs.bris.ac.uk/home/simonm/workshops/OpenCL_lecture3.pdf
const char* kernelS = "\n"
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

dpMatrixMultiplication::dpMatrixMultiplication(float* M1, float* M2, int M1Height, int M1Width, int M2Width, cl_context ctx, cl_command_queue q, int xLocal, int yLocal){
	
	A = M1;
	B = M2;
	N = M1Height;
	P = M1Width;
	M = M2Width;
	context = ctx;
	queue = q;
	localSize[0] = xLocal;
	localSize[1] = yLocal;
	szA = N*P; 
	szB = P*M; 
	szC = N*M; 
	
	C = new float[szC];
	if (!C)
		fprintf(stderr,"error in dynamic allocation");
		
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelS, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "mmul", &err); clErrChk(err);
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
	delete[] C;
}