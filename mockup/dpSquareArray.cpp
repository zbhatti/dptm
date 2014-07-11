#include "dpSquareArray.hpp"
#include "helperFunctions/errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

const char* kernelString = "\n"
"__kernel void squareArray( \n" 
"	__global const float * Ain_d, \n"
"	__global float * Aout_d, \n"
"	const int N){ \n"
"		int idx = get_global_id(0); \n"
"		int idy = get_global_id(1); \n"
"		int idz = get_global_id(2); \n"
"		int Index = idx + idy*get_global_size(0) + idz*get_global_size(1)*get_global_size(2); \n"
"		if (Index<N){ \n"
"			Aout_d[Index] = Ain_d[Index] * Ain_d[Index]; \n"
"		} \n"
"	} \n";

dpSquareArray::dpSquareArray(float* input, int inputSize, cl_context ctx, cl_command_queue q, int xLocal){
	Ain = input; //point to same place argument was pointing to
	Asize = inputSize;
	Aout = new float[Asize];
	if (!Aout)
		fprintf(stderr, "error in dynamic allocation");
	context = ctx;
	queue = q;
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "squareArray", &err); clErrChk(err);
	localSize[0] = xLocal;
	globalSize[0] = Asize;
}

void dpSquareArray::memoryCopyOut(){
	Ain_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err); clErrChk(err);
	Aout_d = clCreateBuffer(context, CL_MEM_READ_WRITE, Asize*sizeof(float), NULL, &err); clErrChk(err);
	clErrChk(clEnqueueWriteBuffer(queue, Ain_d, CL_TRUE, 0, Asize*sizeof(float), Ain, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpSquareArray::plan(){
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &Ain_d));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &Aout_d));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &Asize));
}

void dpSquareArray::execute(){
	clErrChk(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpSquareArray::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, Aout_d, CL_TRUE, 0, Asize*sizeof(float), Aout, 0, NULL, NULL));
	clErrChk(clFinish(queue));
}

void dpSquareArray::cleanUp(){
	//printf("%f^2 = %f\n",Ain[Asize-1],Aout[Asize-1]);
	clErrChk(clReleaseKernel(kernel));
	clErrChk(clReleaseProgram(program));
	clErrChk(clReleaseMemObject(Ain_d));
	clErrChk(clReleaseMemObject(Aout_d));
	delete[] Aout;
}
