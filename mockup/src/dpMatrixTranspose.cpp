/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 */
 
/* Matrix transpose with Cuda 
 * Host code.

 * This example transposes arbitrary-size matrices.  It compares a naive
 * transpose kernel that suffers from non-coalesced writes, to an optimized
 * transpose with fully coalesced memory access and no bank conflicts.  On 
 * a G80 GPU, the optimized transpose can be more than 10x faster for large
 * matrices.
 */

#include "dpMatrixTranspose.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpMatrixTranspose::dpMatrixTranspose(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = TWO_D;

	name = "MatrixTranspose";
	kernelString = "\n"
		"__kernel void simple_copy(__global float *odata, __global float* idata, int offset, int width, int height) \n"
		"{                                                                                                          \n"
		"    unsigned int xIndex = get_global_id(0);                                                                \n"
		"    unsigned int yIndex = get_global_id(1);                                                                \n"
		"                                                                                                           \n"
		"    if (xIndex + offset < width && yIndex < height)                                                        \n"
		"    {                                                                                                      \n"
		"        unsigned int index_in  = xIndex + offset + width * yIndex;                                         \n"
		"        odata[index_in] = idata[index_in];                                                                 \n"
		"    }                                                                                                      \n"
		"}                                                                                                          \n"
		"                                                                                                           \n";
		
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL));
	kernel = clCreateKernel(program, "simple_copy", &err); clErrChk(err);
	
}

void dpMatrixTranspose::setup(int dataMB, int xLocal, int yLocal, int zLocal){

	localSize[0] = xLocal;
	localSize[1] = yLocal;
	localSize[2] = 1;

	for (int i =0; pow(2,i)*pow(2,i)*sizeof(float)/(float) 1048576 <= dataMB;i++){
		size_x = pow(2,i);
		size_y = pow(2,i);
	}
	
	MB = size_x*size_y*sizeof(float)/(float) 1048576;
}

void dpMatrixTranspose::init(){

	dataParameters.push_back(size_x);
	dataParameters.push_back(size_y);
	dataNames.push_back("width");
	dataNames.push_back("height");
	
	mem_size = sizeof(float) * size_x * size_y;
	// allocate and initalize host memory
	h_idata = (float*)malloc(mem_size);
	h_odata = (float*)malloc(mem_size);
	generateMatrix(h_idata, size_x, size_y);

}

void dpMatrixTranspose::memoryCopyOut(){

	d_idata = clCreateBuffer(context, CL_MEM_READ_ONLY, size_x*size_y*sizeof(float), NULL, &err); clErrChk(err);
	d_odata = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_x*size_y*sizeof(float), NULL, &err); clErrChk(err);
	
	clErrChk(clEnqueueWriteBuffer(queue, d_idata, CL_FALSE, 0, size_x*size_y*sizeof(float), h_idata, 0, NULL, NULL));
	clFinish(queue);
}

void dpMatrixTranspose::plan(){
		size_t offset = 0;
		clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &d_odata)); //need to double check the pointers
		clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &d_idata));
		clErrChk(clSetKernelArg(kernel, 2, sizeof(int), &offset));
		clErrChk(clSetKernelArg(kernel, 3, sizeof(int), &size_x));
		clErrChk(clSetKernelArg(kernel, 4, sizeof(int), &size_y));
    globalSize[0] = size_x;
    globalSize[1] = size_y;
}

int dpMatrixTranspose::execute(){
		err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
		clErrChk(err);
		if(err<0)
			return -1;
		clFinish(queue);
		return 0;
}

void dpMatrixTranspose::memoryCopyIn(){
		clErrChk(clEnqueueReadBuffer(queue, d_odata, CL_TRUE, 0, size_x * size_y * sizeof(float), h_odata, 0, NULL, NULL));
		clFinish(queue);
}

void dpMatrixTranspose::cleanUp(){
	free(h_idata);
	free(h_odata);
	clReleaseProgram(program);
	clReleaseMemObject(d_idata);
	clReleaseMemObject(d_odata);
	clReleaseKernel(kernel);
}

void dpMatrixTranspose::generateMatrix(float *A, int height, int width){
	int i, j;
	srand(time(NULL));
	for (j=0; j<height; j++){//rows in A
		for (i=0; i<width; i++){//cols in A
			A[i + width*j] = rand() / (RAND_MAX/99999.9 + 1);
		}
	}
}

