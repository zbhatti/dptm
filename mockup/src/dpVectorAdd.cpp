/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
// *********************************************************************
// oclVectorAdd Notes:  
//
// A simple OpenCL API demo application that implements 
// element by element vector addition between 2 float arrays. 
// *********************************************************************
 
#include "dpVectorAdd.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpVectorAdd::dpVectorAdd(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;
	kernelString = "\n"
	" // OpenCL Kernel Function for element by element vector addition                                                \n"
	"__kernel void VectorAdd(__global const float* a, __global const float* b, __global float* c, int iNumElements)   \n"
	"{                                                                                                                \n"
	"    // get index into global data array                                                                          \n"
	"    int iGID = get_global_id(0);                                                                                 \n"
	"                                                                                                                 \n"
	"    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code                           \n"
	"    if (iGID >= iNumElements)                                                                                    \n"
	"    {                                                                                                            \n"
	"        return;                                                                                                  \n"
	"    }                                                                                                            \n"
	"                                                                                                                 \n"
	"    // add the vector elements                                                                                   \n"
	"    c[iGID] = a[iGID] + b[iGID];                                                                                 \n"
	"}                                                                                                                \n";
	name = "VectorAdd";
	
}

void dpVectorAdd::init(int xLocal,int yLocal,int zLocal){

	localSize[0] = xLocal;
	localSize[1] = yLocal;
	localSize[2] = zLocal;
	
	iNumElements = 262144;
	
	dataParameters.push_back(iNumElements);
	dataNames.push_back("nElements");
	
	// Allocate and initialize host arrays 
	srcA = (float *)malloc(sizeof(cl_float) * iNumElements);
	srcB = (float *)malloc(sizeof(cl_float) * iNumElements);
	dst = (float *)malloc(sizeof(cl_float) * iNumElements);
	generateArray(srcA, iNumElements);
	generateArray(srcB, iNumElements);

	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL));
	kernel = clCreateKernel(program, "VectorAdd", &err); clErrChk(err);
	
}

void dpVectorAdd::memoryCopyOut(){
	cmDevSrcA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * iNumElements, NULL, &err); clErrChk(err);
	cmDevSrcB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * iNumElements, NULL, &err); clErrChk(err);
	cmDevDst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * iNumElements, NULL, &err); clErrChk(err);
	
	clErrChk(clEnqueueWriteBuffer(queue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * iNumElements, srcA, 0, NULL, NULL));
	clErrChk(clEnqueueWriteBuffer(queue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_float) * iNumElements, srcB, 0, NULL, NULL));
	clFinish(queue);
}

void dpVectorAdd::plan(){
	// Set the Argument values
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&cmDevSrcB));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&cmDevDst));
	clErrChk(clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&iNumElements));
	globalSize[0] = iNumElements;
}

void dpVectorAdd::execute(){
	clErrChk(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL));
	clFinish(queue);
}

void dpVectorAdd::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, cmDevDst, CL_TRUE, 0, sizeof(cl_float) * iNumElements, dst, 0, NULL, NULL));
	clFinish(queue);
}

void dpVectorAdd::cleanUp(){
	// Cleanup and leave
	clReleaseKernel(kernel);  
	clReleaseProgram(program);
	clReleaseMemObject(cmDevSrcA);
	clReleaseMemObject(cmDevSrcB);
	clReleaseMemObject(cmDevDst);
	free(srcA); 
	free(srcB);
	free (dst);
}

void dpVectorAdd::generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}