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
 
#include "dpMemory.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpMemory::dpMemory(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;

	kernelString = 																																						 "\n"
	" // OpenCL Kernel Function for element by element copy                                     \n"        
	"__kernel void Memory(__global float* a, __global float* b, int iNumElements){              \n"                                                                            
	"    int iGID = get_global_id(0);                                                           \n"        
	"                                                                                           \n"        
	"    if (iGID < iNumElements)                                                               \n"        
	"        b[iGID] = a[iGID];                                                                }\n";
	
	name = "Memory";
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	err =clBuildProgram(program, 0, NULL, NULL, NULL, NULL); 
	clErrChk(err);
	programCheck(err, context, program);
	kernel = clCreateKernel(program, "Memory", &err); clErrChk(err);
}

void dpMemory::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = xLocal;
	localSize[1] = 1;
	localSize[2] = 1;
	
	Asize = dataMB*1048576/sizeof(cl_float);
	
	MB= Asize*sizeof(cl_float)/1048576;
}

void dpMemory::init(){
	dataParameters.push_back(Asize);
	dataNames.push_back("nElements");
	
	// Allocate and initialize host arrays 
	srcA = (float *)malloc(sizeof(cl_float) * Asize);
	dst = (float *)malloc(sizeof(cl_float) * Asize);
	generateArray(srcA, Asize);
}

void dpMemory::memoryCopyOut(){
	cmDevSrcA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * Asize, NULL, &err); clErrChk(err);
	cmDevDst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * Asize, NULL, &err); clErrChk(err);
	
	clErrChk(clEnqueueWriteBuffer(queue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * Asize, srcA, 0, NULL, NULL));
	clFinish(queue);
}

void dpMemory::plan(){
	// Set the Argument values
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&cmDevSrcA));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&cmDevDst));
	clErrChk(clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&Asize));
	globalSize[0] = Asize;
}

int dpMemory::execute(){
	err=clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL);
	clErrChk(err);
	if(err<0)
		return -1;
	clFinish(queue);
	return 0;
}

void dpMemory::memoryCopyIn(){
	clErrChk(clEnqueueReadBuffer(queue, cmDevDst, CL_TRUE, 0, sizeof(cl_float) * Asize, dst, 0, NULL, NULL));
	clFinish(queue);
}

void dpMemory::cleanUp(){
	// Cleanup and leave
	clReleaseKernel(kernel);  
	clReleaseProgram(program);
	clReleaseMemObject(cmDevSrcA);
	clReleaseMemObject(cmDevDst);
	free(srcA);
	free (dst);
}

void dpMemory::generateArray(float *A, int N){
	int i;
	srand(time(NULL));
	for (i=0; i < N; i++){
		A[i]=rand() / (RAND_MAX/99999.9 + 1);
	}
}