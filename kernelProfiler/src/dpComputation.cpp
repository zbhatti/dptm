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
 
#include "dpComputation.hpp"
#include "errorCheck.hpp"
#define clErrChk(ans) { clAssert((ans), __FILE__, __LINE__); }

dpComputation::dpComputation(cl_context ctx, cl_command_queue q){
	context = ctx;
	queue = q;
	workDimension = ONE_D;

	kernelString = "\n"
	" // OpenCL kernel for local computation        \n"
	"__kernel void Computation(float a, float b)   	\n"
	"{                                              \n"
	"    float c = a*b;                            }\n";
	
	name = "Computation";
	
	program = clCreateProgramWithSource(context, 1, (const char **) &kernelString, NULL, &err); clErrChk(err);
	clErrChk(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	kernel = clCreateKernel(program, "Computation", &err); clErrChk(err);
}

void dpComputation::setup(int dataMB, int xLocal, int yLocal, int zLocal){
	localSize[0] = xLocal;
	localSize[1] = 1;
	localSize[2] = 1;
	
	MB= dataMB;
}

void dpComputation::init(){
	// Allocate and initialize host arrays 
	srand(time(NULL));
	a = rand() / (RAND_MAX/1000);
	b = rand() / (RAND_MAX/1000);
	
	dataParameters.push_back(a);
	dataParameters.push_back(b);
	dataNames.push_back("a");
	dataNames.push_back("b");
}

void dpComputation::memoryCopyOut(){}

void dpComputation::plan(){
	// Set the Argument values
	clErrChk(clSetKernelArg(kernel, 0, sizeof(cl_float), (void*)&a));
	clErrChk(clSetKernelArg(kernel, 1, sizeof(cl_float), (void*)&b));
	globalSize[0] = 512;
}

int dpComputation::execute(){
	err=clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, localSize, 0, NULL, NULL);
	clErrChk(err);
	if(err<0)
		return -1;
	clFinish(queue);
	return 0;
}

void dpComputation::memoryCopyIn(){}

void dpComputation::cleanUp(){
	// Cleanup and leave
	clReleaseKernel(kernel);  
	clReleaseProgram(program);
}